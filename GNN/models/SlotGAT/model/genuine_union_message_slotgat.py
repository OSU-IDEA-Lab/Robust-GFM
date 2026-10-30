"""Direct union-message-equivalent conditional SlotGAT for Freebase NC.

The raw context plan remains variant-specific.  Every accepted context activates
one shared destination-CSR schedule reproducing the direct ordered-pair records
of the physical Exact-2 SlotGAT graph, its generated missing-reverse records,
and self records.
The layer therefore normalizes attention over the same incoming records as the
union graph rather than replacing a shortcut neighborhood by one factorized
path aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from freebase_block_conditional import FreebaseBlockContextPlan, NodeTypeLayout
from freebase_union_messages import UnionMessageSchedule
from genuine_block_conditional_slotgat import (
    GlobalSlotAttention,
    sparse_identity_aware_linear,
)


@dataclass(frozen=True)
class _RuntimeUnionMessages:
    source: torch.Tensor
    channel: torch.Tensor
    blocks: tuple[tuple[int, int, int, int, torch.Tensor], ...]
    message_program_hash: str


def _segment_reduce_available() -> bool:
    return hasattr(torch, "segment_reduce")


def _segment_softmax_and_sum(
    *,
    messages: torch.Tensor,
    logits: torch.Tensor,
    degrees: torch.Tensor,
    attention_dropout: float,
    training: bool,
) -> torch.Tensor:
    """Softmax over each complete destination segment and sum its messages."""
    if int(degrees.sum().item()) != int(messages.shape[0]):
        raise ValueError("Destination degrees do not cover the message block")
    if bool(torch.any(degrees <= 0)):
        raise ValueError("Every destination block row must contain a self message")

    if _segment_reduce_available():
        maxima = torch.segment_reduce(logits, reduce="max", lengths=degrees, axis=0)
        centered = logits - torch.repeat_interleave(maxima, degrees, dim=0)
        exponentials = torch.exp(centered)
        denominators = torch.segment_reduce(
            exponentials, reduce="sum", lengths=degrees, axis=0
        )
        attention = exponentials / torch.repeat_interleave(
            denominators, degrees, dim=0
        )
        attention = F.dropout(
            attention, p=float(attention_dropout), training=training
        )
        weighted = attention[:, None, :, None] * messages
        return torch.segment_reduce(
            weighted, reduce="sum", lengths=degrees, axis=0
        )

    # Compatibility fallback for older PyTorch releases.  Rows with equal
    # indegree are reduced together; all incoming records for a destination are
    # still normalized jointly.
    outputs = messages.new_empty(
        (degrees.numel(),) + tuple(messages.shape[1:])
    )
    starts = torch.cumsum(degrees, dim=0) - degrees
    for degree_value in torch.unique(degrees, sorted=True).detach().cpu().tolist():
        degree = int(degree_value)
        nodes = torch.nonzero(degrees == degree, as_tuple=False).flatten()
        positions = starts[nodes, None] + torch.arange(
            degree, dtype=torch.int64, device=degrees.device
        )[None, :]
        bucket_logits = logits.index_select(0, positions.reshape(-1)).reshape(
            nodes.numel(), degree, logits.shape[1]
        )
        attention = torch.softmax(bucket_logits, dim=1)
        attention = F.dropout(
            attention, p=float(attention_dropout), training=training
        )
        bucket_messages = messages.index_select(0, positions.reshape(-1)).reshape(
            nodes.numel(), degree, *messages.shape[1:]
        )
        aggregate = (
            attention[:, :, None, :, None] * bucket_messages
        ).sum(dim=1)
        outputs.index_copy_(0, nodes, aggregate)
    return outputs


class ChunkedDirectUnionSlotLayer(nn.Module):
    """One relation-aware SlotGAT layer over exact direct union messages."""

    def __init__(
        self,
        *,
        num_slots: int,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        num_channels: int,
        edge_dim: int,
        feature_dropout: float,
        attention_dropout: float,
        negative_slope: float,
        activation: bool,
        residual: bool,
        checkpoint_chunks: bool,
        parity: bool = False,
    ) -> None:
        super().__init__()
        if int(output_dim) % int(num_heads) != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        # ``parity``: the upstream slotGAT details that the IMDb attribution
        # (model/parity_slotgat.py) found to matter or
        # that are free -- no per-slot projection bias, upstream edge-feature path,
        # xavier_normal(gain sqrt 2) init, projected residual where widths differ.
        # NOT ported: the attention residual (needs every message's attention from
        # the previous layer -- 750M x 8 floats on the union schedule; worth nothing
        # on IMDb) and the upstream width (hidden*heads per slot; 8x the messages).
        self.parity = bool(parity)
        self.num_slots = int(num_slots)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.output_dim // self.num_heads
        self.num_channels = int(num_channels)
        self.feature_dropout = float(feature_dropout)
        self.attention_dropout = float(attention_dropout)
        self.negative_slope = float(negative_slope)
        self.use_activation = bool(activation)
        self.use_residual = bool(residual and input_dim == output_dim)
        self.residual_projection = None
        if self.parity and residual and input_dim != output_dim:
            self.residual_projection = nn.Parameter(
                torch.empty(self.num_slots, self.input_dim, self.output_dim)
            )
            self.use_residual = True
        self.checkpoint_chunks = bool(checkpoint_chunks)

        self.slot_weight = nn.Parameter(
            torch.empty(
                self.num_slots,
                self.num_heads,
                self.input_dim,
                self.head_dim,
            )
        )
        self.slot_bias = (
            None
            if self.parity
            else nn.Parameter(torch.zeros(self.num_slots, self.num_heads, self.head_dim))
        )
        self.relation_embedding = nn.Parameter(
            torch.empty(self.num_channels, int(edge_dim))
        )
        # upstream: Embedding(E, edge_feats) -> Linear(edge_feats, edge_feats*heads)
        # -> attn_e[heads, edge_feats]; re-implementation: -> heads*head_dim / head_dim
        self.relation_width = int(edge_dim) if self.parity else self.head_dim
        self.relation_projection = nn.Linear(
            int(edge_dim), self.num_heads * self.relation_width, bias=False
        )
        self.attention_source = nn.Parameter(
            torch.empty(self.num_heads, self.num_slots, self.head_dim)
        )
        self.attention_destination = nn.Parameter(
            torch.empty(self.num_heads, self.num_slots, self.head_dim)
        )
        self.attention_relation = nn.Parameter(
            torch.empty(self.num_heads, self.relation_width)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.parity:
            gain = nn.init.calculate_gain("relu")
            # upstream: fc / attn_l / attn_r / attn_e / fc_e / res_fc xavier_normal(gain),
            # edge embedding at nn.Embedding's default N(0, 1)
            nn.init.xavier_normal_(self.slot_weight.view(self.num_slots, self.input_dim, -1), gain=gain)
            nn.init.normal_(self.relation_embedding)
            nn.init.xavier_normal_(self.relation_projection.weight, gain=gain)
            # upstream attention vectors are [1, heads, width]; xavier's fan computation
            # depends on that shape, so draw with it and copy (entries are iid anyway)
            for param in (self.attention_source, self.attention_destination, self.attention_relation):
                draw = torch.empty(1, self.num_heads, param.numel() // self.num_heads)
                nn.init.xavier_normal_(draw, gain=gain)
                with torch.no_grad():
                    param.copy_(draw.reshape(param.shape))
            if self.residual_projection is not None:
                nn.init.xavier_normal_(self.residual_projection, gain=gain)
            return
        nn.init.xavier_uniform_(self.slot_weight)
        nn.init.zeros_(self.slot_bias)
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.relation_projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        nn.init.xavier_uniform_(self.attention_relation)

    def _one_block(
        self,
        projected: torch.Tensor,
        source_scores: torch.Tensor,
        destination_scores: torch.Tensor,
        relation_scores: torch.Tensor,
        source: torch.Tensor,
        channel: torch.Tensor,
        degrees: torch.Tensor,
        destination_start: int,
    ) -> torch.Tensor:
        source_state = projected.index_select(0, source)
        edge_source_scores = source_scores.index_select(0, source)
        destination_nodes = torch.arange(
            int(destination_start),
            int(destination_start) + int(degrees.numel()),
            dtype=torch.int64,
            device=projected.device,
        )
        edge_destination_scores = torch.repeat_interleave(
            destination_scores.index_select(0, destination_nodes),
            degrees,
            dim=0,
        )
        edge_relation_scores = relation_scores.index_select(0, channel)
        logits = F.leaky_relu(
            edge_source_scores + edge_destination_scores + edge_relation_scores,
            negative_slope=self.negative_slope,
        )
        return _segment_softmax_and_sum(
            messages=source_state,
            logits=logits,
            degrees=degrees,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )

    def forward(
        self,
        slots: torch.Tensor,
        runtime: _RuntimeUnionMessages,
    ) -> torch.Tensor:
        residual = slots
        slots = F.dropout(slots, p=self.feature_dropout, training=self.training)
        projected = torch.einsum("ntd,thdk->nthk", slots, self.slot_weight)
        if self.slot_bias is not None:
            projected = projected + self.slot_bias[None, :, :, :]
        source_scores = torch.einsum(
            "nthk,htk->nh", projected, self.attention_source
        )
        destination_scores = torch.einsum(
            "nthk,htk->nh", projected, self.attention_destination
        )
        relation_vectors = self.relation_projection(
            self.relation_embedding
        ).view(self.num_channels, self.num_heads, self.relation_width)
        relation_scores = torch.einsum(
            "chk,hk->ch", relation_vectors, self.attention_relation
        )

        output_chunks: list[torch.Tensor] = []
        for destination_start, destination_stop, edge_start, edge_stop, degrees in runtime.blocks:
            source = runtime.source[edge_start:edge_stop]
            channel = runtime.channel[edge_start:edge_stop]

            # The loop variables MUST be bound as defaults. With
            # use_reentrant=False, `checkpoint` re-executes this function during
            # the backward pass -- after the loop has finished -- so a closure
            # over `source`/`channel`/`degrees`/`destination_start` would
            # recompute every block using the LAST block's tensors. That is a
            # silent wrong-gradient bug whenever the shapes happen to agree, and
            # a hard failure when they do not:
            #   saved metadata: shape [249405, 8, 1, 8]
            #   recomputed metadata: shape [34932, 8, 1, 8]
            def compute(
                projected_value: torch.Tensor,
                source_score_value: torch.Tensor,
                destination_score_value: torch.Tensor,
                relation_score_value: torch.Tensor,
                source=source,
                channel=channel,
                degrees=degrees,
                destination_start=destination_start,
            ) -> torch.Tensor:
                return self._one_block(
                    projected_value,
                    source_score_value,
                    destination_score_value,
                    relation_score_value,
                    source,
                    channel,
                    degrees,
                    destination_start,
                )

            if self.training and self.checkpoint_chunks:
                aggregate = checkpoint(
                    compute,
                    projected,
                    source_scores,
                    destination_scores,
                    relation_scores,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                aggregate = compute(
                    projected,
                    source_scores,
                    destination_scores,
                    relation_scores,
                )
            expected = destination_stop - destination_start
            if aggregate.shape[0] != expected:
                raise RuntimeError("Union message block returned the wrong node count")
            output_chunks.append(aggregate)

        aggregate = torch.cat(output_chunks, dim=0)
        if aggregate.shape[0] != slots.shape[0]:
            raise RuntimeError("Union message blocks do not cover every node")
        output = aggregate.reshape(
            slots.shape[0], self.num_slots, self.output_dim
        )
        if self.use_residual:
            if self.residual_projection is not None:
                # upstream res_fc: projects the (dropout-applied) layer input per slot
                output = output + torch.einsum("ntd,tdo->nto", slots, self.residual_projection)
            else:
                output = output + residual
        if self.use_activation:
            output = F.elu(output)
        return output


class ExactUnionMessageConditionalSlotGAT(nn.Module):
    """Distinct raw contexts with the direct Exact-2 propagation schedule."""

    def __init__(
        self,
        *,
        layout: NodeTypeLayout,
        in_dims: Sequence[int],
        schedule: UnionMessageSchedule,
        num_classes: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        edge_feats: int = 64,
        dropout_feat: float = 0.5,
        dropout_attn: float = 0.2,
        slope: float = 0.05,
        aggregator: str = "SA",
        sa_attention_dim: int = 3,
        require_identity_sparse: bool = True,
        message_chunk_edges: int = 1_000_000,
        checkpoint_chunks: bool = True,
        schedule_on_cpu: bool = False,
        parity: bool = False,
    ) -> None:
        super().__init__()
        self.parity = bool(parity)
        if aggregator != "SA":
            raise ValueError("Union-equivalent Freebase SlotGAT supports aggregator=SA")
        if len(in_dims) != len(layout.counts):
            raise ValueError("Feature type count differs from the context layout")
        if schedule.total_nodes != layout.total:
            raise ValueError("Union message schedule node count differs from layout")
        self.layout = layout
        self.schedule = schedule
        self.num_slots = len(layout.counts)
        self.num_channels = int(schedule.num_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.require_identity_sparse = bool(require_identity_sparse)
        self.message_chunk_edges = int(message_chunk_edges)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.schedule_on_cpu = bool(schedule_on_cpu)
        self.input_projections = nn.ModuleList(
            [nn.Linear(int(in_dim), self.hidden_dim) for in_dim in in_dims]
        )
        layers: list[nn.Module] = []
        current_dim = self.hidden_dim
        for layer_index in range(int(num_layers)):
            is_last = layer_index == int(num_layers) - 1
            output_dim = self.num_classes if is_last else self.hidden_dim
            heads = 1 if is_last else int(num_heads)
            layers.append(
                ChunkedDirectUnionSlotLayer(
                    num_slots=self.num_slots,
                    input_dim=current_dim,
                    output_dim=output_dim,
                    num_heads=heads,
                    num_channels=self.num_channels,
                    edge_dim=int(edge_feats),
                    feature_dropout=float(dropout_feat),
                    attention_dropout=float(dropout_attn),
                    negative_slope=float(slope),
                    activation=not is_last,
                    # upstream: no residual on the input layer, one on every later
                    # layer (projected on the output layer); the re-implementation
                    # had none on the output layer
                    residual=(layer_index > 0) if self.parity else (not is_last),
                    checkpoint_chunks=self.checkpoint_chunks,
                    parity=self.parity,
                )
            )
            current_dim = output_dim
        self.layers = nn.ModuleList(layers)
        if self.parity:
            # upstream "SA" readout: l2byslot -> macroLinear/macroSemanticVec slot
            # attention -> L2-normalised logits (the dominant factor on IMDb: -2.8 pts
            # on the matched control without it)
            self.macroLinear = nn.Linear(self.num_classes, int(sa_attention_dim), bias=True)
            self.macroSemanticVec = nn.Parameter(torch.empty(int(sa_attention_dim), 1))
            self.register_buffer("epsilon", torch.tensor([1e-12]))
            nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414)
            nn.init.normal_(self.macroLinear.bias, std=1.414 * math.sqrt(1 / self.macroLinear.bias.numel()))
            nn.init.normal_(self.macroSemanticVec, std=1)
            for fc in self.input_projections:
                nn.init.xavier_normal_(fc.weight, gain=1.414)
            self.slot_attention = None
        else:
            self.slot_attention = GlobalSlotAttention(
                slot_dim=self.num_classes,
                attention_dim=int(sa_attention_dim),
            )
        self._runtime_cache: dict[str, _RuntimeUnionMessages] = {}

    def _readout(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.parity:
            return self.slot_attention(slots)
        logits = slots / torch.max(torch.norm(slots, dim=2, keepdim=True), self.epsilon)
        slot_scores = (torch.tanh(self.macroLinear(logits)) @ self.macroSemanticVec).mean(0, keepdim=True)
        weights = F.softmax(slot_scores, dim=1)                                   # [1, T, 1]
        out = (logits * weights).sum(1)
        out = out / torch.max(torch.norm(out, dim=1, keepdim=True), self.epsilon)
        return out, weights.flatten()

    def _initialize_slots(self, features_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features_list) != self.num_slots:
            raise ValueError("Feature matrix count differs from node-type count")
        reference = features_list[0]
        slots = torch.zeros(
            self.layout.total,
            self.num_slots,
            self.hidden_dim,
            dtype=reference.dtype,
            device=reference.device,
        )
        for node_type, (features, projection) in enumerate(
            zip(features_list, self.input_projections)
        ):
            transformed = sparse_identity_aware_linear(
                features,
                projection,
                require_identity_sparse=self.require_identity_sparse,
            )
            start = int(self.layout.offsets[node_type])
            stop = start + int(self.layout.counts[node_type])
            if transformed.shape[0] != stop - start:
                raise ValueError("Feature row count differs from node-type layout")
            slots[start:stop, node_type, :] = transformed
        return slots

    def _runtime(self, device: torch.device) -> _RuntimeUnionMessages:
        key = str(device)
        cached = self._runtime_cache.get(key)
        if cached is not None:
            return cached
        target_device = torch.device("cpu") if self.schedule_on_cpu else device
        # The schedule arrays are read-only memory maps. ``torch.tensor`` makes
        # an owned copy and avoids undefined behavior / non-writable-array
        # warnings from ``torch.as_tensor``.
        source = torch.tensor(
            np.asarray(self.schedule.source), dtype=torch.int32, device=target_device
        )
        channel = torch.tensor(
            np.asarray(self.schedule.channel, dtype=np.int32),
            dtype=torch.int32,
            device=target_device,
        )
        if target_device.type == "cpu" and device.type != "cpu":
            # A CPU-resident full schedule is sliced and transferred blockwise.
            source = source.pin_memory()
            channel = channel.pin_memory()
        blocks: list[tuple[int, int, int, int, torch.Tensor]] = []
        for destination_start, destination_stop in self.schedule.destination_blocks(
            self.message_chunk_edges
        ):
            edge_start = int(self.schedule.indptr[destination_start])
            edge_stop = int(self.schedule.indptr[destination_stop])
            degrees = np.diff(
                np.asarray(
                    self.schedule.indptr[destination_start : destination_stop + 1],
                    dtype=np.int64,
                )
            )
            degree_tensor = torch.as_tensor(
                degrees, dtype=torch.int64, device=device
            )
            blocks.append(
                (
                    int(destination_start),
                    int(destination_stop),
                    edge_start,
                    edge_stop,
                    degree_tensor,
                )
            )
        runtime = _RuntimeUnionMessages(
            source=source,
            channel=channel,
            blocks=tuple(blocks),
            message_program_hash=self.schedule.message_program_hash,
        )
        self._runtime_cache[key] = runtime
        return runtime

    def forward(
        self,
        features_list: Sequence[torch.Tensor],
        context_plan: FreebaseBlockContextPlan,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        self.schedule.validate_against_plan(context_plan)
        slots = self._initialize_slots(features_list)
        runtime = self._runtime(slots.device)
        if runtime.source.device != slots.device:
            # Keep the full schedule on CPU and transfer one block at a time.
            # Replace the runtime with a lightweight device view for each layer.
            device_blocks: list[tuple[int, int, int, int, torch.Tensor]] = []
            output = slots
            for layer in self.layers:
                chunks: list[torch.Tensor] = []
                for destination_start, destination_stop, edge_start, edge_stop, degrees in runtime.blocks:
                    local_runtime = _RuntimeUnionMessages(
                        source=runtime.source[edge_start:edge_stop].to(
                            slots.device, non_blocking=True
                        ),
                        channel=runtime.channel[edge_start:edge_stop].to(
                            slots.device, non_blocking=True
                        ),
                        blocks=((
                            destination_start,
                            destination_stop,
                            0,
                            edge_stop - edge_start,
                            degrees,
                        ),),
                        message_program_hash=runtime.message_program_hash,
                    )
                    # A layer expects complete node coverage, so CPU schedule mode
                    # uses the regular path below only when the full schedule is on
                    # device.  Fail closed rather than silently changing semantics.
                    chunks.append(local_runtime.source.new_empty((0,)))
                raise RuntimeError(
                    "--union-schedule-cpu is not supported for the current multi-layer "
                    "implementation; keep the compact int32 schedule on GPU."
                )
        for layer in self.layers:
            slots = layer(slots, runtime)
        logits, slot_weights = self._readout(slots)
        return logits, {
            "slot_attention": slot_weights,
            "union_message_program_hash": self.schedule.message_program_hash,
            "raw_context_plan_hash": context_plan.context_plan_hash,
        }
