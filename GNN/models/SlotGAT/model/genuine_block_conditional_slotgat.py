"""Genuine recoverability-conditional SlotGAT for Freebase NC.

Every forward pass consumes a variant-specific ``FreebaseBlockContextPlan``.
The unchanged input activates a semantic relation through its ordered base-edge
composition.  The Exact-2 input activates the same propagation only after its
physical shortcut relation has been verified against that composition.  The
raw/context plans therefore remain distinct, while the learned propagation
operator is transformation closed.

The implementation never materializes the potentially tens of millions of
shortcut endpoint edges.  A semantic shortcut is evaluated as a factorized
sequence of deterministic relation-wise attention reductions.  The final path
state contributes one record per supported endpoint node.  Ordinary Freebase
base relations continue to contribute one record per physical base edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from freebase_block_conditional import (
    BlockRelation,
    FreebaseBlockContextPlan,
    NodeTypeLayout,
)


def _is_sparse(tensor: torch.Tensor) -> bool:
    return tensor.layout != torch.strided


def _is_sparse_identity(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        return False
    if not _is_sparse(tensor):
        return bool(
            tensor.shape[0] <= 4096
            and torch.equal(
                tensor.detach().cpu(),
                torch.eye(tensor.shape[0], dtype=tensor.dtype),
            )
        )
    coo = tensor.to_sparse_coo().coalesce()
    if coo._nnz() != tensor.shape[0]:
        return False
    indices = coo.indices().detach().cpu()
    values = coo.values().detach().cpu()
    expected = torch.arange(tensor.shape[0], dtype=indices.dtype)
    return bool(
        torch.equal(indices[0], expected)
        and torch.equal(indices[1], expected)
        and torch.all(values == 1)
    )


def sparse_identity_aware_linear(
    tensor: torch.Tensor,
    layer: nn.Linear,
    *,
    require_identity_sparse: bool,
) -> torch.Tensor:
    """Project dense features or sparse identity features deterministically."""
    if not _is_sparse(tensor):
        return layer(tensor)
    if _is_sparse_identity(tensor):
        output = layer.weight.t()
        if layer.bias is not None:
            output = output + layer.bias
        return output
    if require_identity_sparse:
        raise ValueError(
            "Strict deterministic Freebase mode expects sparse feature matrices "
            "to be identities. A non-identity sparse feature was found."
        )
    coo = tensor.to_sparse_coo().coalesce()
    output = torch.sparse.mm(coo, layer.weight.t())
    if layer.bias is not None:
        output = output + layer.bias
    return output


class _DeterministicGather(torch.autograd.Function):
    """Gather rows with deterministic repeated-index gradient accumulation."""

    @staticmethod
    def forward(ctx, value, index, bucket_specs, num_rows):
        ctx.bucket_specs = bucket_specs
        ctx.num_rows = int(num_rows)
        return value.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_output):
        grad_value = torch.zeros(
            (ctx.num_rows,) + tuple(grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        for rows, positions in ctx.bucket_specs:
            gradients = grad_output[positions].sum(dim=1)
            # ``rows`` are unique within and across buckets.
            grad_value.index_copy_(0, rows, gradients)
        return grad_value, None, None, None


def deterministic_gather(
    value: torch.Tensor,
    index: torch.Tensor,
    bucket_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return _DeterministicGather.apply(value, index, bucket_specs, value.shape[0])


def build_index_bucket_specs(
    index: torch.Tensor,
    *,
    num_rows: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Group rows with equal gather multiplicity in canonical row order."""
    if index.dtype != torch.int64:
        raise TypeError("Gather index must be int64")
    if index.numel() == 0:
        return tuple()
    permutation = torch.argsort(index, stable=True)
    sorted_index = index[permutation]
    counts = torch.bincount(sorted_index, minlength=int(num_rows))
    starts = torch.cumsum(counts, dim=0) - counts
    specs: list[tuple[torch.Tensor, torch.Tensor]] = []
    positive_counts = torch.unique(counts[counts > 0], sorted=True)
    for count_value in positive_counts.detach().cpu().tolist():
        count = int(count_value)
        rows = torch.nonzero(counts == count, as_tuple=False).flatten()
        positions = starts[rows, None] + torch.arange(
            count, device=index.device, dtype=torch.int64
        )[None, :]
        specs.append((rows, permutation[positions]))
    return tuple(specs)


@dataclass(frozen=True)
class RuntimeEdgeDirection:
    channel: int
    source: torch.Tensor
    destination: torch.Tensor
    source_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    destination_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    active_destinations: torch.Tensor


@dataclass(frozen=True)
class RuntimePathDirection:
    relation_name: str
    direction: int
    output_channel: int
    step_channels: tuple[int, ...]
    support: torch.Tensor
    support_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class RuntimeContext:
    context_hash: str
    semantic_hash: str
    base_src: torch.Tensor
    base_dst: torch.Tensor
    base_channel: torch.Tensor
    base_source_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    edge_by_channel: tuple[RuntimeEdgeDirection | None, ...]
    path_directions: tuple[RuntimePathDirection, ...]
    all_destination: torch.Tensor
    all_channel: torch.Tensor
    all_destination_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    all_channel_specs: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class GlobalSlotAttention(nn.Module):
    """Learn global importance weights over node-type representation slots."""

    def __init__(self, slot_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(int(slot_dim), int(attention_dim))
        self.query = nn.Parameter(torch.empty(int(attention_dim)))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        bound = 1.0 / math.sqrt(max(1, int(attention_dim)))
        nn.init.uniform_(self.query, -bound, bound)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.tanh(self.projection(slots))
        scores = torch.einsum("ntd,d->nt", scores, self.query)
        weights = torch.softmax(scores.mean(dim=0), dim=0)
        return torch.einsum("t,ntc->nc", weights, slots), weights


class DeterministicRecoverabilityConditionalSlotLayer(nn.Module):
    """One SlotGAT layer over base edges and factorized semantic paths."""

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
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by num_heads={num_heads}"
            )
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

        self.slot_weight = nn.Parameter(
            torch.empty(
                self.num_slots,
                self.num_heads,
                self.input_dim,
                self.head_dim,
            )
        )
        self.slot_bias = nn.Parameter(
            torch.zeros(self.num_slots, self.num_heads, self.head_dim)
        )
        self.relation_embedding = nn.Parameter(
            torch.empty(self.num_channels, int(edge_dim))
        )
        self.relation_projection = nn.Linear(
            int(edge_dim), self.num_heads * self.head_dim, bias=False
        )
        self.attention_source = nn.Parameter(
            torch.empty(self.num_heads, self.num_slots, self.head_dim)
        )
        self.attention_destination = nn.Parameter(
            torch.empty(self.num_heads, self.num_slots, self.head_dim)
        )
        self.attention_relation = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.slot_weight)
        nn.init.zeros_(self.slot_bias)
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.relation_projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        nn.init.xavier_uniform_(self.attention_relation)

    @staticmethod
    def _complete_destination_reduce(
        *,
        messages: torch.Tensor,
        logits: torch.Tensor,
        destination_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_nodes: int,
        attention_dropout: float,
        training: bool,
    ) -> torch.Tensor:
        node_chunks: list[torch.Tensor] = []
        aggregate_chunks: list[torch.Tensor] = []
        for nodes, positions in destination_specs:
            bucket_logits = logits[positions]
            attention = torch.softmax(bucket_logits, dim=1)
            attention = F.dropout(
                attention, p=float(attention_dropout), training=training
            )
            bucket_messages = messages[positions]
            aggregate = (
                attention[:, :, None, :, None] * bucket_messages
            ).sum(dim=1)
            node_chunks.append(nodes)
            aggregate_chunks.append(aggregate)
        all_nodes = torch.cat(node_chunks, dim=0)
        all_aggregates = torch.cat(aggregate_chunks, dim=0)
        order = torch.argsort(all_nodes)
        expected = torch.arange(num_nodes, device=messages.device, dtype=torch.int64)
        if not torch.equal(all_nodes[order], expected):
            raise RuntimeError(
                "Every node must have an incoming record; the self channel should "
                "guarantee complete coverage"
            )
        return all_aggregates[order]

    @staticmethod
    def _partial_destination_reduce(
        *,
        messages: torch.Tensor,
        logits: torch.Tensor,
        destination_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_nodes: int,
        attention_dropout: float,
        training: bool,
    ) -> torch.Tensor:
        """Reduce one relation step and return zeros outside its support."""
        output = torch.zeros(
            (num_nodes,) + tuple(messages.shape[1:]),
            dtype=messages.dtype,
            device=messages.device,
        )
        for nodes, positions in destination_specs:
            bucket_logits = logits[positions]
            attention = torch.softmax(bucket_logits, dim=1)
            attention = F.dropout(
                attention, p=float(attention_dropout), training=training
            )
            bucket_messages = messages[positions]
            aggregate = (
                attention[:, :, None, :, None] * bucket_messages
            ).sum(dim=1)
            output.index_copy_(0, nodes, aggregate)
        return output

    def _relation_step(
        self,
        *,
        state: torch.Tensor,
        destination_query_scores: torch.Tensor,
        relation_scores: torch.Tensor,
        runtime: RuntimeEdgeDirection,
    ) -> torch.Tensor:
        source_state = deterministic_gather(
            state, runtime.source, runtime.source_specs
        )
        state_source_scores = torch.einsum(
            "nthk,htk->nh", state, self.attention_source
        )
        source_scores = deterministic_gather(
            state_source_scores, runtime.source, runtime.source_specs
        )
        destination_scores = deterministic_gather(
            destination_query_scores,
            runtime.destination,
            runtime.destination_specs,
        )
        channel_score = relation_scores[runtime.channel][None, :]
        logits = F.leaky_relu(
            source_scores + destination_scores + channel_score,
            negative_slope=self.negative_slope,
        )
        return self._partial_destination_reduce(
            messages=source_state,
            logits=logits,
            destination_specs=runtime.destination_specs,
            num_nodes=state.shape[0],
            attention_dropout=self.attention_dropout,
            training=self.training,
        )

    def forward(self, slots: torch.Tensor, context: RuntimeContext) -> torch.Tensor:
        residual = slots
        slots = F.dropout(slots, p=self.feature_dropout, training=self.training)
        projected = torch.einsum(
            "ntd,thdk->nthk", slots, self.slot_weight
        ) + self.slot_bias[None, :, :, :]

        source_scores = torch.einsum(
            "nthk,htk->nh", projected, self.attention_source
        )
        destination_scores = torch.einsum(
            "nthk,htk->nh", projected, self.attention_destination
        )
        all_relation_vectors = self.relation_projection(
            self.relation_embedding
        ).view(self.num_channels, self.num_heads, self.head_dim)
        all_relation_scores = torch.einsum(
            "chk,hk->ch", all_relation_vectors, self.attention_relation
        )

        base_messages = deterministic_gather(
            projected, context.base_src, context.base_source_specs
        )
        base_source_scores = deterministic_gather(
            source_scores, context.base_src, context.base_source_specs
        )

        message_chunks = [base_messages]
        source_score_chunks = [base_source_scores]
        for path in context.path_directions:
            state = projected
            for step_channel in path.step_channels:
                runtime = context.edge_by_channel[step_channel]
                if runtime is None:
                    raise RuntimeError(
                        f"No base-edge runtime exists for semantic step channel "
                        f"{step_channel} in {path.relation_name}"
                    )
                state = self._relation_step(
                    state=state,
                    destination_query_scores=destination_scores,
                    relation_scores=all_relation_scores,
                    runtime=runtime,
                )
            path_messages = deterministic_gather(
                state, path.support, path.support_specs
            )
            path_source_scores = torch.einsum(
                "mthk,htk->mh", path_messages, self.attention_source
            )
            message_chunks.append(path_messages)
            source_score_chunks.append(path_source_scores)

        messages = torch.cat(message_chunks, dim=0)
        edge_source_scores = torch.cat(source_score_chunks, dim=0)
        edge_destination_scores = deterministic_gather(
            destination_scores,
            context.all_destination,
            context.all_destination_specs,
        )
        relation_scores = deterministic_gather(
            all_relation_scores,
            context.all_channel,
            context.all_channel_specs,
        )
        logits = F.leaky_relu(
            edge_source_scores + edge_destination_scores + relation_scores,
            negative_slope=self.negative_slope,
        )
        aggregate = self._complete_destination_reduce(
            messages=messages,
            logits=logits,
            destination_specs=context.all_destination_specs,
            num_nodes=slots.shape[0],
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        output = aggregate.reshape(
            slots.shape[0], self.num_slots, self.output_dim
        )
        if self.use_residual:
            output = output + residual
        if self.use_activation:
            output = F.elu(output)
        return output


class GenuineBlockConditionalSlotGAT(nn.Module):
    """SlotGAT with distinct raw contexts and factorized path propagation."""

    def __init__(
        self,
        *,
        layout: NodeTypeLayout,
        in_dims: Sequence[int],
        num_channels: int,
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
    ) -> None:
        super().__init__()
        if len(in_dims) != len(layout.counts):
            raise ValueError(
                f"Expected {len(layout.counts)} feature matrices, got {len(in_dims)}"
            )
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if aggregator != "SA":
            raise ValueError(
                "The recoverability-conditional implementation supports the "
                "paper's Slot Attention aggregator only (--aggregator SA)."
            )
        self.layout = layout
        self.num_slots = len(layout.counts)
        self.num_channels = int(num_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.require_identity_sparse = bool(require_identity_sparse)
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
                DeterministicRecoverabilityConditionalSlotLayer(
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
                    residual=not is_last,
                )
            )
            current_dim = output_dim
        self.layers = nn.ModuleList(layers)
        self.slot_attention = GlobalSlotAttention(
            slot_dim=self.num_classes,
            attention_dim=int(sa_attention_dim),
        )
        self._runtime_cache: dict[tuple[str, str], RuntimeContext] = {}

    def _initialize_slots(self, features_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features_list) != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} feature matrices, got {len(features_list)}"
            )
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
            expected_rows = self.layout.counts[node_type]
            if transformed.shape[0] != expected_rows:
                raise ValueError(
                    f"Feature row mismatch for node type {node_type}: "
                    f"expected {expected_rows}, got {transformed.shape[0]}"
                )
            start = self.layout.offsets[node_type]
            stop = start + expected_rows
            slots[start:stop, node_type, :] = transformed
        return slots

    @staticmethod
    def _edge_runtime(
        *,
        channel: int,
        source: torch.Tensor,
        destination: torch.Tensor,
        total_nodes: int,
    ) -> RuntimeEdgeDirection:
        destination_specs = build_index_bucket_specs(
            destination, num_rows=total_nodes
        )
        active = torch.cat([nodes for nodes, _ in destination_specs], dim=0)
        active = torch.sort(active).values
        return RuntimeEdgeDirection(
            channel=int(channel),
            source=source,
            destination=destination,
            source_specs=build_index_bucket_specs(source, num_rows=total_nodes),
            destination_specs=destination_specs,
            active_destinations=active,
        )

    def _runtime_context(
        self,
        plan: FreebaseBlockContextPlan,
        device: torch.device,
    ) -> RuntimeContext:
        if plan.layout != self.layout:
            raise ValueError("Context-plan node layout does not match the model")
        if plan.num_channels != self.num_channels:
            raise ValueError("Context-plan channel catalog does not match the model")

        # This branch is evaluated for every distinct raw plan.  It is the
        # conditional-graph context check; propagation is activated only after
        # the physical realization is certified.
        plan.verify_runtime_context()
        cache_key = (plan.context_plan_hash, str(device))
        cached = self._runtime_cache.get(cache_key)
        if cached is not None:
            return cached

        # Plans may be backed by read-only memory maps.  Copy before wrapping
        # them as tensors so PyTorch never receives a non-writable NumPy view.
        base_src = torch.as_tensor(
            np.array(plan.base_src, dtype=np.int64, copy=True),
            dtype=torch.int64,
            device=device,
        )
        base_dst = torch.as_tensor(
            np.array(plan.base_dst, dtype=np.int64, copy=True),
            dtype=torch.int64,
            device=device,
        )
        base_channel = torch.as_tensor(
            np.array(plan.base_channel, dtype=np.int64, copy=True),
            dtype=torch.int64,
            device=device,
        )

        edge_by_channel: list[RuntimeEdgeDirection | None] = [
            None for _ in range(self.num_channels)
        ]
        for channel in range(self.num_channels):
            mask = base_channel == int(channel)
            if not bool(mask.any()):
                continue
            source = base_src[mask]
            destination = base_dst[mask]
            edge_by_channel[channel] = self._edge_runtime(
                channel=channel,
                source=source,
                destination=destination,
                total_nodes=self.layout.total,
            )

        path_directions: list[RuntimePathDirection] = []
        destination_chunks = [base_dst]
        channel_chunks = [base_channel]
        for relation in sorted(plan.block_relations, key=lambda item: item.name):
            for direction in (1, -1):
                if direction == 1:
                    output_channel = int(relation.forward_channel)
                    step_channels = tuple(int(x) for x in relation.forward_steps)
                    support_np = relation.forward_support
                else:
                    output_channel = int(relation.reverse_channel)
                    step_channels = tuple(int(x) for x in relation.reverse_steps)
                    support_np = relation.reverse_support
                support = torch.as_tensor(
                    np.asarray(support_np), dtype=torch.int64, device=device
                )
                if support.numel() == 0:
                    raise ValueError(
                        f"Conditional mapping {relation.name} has empty support"
                    )
                for step_channel in step_channels:
                    if edge_by_channel[step_channel] is None:
                        raise ValueError(
                            f"Conditional mapping {relation.name} references empty "
                            f"base channel {step_channel}"
                        )
                runtime = RuntimePathDirection(
                    relation_name=relation.name,
                    direction=direction,
                    output_channel=output_channel,
                    step_channels=step_channels,
                    support=support,
                    support_specs=build_index_bucket_specs(
                        support, num_rows=self.layout.total
                    ),
                )
                path_directions.append(runtime)
                destination_chunks.append(support)
                channel_chunks.append(
                    torch.full(
                        (support.numel(),),
                        output_channel,
                        dtype=torch.int64,
                        device=device,
                    )
                )

        all_destination = torch.cat(destination_chunks, dim=0)
        all_channel = torch.cat(channel_chunks, dim=0)
        context = RuntimeContext(
            context_hash=plan.context_plan_hash,
            semantic_hash=plan.semantic_program_hash,
            base_src=base_src,
            base_dst=base_dst,
            base_channel=base_channel,
            base_source_specs=build_index_bucket_specs(
                base_src, num_rows=self.layout.total
            ),
            edge_by_channel=tuple(edge_by_channel),
            path_directions=tuple(path_directions),
            all_destination=all_destination,
            all_channel=all_channel,
            all_destination_specs=build_index_bucket_specs(
                all_destination, num_rows=self.layout.total
            ),
            all_channel_specs=build_index_bucket_specs(
                all_channel, num_rows=self.num_channels
            ),
        )
        self._runtime_cache[cache_key] = context
        return context

    def forward(
        self,
        features_list: Sequence[torch.Tensor],
        plan: FreebaseBlockContextPlan,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slots = self._initialize_slots(features_list)
        context = self._runtime_context(plan, slots.device)
        for layer in self.layers:
            slots = layer(slots, context)
        logits, slot_weights = self.slot_attention(slots)
        return logits, {
            "slot_attention": slot_weights,
            "context_plan_hash": torch.tensor(
                0, device=logits.device, dtype=torch.int64
            ),
        }
