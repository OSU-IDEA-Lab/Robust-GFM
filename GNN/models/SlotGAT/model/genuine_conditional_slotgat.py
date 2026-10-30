"""Genuine raw-context conditional SlotGAT.

The model accepts a variant-specific :class:`ConditionalContextPlan` on every
forward pass.  Direct Movie--Role and Movie--Link--Role contexts are interpreted
inside the conditional layer; no common DGL adjacency is constructed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from conditional_bias import CHANNEL_NAMES, ConditionalContextPlan, NodeTypeLayout


def _is_sparse(tensor: torch.Tensor) -> bool:
    return tensor.layout != torch.strided


def sparse_aware_linear(tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """Apply ``nn.Linear`` to dense or COO/CSR feature matrices."""
    if not _is_sparse(tensor):
        return layer(tensor)
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
    output = torch.sparse.mm(tensor, layer.weight.t())
    if layer.bias is not None:
        output = output + layer.bias
    return output


class _DeterministicIndexSelect(torch.autograd.Function):
    """Index-select with a deterministic repeated-index backward pass."""

    @staticmethod
    def forward(ctx, value, index, bucket_specs, num_rows):
        ctx.bucket_specs = bucket_specs
        ctx.num_rows = int(num_rows)
        ctx.trailing_shape = tuple(value.shape[1:])
        return value.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_output):
        node_chunks = []
        gradient_chunks = []
        for nodes, edge_indices in ctx.bucket_specs:
            # Every edge belongs to exactly one row bucket. Summation order is
            # fixed by the precomputed edge-index matrix.
            gradients = grad_output[edge_indices].sum(dim=1)
            node_chunks.append(nodes)
            gradient_chunks.append(gradients)
        all_nodes = torch.cat(node_chunks, dim=0)
        all_gradients = torch.cat(gradient_chunks, dim=0)
        order = torch.argsort(all_nodes)
        ordered_nodes = all_nodes[order]
        expected = torch.arange(
            ctx.num_rows, device=grad_output.device, dtype=torch.int64
        )
        if not torch.equal(ordered_nodes, expected):
            raise RuntimeError(
                "Deterministic index-select buckets do not cover every row"
            )
        return all_gradients[order], None, None, None


def deterministic_index_select(
    value: torch.Tensor,
    index: torch.Tensor,
    bucket_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return _DeterministicIndexSelect.apply(
        value, index, bucket_specs, value.shape[0]
    )


class DeterministicConditionalSlotLayer(nn.Module):
    """One SlotGAT layer with deterministic degree-bucket aggregation.

    The aggregation avoids DGL scatter/reduce kernels.  Incoming messages are
    sorted canonically and destinations with equal degree are processed as
    dense batches.  This keeps the computation deterministic while preserving
    attention normalization over each destination's actual neighborhood.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        num_edge_types: int,
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
        self.edge_embedding = nn.Embedding(int(num_edge_types), int(edge_dim))
        self.edge_projection = nn.Linear(
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
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        nn.init.xavier_uniform_(self.attention_relation)
        nn.init.zeros_(self.slot_bias)

    @staticmethod
    def _degree_bucket_reduce(
        *,
        edge_messages: torch.Tensor,
        dst: torch.Tensor,
        logits: torch.Tensor,
        num_nodes: int,
        attention_dropout: float,
        training: bool,
        bucket_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Softmax and sum over incoming edges without atomic scatter."""
        node_chunks: list[torch.Tensor] = []
        aggregate_chunks: list[torch.Tensor] = []
        for nodes, edge_indices in bucket_specs:
            bucket_logits = logits[edge_indices]  # [B, degree, heads]
            attention = torch.softmax(bucket_logits, dim=1)
            attention = F.dropout(
                attention,
                p=attention_dropout,
                training=training,
            )
            messages = edge_messages[edge_indices]  # [B, degree, slots, heads, dim]
            aggregate = (
                attention[:, :, None, :, None] * messages
            ).sum(dim=1)
            node_chunks.append(nodes)
            aggregate_chunks.append(aggregate)

        all_nodes = torch.cat(node_chunks, dim=0)
        all_aggregates = torch.cat(aggregate_chunks, dim=0)
        order = torch.argsort(all_nodes)
        ordered_nodes = all_nodes[order]
        expected = torch.arange(num_nodes, device=dst.device, dtype=torch.int64)
        if not torch.equal(ordered_nodes, expected):
            raise RuntimeError("Degree buckets did not cover every node exactly once")
        return all_aggregates[order]

    def forward(
        self,
        slots: torch.Tensor,
        *,
        src: torch.Tensor,
        dst: torch.Tensor,
        channels: torch.Tensor,
        destination_bucket_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        source_bucket_specs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
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
        channel_one_hot = F.one_hot(
            channels, num_classes=self.edge_embedding.num_embeddings
        ).to(dtype=projected.dtype)
        relation_embeddings = channel_one_hot @ self.edge_embedding.weight
        relation_vectors = self.edge_projection(relation_embeddings).view(
            -1, self.num_heads, self.head_dim
        )
        relation_scores = torch.einsum(
            "ehk,hk->eh", relation_vectors, self.attention_relation
        )
        edge_source_scores = deterministic_index_select(
            source_scores, src, source_bucket_specs
        )
        edge_destination_scores = deterministic_index_select(
            destination_scores, dst, destination_bucket_specs
        )
        edge_messages = deterministic_index_select(
            projected, src, source_bucket_specs
        )
        logits = edge_source_scores + edge_destination_scores + relation_scores
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        aggregated = self._degree_bucket_reduce(
            edge_messages=edge_messages,
            dst=dst,
            logits=logits,
            num_nodes=slots.shape[0],
            attention_dropout=self.attention_dropout,
            training=self.training,
            bucket_specs=destination_bucket_specs,
        )
        output = aggregated.reshape(
            slots.shape[0], self.num_slots, self.output_dim
        )
        if self.use_residual:
            output = output + residual
        if self.use_activation:
            output = F.elu(output)
        return output


def build_bucket_specs(index: torch.Tensor, num_rows: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Group the rows of ``index`` (one entry per message) by degree.

    Returns ``((nodes, edge_indices[B, degree]), ...)`` sorted by degree; every row must
    occur at least once (the plans guarantee a self message per node).
    """
    device = index.device
    permutation = torch.argsort(index)
    sorted_index = index[permutation]
    counts = torch.bincount(sorted_index, minlength=int(num_rows))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten()[:10]
        raise RuntimeError(
            "Every node must occur through a conditional self message; "
            f"missing examples={missing.detach().cpu().tolist()}"
        )
    starts = torch.cumsum(counts, dim=0) - counts
    specs = []
    for degree_value in torch.unique(counts, sorted=True).detach().cpu().tolist():
        degree = int(degree_value)
        nodes = torch.nonzero(counts == degree, as_tuple=False).flatten()
        offsets = torch.arange(degree, device=device, dtype=torch.int64)
        sorted_positions = starts[nodes, None] + offsets[None, :]
        edge_indices = permutation[sorted_positions]
        specs.append((nodes, edge_indices))
    return tuple(specs)


class GlobalSlotAttention(nn.Module):
    """Learn one global importance weight for each semantic slot."""

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
        global_scores = scores.mean(dim=0)
        weights = torch.softmax(global_scores, dim=0)
        output = torch.einsum("t,ntc->nc", weights, slots)
        return output, weights


class GenuineConditionalSlotGAT(nn.Module):
    """SlotGAT whose forward input remains a distinct raw context plan."""

    def __init__(
        self,
        *,
        layout: NodeTypeLayout,
        in_dims: Sequence[int],
        num_classes: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        edge_feats: int,
        dropout_feat: float,
        dropout_attn: float,
        slope: float,
        aggregator: str,
        sa_attention_dim: int,
    ) -> None:
        super().__init__()
        if len(in_dims) != len(layout.counts):
            raise ValueError(
                f"Expected {len(layout.counts)} feature matrices, got {len(in_dims)}"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if aggregator != "SA":
            raise ValueError(
                "The genuine conditional implementation currently supports the "
                "paper's Slot Attention aggregator only (--aggregator SA)."
            )
        self.layout = layout
        self.num_slots = len(layout.counts)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.input_projections = nn.ModuleList(
            [nn.Linear(int(in_dim), self.hidden_dim) for in_dim in in_dims]
        )

        # Geometry mirrors the reference ``model/gnn.py:slotGAT`` exactly, so the
        # invariant arm is capacity-matched to the published baseline:
        #   one input layer  : hidden_dim         -> hidden_dim * num_heads
        #   num_layers-1 hid : hidden_dim*heads   -> hidden_dim * num_heads
        #   one output layer : hidden_dim*heads   -> num_classes, 1 head
        # i.e. num_layers + 1 message-passing layers in total. The previous
        # version built only ``num_layers`` layers at width ``hidden_dim``,
        # which gave the invariant arm one layer fewer and ~3.4x fewer
        # parameters than the baseline it was compared against.
        layers: list[nn.Module] = []
        total_layers = int(num_layers) + 1
        wide_dim = self.hidden_dim * int(num_heads)
        current_dim = self.hidden_dim
        for layer_index in range(total_layers):
            is_first = layer_index == 0
            is_last = layer_index == total_layers - 1
            output_dim = self.num_classes if is_last else wide_dim
            heads = 1 if is_last else int(num_heads)
            layers.append(
                DeterministicConditionalSlotLayer(
                    num_slots=self.num_slots,
                    input_dim=current_dim,
                    output_dim=output_dim,
                    num_heads=heads,
                    num_edge_types=len(CHANNEL_NAMES),
                    edge_dim=int(edge_feats),
                    feature_dropout=float(dropout_feat),
                    attention_dropout=float(dropout_attn),
                    negative_slope=float(slope),
                    activation=not is_last,
                    # The reference gives the input projection no residual and
                    # every later layer one; the layer itself drops it when the
                    # input and output widths differ.
                    residual=not is_first,
                )
            )
            current_dim = output_dim
        self.layers = nn.ModuleList(layers)
        self.slot_attention = GlobalSlotAttention(
            slot_dim=self.num_classes,
            attention_dim=int(sa_attention_dim),
        )
        # Sparse CUDA matrix multiplication can use reduction kernels whose
        # bitwise behavior varies across launches. IMDb is small enough to cache
        # dense feature matrices once per model, then use deterministic GEMMs.
        self._dense_feature_cache: dict[int, torch.Tensor] = {}
        # Per-model, per-raw-plan tensor cache.  Equal semantic tensors may be
        # produced, but each cache entry is reached through a distinct raw plan.
        self._context_tensor_cache: dict[
            tuple[str, str],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                tuple[tuple[torch.Tensor, torch.Tensor], ...],
                tuple[tuple[torch.Tensor, torch.Tensor], ...],
            ],
        ] = {}

    def _initialize_slots(self, features_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features_list) != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} feature matrices, got {len(features_list)}"
            )
        reference = features_list[0]
        dtype = reference.dtype
        device = reference.device
        slots = torch.zeros(
            self.layout.total,
            self.num_slots,
            self.hidden_dim,
            dtype=dtype,
            device=device,
        )
        for node_type, (features, projection) in enumerate(
            zip(features_list, self.input_projections)
        ):
            if _is_sparse(features):
                dense_features = self._dense_feature_cache.get(node_type)
                if dense_features is None:
                    dense_features = features.to_dense()
                    self._dense_feature_cache[node_type] = dense_features
                transformed = projection(dense_features)
            else:
                transformed = projection(features)
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

    def _context_tensors(
        self,
        plan: ConditionalContextPlan,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        if plan.layout != self.layout:
            raise ValueError("Context-plan node layout does not match the model")
        # The cache key MUST include the emission mode and the semantic hash.
        # A conditional plan and a physical plan for the same variant share a
        # context_plan_hash (they have identical witnesses and differ only in how
        # those witnesses are emitted), so keying on it alone would silently hand
        # a physical run the invariant run's message tensors.
        cache_key = (
            plan.context_plan_hash,
            getattr(plan, "mode", "semantic_closure"),
            plan.semantic_propagation_hash,
            str(device),
        )
        cached = self._context_tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        # This call branches on direct/via-Link witness kinds and validates the
        # raw context.  It does not read a prebuilt semantic DGL graph.
        src_np, dst_np, channel_np = plan.emit_semantic_numpy(verify=True)
        src = torch.as_tensor(src_np, dtype=torch.int64, device=device)
        dst = torch.as_tensor(dst_np, dtype=torch.int64, device=device)
        channels = torch.as_tensor(channel_np, dtype=torch.int64, device=device)
        destination_bucket_specs = build_bucket_specs(dst, self.layout.total)
        source_bucket_specs = build_bucket_specs(src, self.layout.total)
        tensors = (
            src,
            dst,
            channels,
            destination_bucket_specs,
            source_bucket_specs,
        )
        self._context_tensor_cache[cache_key] = tensors
        return tensors

    def forward(
        self,
        features_list: Sequence[torch.Tensor],
        context_plan: ConditionalContextPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self._initialize_slots(features_list)
        (
            src,
            dst,
            channels,
            destination_bucket_specs,
            source_bucket_specs,
        ) = self._context_tensors(context_plan, slots.device)
        for layer in self.layers:
            slots = layer(
                slots,
                src=src,
                dst=dst,
                channels=channels,
                destination_bucket_specs=destination_bucket_specs,
                source_bucket_specs=source_bucket_specs,
            )
        logits, slot_weights = self.slot_attention(slots)
        return logits, slot_weights
