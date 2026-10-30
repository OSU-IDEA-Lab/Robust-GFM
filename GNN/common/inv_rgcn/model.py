from __future__ import annotations

from typing import Optional, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F


class BasisRelationWeights(nn.Module):
    def __init__(self, num_relations: int, in_dim: int, out_dim: int, num_bases: int):
        super().__init__()
        self.num_relations = int(num_relations)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_bases = min(int(num_bases), self.num_relations)
        self.bases = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.empty(self.num_relations, self.num_bases))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeff)

    def forward(self) -> torch.Tensor:
        return torch.einsum("rb,bio->rio", self.coeff, self.bases)


class ConditionalRGCNLayer(nn.Module):
    """RGCN layer over a compiled conditional message program.

    The physical graph is not rewritten. Context matching is performed by the
    preprocessing compiler, while this layer propagates only on canonical semantic
    edges. Relation parameters are shared by semantic relation name across variants.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int,
        dropout: float = 0.0,
        activation: bool = True,
        use_self_loop: bool = True,
        norm: str = "mean",
    ) -> None:
        super().__init__()
        if norm not in {"mean", "sum"}:
            raise ValueError("norm must be 'mean' or 'sum'")
        self.norm = norm
        self.rel_weights = BasisRelationWeights(num_relations, in_dim, out_dim, num_bases)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False) if use_self_loop else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.num_relations = int(num_relations)
        self.out_dim = int(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = torch.ones(edge_type.numel(), dtype=x.dtype, device=x.device)
        src_all, dst_all = edge_index
        weights = self.rel_weights()
        out = x.new_zeros((x.shape[0], self.out_dim))

        # A fixed relation loop and canonical edge ordering make random-number and
        # accumulation order identical across equivalent variants.
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            ew = edge_weight[mask].to(x.dtype)
            msg = x.index_select(0, src).matmul(weights[rel])
            if self.norm == "mean":
                # Relation-specific destination normalization (mean over in-edges).
                deg = torch.bincount(dst, minlength=x.shape[0]).clamp_min_(1).to(x.dtype)
                msg = msg * (ew / deg.index_select(0, dst)).unsqueeze(1)
            else:
                # Unnormalized sum, matching DGL RelGraphConv's default. On DBLP the
                # ranked destination type (conference) has in-degree ~1531, so the mean
                # divides away the popularity prior that a venue-ranking task depends on.
                msg = msg * ew.unsqueeze(1)
            # edge rows are canonicalized as (relation, destination, source), so
            # destinations are consecutive here. Segment reduction avoids atomic
            # scatter accumulation and is reproducible on deterministic backends.
            unique_dst, counts = torch.unique_consecutive(dst, return_counts=True)
            if hasattr(torch, "segment_reduce"):
                summed = torch.segment_reduce(msg, reduce="sum", lengths=counts)
                out[unique_dst] = out.index_select(0, unique_dst) + summed
            else:  # compatibility fallback for older PyTorch releases
                out.index_add_(0, dst, msg)

        if self.self_loop is not None:
            out = out + self.self_loop(x)
        out = out + self.bias
        if self.activation:
            out = F.relu(out)
        return self.dropout(out)


class ConditionalRGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_bases: int,
        dropout: float,
        input_features: Optional[torch.Tensor] = None,
        norm: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        if input_features is None:
            self.node_embedding = nn.Embedding(num_nodes, in_dim)
            self.register_buffer("fixed_features", None)
        else:
            if input_features.shape[0] != num_nodes:
                raise ValueError("input feature row count mismatch")
            self.node_embedding = None
            self.register_buffer("fixed_features", input_features.float())
            self.input_projection = (
                nn.Identity()
                if input_features.shape[1] == in_dim
                else nn.Linear(input_features.shape[1], in_dim, bias=False)
            )

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)
        self.layers = nn.ModuleList(
            [
                ConditionalRGCNLayer(
                    dims[i],
                    dims[i + 1],
                    num_relations=num_relations,
                    num_bases=num_bases,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                    activation=i < num_layers - 1,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.node_embedding is not None:
            nn.init.xavier_uniform_(self.node_embedding.weight)
        if hasattr(self, "input_projection") and isinstance(self.input_projection, nn.Linear):
            nn.init.xavier_uniform_(self.input_projection.weight)

    def initial_features(self) -> torch.Tensor:
        if self.node_embedding is not None:
            return self.node_embedding.weight
        return self.input_projection(self.fixed_features)

    def forward(
        self,
        semantic_edge_index: torch.Tensor,
        semantic_edge_type: torch.Tensor,
        semantic_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h = self.initial_features()
        for layer in self.layers:
            h = layer(h, semantic_edge_index, semantic_edge_type, semantic_edge_weight)
        return h


class LinkPredictor(nn.Module):
    def __init__(self, dim: int, mode: str = "dot") -> None:
        super().__init__()
        if mode not in {"dot", "distmult"}:
            raise ValueError("mode must be 'dot' or 'distmult'")
        self.mode = mode
        self.relation = nn.Parameter(torch.ones(dim)) if mode == "distmult" else None

    def forward(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        lhs = h.index_select(0, pairs[:, 0])
        rhs = h.index_select(0, pairs[:, 1])
        if self.relation is not None:
            lhs = lhs * self.relation
        return (lhs * rhs).sum(dim=-1)
