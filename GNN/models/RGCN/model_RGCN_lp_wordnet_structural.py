"""WordNet conditional-structural RGCN encoder and DistMult decoder.

The semantic edge table is sorted by ``(destination, relation, source)`` during
preprocessing.  That ordering makes every destination's incoming messages one
contiguous segment.  The layer can therefore reproduce the original WordNet
RGCN global incoming-degree normalization without CUDA atomic scatter adds.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DeterministicBasisRGCNLayer(nn.Module):
    """One basis-decomposed RGCN layer with deterministic segment aggregation.

    For basis ``b`` the usual RGCN sum can be rearranged as

    ``sum_e a[type(e), b] * x[src(e)]``

    followed by projection with basis matrix ``V_b``.  Because edges are sorted
    by destination, the weighted source rows are reduced with one contiguous
    segment sum per destination.  No two CUDA threads write to the same output
    row, so equivalent variants follow the same numerical reduction order.

    ``edge_chunk_size`` is retained only for command-line/checkpoint
    compatibility with earlier generated runners.  It is not needed by the
    segment implementation, which materializes at most one ``E x hidden``
    weighted tensor per basis (the same asymptotic working set as the former
    chunked implementation but without atomic ``index_add_``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int,
        edge_chunk_size: int = 200_000,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("RGCN channel sizes must be positive")
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.num_bases = min(int(num_bases), self.num_relations)
        self.edge_chunk_size = int(edge_chunk_size)  # backward-compatible config

        self.weight = nn.Parameter(
            torch.empty(self.num_bases, self.in_channels, self.out_channels)
        )
        self.comp = nn.Parameter(torch.empty(self.num_relations, self.num_bases))
        self.root = nn.Parameter(torch.empty(self.in_channels, self.out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.comp)
        nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _segment_sum_by_destination(
        weighted: torch.Tensor,
        destination: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unique destinations and contiguous sums for sorted rows."""
        if destination.numel() == 0:
            return destination, weighted.new_empty((0, weighted.shape[1]))

        unique_dst, counts = torch.unique_consecutive(
            destination,
            return_counts=True,
        )
        if not hasattr(torch, "segment_reduce"):
            raise RuntimeError(
                "This deterministic WordNet RGCN requires torch.segment_reduce "
                "(PyTorch 2.x). Upgrade PyTorch rather than falling back to a "
                "CUDA atomic scatter implementation."
            )
        reduced = torch.segment_reduce(weighted, reduce="sum", lengths=counts)
        return unique_dst, reduced

    def _basis_contribution(
        self,
        src_x: torch.Tensor,
        coefficients: torch.Tensor,
        basis_weight: torch.Tensor,
        unique_dst: torch.Tensor,
        counts: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weighted = src_x * coefficients
        aggregate = torch.segment_reduce(weighted, reduce="sum", lengths=counts)
        projected = aggregate @ basis_weight
        result = projected.new_zeros((num_nodes, self.out_channels))
        # unique_dst is duplicate-free, so index_copy is a deterministic
        # one-writer-per-row operation rather than an atomic scatter.
        return torch.index_copy(result, 0, unique_dst, projected)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        c_i: torch.Tensor,
        root_scale: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        n = x.shape[0]
        neigh = x.new_zeros((n, self.out_channels))

        if src.numel() > 0:
            # Preprocessing guarantees nondecreasing destination IDs.  Edge
            # dropout preserves relative order, so destination segments remain
            # contiguous during training.
            src_x = x.index_select(0, src)
            coeffs = self.comp.index_select(0, edge_type)  # (E, B)
            unique_dst, counts = torch.unique_consecutive(
                dst, return_counts=True
            )

            use_checkpoint = torch.is_grad_enabled() and src_x.requires_grad
            for b in range(self.num_bases):
                args = (
                    src_x,
                    coeffs[:, b : b + 1],
                    self.weight[b],
                    unique_dst,
                    counts,
                    n,
                )
                if use_checkpoint:
                    contribution = checkpoint(
                        self._basis_contribution,
                        *args,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    contribution = self._basis_contribution(*args)
                neigh = neigh + contribution

        out = neigh / c_i.clamp(min=1.0).unsqueeze(1)
        out = out + root_scale.reshape(1, 1) * (x @ self.root)
        if self.bias is not None:
            out = out + self.bias
        return out


# Backward-compatible import name for any external scripts created earlier.
ChunkedBasisRGCNLayer = DeterministicBasisRGCNLayer


class WordNetStructuralRGCN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_message_relations: int,
        num_decoder_relations: int,
        hidden_dim: int = 200,
        num_bases: int = 30,
        edge_dropout_other: float = 0.4,
        root_dropout_loop: float = 0.2,
        edge_chunk_size: int = 200_000,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_message_relations = int(num_message_relations)
        self.num_decoder_relations = int(num_decoder_relations)
        self.hidden_dim = int(hidden_dim)
        self.edge_dropout_other = float(edge_dropout_other)
        self.root_dropout_loop = float(root_dropout_loop)

        self.entity_emb = nn.Embedding(self.num_entities, self.hidden_dim)
        self.rel_emb = nn.Embedding(self.num_decoder_relations, self.hidden_dim)
        self.conv = DeterministicBasisRGCNLayer(
            self.hidden_dim,
            self.hidden_dim,
            self.num_message_relations,
            num_bases,
            edge_chunk_size=edge_chunk_size,
        )
        # Match the original WordNet RGCN initialization order: the convolution
        # initializes itself once in its constructor, after which only the entity
        # and decoder-relation embeddings are initialized.  The previous code
        # reset the convolution twice, consuming a different RNG stream.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if training and self.edge_dropout_other > 0:
            keep = (
                torch.rand(edge_type.numel(), device=edge_type.device)
                >= self.edge_dropout_other
            )
            # Boolean filtering preserves the canonical destination-major order.
            ei = edge_index[:, keep]
            et = edge_type[keep]
        else:
            ei, et = edge_index, edge_type

        if ei.shape[1] == 0:
            degree = torch.ones(
                self.num_entities,
                device=edge_index.device,
                dtype=self.entity_emb.weight.dtype,
            )
        else:
            degree = torch.bincount(
                ei[1], minlength=self.num_entities
            ).to(self.entity_emb.weight.dtype)
            degree.clamp_(min=1.0)
        return ei, et, degree

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        ei, et, degree = self._drop_edges(edge_index, edge_type, training)
        if training and self.root_dropout_loop > 0:
            root_scale = torch.bernoulli(
                torch.full(
                    (1,),
                    1.0 - self.root_dropout_loop,
                    device=edge_index.device,
                    dtype=self.entity_emb.weight.dtype,
                )
            )
        else:
            root_scale = torch.ones(
                1,
                device=edge_index.device,
                dtype=self.entity_emb.weight.dtype,
            )
        return self.conv(self.entity_emb.weight, ei, et, degree, root_scale)

    def score_triples(
        self,
        entity_embs: torch.Tensor,
        triples: torch.Tensor,
    ) -> torch.Tensor:
        h = entity_embs.index_select(0, triples[:, 0])
        r = self.rel_emb(triples[:, 1])
        t = entity_embs.index_select(0, triples[:, 2])
        return (h * r * t).sum(dim=-1)

    def score_tails(
        self,
        entity_embs: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        return (
            entity_embs.index_select(0, heads) * self.rel_emb(relations)
        ) @ entity_embs.T

    def score_heads(
        self,
        entity_embs: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.rel_emb(relations) * entity_embs.index_select(0, tails)
        ) @ entity_embs.T
