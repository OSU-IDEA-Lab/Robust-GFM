"""Memory-only override of ``pain_nc/model.py::PathAggregator.forward`` (installed by ``install()``).

Upstream moves each chunk's path tensors to the GPU (widened to int64) and passes them to ``torch.utils.checkpoint``,
which saves its inputs for the backward recomputation — so the whole path program is kept on the GPU once per PAIN
layer (≈ 20 GB for the 32 M-path IMDb invariant program: OOM on a 32 GB V100). Here the checkpointed function takes the
CPU (host) chunk slices and performs the identical ``.to(device, dtype)`` inside, so the saved inputs live in host
memory. The arithmetic — same ops, dtypes, order, chunk boundaries — is unchanged; every arm uses this override.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from pain_nc.model import PathAggregator


def _forward(self, x: torch.Tensor, graph) -> torch.Tensor:
    total = graph.num_paths
    aggregate = x.new_zeros((graph.num_nodes, self.hidden_dim))
    counts = x.new_zeros((graph.num_nodes, 1)) if self.aggregation == "mean" else None
    edge_type = graph.edge_type.to(x.device, dtype=torch.long, non_blocking=True)
    dev = x.device

    def encode(features, c_path_index, c_lengths, c_path_edge_idx, c_neighbor_mask, c_distances):
        return self._encode_chunk(
            features, edge_type,
            c_path_index.to(dev, dtype=torch.long, non_blocking=True), c_lengths,
            c_path_edge_idx.to(dev, dtype=torch.long, non_blocking=True),
            c_neighbor_mask.to(dev, dtype=torch.long, non_blocking=True),
            c_distances.to(dev, dtype=torch.long, non_blocking=True))

    for start in range(0, total, self.chunk_size):
        stop = min(start + self.chunk_size, total)
        path_index = graph.path_index[..., start:stop]
        path_edge_idx = graph.path_edge_idx[..., start:stop]
        neighbor_mask = graph.neighbor_mask[start:stop]
        distances = graph.distances[start:stop]
        lengths = graph.path_lengths[start:stop]
        roots = graph.mask_index[start:stop].to(dev, dtype=torch.long, non_blocking=True)
        if self.training and self.checkpoint_chunks:
            messages = checkpoint(encode, x, path_index, lengths, path_edge_idx, neighbor_mask, distances, use_reentrant=False)
        else:
            messages = encode(x, path_index, lengths, path_edge_idx, neighbor_mask, distances)
        path_weights = getattr(graph, "path_weights", None)
        if path_weights is not None:
            weights = path_weights[start:stop].to(dev, dtype=messages.dtype, non_blocking=True)
            messages = messages * weights.unsqueeze(1)
        else:
            weights = None
        aggregate, segment_counts = self._deterministic_segment_add(aggregate, roots, messages)
        if counts is not None:
            if weights is None:
                unique_roots = torch.unique_consecutive(roots)
                counts = counts.index_copy(0, unique_roots, counts[unique_roots] + segment_counts)
            else:
                counts, _ = self._deterministic_segment_add(counts, roots, weights.unsqueeze(1))
    if counts is not None:
        aggregate = aggregate / counts.clamp_min(1)
    return aggregate


def install():
    PathAggregator.forward = _forward
