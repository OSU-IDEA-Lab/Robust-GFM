"""Deterministic relational convolution for CMPNN.

``cmpnn.layer.GeneralizedRelationalConv.aggregate`` reduces incoming messages with
``torch_scatter`` (``scatter_add`` / ``scatter_mean`` use CUDA atomics, so the float
summation order — and therefore the last bits of the result — varies run to run), and the
fused ``generalized_rspmm`` path used for ``aggregate_func="sum"`` has an atomic backward
(measured: three identical 20-step runs on the WordNet program give three different
parameter hashes). The project's invariance claim is *bitwise* τ = 1 across independently
trained variants, which needs reductions whose result depends only on the multiset of
inputs. This subclass keeps the layer's message function, PNA statistics and ``combine``
untouched and replaces the reduction with ``torch.segment_reduce`` over edges sorted by
destination — one serial loop per segment, no atomics.

Two execution modes:

* **whole** (``chunk_edges=None``): materialise all messages, one ``segment_reduce`` per
  statistic. Used for IMDb / DBLP (≤ 300k messages).
* **chunked** (``chunk_edges=K``): the dst-sorted edge list is processed in fixed slices
  of K edges; each slice's messages are computed, reduced per destination inside the
  slice, and folded into running partials (sum / sum-of-squares / max / min) in slice
  order, under ``torch.utils.checkpoint`` so only one slice of messages is alive at a
  time. Fixed slice order + per-slice serial reductions = deterministic. Used for WordNet
  (1.6M messages after CMPNN's inverses), where the whole mode OOMs on a 32 GB V100.

The sorted order and segment lengths are computed once per (masked) graph and cached on
the graph object.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils import checkpoint as _ckpt

from cmpnn.layer import GeneralizedRelationalConv


def dst_sorted_layout(graph):
    """Permutation sorting the edge list by destination (stable) + segment lengths.

    Cached on the graph under ``_det_layout``; the boundary self-message of every node is
    appended by ``aggregate`` and accounted for here (one extra message per node, placed
    at the end of its segment), so every segment has length ≥ 1.
    """
    layout = getattr(graph, "_det_layout", None)
    if layout is not None and layout["num_edge"] == graph.num_edge:
        return layout
    node_out = graph.edge_list[:, 1]
    n = graph.num_node
    dst_all = torch.cat([node_out, torch.arange(n, device=graph.device)])
    order = torch.argsort(dst_all, stable=True)
    lengths = torch.bincount(dst_all, minlength=n)
    layout = {"num_edge": int(graph.num_edge), "order": order, "lengths": lengths, "dst_sorted": dst_all[order]}
    graph._det_layout = layout
    return layout


class DeterministicRelationalConv(GeneralizedRelationalConv):
    """Drop-in replacement whose aggregation is order-independent."""

    chunk_edges: Optional[int] = None

    # ------------------------------------------------------------------ whole mode
    def aggregate(self, graph, message):
        layout = dst_sorted_layout(graph)
        order, lengths = layout["order"], layout["lengths"]
        edge_weight = torch.cat(
            [graph.edge_weight, torch.ones(graph.num_node, device=graph.device)]
        ).unsqueeze(-1).unsqueeze(-1)
        msg = (message * edge_weight)[order]              # [E+N, B, D] dst-sorted
        flat = msg.flatten(1)
        shape = (graph.num_node,) + tuple(message.shape[1:])

        def seg(reduce, x):
            return torch.segment_reduce(x, reduce, lengths=lengths, axis=0, unsafe=True).view(shape)

        if self.aggregate_func == "sum":
            return seg("sum", flat)
        if self.aggregate_func == "mean":
            return seg("mean", flat)
        if self.aggregate_func == "max":
            return seg("max", flat)
        if self.aggregate_func == "pna":
            return self._pna_combine(graph, seg("mean", flat), seg("mean", flat * flat), seg("max", flat), seg("min", flat))
        raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

    def _pna_combine(self, graph, mean, sq_mean, mx, mn):
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1
        std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
        features = torch.cat(
            [mean.unsqueeze(-1), mx.unsqueeze(-1), mn.unsqueeze(-1), std.unsqueeze(-1)], dim=-1
        ).flatten(-2)
        scale = degree_out.log()
        scale = scale / scale.mean()
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        return (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

    # ---------------------------------------------------------------- chunked mode
    def _relation_input(self, graph, batch_size):
        if self.dependent:
            rel = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            rel = self.relation.weight.expand(batch_size, -1, -1)
        return rel.transpose(0, 1)                         # [R, B, D]

    def _chunk_messages(self, input, relation_input, node_in, relation, weight):
        """DistMult / TransE messages for one slice of edges: [k, B, D]."""
        node_input = input[node_in]
        edge_input = relation_input[relation]
        if self.message_func == "transe":
            m = edge_input + node_input
        elif self.message_func == "distmult":
            m = edge_input * node_input
        else:
            raise ValueError("chunked mode supports transe/distmult, got `%s`" % self.message_func)
        return m * weight.unsqueeze(-1).unsqueeze(-1)

    def _chunk_reduce(self, input, relation_input, node_in, relation, weight, dst, n):
        """Per-destination partial (sum, sumsq, max, min) of one slice; returns dense [4, n, B*D]."""
        m = self._chunk_messages(input, relation_input, node_in, relation, weight).flatten(1)
        uniq, counts = torch.unique_consecutive(dst, return_counts=True)
        parts = []
        for red in ("sum", "max", "min") if self.aggregate_func == "pna" else ("sum",):
            parts.append(torch.segment_reduce(m, red, lengths=counts, axis=0, unsafe=True))
        if self.aggregate_func == "pna":
            parts.insert(1, torch.segment_reduce(m * m, "sum", lengths=counts, axis=0, unsafe=True))
        out = []
        for i, p in enumerate(parts):
            fill = 0.0 if i < 2 else (float("-inf") if i == 2 else float("inf"))
            dense = torch.full((n, m.shape[1]), fill, dtype=m.dtype, device=m.device)
            dense[uniq] = p                                # unique indices -> deterministic
            out.append(dense)
        return torch.stack(out)

    def message_and_aggregate(self, graph, input):
        if self.chunk_edges is None:
            # whole mode: the stock message() (which appends the boundary) + our aggregate()
            return self.aggregate(graph, self.message(graph, input))
        assert graph.num_relation == self.num_relation
        layout = dst_sorted_layout(graph)
        E, n = int(graph.num_edge), int(graph.num_node)
        order = layout["order"]
        edge_order = order[order < E]                      # dst-sorted edge indices (boundary handled apart)
        node_in, node_out, relation = graph.edge_list.t()
        batch_size = len(graph.query)
        relation_input = self._relation_input(graph, batch_size)
        boundary = graph.boundary if self.set_boundary else input
        BD = input.shape[1] * input.shape[2]
        # running partials in slice order: sum, sumsq, max, min (boundary counts as one message per node)
        b = boundary.flatten(1)
        acc = torch.stack([b, b * b, b, b]) if self.aggregate_func == "pna" else b.unsqueeze(0)
        for start in range(0, E, self.chunk_edges):
            idx = edge_order[start:start + self.chunk_edges]
            part = _ckpt.checkpoint(
                self._chunk_reduce, input, relation_input, node_in[idx], relation[idx],
                graph.edge_weight[idx], node_out[idx], n, use_reentrant=False,
            )
            if self.aggregate_func == "pna":
                acc = torch.stack([acc[0] + part[0], acc[1] + part[1],
                                   torch.maximum(acc[2], part[2]), torch.minimum(acc[3], part[3])])
            else:
                acc = acc + part
        shape = (n,) + tuple(input.shape[1:])
        if self.aggregate_func == "sum":
            return acc[0].view(shape)
        count = layout["lengths"].to(input.dtype).unsqueeze(-1)   # edges + 1 boundary per node
        if self.aggregate_func == "mean":
            return (acc[0] / count).view(shape)
        if self.aggregate_func == "pna":
            mean = (acc[0] / count).view(shape); sq_mean = (acc[1] / count).view(shape)
            return self._pna_combine(graph, mean, sq_mean, acc[2].view(shape), acc[3].view(shape))
        raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
