"""Deterministic SlotGAT link-prediction model for the four-arm WordNet experiment.

A copy of ``SlotGAT/model/parity_slotgat.py:ParitySlotLayer`` (the upstream ``slotGATConv`` computation)
specialised to one node type (one slot) and driven by a plain ``(src, dst, channel)`` edge list, with the
aggregation done as in the RGCN / CMPNN invariant packages: edges sorted by destination, edge softmax and
message sums via ``torch.segment_reduce`` (fixed order, no atomics), gathers via ``index_select`` (whose
CUDA backward is deterministic under ``torch.use_deterministic_algorithms(True)``). The degree-bucket
operator of the NC parity model gives the same numbers but is ~15x slower on the 1.4 M-message program.
Nothing under ``SlotGAT/`` is imported, so the queued NC jobs cannot be affected.

Encoder = entity table -> ParitySlotLayer x (num_layers + output layer) -> concat of every layer's
representation -> LayerNorm (the ``concat_ln`` readout). Decoder = diagonal DistMult over the 11 task relations.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class MessagePlan:
    """Edge list sorted by (dst, channel, src) with per-destination segment lengths.

    Aggregation uses ``torch.segment_reduce`` over the sorted edges (one fixed reduction order per
    destination — the same device-independent scheme as the RGCN / CMPNN invariant packages) and
    ``index_select`` gathers, whose CUDA backward is deterministic under
    ``torch.use_deterministic_algorithms(True)``. Every node has a self message, so lengths > 0.
    """

    def __init__(self, src: torch.Tensor, dst: torch.Tensor, channels: torch.Tensor, num_nodes: int):
        key = (dst.to(torch.int64) * (int(channels.max()) + 1) + channels.to(torch.int64)) * int(num_nodes) + src.to(torch.int64)
        order = torch.argsort(key, stable=True)
        self.src, self.dst, self.channels = src[order].contiguous(), dst[order].contiguous(), channels[order].contiguous()
        self.num_nodes = int(num_nodes)
        counts = torch.bincount(self.dst.cpu(), minlength=self.num_nodes)      # CPU: no deterministic CUDA bincount
        if bool((counts == 0).any()):
            raise RuntimeError("every node needs an incoming self message")
        self.lengths = counts.to(src.device)


# ----------------------------------------------------------------------------- parity layer (T = 1)
class ParitySlotLayer(nn.Module):
    """``slotGATConv`` (upstream) with deterministic aggregation, one slot."""

    def __init__(self, *, in_feats: int, out_feats: int, num_heads: int, num_edge_types: int, edge_feats: int,
                 feat_drop: float, attn_drop: float, negative_slope: float, residual: bool, activation: bool,
                 alpha: float, inputhead: bool = False):
        super().__init__()
        self.inputhead = bool(inputhead)
        self.in_feats, self.out_feats, self.num_heads = int(in_feats), int(out_feats), int(num_heads)
        self.feat_drop, self.attn_drop, self.negative_slope = float(feat_drop), float(attn_drop), float(negative_slope)
        self.use_activation, self.alpha = bool(activation), float(alpha)
        H, O = self.num_heads, self.out_feats
        self.fc = nn.Parameter(torch.empty(1, self.in_feats, O * H))          # upstream fc[T, in, O*H]
        self.attn_l = nn.Parameter(torch.empty(1, H, O))
        self.attn_r = nn.Parameter(torch.empty(1, H, O))
        self.edge_emb = nn.Embedding(int(num_edge_types), int(edge_feats))
        self.fc_e = nn.Linear(int(edge_feats), int(edge_feats) * H, bias=False)
        self.attn_e = nn.Parameter(torch.empty(1, H, int(edge_feats)))
        self.residual = bool(residual)
        self.res_fc = nn.Parameter(torch.empty(1, self.in_feats, O * H)) if (self.residual and self.in_feats != O) else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for p in (self.fc, self.attn_l, self.attn_r, self.attn_e, self.fc_e.weight):
            nn.init.xavier_normal_(p, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc, gain=gain)

    def forward(self, h_flat: torch.Tensor, plan: MessagePlan, res_attn: torch.Tensor | None):
        N, H, O = h_flat.shape[0], self.num_heads, self.out_feats
        h = F.dropout(h_flat, p=self.feat_drop, training=self.training)
        h = h.view(1, N, -1)                                                   # [T=1, N, in]
        feat = torch.bmm(h, self.fc).view(N, H, O)                             # [N, H, O]
        el = (feat * self.attn_l).sum(-1)                                      # [N, H]
        er = (feat * self.attn_r).sum(-1)
        e_feat = self.fc_e(self.edge_emb(plan.channels)).view(-1, H, self.attn_e.shape[-1])
        ee = (e_feat * self.attn_e).sum(-1)                                    # [E, H]
        e = el.index_select(0, plan.src) + er.index_select(0, plan.dst) + ee
        e = F.leaky_relu(e, self.negative_slope)
        # edge softmax over each destination's incoming edges (dst-sorted segments)
        emax = torch.segment_reduce(e, "max", lengths=plan.lengths, axis=0, unsafe=True)      # [N, H]
        ex = torch.exp(e - emax.index_select(0, plan.dst))
        esum = torch.segment_reduce(ex, "sum", lengths=plan.lengths, axis=0, unsafe=True)     # [N, H]
        a = F.dropout(ex / esum.index_select(0, plan.dst), p=self.attn_drop, training=self.training)
        if res_attn is not None and self.alpha > 0:           # upstream order: dropout, then the residual mix
            a = a * (1 - self.alpha) + res_attn * self.alpha
        attn = a.detach()
        msgs = feat.index_select(0, plan.src) * a[..., None]                   # [E, H, O]
        rst = torch.segment_reduce(msgs, "sum", lengths=plan.lengths, axis=0, unsafe=True)    # [N, H, O]
        if self.residual:
            # upstream: projected residual when widths differ, else an identity that broadcasts over heads
            resval = torch.bmm(h, self.res_fc).view(N, H, O) if self.res_fc is not None else h.view(N, 1, O)
            rst = rst + resval
        if self.use_activation:
            rst = F.elu(rst)
        return rst.flatten(1), attn                                            # [N, H*O]


# ----------------------------------------------------------------------------- encoder / decoder
class ParitySlotGATLPEncoder(nn.Module):
    def __init__(self, *, num_entities: int, num_edge_types: int, hidden: int, out_dim: int, num_layers: int,
                 num_heads: int, edge_feats: int, feat_drop: float, attn_drop: float, slope: float, alpha: float = 0.05,
                 readout: str = "concat_ln"):
        super().__init__()
        assert readout in ("concat_ln", "last_raw")
        self.readout = readout
        self.hidden, self.out_dim, self.num_layers = int(hidden), int(out_dim), int(num_layers)
        self.entity_table = nn.Embedding(int(num_entities), int(hidden))
        nn.init.normal_(self.entity_table.weight, std=1.0 / math.sqrt(hidden))
        heads = [int(num_heads)] * int(num_layers) + [1]
        layers = [ParitySlotLayer(in_feats=hidden, out_feats=hidden, num_heads=heads[0], num_edge_types=num_edge_types,
                                  edge_feats=edge_feats, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=slope,
                                  residual=False, activation=True, alpha=alpha, inputhead=True)]
        for l in range(1, int(num_layers)):
            layers.append(ParitySlotLayer(in_feats=hidden * heads[l - 1], out_feats=hidden, num_heads=heads[l],
                                          num_edge_types=num_edge_types, edge_feats=edge_feats, feat_drop=feat_drop,
                                          attn_drop=attn_drop, negative_slope=slope, residual=True, activation=True, alpha=alpha))
        layers.append(ParitySlotLayer(in_feats=hidden * heads[-2], out_feats=out_dim, num_heads=heads[-1],
                                      num_edge_types=num_edge_types, edge_feats=edge_feats, feat_drop=feat_drop,
                                      attn_drop=attn_drop, negative_slope=slope, residual=True, activation=False, alpha=alpha))
        self.layers = nn.ModuleList(layers)
        self.embedding_dim = hidden + num_layers * hidden + out_dim if readout == "concat_ln" else out_dim
        self.ln = nn.LayerNorm(self.embedding_dim) if readout == "concat_ln" else None

    def forward(self, plan: MessagePlan) -> torch.Tensor:
        h = self.entity_table.weight
        embs = [h]
        res_attn = None
        for layer in self.layers[:-1]:
            h, res_attn = layer(h, plan, res_attn)                           # [N, H*hidden]
            embs.append(h.view(h.shape[0], layer.num_heads, -1).mean(1))
        out, _ = self.layers[-1](h, plan, None)                                # [N, out_dim] (1 head)
        if self.readout == "last_raw":
            return out
        return self.ln(torch.cat(embs + [out], dim=1))


class DistMultDiag(nn.Module):
    def __init__(self, num_rel: int, dim: int):
        super().__init__()
        self.rel = nn.Embedding(num_rel, dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, ent, triples):
        return (ent[triples[:, 0]] * self.rel(triples[:, 1]) * ent[triples[:, 2]]).sum(-1)

    # names used by RGCN/wordnet_structural_eval.evaluate_ranking
    def score_tails(self, ent, h, r):
        return (ent[h] * self.rel(r)) @ ent.T

    def score_heads(self, ent, r, t):
        return (self.rel(r) * ent[t]) @ ent.T
