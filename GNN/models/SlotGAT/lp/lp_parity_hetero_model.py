"""Deterministic multi-slot SlotGAT link-prediction model (IMDb / DBLP).

Same construction as ``lp_parity_model.py`` (WordNet, one slot) but with the full slot machinery of
``SlotGAT/model/parity_slotgat.py:ParitySlotLayer`` (a copy, T slots = node types, upstream layouts
reproduced including the 1-head output layer's slot re-slicing), driven by a ``(src, dst, channel)``
edge list, aggregated with ``torch.segment_reduce`` over dst-sorted edges (fixed order, no atomics)
and ``index_select`` gathers (deterministic backward under ``torch.use_deterministic_algorithms``).

Input: node features per type projected into the type's own slot (upstream ``fc_list`` +
``emsen_ft``; IMDb canonical 3008-d table), or — when the bundle has no features (DBLP) — a per-node
embedding table placed in the node's slot. Readout: concat of every layer's head-mean slot vector +
LayerNorm (``concat_ln``). Decoder: diagonal DistMult over the single task relation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from lp_parity_model import MessagePlan  # dst-sorted plan + segment lengths (unchanged)


class HeteroParitySlotLayer(nn.Module):
    def __init__(self, *, num_slots: int, in_per_slot: int, out_feats: int, num_heads: int, num_edge_types: int,
                 edge_feats: int, feat_drop: float, attn_drop: float, negative_slope: float, residual: bool,
                 activation: bool, alpha: float, inputhead: bool = False):
        super().__init__()
        self.inputhead = bool(inputhead)
        self.num_slots, self.in_per_slot, self.out_feats, self.num_heads = int(num_slots), int(in_per_slot), int(out_feats), int(num_heads)
        self.feat_drop, self.attn_drop, self.negative_slope = float(feat_drop), float(attn_drop), float(negative_slope)
        self.use_activation, self.alpha = bool(activation), float(alpha)
        H, O, T = self.num_heads, self.out_feats, self.num_slots
        self.fc = nn.Parameter(torch.empty(T, self.in_per_slot, O * H))
        self.attn_l = nn.Parameter(torch.empty(1, H, O * T))
        self.attn_r = nn.Parameter(torch.empty(1, H, O * T))
        self.edge_emb = nn.Embedding(int(num_edge_types), int(edge_feats))
        self.fc_e = nn.Linear(int(edge_feats), int(edge_feats) * H, bias=False)
        self.attn_e = nn.Parameter(torch.empty(1, H, int(edge_feats)))
        self.residual = bool(residual)
        self.res_fc = nn.Parameter(torch.empty(T, self.in_per_slot, O * H)) if (self.residual and self.in_per_slot != O) else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for p in (self.fc, self.attn_l, self.attn_r, self.attn_e, self.fc_e.weight):
            nn.init.xavier_normal_(p, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc, gain=gain)

    def forward(self, h_flat: torch.Tensor, plan: MessagePlan, res_attn: torch.Tensor | None):
        N, T, H, O = h_flat.shape[0], self.num_slots, self.num_heads, self.out_feats
        h = F.dropout(h_flat, p=self.feat_drop, training=self.training)
        h = h.view(N, H if not self.inputhead else 1, T, -1).permute(2, 0, 1, 3).flatten(2)   # [T, N, in]
        feat = torch.bmm(h, self.fc)                                                            # [T, N, O*H]
        feat = feat.permute(1, 0, 2).view(N, T, H, O).permute(0, 2, 1, 3).reshape(N, H, T * O)  # [N, H, T*O]
        el = (feat * self.attn_l).sum(-1)                                                       # [N, H]
        er = (feat * self.attn_r).sum(-1)
        e_feat = self.fc_e(self.edge_emb(plan.channels)).view(-1, H, self.attn_e.shape[-1])
        ee = (e_feat * self.attn_e).sum(-1)                                                     # [E, H]
        e = F.leaky_relu(el.index_select(0, plan.src) + er.index_select(0, plan.dst) + ee, self.negative_slope)
        emax = torch.segment_reduce(e, "max", lengths=plan.lengths, axis=0, unsafe=True)
        ex = torch.exp(e - emax.index_select(0, plan.dst))
        esum = torch.segment_reduce(ex, "sum", lengths=plan.lengths, axis=0, unsafe=True)
        a = F.dropout(ex / esum.index_select(0, plan.dst), p=self.attn_drop, training=self.training)
        if res_attn is not None and self.alpha > 0:
            a = a * (1 - self.alpha) + res_attn * self.alpha
        attn = a.detach()
        msgs = feat.index_select(0, plan.src) * a[..., None]                                    # [E, H, T*O]
        rst = torch.segment_reduce(msgs, "sum", lengths=plan.lengths, axis=0, unsafe=True)      # [N, H, T*O]
        if self.residual:
            if self.res_fc is not None:
                resval = torch.bmm(h, self.res_fc).permute(1, 0, 2).view(N, T, H, O).permute(0, 2, 1, 3).reshape(N, H, T * O)
            else:
                resval = h.permute(1, 0, 2).reshape(N, -1, T * O)          # upstream Identity: view(N, -1, O*T), broadcast over heads
            rst = rst + resval
        if self.use_activation:
            rst = F.elu(rst)
        return rst.flatten(1), attn                                                             # [N, H*T*O]


class HeteroParitySlotGATLPEncoder(nn.Module):
    def __init__(self, *, node_type: torch.Tensor, num_types: int, feature_dim: int | None, num_edge_types: int,
                 hidden: int, out_dim: int, num_layers: int, num_heads: int, edge_feats: int, feat_drop: float,
                 attn_drop: float, slope: float, alpha: float = 0.05):
        super().__init__()
        self.register_buffer("node_type", node_type.to(torch.int64))
        N, T = int(node_type.numel()), int(num_types)
        self.num_types, self.hidden, self.out_dim, self.num_layers = T, int(hidden), int(out_dim), int(num_layers)
        if feature_dim is not None:
            # upstream fc_list: one Linear per node type, xavier_normal gain 1.414, default bias
            self.fc_list = nn.ModuleList([nn.Linear(int(feature_dim), hidden, bias=True) for _ in range(T)])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)
            self.entity_table = None
        else:
            self.fc_list = None
            self.entity_table = nn.Embedding(N, hidden)
            nn.init.normal_(self.entity_table.weight, std=1.0 / math.sqrt(hidden))
        heads = [int(num_heads)] * int(num_layers) + [1]
        mk = lambda **kw: HeteroParitySlotLayer(num_slots=T, num_edge_types=num_edge_types, edge_feats=edge_feats, feat_drop=feat_drop,
                                                attn_drop=attn_drop, negative_slope=slope, alpha=alpha, **kw)
        layers = [mk(in_per_slot=hidden, out_feats=hidden, num_heads=heads[0], residual=False, activation=True, inputhead=True)]
        for l in range(1, int(num_layers)):
            layers.append(mk(in_per_slot=hidden * heads[l - 1], out_feats=hidden, num_heads=heads[l], residual=True, activation=True))
        layers.append(mk(in_per_slot=hidden * heads[-2], out_feats=out_dim, num_heads=heads[-1], residual=True, activation=False))
        self.layers = nn.ModuleList(layers)
        self.embedding_dim = T * (hidden * (num_layers + 1) + out_dim)
        self.ln = nn.LayerNorm(self.embedding_dim)

    def initial_slots(self, x: torch.Tensor | None) -> torch.Tensor:
        N, T, d = self.node_type.numel(), self.num_types, self.hidden
        slots = x.new_zeros(N, T, d) if x is not None else self.entity_table.weight.new_zeros(N, T, d)
        own = F.one_hot(self.node_type, T).to(slots.dtype)                    # [N, T]
        if self.fc_list is not None:
            proj = torch.stack([fc(x) for fc in self.fc_list], 1)              # [N, T, d]  (every type's projection)
            slots = proj * own[..., None]                                       # keep only the node's own slot
        else:
            slots = self.entity_table.weight[:, None, :] * own[..., None]
        return slots.flatten(1)                                                 # [N, T*d] (slot-major, upstream layout)

    def forward(self, plan: MessagePlan, x: torch.Tensor | None) -> torch.Tensor:
        h = self.initial_slots(x)
        embs = [h]
        res_attn = None
        for layer in self.layers[:-1]:
            h, res_attn = layer(h, plan, res_attn)                             # [N, H*T*hidden]
            embs.append(h.view(h.shape[0], layer.num_heads, -1).mean(1))       # [N, T*hidden]
        out, _ = self.layers[-1](h, plan, None)                                # [N, T*out]
        return self.ln(torch.cat(embs + [out], dim=1))


class DistMultSingle(nn.Module):
    """Diagonal DistMult for one task relation."""

    def __init__(self, dim: int):
        super().__init__()
        self.rel = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.rel, -math.sqrt(6.0 / (dim + 1)), math.sqrt(6.0 / (dim + 1)))

    def score_pairs(self, ent, heads, tails):
        return (ent[heads] * self.rel * ent[tails]).sum(-1)

    def score_candidates(self, ent, heads, cand):
        """heads [Q], cand [Q, C] -> [Q, C]."""
        return ((ent[heads] * self.rel)[:, None, :] * ent[cand]).sum(-1)
