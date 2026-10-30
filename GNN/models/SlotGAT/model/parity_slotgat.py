"""Deterministic, plan-driven port of the upstream ``model/gnn.py:slotGAT``.

Why this exists: on the IMDb year graph the upstream
DGL model scores .6547 while the capacity-matched control built from
``GenuineConditionalSlotGAT`` scores .6173 on the SAME message set.  Reading the two side by
side, the re-implementation departs from the published architecture in six places:

1. readout -- upstream L2-normalises every slot (``l2byslot``), applies slot attention
   (``macroLinear`` with a bias initialised N(0, 1.414/sqrt(d)); ``macroSemanticVec``
   ~ N(0,1)) and then L2-normalises the final logits before the cross-entropy;
   the re-implementation applies slot attention to raw slots and never normalises;
2. residuals -- upstream projects the layer input through ``res_fc[T, in, out*heads]``
   whenever the per-slot input width differs from the PER-HEAD output width, i.e. on every
   hidden layer and on the output layer (1.05M of the 3.09M parameters); the
   re-implementation uses an identity residual on hidden layers and none on the output;
3. attention residual -- upstream mixes each hidden layer's attention with the previous
   layer's (``alpha = 0.05``); the re-implementation has none;
4. edge path -- upstream ``Embedding(num_etypes, edge_feats) -> Linear(edge_feats,
   edge_feats*heads) -> attn_e[heads, edge_feats]``; the re-implementation projects to
   ``heads*head_dim`` and scores against a ``head_dim`` vector;
5. per-slot projection -- upstream has no bias; the re-implementation adds one;
6. initialisation -- upstream xavier_normal with gain sqrt(2) (and 1.414 on the input
   projections / SA); the re-implementation xavier_uniform with gain 1.

This module reproduces the upstream computation exactly (verified with copied weights in
``tests/test_parity_slotgat.py``) while keeping the deterministic degree-bucket aggregation
and the ``ConditionalContextPlan`` interface, so invariant / matched-control / universal /
augmentation arms can all use it.  Each departure can be switched back off through
``ablate`` to attribute the gap.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from conditional_bias import CHANNEL_NAMES, ConditionalContextPlan, NodeTypeLayout
from model.genuine_conditional_slotgat import (
    GenuineConditionalSlotGAT,
    deterministic_index_select,
)

ABLATIONS = ("readout_norm", "out_residual", "attn_residual", "edge_path", "proj_bias", "init")


class ParitySlotLayer(nn.Module):
    """``slotGATConv`` with deterministic aggregation.

    Slot features are carried as ``[N, slots, in_per_slot]``; the layer output is
    ``[N, slots, heads * out]`` (upstream's ``[N, heads, slots*out]`` re-ordered).
    """

    def __init__(self, *, num_slots: int, in_per_slot: int, out_feats: int, num_heads: int,
                 num_edge_types: int, edge_feats: int, feat_drop: float, attn_drop: float,
                 negative_slope: float, residual: bool, activation: bool, alpha: float,
                 ablate: frozenset[str], inputhead: bool = False) -> None:
        super().__init__()
        self.inputhead = bool(inputhead)
        self.num_slots, self.in_per_slot, self.out_feats, self.num_heads = int(num_slots), int(in_per_slot), int(out_feats), int(num_heads)
        self.edge_feats = int(edge_feats)
        self.feat_drop, self.attn_drop, self.negative_slope = float(feat_drop), float(attn_drop), float(negative_slope)
        self.use_activation = bool(activation)
        self.alpha = 0.0 if "attn_residual" in ablate else float(alpha)
        self.ablate = ablate
        H, O, T = self.num_heads, self.out_feats, self.num_slots
        # upstream: fc[T, in, O*H]  (no bias)
        self.fc = nn.Parameter(torch.empty(T, self.in_per_slot, O * H))
        self.fc_bias = nn.Parameter(torch.zeros(T, O * H)) if "proj_bias" in ablate else None
        self.attn_l = nn.Parameter(torch.empty(1, H, O * T))
        self.attn_r = nn.Parameter(torch.empty(1, H, O * T))
        self.edge_emb = nn.Embedding(int(num_edge_types), self.edge_feats)
        if "edge_path" in ablate:   # re-implementation's parametrisation
            self.fc_e = nn.Linear(self.edge_feats, H * O, bias=False)
            self.attn_e = nn.Parameter(torch.empty(1, H, O))
        else:                       # upstream
            self.fc_e = nn.Linear(self.edge_feats, self.edge_feats * H, bias=False)
            self.attn_e = nn.Parameter(torch.empty(1, H, self.edge_feats))
        # Residual. Upstream compares the per-slot input width with the PER-HEAD output
        # width (``_in_dst_feats != out_feats``), so with 8 heads every hidden layer AND the
        # output layer get a projected residual ``res_fc[T, in, O*H]`` -- that is the whole
        # 1.05M-parameter gap between upstream (3,091,147) and the re-implementation
        # (2,037,302), which uses an identity residual when widths match and none otherwise.
        self.residual = bool(residual)
        self.res_fc = None
        if self.residual:
            if "out_residual" in ablate:                       # re-implementation semantics
                self.residual = self.in_per_slot == O * H
            elif self.in_per_slot != O:                        # upstream semantics
                self.res_fc = nn.Parameter(torch.empty(T, self.in_per_slot, O * H))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if "init" in self.ablate:
            for p in (self.fc, self.attn_l, self.attn_r, self.attn_e, self.fc_e.weight, self.edge_emb.weight):
                nn.init.xavier_uniform_(p)
            if self.res_fc is not None:
                nn.init.xavier_uniform_(self.res_fc)
            return
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc, gain=gain)
        # upstream leaves nn.Embedding at its default N(0,1) initialisation

    def forward(self, h_flat: torch.Tensor, *, src, dst, channels, destination_bucket_specs,
                source_bucket_specs, res_attn: torch.Tensor | None):
        """``h_flat``: ``[N, H_prev * T * O_prev]`` in upstream's head-major layout.

        Upstream re-interprets its input with the layer's OWN head count
        (``view(-1, num_heads, T, in/num_heads)``): consistent on the hidden layers, but the
        1-head output layer then slices the 8-head input into T contiguous blocks, i.e. its
        "slots" mix the previous layer's slots.  Reproduced as is -- it is part of the
        published model (``model/conv.py:slotGATConv.forward``).
        """
        N, T, H, O = h_flat.shape[0], self.num_slots, self.num_heads, self.out_feats
        h = F.dropout(h_flat, p=self.feat_drop, training=self.training)
        h = h.view(N, H if not self.inputhead else 1, T, -1).permute(2, 0, 1, 3).flatten(2)   # [T, N, in]
        feat = torch.bmm(h, self.fc)                                               # [T, N, O*H]
        if self.fc_bias is not None:
            feat = feat + self.fc_bias[:, None, :]
        feat = feat.permute(1, 0, 2).view(N, T, H, O).permute(0, 2, 1, 3).reshape(N, H, T * O)  # [N, H, T*O]
        el = (feat * self.attn_l).sum(-1)                                         # [N, H]
        er = (feat * self.attn_r).sum(-1)
        e_feat = self.fc_e(self.edge_emb(channels)).view(-1, H, self.attn_e.shape[-1])
        ee = (e_feat * self.attn_e).sum(-1)                                        # [E, H]
        e = deterministic_index_select(el, src, source_bucket_specs) + \
            deterministic_index_select(er, dst, destination_bucket_specs) + ee
        e = F.leaky_relu(e, self.negative_slope)
        msgs = deterministic_index_select(feat, src, source_bucket_specs)          # [E, H, T*O]
        # edge softmax over each destination's incoming edges + attention residual + dropout
        node_chunks, agg_chunks, attn_chunks, attn_idx = [], [], [], []
        for nodes, edge_indices in destination_bucket_specs:
            a = torch.softmax(e[edge_indices], dim=1)                              # [B, deg, H]
            if res_attn is not None and self.alpha > 0:
                a = a * (1 - self.alpha) + res_attn[edge_indices] * self.alpha
            a = F.dropout(a, p=self.attn_drop, training=self.training)
            agg_chunks.append((a[..., None] * msgs[edge_indices]).sum(1))          # [B, H, T*O]
            node_chunks.append(nodes); attn_chunks.append(a.reshape(-1, H)); attn_idx.append(edge_indices.reshape(-1))
        order = torch.argsort(torch.cat(node_chunks))
        rst = torch.cat(agg_chunks)[order]                                         # [N, H, T*O]
        attn = torch.empty(e.shape, device=e.device, dtype=e.dtype)
        attn[torch.cat(attn_idx)] = torch.cat(attn_chunks)
        if self.residual:
            if self.res_fc is not None:
                resval = torch.bmm(h, self.res_fc).permute(1, 0, 2).view(N, T, H, O).permute(0, 2, 1, 3).reshape(N, H, T * O)
            else:
                resval = h.permute(1, 0, 2).reshape(N, T, H, O).permute(0, 2, 1, 3).reshape(N, H, T * O)
            rst = rst + resval
        if self.use_activation:
            rst = F.elu(rst)
        return rst.flatten(1), attn.detach()                                       # [N, H*T*O]


class ParitySlotGAT(GenuineConditionalSlotGAT):
    """Upstream ``slotGAT`` semantics on the deterministic plan interface."""

    def __init__(self, *, layout: NodeTypeLayout, in_dims: Sequence[int], num_classes: int, hidden_dim: int,
                 num_heads: int, num_layers: int, edge_feats: int, dropout_feat: float, dropout_attn: float,
                 slope: float, aggregator: str, sa_attention_dim: int, alpha: float = 0.05,
                 num_edge_types: int | None = None, ablate: Sequence[str] = ()) -> None:
        bad = set(ablate) - set(ABLATIONS)
        if bad:
            raise ValueError(f"unknown ablations {sorted(bad)}; choose from {ABLATIONS}")
        self.ablate = frozenset(ablate)
        super().__init__(layout=layout, in_dims=in_dims, num_classes=num_classes, hidden_dim=hidden_dim,
                         num_heads=num_heads, num_layers=num_layers, edge_feats=edge_feats, dropout_feat=dropout_feat,
                         dropout_attn=dropout_attn, slope=slope, aggregator=aggregator, sa_attention_dim=sa_attention_dim)
        del self.layers, self.slot_attention
        T = self.num_slots
        E = int(num_edge_types) if num_edge_types else len(CHANNEL_NAMES)
        heads = [int(num_heads)] * int(num_layers) + [1]
        layers = [ParitySlotLayer(num_slots=T, in_per_slot=hidden_dim, out_feats=hidden_dim, num_heads=heads[0],
                                  num_edge_types=E, edge_feats=edge_feats, feat_drop=dropout_feat, attn_drop=dropout_attn,
                                  negative_slope=slope, residual=False, activation=True, alpha=alpha, ablate=self.ablate, inputhead=True)]
        for l in range(1, int(num_layers)):
            layers.append(ParitySlotLayer(num_slots=T, in_per_slot=hidden_dim * heads[l - 1], out_feats=hidden_dim, num_heads=heads[l],
                                          num_edge_types=E, edge_feats=edge_feats, feat_drop=dropout_feat, attn_drop=dropout_attn,
                                          negative_slope=slope, residual=True, activation=True, alpha=alpha, ablate=self.ablate))
        layers.append(ParitySlotLayer(num_slots=T, in_per_slot=hidden_dim * heads[-2], out_feats=num_classes, num_heads=heads[-1],
                                      num_edge_types=E, edge_feats=edge_feats, feat_drop=dropout_feat, attn_drop=dropout_attn,
                                      negative_slope=slope, residual=True, activation=False, alpha=alpha, ablate=self.ablate))
        self.layers = nn.ModuleList(layers)
        # readout (upstream "SA" aggregator)
        self.macroLinear = nn.Linear(num_classes, int(sa_attention_dim), bias=True)
        self.macroSemanticVec = nn.Parameter(torch.empty(int(sa_attention_dim), 1))
        self.register_buffer("epsilon", torch.tensor([1e-12]))
        self.reset_readout_parameters()

    def reset_readout_parameters(self) -> None:
        if "init" in self.ablate:
            nn.init.xavier_uniform_(self.macroLinear.weight); nn.init.zeros_(self.macroLinear.bias)
            bound = 1.0 / math.sqrt(max(1, self.macroSemanticVec.shape[0])); nn.init.uniform_(self.macroSemanticVec, -bound, bound)
            return
        nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414)
        nn.init.normal_(self.macroLinear.bias, std=1.414 * math.sqrt(1 / self.macroLinear.bias.numel()))
        nn.init.normal_(self.macroSemanticVec, std=1)
        for fc in self.input_projections:      # upstream fc_list: xavier_normal gain 1.414, default bias
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    # ------------------------------------------------------------------ forward
    def forward_edges(self, features_list: Sequence[torch.Tensor], src, dst, channels,
                      destination_bucket_specs, source_bucket_specs) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self._initialize_slots(features_list)                              # [N, T, hidden]
        h = slots.flatten(1)                                                       # upstream input layout [N, T*hidden]
        res_attn = None
        for layer in self.layers[:-1]:
            h, res_attn = layer(h, src=src, dst=dst, channels=channels, destination_bucket_specs=destination_bucket_specs,
                                source_bucket_specs=source_bucket_specs, res_attn=res_attn)
        logits, _ = self.layers[-1](h, src=src, dst=dst, channels=channels, destination_bucket_specs=destination_bucket_specs,
                                    source_bucket_specs=source_bucket_specs, res_attn=None)   # [N, T*C] (1 head)
        logits = logits.view(-1, self.num_slots, self.num_classes)
        if "readout_norm" not in self.ablate:
            logits = logits / torch.max(torch.norm(logits, dim=2, keepdim=True), self.epsilon)   # l2byslot
        slot_scores = (torch.tanh(self.macroLinear(logits)) @ self.macroSemanticVec).mean(0, keepdim=True)  # [1, T, 1]
        slot_weights = F.softmax(slot_scores, dim=1)
        out = (logits * slot_weights).sum(1)                                        # [N, C]
        if "readout_norm" not in self.ablate:
            out = out / torch.max(torch.norm(out, dim=1, keepdim=True), self.epsilon)
        return out, slot_weights.flatten()

    def forward(self, features_list: Sequence[torch.Tensor], context_plan: ConditionalContextPlan):
        src, dst, channels, dbs, sbs = self._context_tensors(context_plan, features_list[0].device)
        return self.forward_edges(features_list, src, dst, channels, dbs, sbs)

    # ------------------------------------------------------------------ weight import (tests)
    @torch.no_grad()
    def load_from_upstream(self, upstream) -> None:
        """Copy the weights of a ``model/gnn.py:slotGAT`` with the same geometry."""
        for mine, theirs in zip(self.input_projections, upstream.fc_list):
            mine.weight.copy_(theirs.weight); mine.bias.copy_(theirs.bias)
        assert len(self.layers) == len(upstream.gat_layers)
        for mine, theirs in zip(self.layers, upstream.gat_layers):
            mine.fc.copy_(theirs.fc); mine.attn_l.copy_(theirs.attn_l); mine.attn_r.copy_(theirs.attn_r)
            mine.attn_e.copy_(theirs.attn_e); mine.fc_e.weight.copy_(theirs.fc_e.weight); mine.edge_emb.weight.copy_(theirs.edge_emb.weight)
            if mine.res_fc is not None:
                mine.res_fc.copy_(theirs.res_fc)
        self.macroLinear.weight.copy_(upstream.macroLinear.weight); self.macroLinear.bias.copy_(upstream.macroLinear.bias)
        self.macroSemanticVec.copy_(upstream.macroSemanticVec)
