"""PathNet node classifier for the INV-GNN four-arm experiments.

Derived from ``pathnet/model.py::PathNetNC`` of the PyTorch port (a faithful port of upstream zjunet/PathNet
``PathNet``, IJCAI 2022, which also fixes an upstream heterophily indexing bug) with two changes, both towards
upstream / stock PathNet:

1. ``structure_only`` is gone: the port's default zeroed the ego in the final ``cat(ego, h_agg)`` and added
   a LayerNorm to make PathNet *more* structure-sensitive on IMDb — a change to the baseline that inflates the
   originals' cross-variant variance. Stock PathNet concatenates the ego embedding.
2. The attention query is the raw ``fc0`` ego embedding (upstream ``PathNet``), not ``nets[0](fc0(ego))``.

Everything else (``fc0`` → distance-specific ``nets[d]`` → LSTM read tail→ego → softmax attention over walks →
mean → ``fc2`` on ``[ego, h_agg]``) is unchanged. ``PathNetHomoNC`` ports upstream's ``PathNet_homo``, the variant
upstream runs on homophilous graphs; it is the model used for IMDb and Freebase here. All gathers are advanced indexing (deterministic backward under
``torch.use_deterministic_algorithms(True)``); the LSTM is cuDNN with ``CUBLAS_WORKSPACE_CONFIG`` set.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathNetNC(nn.Module):
    def __init__(self, feature_length: int, hidden_size: int, out_size: int, walk_length: int, dropout: float = 0.7):
        super().__init__()
        self.hidden_size, self.walk_length, self.dropout = hidden_size, walk_length, dropout
        self.fc0 = nn.Linear(feature_length, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, out_size)
        self.nets = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(walk_length)])
        self.attw = nn.Linear(2 * hidden_size, 1)
        self.Lrelu = nn.LeakyReLU()
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor, walks: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        """x [N, F]; walks / dists [T, W, L] (walks[:, :, 0] = target). Returns logits [T, C]."""
        T, W, L = walks.shape
        h = self.fc0(x)                                                        # [N, hid]
        nei = h[walks.reshape(-1)]                                             # [T*W*L, hid]
        stacked = torch.stack([net(nei) for net in self.nets], dim=1)          # [T*W*L, L, hid]
        d = dists.reshape(-1).clamp(0, self.walk_length - 1)
        nei = stacked[torch.arange(stacked.size(0), device=walks.device), d].view(T * W, L, self.hidden_size)
        ego = h[walks[:, 0, 0]]                                                # [T, hid] raw fc0 ego (upstream)
        seq = torch.flip(nei, dims=[1]).transpose(0, 1)                        # [L, T*W, hid], tail -> ego
        seq = F.dropout(seq, p=self.dropout, training=self.training)
        _, (h_n, _) = self.LSTM(seq)
        h_n = h_n.squeeze(0).view(T, W, self.hidden_size)
        att = F.softmax(self.Lrelu(self.attw(torch.cat((h_n, ego[:, None, :].expand(T, W, -1)), dim=-1))), dim=1)
        h_agg = torch.mean(att * h_n, dim=1)                                   # [T, hid]
        self._diag_norms = (float(ego.detach().norm(dim=1).mean()), float(h_agg.detach().norm(dim=1).mean()))
        out = F.dropout(torch.cat((ego, h_agg), dim=1), p=self.dropout, training=self.training)
        return self.fc2(out)


class PathNetHomoNC(nn.Module):
    """Port of upstream zjunet/PathNet ``PathNet_homo`` (PathNet_run.py; the model upstream runs on its homophilous
    datasets): ReLU after ``fc0`` and after the distance-specific ``nets[d]``; LSTM reads the walk ego -> tail; walk
    gate ``(1 + attw([h_n, ego_walk])) * h_n`` (no softmax over walks); mean over walks; ``fc2`` on ``[relu(fc0(ego)), h_agg]``.
    Upstream applies dropout to the per-position inputs before the LSTM and to the concat, as here."""

    def __init__(self, feature_length: int, hidden_size: int, out_size: int, walk_length: int, dropout: float = 0.7):
        super().__init__()
        self.hidden_size, self.walk_length, self.dropout = hidden_size, walk_length, dropout
        self.fc0 = nn.Linear(feature_length, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, out_size)
        self.nets = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(walk_length)])
        self.attw = nn.Linear(2 * hidden_size, 1)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor, walks: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        T, W, L = walks.shape
        h = F.relu(self.fc0(x))
        nei = h[walks.reshape(-1)]
        stacked = torch.stack([net(nei) for net in self.nets], dim=1)
        d = dists.reshape(-1).clamp(0, self.walk_length - 1)
        nei = F.relu(stacked[torch.arange(stacked.size(0), device=walks.device), d]).view(T * W, L, self.hidden_size)
        ego_walk = nei[:, 0, :]                                                # [T*W, hid]
        seq = F.dropout(nei.transpose(0, 1), p=self.dropout, training=self.training)   # [L, T*W, hid], ego -> tail
        _, (h_n, _) = self.LSTM(seq)
        h_n = h_n.squeeze(0)                                                   # [T*W, hid]
        h_n = (1 + self.attw(torch.cat((h_n, ego_walk), dim=-1))) * h_n
        h_agg = h_n.view(T, W, self.hidden_size).mean(dim=1)
        ego = h[walks[:, 0, 0]]
        self._diag_norms = (float(ego.detach().norm(dim=1).mean()), float(h_agg.detach().norm(dim=1).mean()))
        out = F.dropout(torch.cat((ego, h_agg), dim=1), p=self.dropout, training=self.training)
        return self.fc2(out)
