"""PathNet node-classification model.

Direct adaptation of ``PathNet`` / ``PathNet_homo`` from the upstream
``PathNet_run.py`` (zjunet/PathNet, IJCAI 2022). The upstream forward pass is
tangled with its transductive full-graph data pipeline (it selects a split with
boolean masks and re-derives per-walk index tensors on the fly). Here we keep the
*same computation* but express it over an explicit walk tensor so it can be
reused for IMDB / Freebase exactly like ``PathNNs_expressive``:

For a set of ``T`` target nodes we take ``W`` random walks of length ``L`` each
(node-id sequences beginning at the target) plus a per-node "distance to target"
index (``0`` for the ego, capped at ``L-1``). The model:

  1. embeds node features with ``fc0`` (``feature_length -> hidden``);
  2. applies a *distance-specific* linear map (``nets[dist]``) to every node on a
     walk (this is PathNet's structure-aware step);
  3. runs a shared ``LSTM`` over each walk (read tail -> ego, matching the
     upstream ``torch.flip`` on the sequence axis);
  4. aggregates the ``W`` walk summaries per target with an attention weight that
     compares each walk summary against the ego embedding;
  5. concatenates the ego embedding and classifies with ``fc2``.

``variant='hetero'`` reproduces ``PathNet`` (softmax attention, no ReLU after
``fc0``); ``variant='homo'`` reproduces ``PathNet_homo`` (ReLU + ``1 + att``
gating) which upstream uses for the homophilous citation graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathNetNC(nn.Module):
    def __init__(
        self,
        feature_length: int,
        hidden_size: int,
        out_size: int,
        walk_length: int,
        dropout: float = 0.7,
        variant: str = "hetero",
        structure_only: bool = False,
    ) -> None:
        super().__init__()
        if variant not in ("hetero", "homo"):
            raise ValueError(f"Unknown variant {variant!r}; use 'hetero' or 'homo'.")
        self.feature_length = feature_length
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.walk_length = walk_length
        self.dropout = dropout
        self.variant = variant
        # When True the *separate* raw-ego classifier shortcut ``cat(ego, h_agg)``
        # is dropped, so predictions flow only through the walk/path encoder (the
        # ego features still enter via the LSTM sequence, keeping training stable).
        # This makes PathNet sensitive to graph structure on feature-rich graphs
        # (e.g. IMDB, where the movie bag-of-words otherwise dominates and makes
        # every structural variant collapse to the same output). Because the walk
        # neighbourhood differs across variants while the ego features are shared,
        # the structural differences between variants now drive the output.
        self.structure_only = structure_only

        self.fc0 = nn.Linear(feature_length, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, out_size)
        # One linear per distance-to-target bucket (0 .. walk_length-1).
        self.nets = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(walk_length)]
        )
        self.attw = nn.Linear(2 * hidden_size, 1)
        self.Lrelu = nn.LeakyReLU()
        # In structure_only mode the path signal can be tiny (walks that traverse
        # feature-poor node types), which leaves the aggregated representation
        # near-constant across nodes and parks training on the majority class.
        # Normalising the aggregate rescales that small per-node variation so the
        # classifier can discriminate and training escapes the plateau.
        self.agg_norm = nn.LayerNorm(hidden_size) if structure_only else None

        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(
        self,
        x: torch.Tensor,
        walks: torch.Tensor,
        dists: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits ``[T, out_size]`` for the ``T`` target nodes.

        Args:
            x:     ``[N, feature_length]`` features for every node in the graph.
            walks: ``[T, W, L]`` long tensor of node ids per walk (``walks[:, :, 0]``
                   is the target/ego node, repeated across the ``W`` walks).
            dists: ``[T, W, L]`` long tensor of distance-to-target buckets in
                   ``[0, L-1]`` aligned with ``walks``.
        """
        T, W, L = walks.shape
        device = walks.device

        h = self.fc0(x)  # [N, hid]
        if self.variant == "homo":
            h = F.relu(h)

        # Node embeddings along every walk position.
        nei = h[walks.reshape(-1)]  # [T*W*L, hid]

        # Distance-specific linear map: stack all buckets then gather per node.
        stacked = torch.stack([net(nei) for net in self.nets], dim=1)  # [T*W*L, L, hid]
        flat_d = dists.reshape(-1).clamp_(0, self.walk_length - 1)
        rows = torch.arange(stacked.size(0), device=device)
        nei = stacked[rows, flat_d]  # [T*W*L, hid]
        if self.variant == "homo":
            nei = F.relu(nei)
        nei = nei.view(T * W, L, self.hidden_size)

        # Ego embedding of each walk (position 0), used as the attention query.
        ego_walk = nei.view(T, W, L, self.hidden_size)[:, :, 0, :]  # [T, W, hid]

        # LSTM reads tail -> ego (upstream flips the sequence axis). The ego's own
        # features stay in the sequence, which keeps a strong, learnable signal;
        # in ``structure_only`` mode we only remove the *separate* raw-ego
        # shortcut below, so the path encoding is the sole route to the classifier.
        seq = torch.flip(nei, dims=[1]).transpose(0, 1)  # [L, T*W, hid]
        seq = F.dropout(seq, p=self.dropout, training=self.training)
        _, (h_n, _) = self.LSTM(seq)
        h_n = h_n.squeeze(0).view(T, W, self.hidden_size)  # [T, W, hid]

        # Attention over the W walk summaries, compared against the ego.
        cat_res = torch.cat((h_n, ego_walk), dim=-1)  # [T, W, 2*hid]
        if self.variant == "homo":
            att = self.attw(cat_res)
            h_n = (1.0 + att) * h_n
        else:
            att = F.softmax(self.Lrelu(self.attw(cat_res)), dim=1)  # over walks
            h_n = att * h_n
        h_agg = torch.mean(h_n, dim=1)  # [T, hid]
        if self.agg_norm is not None:
            h_agg = self.agg_norm(h_agg)

        ego = h[walks[:, 0, 0]]  # [T, hid] target embedding
        if self.structure_only:
            # Drop the *separate* raw-ego shortcut so the target's own features
            # cannot bypass the path encoder: the classifier sees only the
            # walk/attention aggregation, which jointly encodes the ego together
            # with the reachable neighbourhood. Because the neighbourhood differs
            # across the structural variants (while the ego features are shared),
            # this makes PathNet sensitive to graph structure without the training
            # instability of hiding the ego features entirely.
            ego = torch.zeros_like(ego)
        out = torch.cat((ego, h_agg), dim=1)  # [T, 2*hid]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.fc2(out)  # [T, out_size]
