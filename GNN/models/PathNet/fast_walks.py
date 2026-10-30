"""Vectorised per-epoch walk sampler (upstream ``gen_epoch_merw.cpp`` / ``gen.cpp`` protocol: a fresh walk set every epoch).

Same transition kernel as ``pathnet.walks.build_transition_rows`` (MERW: P_ij ∝ A_ij psi_j; rw: uniform),
same dead-end rule (stay put), same exact BFS distance buckets as ``walk_dist.bfs_distance_buckets``; only the sampling is
vectorised (inverse-CDF on a CSR of cumulative probabilities) so 1,000 epochs of walks are cheap. RNG: numpy
``default_rng([seed, epoch])`` — a pure function of (seed, epoch), so the invariant arm draws identical walks on every variant.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pathnet.walks import build_symmetric_adj, build_transition_rows  # MERW / RW kernel of the PathNet port


class EpochWalker:
    def __init__(self, pairs: np.ndarray, num_nodes: int, targets: np.ndarray, num_walks: int, walk_length: int, walk_type: str):
        A = build_symmetric_adj(pairs, num_nodes); A.sort_indices()
        nbrs, probs = build_transition_rows(A, walk_type)
        self.deg = np.array([len(n) for n in nbrs], dtype=np.int64)
        self.ptr = np.r_[0, np.cumsum(self.deg)]
        self.nb = np.concatenate(nbrs).astype(np.int64)
        cdf = np.concatenate([np.cumsum(p) for p in probs])
        ends = self.ptr[1:] - 1
        cdf[ends[self.deg > 0]] = 1.0 + 1e-12                                   # guard the last bucket against round-off
        self.cdf = cdf + np.repeat(np.arange(num_nodes, dtype=np.float64), self.deg)   # row r lives in (r, r+1]
        self.targets, self.W, self.L, self.N = np.asarray(targets, np.int64), num_walks, walk_length, num_nodes
        A2 = (A @ A).tocsr(); A2.data[:] = 1
        self.nbr_keys = np.sort(A.tocoo().row.astype(np.int64) * num_nodes + A.tocoo().col)
        c2 = A2.tocoo(); self.two_keys = np.sort(c2.row.astype(np.int64) * num_nodes + c2.col)

    @staticmethod
    def _member(keys, q):
        k = np.searchsorted(keys, q); k[k >= keys.size] = keys.size - 1
        return keys[k] == q

    def sample(self, seed: int, epoch: int):
        rng = np.random.default_rng([int(seed), int(epoch)])
        T, W, L = len(self.targets), self.W, self.L
        walks = np.empty((T, W, L), np.int64); walks[:, :, 0] = self.targets[:, None]
        cur = np.repeat(self.targets, W)
        for s in range(1, L):
            u = rng.random(cur.size)
            pos = np.searchsorted(self.cdf, cur + u, side="left")
            pos = np.clip(pos, self.ptr[cur], np.maximum(self.ptr[cur + 1] - 1, self.ptr[cur]))
            nxt = np.where(self.deg[cur] > 0, self.nb[np.minimum(pos, self.nb.size - 1)], cur)
            walks[:, :, s] = nxt.reshape(T, W); cur = nxt
        t = np.broadcast_to(walks[:, :, :1], walks.shape)
        q = t * self.N + walks
        d = np.minimum(np.broadcast_to(np.arange(L), walks.shape), L - 1).copy()
        d[(d >= 3) & self._member(self.two_keys, q)] = 2
        d[self._member(self.nbr_keys, q)] = 1
        d[walks == t] = 0
        return walks, d
