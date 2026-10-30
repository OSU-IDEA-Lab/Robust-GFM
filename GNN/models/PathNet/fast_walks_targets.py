"""``fast_walks.EpochWalker`` for large graphs with hubs (Freebase: 180k nodes): no 2-hop table. The distance-2 test for a
walk node v at position 3 is done per query (target t, v) by scanning the smaller of N(t), N(v) against the sorted
neighbour-key table of the other — the same answer as ``walk_dist.bfs_distance_buckets`` (verified bitwise on the stored
Freebase BFS walk files). Same transition kernel / RNG as ``EpochWalker``."""
from __future__ import annotations

import numpy as np

from fast_walks import EpochWalker
from pathnet.walks import build_symmetric_adj, build_transition_rows  # MERW / RW kernel of the PathNet port


class EpochWalkerT(EpochWalker):
    def __init__(self, pairs, num_nodes, targets, num_walks, walk_length, walk_type):
        A = build_symmetric_adj(pairs, num_nodes); A.sort_indices()
        nbrs, probs = build_transition_rows(A, walk_type)
        self.deg = np.array([len(n) for n in nbrs], dtype=np.int64)
        self.ptr = np.r_[0, np.cumsum(self.deg)]
        self.nb = np.concatenate(nbrs).astype(np.int64)
        cdf = np.concatenate([np.cumsum(p) for p in probs])
        ends = self.ptr[1:] - 1
        cdf[ends[self.deg > 0]] = 1.0 + 1e-12
        self.cdf = cdf + np.repeat(np.arange(num_nodes, dtype=np.float64), self.deg)
        self.targets, self.W, self.L, self.N = np.asarray(targets, np.int64), num_walks, walk_length, num_nodes
        self.nbr_keys = np.repeat(np.arange(num_nodes, dtype=np.int64), self.deg) * num_nodes + self.nb   # sorted (CSR order)

    def build_two_hop_bits(self, chunk: int = 64):
        """R2[i] = packed bitset of nodes within 2 hops of target i (A[targets] @ A, chunked to bound memory). Exact."""
        import scipy.sparse as sp
        N = self.N
        A = sp.csr_matrix((np.ones(self.nb.size, np.float32), self.nb, self.ptr), shape=(N, N))
        self.R2 = np.zeros((len(self.targets), (N + 7) // 8), np.uint8)
        for s in range(0, len(self.targets), chunk):
            B = (A[self.targets[s:s + chunk]] @ A).tocsr()
            for i in range(B.shape[0]):
                cols = B.indices[B.indptr[i]:B.indptr[i + 1]]
                np.bitwise_or.at(self.R2[s + i], cols >> 3, (1 << (cols & 7)).astype(np.uint8))
        self.tpos = np.full(N, -1, np.int64); self.tpos[self.targets] = np.arange(len(self.targets))

    def _common_neighbour(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        if hasattr(self, "R2"):
            return ((self.R2[self.tpos[t], v >> 3] >> (v & 7).astype(np.uint8)) & 1).astype(bool)
        """bool per (t, v): N(t) ∩ N(v) ≠ ∅."""
        small = np.where(self.deg[t] <= self.deg[v], t, v); other = np.where(self.deg[t] <= self.deg[v], v, t)
        cnt = self.deg[small]
        q = np.repeat(np.arange(t.size), cnt)
        starts = np.repeat(self.ptr[small], cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        x = self.nb[starts + off]
        hit = self._member(self.nbr_keys, other[q] * self.N + x)
        out = np.zeros(t.size, bool); out[q[hit]] = True
        return out

    def distances(self, walks: np.ndarray) -> np.ndarray:
        T, W, L = walks.shape
        t = np.broadcast_to(walks[:, :, :1], walks.shape)
        q = t * self.N + walks
        d = np.minimum(np.broadcast_to(np.arange(L), walks.shape), L - 1).copy()
        is_nbr = self._member(self.nbr_keys, q)
        cand = (d >= 3) & ~is_nbr & (walks != t)
        if L >= 4:                                                             # exact witness: walk node at pos 2 adjacent to t
            wit = np.zeros_like(cand); wit[:, :, 3] = cand[:, :, 3] & is_nbr[:, :, 2]
            d[wit] = 2; cand &= ~wit
        if cand.any():                                                         # memoised across epochs (pure function of the graph)
            keys, inv = np.unique(q[cand], return_inverse=True)
            if not hasattr(self, "_ck"): self._ck, self._cv = np.zeros(0, np.int64), np.zeros(0, bool)
            known = self._member(self._ck, keys) if self._ck.size else np.zeros(keys.size, bool)
            two = np.zeros(keys.size, bool)
            if known.any(): two[known] = self._cv[np.searchsorted(self._ck, keys[known])]
            new = keys[~known]
            if new.size:
                two[~known] = nv = self._common_neighbour(new // self.N, new % self.N)
                ck = np.concatenate([self._ck, new]); o = np.argsort(ck, kind="stable")
                self._ck, self._cv = ck[o], np.concatenate([self._cv, nv])[o]
            d[cand] = np.where(two[inv], 2, d[cand])
        d[is_nbr] = 1
        d[walks == t] = 0
        return d

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
        return walks, self.distances(walks)
