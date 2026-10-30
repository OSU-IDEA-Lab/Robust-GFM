"""Shared MAGNN metapath-instance compilation primitives (Freebase v2 and IMDb compilers).

Set semantics on endpoint pairs, canonical order, and deterministic hash-keyed caps:
``sample_instances`` (symmetric 3-node metapaths from a BOOK/movie x Y pair matrix) and
``sample_long_paths`` (2k+1-node metapaths over oriented step matrices, drawn by an exact
hierarchical sampler). See preprocess_freebase_magnn_v2.py for the rationale.
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp


def h64(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def binarize(m: sp.csr_matrix) -> sp.csr_matrix:
    m = m.tocsr(); m.sum_duplicates(); m.data[:] = 1; m.eliminate_zeros()
    m.sort_indices()
    return m.astype(np.int8)


def pair_hash(m: sp.csr_matrix) -> str:
    coo = m.tocoo(); order = np.lexsort((coo.col, coo.row))
    return sha(np.stack([coo.row[order], coo.col[order]], 1).astype(np.int64))


def sample_instances(M: sp.csr_matrix, MT: sp.csr_matrix, sources: np.ndarray, K: int, mp_name: str, seed: int):
    """For each source b: instances (b, y, b') = all pairs of (y in N(b), b' in N_T(y)),
    in canonical (y asc, b' asc) order; keep all if ≤ K, else K chosen by a seeded RNG
    (seed = h64(seed, mp_name, b)) — deterministic in the semantic instance set."""
    rows, per_source = [], []
    indptr, indices = M.indptr, M.indices
    tindptr, tindices = MT.indptr, MT.indices
    for b in sources:
        ys = indices[indptr[b]:indptr[b + 1]]                         # sorted ascending
        deg = tindptr[ys + 1] - tindptr[ys]
        total = int(deg.sum())
        if total == 0:
            per_source.append(0); continue
        if total <= K:
            y_rep = np.repeat(ys, deg)
            bp = np.concatenate([tindices[tindptr[y]:tindptr[y + 1]] for y in ys])
        else:
            rng = np.random.default_rng(h64(seed, mp_name, int(b)))
            pick = np.sort(rng.choice(total, size=K, replace=False))
            cum = np.cumsum(deg)
            yi = np.searchsorted(cum, pick, side="right")
            off = pick - (cum[yi] - deg[yi])
            y_rep = ys[yi]
            bp = tindices[tindptr[y_rep] + off]
        inst = np.stack([np.full(len(y_rep), b, dtype=np.int64), y_rep.astype(np.int64), bp.astype(np.int64)], 1)
        rows.append(inst); per_source.append(len(inst))
    arr = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.int64)
    return arr, np.asarray(per_source, dtype=np.int64)


def sample_long_paths(mats, sources: np.ndarray, K: int, mp_name: str, seed: int):
    """Instances of BOOK -m1-> ... -mk-> Y -mkᵀ-> ... -m1ᵀ-> BOOK, each path exactly once.

    Level matrices L_1..L_2k = m1..mk, mkᵀ..m1ᵀ. Path counts per node at each level are
    exact integers (sparse products); per source, K path indices are drawn uniformly by a
    seeded RNG (seed = h64(seed, mp_name, b)) and decoded hop by hop — deterministic given
    identical base edges, never dependent on enumeration order.
    """
    hops = [binarize(m) for m in mats] + [binarize(m.T.tocsr()) for m in reversed(mats)]
    n = hops[0].shape[0]
    counts = [None] * (len(hops) + 1)
    counts[-1] = np.ones(n, dtype=np.int64)
    for j in range(len(hops) - 1, -1, -1):
        counts[j] = np.asarray(hops[j].astype(np.int64) @ counts[j + 1]).ravel()
    # per hop: edge weights = path counts below each neighbour, and their global running sum, so
    # a draw can be decoded for every sample at once with one searchsorted (vectorised)
    W = [counts[j + 1][H.indices].astype(np.int64) for j, H in enumerate(hops)]
    C = [np.cumsum(w) for w in W]
    rows, per_source = [], []
    src_u, src_b = [], []
    for b in sources:
        total = int(counts[0][b])
        if total == 0:
            per_source.append(0); continue
        rng = np.random.default_rng(h64(seed, mp_name, int(b)))
        u = np.sort(rng.choice(total, size=K, replace=False)) if total > K else np.arange(total)
        src_u.append(u); src_b.append(np.full(len(u), int(b), dtype=np.int64)); per_source.append(len(u))
    if not src_u:
        return np.empty((0, len(hops) + 1), dtype=np.int64), np.asarray(per_source, dtype=np.int64)
    u = np.concatenate(src_u).astype(np.int64); cur = np.concatenate(src_b); path = [cur]
    for j, H in enumerate(hops):
        start = H.indptr[cur]                                   # first edge of each sample's node
        base = np.where(start > 0, C[j][np.maximum(start - 1, 0)], 0)
        pos = np.searchsorted(C[j], base + u, side="right")     # edge holding the u-th path
        nxt = H.indices[pos]
        u = u - (C[j][pos] - W[j][pos] - base)
        cur = nxt; path.append(cur)
    rows.append(np.stack(path, 1))
    width = len(hops) + 1
    arr = np.concatenate(rows) if rows else np.empty((0, width), dtype=np.int64)
    return arr, np.asarray(per_source, dtype=np.int64)


