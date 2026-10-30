"""SeHGNN channel compiler with a deterministic cap.

A *channel* is SeHGNN's precomputed feature for one metapath rooted at the target type: the
nested mean ``rownorm(S_1) · rownorm(S_2) ⋯ rownorm(S_k) · X_{end}`` (``hg_propagate_feat_dgl``
with ``fn.mean`` per hop). We compile the sparse operator ``R = rownorm(S_1)[targets] ⋯
rownorm(S_k)`` restricted to the end type's columns; the runner applies it to features
(IMDb) or to a learned per-type embedding (Freebase).

Step matrices are GLOBAL ``n × n`` binarised CSR matrices (rows of the step's source type,
columns of its destination type), so that a channel is fully described by its step list.

Cap rule (Freebase; ``cap=0`` disables it for IMDb): for target ``t``, if the number of walks
``t → … → z`` through the steps is ≤ C the row is the exact nested mean; otherwise it is the
mean of C walks drawn by a seeded k-stage uniform sampler (``y_1 ~ U(N_1(t))``,
``y_2 ~ U(N_2(y_1))``, …; a draw that reaches a node with no next-step neighbour contributes
the zero vector, exactly as ``fn.mean`` over an empty neighbourhood does), an unbiased
estimate of the nested mean. The RNG is seeded by ``h64(cap_seed, channel key, t)`` — a
function of the semantic walk set only, never of the physical layout (docs/THEORY.md §5).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
_MAGNN = _HERE.parent.parent / "MAGNN"
if str(_MAGNN) not in sys.path:
    sys.path.insert(0, str(_MAGNN))
from magnn_compile_common import binarize, h64, pair_hash, sha  # noqa: E402

FLAVORS = ("full", "full_rel", "restricted", "restricted_union")


# ----------------------------------------------------------------------------- linear algebra
def row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    m = m.tocsr().astype(np.float64)
    s = np.asarray(m.sum(axis=1)).ravel()
    inv = np.zeros_like(s)
    nz = s > 0
    inv[nz] = 1.0 / s[nz]
    out = sp.diags(inv) @ m
    out = out.tocsr()
    out.sort_indices()
    return out


def typed_union(mats: dict[str, sp.csr_matrix], rel_types: dict[str, tuple[int, int]], n: int) -> dict[tuple[int, int], sp.csr_matrix]:
    """Type-collapsed undirected adjacency: (a, b) -> binarised union of every relation with
    typed endpoints (a, b) plus the transpose of every relation with endpoints (b, a)."""
    acc: dict[tuple[int, int], list[sp.csr_matrix]] = {}
    for name, m in mats.items():
        a, b = rel_types[name]
        acc.setdefault((a, b), []).append(m)
        acc.setdefault((b, a), []).append(m.T.tocsr())
    out = {}
    for pair, parts in acc.items():
        total = parts[0].astype(np.int8)
        for p in parts[1:]:
            total = total + p.astype(np.int8)
        total = binarize(total)
        if total.nnz:
            out[pair] = total
    return out


# ----------------------------------------------------------------------------- channel spec
@dataclass
class ChannelSpec:
    key: str                       # SeHGNN feature key; for Freebase it must END with the source-type digit
    node_types: list[int]          # [t0 (target), t1, ..., tk (feature source)]
    steps: list[sp.csr_matrix]     # k global n x n binarised CSR matrices
    step_names: list[str] = field(default_factory=list)


def compile_channel(spec: ChannelSpec, targets: np.ndarray, node_type: np.ndarray, cap: int, cap_seed: int) -> tuple[sp.csr_matrix, dict]:
    """Return (R [len(targets) x n_end] CSR float32 in LOCAL end-type ids, audit dict)."""
    n = len(node_type)
    end_type = spec.node_types[-1]
    end_nodes = np.where(node_type == end_type)[0]
    glob2loc = np.full(n, -1, dtype=np.int64)
    glob2loc[end_nodes] = np.arange(len(end_nodes))
    T = len(targets)
    if not spec.steps:                                    # identity channel
        rows = np.arange(T); cols = glob2loc[targets]
        assert (cols >= 0).all(), "identity channel: targets are not of the end type"
        R = sp.csr_matrix((np.ones(T, dtype=np.float32), (rows, cols)), shape=(T, len(end_nodes)))
        return R, {"exact_rows": T, "sampled_rows": 0, "empty_rows": 0, "walks_max": 1}
    steps = [s.tocsr() for s in spec.steps]
    # walk counts per target (float64: only compared against the cap)
    v = np.ones(n, dtype=np.float64)
    for s in reversed(steps):
        v = s.astype(np.float64) @ v
    counts = v[targets]
    heavy = (counts > cap) if cap and cap > 0 else np.zeros(T, dtype=bool)
    light = ~heavy
    norm = [row_normalize(s) for s in steps]
    R = sp.csr_matrix((T, n), dtype=np.float64)
    # exact rows: left-to-right products keep every intermediate bounded by the walk count
    if light.any():
        rows_l = np.where(light)[0]
        acc = norm[0][targets[rows_l]]
        for s in norm[1:]:
            acc = acc @ s
        acc = acc.tocoo()
        R = R + sp.csr_matrix((acc.data, (rows_l[acc.row], acc.col)), shape=(T, n))
    # sampled rows
    if heavy.any():
        indptr = [s.indptr for s in steps]; indices = [s.indices for s in steps]
        rr, cc, ww = [], [], []
        for i in np.where(heavy)[0]:
            t = int(targets[i])
            rng = np.random.default_rng(h64(cap_seed, spec.key, t))
            cur = np.full(cap, t, dtype=np.int64)
            alive = np.ones(cap, dtype=bool)
            for ip, ix in zip(indptr, indices):
                deg = ip[cur + 1] - ip[cur]
                alive &= deg > 0
                pick = ip[cur] + np.floor(rng.random(cap) * np.maximum(deg, 1)).astype(np.int64)
                nxt = ix[np.minimum(pick, len(ix) - 1)]
                cur = np.where(alive, nxt, cur)
            z = cur[alive]
            if len(z):
                uniq, cnt = np.unique(z, return_counts=True)
                rr.append(np.full(len(uniq), i, dtype=np.int64)); cc.append(uniq); ww.append(cnt / float(cap))
        if rr:
            R = R + sp.csr_matrix((np.concatenate(ww), (np.concatenate(rr), np.concatenate(cc))), shape=(T, n))
    R = R.tocsr()
    R.sum_duplicates(); R.sort_indices()
    # restrict columns to the end type (all mass must already be there)
    coo = R.tocoo()
    loc = glob2loc[coo.col]
    if (loc < 0).any():
        raise RuntimeError(f"channel {spec.key}: walk ended outside type {end_type}")
    out = sp.csr_matrix((coo.data.astype(np.float32), (coo.row, loc)), shape=(T, len(end_nodes)))
    out.sum_duplicates(); out.sort_indices()
    empty = int(T - len(np.unique(coo.row)))
    return out, {"exact_rows": int(light.sum()), "sampled_rows": int(heavy.sum()), "empty_rows": empty,
                 "walks_max": float(counts.max()) if T else 0.0}


def channel_hash(R: sp.csr_matrix) -> str:
    R = R.tocsr(); R.sort_indices()
    h = hashlib.sha256()
    h.update(str(R.shape).encode())
    h.update(np.ascontiguousarray(R.indptr.astype(np.int64)).tobytes())
    h.update(np.ascontiguousarray(R.indices.astype(np.int64)).tobytes())
    h.update(np.ascontiguousarray(R.data.astype(np.float32)).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- catalogs
def full_type_paths(adj: dict[tuple[int, int], sp.csr_matrix], target_type: int, K: int) -> list[tuple[list[int], list[sp.csr_matrix]]]:
    """Every type path of length <= K from the target over the typed adjacency (SeHGNN's native
    ``hg_propagate_feat_dgl`` key set: all walks t0 -> ... -> tk, keys read target-first)."""
    nbrs: dict[int, list[int]] = {}
    for (a, b) in adj:
        nbrs.setdefault(a, []).append(b)
    for a in nbrs:
        nbrs[a] = sorted(set(nbrs[a]))
    out = [([target_type], [])]
    frontier = [([target_type], [])]
    for _ in range(K):
        nxt = []
        for path, steps in frontier:
            for b in nbrs.get(path[-1], []):
                nxt.append((path + [b], steps + [adj[(path[-1], b)]]))
        out.extend(nxt); frontier = nxt
    return out


def prefix_closure(metapaths: Iterable[tuple[str, list[int], list[sp.csr_matrix], list[str]]]):
    """MAGNN metapath (name, node_types, oriented steps to the END, step names) -> every prefix
    (length 1..k) as its own channel: the per-hop information MAGNN's instance encoder sees."""
    out = []
    for name, node_types, steps, step_names in metapaths:
        for j in range(1, len(steps) + 1):
            out.append((name, j, node_types[: j + 1], steps[:j], step_names[:j]))
    return out


# ----------------------------------------------------------------------------- bundle I/O
def save_bundle(out: Path, *, dataset: str, variant: str, graph: str, flavor: str, K: int, cap: int, cap_seed: int,
                target_type: int, node_type: np.ndarray, targets: np.ndarray, labels: np.ndarray, split: dict,
                channels: list[tuple[ChannelSpec, sp.csr_matrix, dict]], label_channels: list[tuple[str, int]],
                features: dict[int, np.ndarray] | None, hashes: dict, extra: dict | None = None, t0: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "channels").mkdir(exist_ok=True)
    recs = []
    for i, (spec, R, audit) in enumerate(channels):
        sp.save_npz(out / "channels" / f"{i}.npz", R, compressed=True)
        recs.append({"index": i, "key": spec.key, "node_types": [int(t) for t in spec.node_types], "end_type": int(spec.node_types[-1]),
                     "steps": spec.step_names, "shape": [int(x) for x in R.shape], "nnz": int(R.nnz), "hash": channel_hash(R), **audit})
    np.save(out / "targets.npy", targets.astype(np.int64)); np.save(out / "labels.npy", labels.astype(np.int64))
    np.save(out / "node_types.npy", node_type.astype(np.int64))
    np.savez(out / "split.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in split.items()})
    feature_hash = None
    if features is not None:
        hh = hashlib.sha256()
        for t in sorted(features):
            np.save(out / f"features_{t}.npy", features[t].astype(np.float32))
            hh.update(str(t).encode()); hh.update(np.ascontiguousarray(features[t].astype(np.float32)).tobytes())
        feature_hash = hh.hexdigest()
    program = hashlib.sha256("\n".join(f"{r['key']}:{r['end_type']}:{r['hash']}" for r in sorted(recs, key=lambda r: r["key"])).encode()
                             + json.dumps(sorted(label_channels)).encode()).hexdigest()
    supervision = sha(np.concatenate([targets.astype(np.int64), labels.astype(np.int64)] + [np.asarray(split[k], dtype=np.int64) for k in ("train_idx", "val_idx", "test_idx")]))
    manifest = {"dataset": dataset, "variant": variant, "graph": graph, "flavor": flavor, "K": K, "cap": cap, "cap_seed": cap_seed,
                "target_type": int(target_type), "num_types": int(node_type.max()) + 1,
                "type_counts": [int((node_type == t).sum()) for t in range(int(node_type.max()) + 1)],
                "num_targets": int(len(targets)), "num_classes": int(labels.max()) + 1, "num_channels": len(recs),
                "channels": recs, "label_channels": [{"key": k, "channel": int(i)} for k, i in label_channels],
                "program_hash": program, "supervision_hash": supervision, "feature_hash": feature_hash, **hashes,
                "seconds": (time.time() - t0) if t0 else None, **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


class Bundle:
    """Loader for a compiled channel bundle."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.targets = np.load(self.dir / "targets.npy")
        self.labels = np.load(self.dir / "labels.npy")
        self.node_type = np.load(self.dir / "node_types.npy")
        z = np.load(self.dir / "split.npz")
        self.split = {k: z[k] for k in ("train_idx", "val_idx", "test_idx")}
        self.num_classes = int(self.manifest["num_classes"])
        self.target_type = int(self.manifest["target_type"])
        self.keys = [c["key"] for c in self.manifest["channels"]]
        self.end_type = {c["key"]: int(c["end_type"]) for c in self.manifest["channels"]}
        self.label_channels = {lc["key"]: int(lc["channel"]) for lc in self.manifest["label_channels"]}
        self.features = None
        if self.manifest.get("feature_hash"):
            self.features = {t: np.load(self.dir / f"features_{t}.npy") for t in range(self.manifest["num_types"])}

    def channel(self, i: int) -> sp.csr_matrix:
        return sp.load_npz(self.dir / "channels" / f"{i}.npz").tocsr()

    def channels(self) -> dict[str, sp.csr_matrix]:
        return {c["key"]: self.channel(c["index"]) for c in self.manifest["channels"]}

    @property
    def program_hash(self) -> str:
        return self.manifest["program_hash"]
