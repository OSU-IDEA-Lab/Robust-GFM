"""Undirected 0/1 walk-graph pairs of a Freebase store — the exact construction of ``build_freebase_walks.py`` (every
channel, self-pairs dropped), factored out so the per-epoch runner can rebuild it in-process."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent / "RGCN"), str(HERE.parent.parent / "common")):
    if p not in sys.path: sys.path.insert(0, p)
from freebase_block_common import load_csr, load_json  # noqa: E402


def store_pairs(root: Path) -> np.ndarray:
    cat = load_json(root / "relation_catalog.json")
    n = int(np.load(root / "node_types.npy", mmap_mode="r").size)
    keys = []
    for rec in sorted(cat["channels"], key=lambda r: int(r["channel_id"])):
        indptr, indices = load_csr(root / rec["directory"], mmap=False)
        dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
        src = indices.astype(np.int64)
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        keys.append(np.unique((lo * n + hi)[lo != hi]))
    keys = np.unique(np.concatenate(keys))
    return np.stack([keys // n, keys % n], 1)
