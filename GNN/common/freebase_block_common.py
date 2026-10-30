#!/usr/bin/env python3
"""Shared preprocessing and audit utilities for block-conditional Freebase RGCN.

The mappings are relation-family local.  A mined semantic relation is resolved
from its authoritative base-relation composition inside each graph variant.
No graph-wide isomorphism or one global node permutation is assumed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class RelationInfo:
    rid: int
    start: int
    end: int
    name: str


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(list(arr.shape)).encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def pair_hash64(src: int, dst: int) -> int:
    packed = struct.pack("<qq", int(src), int(dst))
    return int.from_bytes(hashlib.blake2b(packed, digest_size=8).digest(), "little")


def parse_info(path: Path) -> tuple[dict[int, str], dict[int, RelationInfo]]:
    node_types: dict[int, str] = {}
    relations: dict[int, RelationInfo] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        if text == "node.dat":
            section = "nodes"
            continue
        if text == "link.dat":
            section = "links"
            continue
        if text in {"label.dat", "label.dat.test"}:
            section = "labels"
            continue
        if (
            text.startswith("TYPE\tMEANING")
            or text.startswith("LINK\tSTART\tEND\tMEANING")
            or text.startswith("TYPE\tCLASS\tMEANING")
            or text.startswith("Attribute Dimension:")
            or text.startswith("Targeting:")
            or text.startswith("---")
        ):
            continue
        parts = [x for x in text.split("\t") if x != ""]
        if section == "nodes" and len(parts) >= 2:
            node_types[int(parts[0])] = parts[-1]
        elif section == "links" and len(parts) >= 4:
            rid = int(parts[0])
            relations[rid] = RelationInfo(rid, int(parts[1]), int(parts[2]), parts[-1])
    if not relations:
        raise ValueError(f"No relation schema in {path}")
    return node_types, relations


def read_nodes(path: Path) -> tuple[list[str], np.ndarray]:
    rows: list[tuple[int, str, int]] = []
    max_id = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad node.dat line {line_no}: {raw!r}")
            nid, name, ntype = int(parts[0]), parts[1], int(parts[2])
            rows.append((nid, name, ntype))
            max_id = max(max_id, nid)
    names = [""] * (max_id + 1)
    types = np.full(max_id + 1, -1, dtype=np.int16)
    for nid, name, ntype in rows:
        names[nid] = name
        types[nid] = ntype
    if np.any(types < 0):
        raise ValueError("node.dat ids must be contiguous")
    return names, types


def read_labels(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Bad label.dat line {line_no}: {raw!r}")
            rows.append((int(parts[0]), int(parts[2]), int(parts[3])))
    return rows


def make_split(
    labels: Sequence[tuple[int, int, int]],
    node_types: np.ndarray,
    center_type: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    labeled = [(nid, cls) for nid, typ, cls in labels if int(typ) == center_type and int(node_types[nid]) == center_type]
    if not labeled:
        raise ValueError(f"No labels for center type {center_type}")
    target_nodes = np.asarray(sorted(nid for nid, _ in labeled), dtype=np.int64)
    by_node = {int(nid): int(cls) for nid, cls in labeled}
    y = np.asarray([by_node[int(n)] for n in target_nodes], dtype=np.int64)
    idx = np.arange(len(target_nodes), dtype=np.int64)
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and test_ratio > 0):
        raise ValueError("train/val/test ratios must be positive")
    train_idx, hold = train_test_split(
        idx, test_size=1.0 - train_ratio, random_state=int(seed), shuffle=True, stratify=y
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        hold, test_size=rel_test, random_state=int(seed), shuffle=True, stratify=y[hold]
    )
    local = {
        "train_idx": np.asarray(sorted(train_idx), dtype=np.int64),
        "val_idx": np.asarray(sorted(val_idx), dtype=np.int64),
        "test_idx": np.asarray(sorted(test_idx), dtype=np.int64),
    }
    global_idx = {k: target_nodes[v] for k, v in local.items()}
    return y, target_nodes, local, global_idx


def expected_variants(up_to: str) -> list[str]:
    if up_to == "exact_2":
        return ["unchanged", "exact_2"]
    if up_to == "exact_3":
        return ["unchanged", "exact_3"]
    if up_to == "range_2_3":
        return ["unchanged", "exact_2", "exact_3", "range_2_3"]
    raise ValueError(f"Unsupported up-to variant: {up_to}")


def load_catalog(package_root: Path, up_to: str, set_mode: str, scope: str) -> dict[str, Any]:
    path = package_root / "catalogs" / up_to / set_mode / f"{scope}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    catalog = load_json(path)
    if catalog.get("format_version") != "freebase_rgcn_block_catalog_v1":
        raise ValueError(f"Unexpected catalog format in {path}")
    return catalog


def selected_dependency_names(catalog: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for rel in catalog["semantic_relations"]:
        if rel["kind"] == "base":
            names.add(str(rel["name"]))
        else:
            names.update(str(x) for x in rel["composition"])
    return names


def read_graph(
    *,
    data_dir: Path,
    relation_schema: Mapping[int, RelationInfo],
    needed_base_names: set[str],
    mapped_names: set[str],
    full_physical_dir: Path | None = None,
) -> tuple[
    dict[str, dict[int, tuple[int, ...]]],
    dict[str, dict[str, int]],
    int,
]:
    """Read only needed base edges and stream physical shortcut provenance."""
    base_sets: dict[str, dict[int, set[int]]] = {
        name: defaultdict(set) for name in needed_base_names
    }
    physical: dict[str, dict[str, int]] = {
        name: {"raw_count": 0, "xor64": 0, "sum64": 0} for name in mapped_names
    }
    handles: dict[str, Any] = {}
    if full_physical_dir is not None:
        full_physical_dir.mkdir(parents=True, exist_ok=True)
        for name in mapped_names:
            handles[name] = (full_physical_dir / f"{safe_name(name)}.pairs.i32").open("wb")
    total = 0
    try:
        with (data_dir / "link.dat").open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                parts = raw.rstrip("\n").split("\t")
                if len(parts) < 3:
                    raise ValueError(f"Bad link.dat line {line_no}: {raw!r}")
                src, dst, rid = int(parts[0]), int(parts[1]), int(parts[2])
                spec = relation_schema.get(rid)
                if spec is None:
                    raise KeyError(f"Relation id {rid} absent from info.dat")
                total += 1
                if spec.name in base_sets:
                    base_sets[spec.name][src].add(dst)
                if spec.name in physical:
                    value = pair_hash64(src, dst)
                    rec = physical[spec.name]
                    rec["raw_count"] += 1
                    rec["xor64"] ^= value
                    rec["sum64"] = (rec["sum64"] + value) & MASK64
                    if spec.name in handles:
                        handles[spec.name].write(struct.pack("<ii", src, dst))
    finally:
        for h in handles.values():
            h.close()
    base = {
        name: {int(src): tuple(sorted(dsts)) for src, dsts in table.items()}
        for name, table in base_sets.items()
    }
    return base, physical, total


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def base_fingerprints(base: Mapping[str, Mapping[int, Sequence[int]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(base):
        h = hashlib.sha256()
        count = 0
        xor64 = 0
        sum64 = 0
        for src in sorted(base[name]):
            for dst in base[name][src]:
                h.update(struct.pack("<ii", int(src), int(dst)))
                value = pair_hash64(src, dst)
                count += 1
                xor64 ^= value
                sum64 = (sum64 + value) & MASK64
        result[name] = {
            "unique_count": count,
            "ordered_sha256": h.hexdigest(),
            "xor64": xor64,
            "sum64": sum64,
        }
    return result


class RelationResolver:
    def __init__(self, base: Mapping[str, Mapping[int, Sequence[int]]]) -> None:
        self.base = base

    def neighbors(self, relation: str, src: int) -> tuple[int, ...]:
        return tuple(self.base.get(relation, {}).get(int(src), ()))

    def compose(self, composition: Sequence[str], src: int) -> tuple[int, ...]:
        frontier: set[int] = {int(src)}
        for relation in composition:
            nxt: set[int] = set()
            table = self.base.get(str(relation))
            if table is None:
                raise KeyError(f"Missing base relation needed by composition: {relation}")
            for node in sorted(frontier):
                nxt.update(table.get(node, ()))
            frontier = nxt
            if not frontier:
                break
        return tuple(sorted(frontier))


def build_outgoing_csr(
    *,
    num_nodes: int,
    source_nodes: np.ndarray,
    relation: Mapping[str, Any],
    resolver: RelationResolver,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build source-major unique semantic endpoint blocks.

    Every source node is one local block.  Mapped families are only projected
    inside their own source-local endpoint block; no whole-graph mapping is used.
    """
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    pieces: list[np.ndarray] = []
    h = hashlib.sha256()
    block_h = hashlib.sha256()
    nonempty = 0
    max_degree = 0
    total = 0
    source_set = set(int(x) for x in source_nodes.tolist())
    for src in range(num_nodes):
        if src in source_set:
            if relation["kind"] == "base":
                targets = resolver.neighbors(str(relation["name"]), src)
            else:
                targets = resolver.compose(relation["composition"], src)
        else:
            targets = ()
        arr = np.asarray(targets, dtype=np.int32)
        if arr.size:
            # Resolver output is sorted unique; retain defensive normalization.
            arr = np.unique(arr)
            pieces.append(arr)
            nonempty += 1
            max_degree = max(max_degree, int(arr.size))
            total += int(arr.size)
            h.update(struct.pack("<ii", src, int(arr.size)))
            h.update(arr.tobytes(order="C"))
            block_h.update(struct.pack("<ii", src, int(arr.size)))
            block_h.update(hashlib.sha256(arr.tobytes(order="C")).digest())
        indptr[src + 1] = total
    indices = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32)
    stats = {
        "num_source_blocks": int(len(source_nodes)),
        "num_nonempty_source_blocks": int(nonempty),
        "num_semantic_pairs": int(total),
        "max_block_degree": int(max_degree),
        "mean_nonempty_block_degree": float(total / nonempty) if nonempty else 0.0,
        "semantic_pair_sha256": h.hexdigest(),
        "source_block_sha256": block_h.hexdigest(),
    }
    return indptr, indices, stats


def transpose_csr(num_nodes: int, indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transpose a binary CSR relation in compiled SciPy code.

    This avoids a Python loop over tens of millions of Freebase endpoint pairs.
    """
    if indptr.shape != (num_nodes + 1,):
        raise ValueError("Bad indptr shape")
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - dependency is required by package
        raise RuntimeError("scipy is required for scalable Freebase CSR transposition") from exc
    data = np.ones(indices.size, dtype=np.int8)
    matrix = sp.csr_matrix((data, indices.astype(np.int32, copy=False), indptr), shape=(num_nodes, num_nodes))
    transposed = matrix.transpose().tocsr()
    transposed.sum_duplicates()
    transposed.sort_indices()
    return transposed.indptr.astype(np.int64, copy=False), transposed.indices.astype(np.int32, copy=False)


def csr_hash(indptr: np.ndarray, indices: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(indptr.dtype).encode())
    h.update(str(indices.dtype).encode())
    h.update(json.dumps([len(indptr), len(indices)]).encode())
    h.update(np.ascontiguousarray(indptr).tobytes())
    h.update(np.ascontiguousarray(indices).tobytes())
    return h.hexdigest()


def save_csr(directory: Path, indptr: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "indptr.npy", indptr)
    np.save(directory / "indices.npy", indices)
    return {
        "indptr": str((directory / "indptr.npy").name),
        "indices": str((directory / "indices.npy").name),
        "num_edges": int(indices.size),
        "csr_sha256": csr_hash(indptr, indices),
    }


def load_csr(directory: Path, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    return np.load(directory / "indptr.npy", mmap_mode=mode), np.load(directory / "indices.npy", mmap_mode=mode)


def verify_full_physical_pairs(
    pair_file: Path,
    *,
    num_nodes: int,
    expected_indptr: np.ndarray,
    expected_indices: np.ndarray,
) -> dict[str, Any]:
    raw = np.fromfile(pair_file, dtype=np.int32)
    if raw.size % 2:
        raise ValueError(f"Corrupt pair file: {pair_file}")
    pairs = raw.reshape(-1, 2)
    if pairs.size == 0:
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        indices = np.empty(0, dtype=np.int32)
    else:
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
        keep = np.ones(len(pairs), dtype=bool)
        keep[1:] = np.any(pairs[1:] != pairs[:-1], axis=1)
        pairs = pairs[keep]
        counts = np.bincount(pairs[:, 0].astype(np.int64), minlength=num_nodes)
        indptr = np.empty(num_nodes + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = pairs[:, 1].astype(np.int32, copy=False)
    equal = bool(np.array_equal(indptr, expected_indptr) and np.array_equal(indices, expected_indices))
    return {
        "unique_physical_pairs": int(indices.size),
        "physical_csr_sha256": csr_hash(indptr, indices),
        "matches_composed_semantic_blocks": equal,
    }


def set_all_seeds(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
