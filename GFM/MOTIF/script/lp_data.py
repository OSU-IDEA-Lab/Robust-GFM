"""Model-independent graph/query checks adapted from SCR's script/lp_common.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parents[1] / "GNN"  # Robust-GFM/GNN: dataset builders shared with the GNN experiments
CHECKSUMS = ROOT / "rai-datasets/SHA256SUMS"
RAW_ROOT = ROOT / "rai-datasets/raw"
PREPROCESSED_ROOT = ROOT / "rai-datasets/preprocessed"
BUILD_ROOT = PREPROCESSED_ROOT / "_build"
TASK_VARIANTS = {"movie_year": ("v1", "v2", "v3", "v4"), "movie_director": ("v1", "v3")}
TARGET_NAMES = {"movie_year": "movie_to_year", "movie_director": "movie_to_director"}
DBLP_TASK_VARIANTS = {"paper_conference": ("v1", "v2", "v3")}


def variants_for(family, include_universal=False):
    return tuple(family) + (("universal",) if include_universal else ())


def verify_raw_files(checksums_file, raw_dir, dataset, names, canonical_names=None):
    checksums = {}
    for line in Path(checksums_file).read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            checksums[fields[1]] = fields[0]
    hashes = {}
    for name in names:
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw {dataset} file: {path}")
        hashes[name] = file_hash(path)
        canonical = (canonical_names or {}).get(name, name)
        if hashes[name] != checksums.get(f"raw/{dataset}/{canonical}"):
            raise ValueError(f"Raw {dataset} checksum differs from the canonical source: {name}")
    return hashes


def check_dependencies(parser, building):
    required = {"torch": "torch"}
    if building:
        required.update({"pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy"})
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        parser.error("Missing preprocessing dependencies: " + ", ".join(missing)
                     + ". Install requirements-preprocess.txt in your venv.")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def forward_edges(edge_index, edge_type, names):
    """Verify mirror pairs, retain one direction, and remap ids for internal inverses.

    Returns [E, 3] in MOTIF's (head, tail, relation) order, including empty relation
    slots in the vocabulary. Repeated facts are rejected, not silently weighted.
    """
    rows = np.column_stack((edge_index.T, edge_type)).astype(np.int64)
    if len(np.unique(rows, axis=0)) != len(rows):
        raise ValueError("Duplicate typed edges in source graph")
    kept, pieces = [], []
    seen = set()
    for rid, name in enumerate(names):
        if name in seen:
            continue
        left, right = name.split("_to_")
        reverse = f"{right}_to_{left}"
        if reverse not in names or reverse == name:
            raise ValueError(f"Missing distinct reverse relation for {name}")
        rev_id = names.index(reverse)
        a = set(map(tuple, rows[rows[:, 2] == rid, :2]))
        b = set(map(tuple, rows[rows[:, 2] == rev_id, :2][:, ::-1]))
        if a != b:
            raise ValueError(f"Reverse edges do not match: {name}/{reverse}")
        new_id = len(kept)
        kept.append(name)
        part = rows[rows[:, 2] == rid].copy()
        part[:, 2] = new_id
        pieces.append(part)
        seen.update((name, reverse))
    out = np.concatenate(pieces)
    order = np.lexsort((out[:, 0], out[:, 1], out[:, 2]))
    return out[order], kept


def validate_queries(queries, node_type, target_type, head_type=0):
    """Check head-disjoint supervision and the exact 20-candidate contract."""
    heads = []
    for split in ("train", "val", "test"):
        pos, neg = queries[f"{split}_pos"], queries[f"{split}_neg"]
        if pos.ndim != 2 or pos.shape[1] != 2 or neg.shape != (len(pos), 19):
            raise ValueError(f"Invalid {split} positive/negative dimensions")
        if not len(pos) or len(np.unique(pos[:, 0])) != len(pos):
            raise ValueError(f"Empty or repeated head queries in {split}")
        if min(pos.min(), neg.min()) < 0 or max(pos.max(), neg.max()) >= len(node_type):
            raise ValueError("Query node id out of bounds")
        if np.any(node_type[pos[:, 0]] != head_type):
            raise ValueError("Query head is outside the task head node type")
        if np.any(node_type[pos[:, 1]] != target_type) or np.any(node_type[neg] != target_type):
            raise ValueError("Candidate is outside the target node type")
        if np.any(neg == pos[:, 1, None]):
            raise ValueError("A negative equals the positive target")
        heads.extend(pos[:, 0].tolist())
    if len(set(heads)) != len(heads):
        raise ValueError("Heads overlap across training/validation/test")


def full_target_negatives(positives, pool):
    """Every other target, in fixed pool order; positive is scored first."""
    pool = np.asarray(pool, dtype=np.int64)
    if pool.ndim != 1 or len(pool) != 20 or len(np.unique(pool)) != 20:
        raise ValueError("Expected exactly 20 distinct conference candidates")
    if not np.isin(positives[:, 1], pool).all():
        raise ValueError("Positive target missing from conference pool")
    all_targets = np.broadcast_to(pool, (len(positives), len(pool)))
    return all_targets[all_targets != positives[:, 1, None]].reshape(len(positives), 19)


def validate_full_target_queries(queries, node_type, pool):
    """Validate DBLP's flattened negatives without sampling or trusting row order."""
    normalized = {}
    for split in ("train", "val", "test"):
        pos, flat = queries[f"{split}_pos"], queries[f"{split}_neg"]
        if pos.ndim != 2 or pos.shape[1] != 2 or not len(pos):
            raise ValueError(f"Invalid {split} positive pairs")
        neg = full_target_negatives(pos, pool)
        expected = {(int(h), int(t)) for h, row in zip(pos[:, 0], neg) for t in row}
        if flat.shape != (len(pos) * 19, 2) or set(map(tuple, flat.tolist())) != expected:
            raise ValueError(f"{split} source negatives are not the full target complement")
        normalized[f"{split}_pos"], normalized[f"{split}_neg"] = pos, neg
    validate_queries(normalized, node_type, target_type=3, head_type=1)
    return normalized


def audit_dblp_heldout(edges, names, queries):
    """Reject held-out P-C edges and paper-specific Area context in train scope."""
    heldout = np.concatenate([queries[f"{s}_pos"] for s in ("val", "test")])
    pairs = set(map(tuple, heldout.tolist()))
    for relation, head_column, tail_column in (("paper_to_conference", 0, 1),
                                               ("conference_to_paper", 1, 0)):
        if relation in names:
            rows = edges[edges[:, 2] == names.index(relation)]
            if any((int(row[head_column]), int(row[tail_column])) in pairs for row in rows):
                raise ValueError(f"Held-out label leaked via {relation}")
    papers = set(heldout[:, 0].tolist())
    for relation, column in (("paper_to_area", 0), ("area_to_paper", 1)):
        if relation in names and any(int(row[column]) in papers for row in edges[edges[:, 2] == names.index(relation)]):
            raise ValueError("Held-out paper has training-only Area context")


def audit_heldout(edges, names, queries, task, movie_offset, link_offset):
    """Reject held-out target edges on movies AND their declared Link copies."""
    triples = set(map(tuple, edges.tolist()))
    suffix = "year" if task == "movie_year" else "director"
    for split in ("val", "test"):
        for movie, target in queries[f"{split}_pos"]:
            for kind, head in (("movie", movie), ("link", movie - movie_offset + link_offset)):
                name = f"{kind}_to_{suffix}"
                if name in names and (head, target, names.index(name)) in triples:
                    raise ValueError(f"Held-out label leaked via {name}: {movie}->{target}")
