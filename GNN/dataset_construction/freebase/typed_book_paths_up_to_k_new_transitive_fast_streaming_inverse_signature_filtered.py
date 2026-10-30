#!/usr/bin/env python3
"""
Fast typed transitive-edge discovery for legacy FB-style datasets.

This version finds NEW transitive endpoint relations from typed chains.  It can
optionally use generated inverse relations inside the chain, but it filters
against only ORIGINAL one-hop link.dat relations.

For each typed path

    v0 -r1-> v1 -r2-> ... -rk-> vk

it materializes

    T = r1 @ r2 @ ... @ rk

and emits the composition only if T is nonempty and, by default:

  1. there is NO original one-hop relation with the same start/end node-type
     signature; and
  2. T is NOT a subset of any original one-hop relation with the same start/end
     node-type signature.

The signature filter is stricter than the subset filter. It catches cases where
the schema already has a relation position such as BOOK -> FILM, even if the
particular composed edge set is not a subset of the existing BOOK -> FILM edge
set.

Outputs are streamed row-by-row to JSONL and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class RelationSpec:
    name: str
    start_type: int
    end_type: int
    base_name: str
    is_inverse: bool = False


# -----------------------------
# Parsing
# -----------------------------

def parse_info(info_path: Path):
    node_type_names: List[str] = []
    relation_rows_by_id: Dict[int, dict] = {}
    label_rows: List[dict] = []
    section = None

    with info_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped == "node.dat":
                section = "nodes"
                continue
            if stripped == "link.dat":
                section = "links"
                continue
            if stripped in {"label.dat", "label.dat.test"}:
                section = "labels"
                continue
            if (
                stripped.startswith("TYPE\tMEANING")
                or stripped.startswith("LINK\tSTART\tEND\tMEANING")
                or stripped.startswith("TYPE\tCLASS\tMEANING")
                or stripped.startswith("Attribute Dimension:")
                or stripped.startswith("Targeting:")
                or stripped.startswith("-")
            ):
                continue

            parts = [p for p in stripped.split("\t") if p != ""]
            if section == "nodes" and len(parts) >= 2:
                tid = int(parts[0])
                while len(node_type_names) <= tid:
                    node_type_names.append(f"type_{len(node_type_names)}")
                node_type_names[tid] = parts[-1]
            elif section == "links" and len(parts) >= 4:
                rid = int(parts[0])
                relation_rows_by_id[rid] = {
                    "id": rid,
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "name": parts[-1],
                }
            elif section == "labels" and len(parts) >= 3:
                label_rows.append({"type": int(parts[0]), "class": int(parts[1]), "name": parts[-1]})

    if not relation_rows_by_id:
        raise ValueError(f"No relation rows parsed from {info_path}")

    relation_rows = [relation_rows_by_id[i] for i in sorted(relation_rows_by_id)]
    return node_type_names, relation_rows, label_rows


def parse_nodes(node_path: Path):
    node_names: List[str] = []
    node_types: List[int] = []
    with node_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad node line {i}: {line!r}")
            node_names.append(parts[1])
            node_types.append(int(parts[2]))
    return {
        "node_names": node_names,
        "node_type_by_node_id": np.asarray(node_types, dtype=np.int32),
        "num_nodes": len(node_names),
    }


def parse_edges(link_path: Path):
    srcs: List[int] = []
    dsts: List[int] = []
    rels: List[int] = []
    with link_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad link line {i}: {line!r}")
            srcs.append(int(parts[0]))
            dsts.append(int(parts[1]))
            rels.append(int(parts[2]))
    return np.asarray(srcs, dtype=np.int32), np.asarray(dsts, dtype=np.int32), np.asarray(rels, dtype=np.int32)


# -----------------------------
# Sparse matrices
# -----------------------------

def binarize_matrix(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr(copy=True)
    if M.nnz:
        M.data[:] = 1
        M.eliminate_zeros()
    return M


def build_relation_matrices(
    num_nodes: int,
    srcs: np.ndarray,
    dsts: np.ndarray,
    rels: np.ndarray,
    relation_rows: Sequence[dict],
    add_inverse_relations: bool,
):
    relation_matrices: Dict[str, sp.csr_matrix] = {}
    relation_specs: Dict[str, RelationSpec] = {}
    original_relation_names: List[str] = []

    for row in relation_rows:
        rid = int(row["id"])
        name = row["name"]
        mask = rels == rid
        if np.any(mask):
            data = np.ones(int(mask.sum()), dtype=np.int8)
            M = sp.coo_matrix(
                (data, (srcs[mask], dsts[mask])),
                shape=(num_nodes, num_nodes),
                dtype=np.int8,
            ).tocsr()
            M.sum_duplicates()
            M = binarize_matrix(M)
        else:
            M = sp.csr_matrix((num_nodes, num_nodes), dtype=np.int8)

        relation_matrices[name] = M
        relation_specs[name] = RelationSpec(name, int(row["start"]), int(row["end"]), name, False)
        original_relation_names.append(name)

        if add_inverse_relations:
            inv = f"{name}__inv"
            relation_matrices[inv] = M.transpose().tocsr()
            relation_specs[inv] = RelationSpec(inv, int(row["end"]), int(row["start"]), name, True)

    return relation_matrices, relation_specs, original_relation_names


def multiply_extend(prefix: sp.csr_matrix, next_rel: sp.csr_matrix, group_binary: bool) -> sp.csr_matrix:
    out = (prefix @ next_rel).tocsr()
    return binarize_matrix(out) if group_binary else out


def summarize_relation_matrix(M: sp.csr_matrix):
    M = M.tocsr()
    row_nnz = np.diff(M.indptr)
    Mcsc = M.tocsc()
    col_nnz = np.diff(Mcsc.indptr)
    return {
        "total_paths": int(M.sum()),
        "unique_endpoint_pairs": int(M.nnz),
        "start_nodes_with_path": int(np.count_nonzero(row_nnz)),
        "end_nodes_with_path": int(np.count_nonzero(col_nnz)),
        "max_paths_for_any_pair": int(M.data.max()) if M.nnz else 0,
        "one_to_one_pairs": int(np.count_nonzero(M.data == 1)) if M.nnz else 0,
        "one_to_many_start_nodes": int(np.count_nonzero(row_nnz > 1)),
        "many_to_one_end_nodes": int(np.count_nonzero(col_nnz > 1)),
    }


# -----------------------------
# Existing-relation filters
# -----------------------------

def build_original_base_relation_index(
    relation_matrices: Dict[str, sp.csr_matrix],
    relation_specs: Dict[str, RelationSpec],
    original_relation_names: Sequence[str],
) -> Dict[Tuple[int, int], List[Tuple[str, sp.csr_matrix]]]:
    """Index original, non-generated one-hop link.dat relations by type signature."""
    by_sig: Dict[Tuple[int, int], List[Tuple[str, sp.csr_matrix]]] = defaultdict(list)
    for name in original_relation_names:
        spec = relation_specs[name]
        by_sig[(int(spec.start_type), int(spec.end_type))].append((name, binarize_matrix(relation_matrices[name])))
    return by_sig


def find_existing_type_signature_match(
    start_type: int,
    end_type: int,
    base_relations_by_signature: Dict[Tuple[int, int], List[Tuple[str, sp.csr_matrix]]],
):
    """Find whether any original one-hop relation has this start/end type signature."""
    matches = base_relations_by_signature.get((int(start_type), int(end_type)), [])
    if not matches:
        return False, "", 0
    return True, " ; ".join(name for name, _ in matches), len(matches)


def is_subset_of_base_relation(T_bin: sp.csr_matrix, base_bin: sp.csr_matrix) -> bool:
    """Return True iff every nonzero edge in T_bin also appears in base_bin."""
    # T \ base = T - (T ∩ base).  Since both are binary, nnz == 0 means subset.
    overlap = T_bin.multiply(base_bin)
    extra = T_bin - overlap
    extra.eliminate_zeros()
    return extra.nnz == 0


def find_existing_relation_subset_match(
    T: sp.csr_matrix,
    start_type: int,
    end_type: int,
    base_relations_by_signature: Dict[Tuple[int, int], List[Tuple[str, sp.csr_matrix]]],
):
    """Find original one-hop relation R with same type signature such that T ⊆ R.

    This is exact subset filtering, not approximate and not block-level. If T == R,
    that is also a subset and is filtered.
    """
    T_bin = binarize_matrix(T)
    for base_name, base_bin in base_relations_by_signature.get((int(start_type), int(end_type)), []):
        if is_subset_of_base_relation(T_bin, base_bin):
            return True, base_name, int(T_bin.nnz), int(base_bin.nnz)
    return False, "", int(T_bin.nnz), 0


# -----------------------------
# Type-aware path enumeration
# -----------------------------

def is_inverse_name(a: str, b: str) -> bool:
    return a == f"{b}__inv" or b == f"{a}__inv"


def find_node_type_id(node_type_names: Sequence[str], requested: str):
    q = requested.strip()
    if q.upper() == "NONE":
        return None
    if q.isdigit():
        tid = int(q)
        if not (0 <= tid < len(node_type_names)):
            raise ValueError(f"Node type id {tid} out of range")
        return tid
    matches = [i for i, n in enumerate(node_type_names) if n == q]
    if not matches:
        matches = [i for i, n in enumerate(node_type_names) if n.lower() == q.lower()]
    if len(matches) != 1:
        available = ", ".join(f"{i}:{n}" for i, n in enumerate(node_type_names))
        raise ValueError(f"Could not resolve node type {q!r}. Available: {available}")
    return matches[0]


def reduced_composition_name(chain: Sequence[str]) -> str:
    # Example: ORGANIZATION-to-MUSIC ; MUSIC-in-BOOK -> ORGANIZATION-to-MUSIC-in-BOOK
    if not chain:
        return ""
    def clean(x: str) -> str:
        return x[:-5] + "__inv" if x.endswith("__inv") else x
    out = clean(chain[0])
    for r in chain[1:]:
        nxt = clean(r)
        # Try to remove repeated type prefix from the next relation name.
        pieces = nxt.split("-", 1)
        suffix = pieces[1] if len(pieces) == 2 else nxt
        out += "-" + suffix
    return out


# -----------------------------
# Streaming writers
# -----------------------------

SUMMARY_FIELDS = [
    "composition",
    "reduced_composition_name",
    "typed_composition",
    "hop_count",
    "node_type_path",
    "node_type_name_path",
    "relation_type_signature",
    "relation_type_name_signature",
    "new_transitive_edge",
    "filtered_existing_type_signature",
    "existing_type_signature_relations",
    "existing_type_signature_relation_count",
    "filtered_existing_subset",
    "existing_subset_relation",
    "subset_transitive_edge_count",
    "subset_base_edge_count",
    "total_paths",
    "unique_endpoint_pairs",
    "start_nodes_with_path",
    "end_nodes_with_path",
    "max_paths_for_any_pair",
    "one_to_one_pairs",
    "one_to_many_start_nodes",
    "many_to_one_end_nodes",
]


def make_json_safe(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Find new typed transitive edges, filtering subsets of original one-hop relations.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-hops", type=int, required=True)
    ap.add_argument("--min-hops", type=int, default=2)
    ap.add_argument("--must-include-node-type", type=str, default="BOOK")
    ap.add_argument("--relations", type=str, default="", help="Optional comma-separated relation subset to enumerate.")
    ap.add_argument("--add-inverse-relations", action="store_true", help="Allow generated __inv relations in compositions.")
    ap.add_argument("--group-binary", action="store_true", default=True, help="Binarize after each hop; default behavior.")
    ap.add_argument("--prune-immediate-inverse-cancel", action="store_true")
    ap.add_argument("--prune-consecutive-duplicates", action="store_true")
    ap.add_argument("--prune-all-same-relation", action="store_true")
    ap.add_argument("--no-filter-existing-signature", action="store_true", help="Disable the default strict filter that skips any composition whose start/end node-type signature already exists as an original one-hop relation.")
    ap.add_argument("--no-filter-existing-subset", action="store_true", help="Disable subset filtering against original one-hop relations.")
    ap.add_argument("--max-output-rows", type=int, default=0, help="Stop after this many emitted rows; 0 means no limit.")
    ap.add_argument("--flush-every", type=int, default=1, help="Flush output every N emitted rows. Default 1 = after every row.")
    args = ap.parse_args()

    if args.min_hops < 1 or args.min_hops > args.max_hops:
        raise ValueError("Require 1 <= --min-hops <= --max-hops")

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "typed_book_paths_up_to_k_new_transitive_subset_filtered_summary.jsonl"
    csv_path = args.outdir / "typed_book_paths_up_to_k_new_transitive_subset_filtered_summary.csv"
    overview_path = args.outdir / "typed_book_paths_up_to_k_new_transitive_subset_filtered_overview.json"

    # Overwrite previous outputs for a clean run.
    for p in [jsonl_path, csv_path, overview_path]:
        if p.exists():
            p.unlink()

    node_type_names, relation_rows, _label_rows = parse_info(args.data_dir / "info.dat")
    nodes = parse_nodes(args.data_dir / "node.dat")
    srcs, dsts, rels = parse_edges(args.data_dir / "link.dat")

    relation_matrices, relation_specs, original_relation_names = build_relation_matrices(
        int(nodes["num_nodes"]), srcs, dsts, rels, relation_rows, add_inverse_relations=args.add_inverse_relations
    )
    base_relations_by_signature = build_original_base_relation_index(
        relation_matrices, relation_specs, original_relation_names
    )

    if args.relations.strip():
        relation_names = [r.strip() for r in args.relations.split(",") if r.strip()]
        missing = [r for r in relation_names if r not in relation_matrices]
        if missing:
            raise ValueError(f"Unknown relation(s): {missing}")
        if not args.add_inverse_relations:
            relation_names = [r for r in relation_names if not r.endswith("__inv")]
    else:
        if args.add_inverse_relations:
            relation_names = sorted(relation_matrices.keys())
        else:
            relation_names = sorted(original_relation_names)

    required_type = find_node_type_id(node_type_names, args.must_include_node_type)

    by_start: Dict[int, List[str]] = defaultdict(list)
    for r in relation_names:
        by_start[int(relation_specs[r].start_type)].append(r)

    counters = Counter()
    emitted = 0

    def should_stop() -> bool:
        return args.max_output_rows > 0 and emitted >= args.max_output_rows

    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        def emit_row(row: dict):
            nonlocal emitted
            emitted += 1
            safe = make_json_safe(row)
            jf.write(json.dumps(safe) + "\n")
            writer.writerow({k: safe.get(k, "") for k in SUMMARY_FIELDS})
            if emitted % max(args.flush_every, 1) == 0:
                jf.flush()
                cf.flush()

        def dfs(seq: Tuple[str, ...], current_type: int, type_path: Tuple[int, ...], M: sp.csr_matrix):
            if should_stop():
                return
            if seq:
                counters["typed_prefixes_seen"] += 1
                if args.prune_immediate_inverse_cancel and any(is_inverse_name(a, b) for a, b in zip(seq, seq[1:])):
                    counters["pruned_immediate_inverse_cancel"] += 1
                    return
                if args.prune_consecutive_duplicates and any(a == b for a, b in zip(seq, seq[1:])):
                    counters["pruned_consecutive_duplicates"] += 1
                    return
                if M.nnz == 0:
                    counters["pruned_empty_prefix"] += 1
                    return

                if len(seq) >= args.min_hops:
                    counters["candidate_compositions"] += 1
                    if args.prune_all_same_relation and len(set(seq)) == 1:
                        counters["pruned_all_same_relation"] += 1
                    elif required_type is not None and required_type not in type_path:
                        counters["pruned_missing_required_node_type"] += 1
                    else:
                        signature_match = False
                        signature_names = ""
                        signature_match_count = 0
                        if not args.no_filter_existing_signature:
                            signature_match, signature_names, signature_match_count = find_existing_type_signature_match(
                                type_path[0], type_path[-1], base_relations_by_signature
                            )
                        if signature_match:
                            counters["skipped_existing_type_signature"] += 1
                        else:
                            subset_match = False
                            subset_name = ""
                            t_edges = int(binarize_matrix(M).nnz)
                            base_edges = 0
                            if not args.no_filter_existing_subset:
                                subset_match, subset_name, t_edges, base_edges = find_existing_relation_subset_match(
                                    M, type_path[0], type_path[-1], base_relations_by_signature
                                )
                            if subset_match:
                                counters["skipped_existing_subset_base_relation"] += 1
                            else:
                                nt_names = [
                                    node_type_names[t] if 0 <= t < len(node_type_names) else f"type_{t}"
                                    for t in type_path
                                ]
                                summary = summarize_relation_matrix(M)
                                row = {
                                    "composition": " ; ".join(seq),
                                    "reduced_composition_name": reduced_composition_name(seq),
                                    "typed_composition": " ; ".join(
                                        f"{nt_names[i]}-[{r}]->{nt_names[i + 1]}" for i, r in enumerate(seq)
                                    ),
                                    "hop_count": len(seq),
                                    "node_type_path": " ; ".join(map(str, type_path)),
                                    "node_type_name_path": " ; ".join(nt_names),
                                    "relation_type_signature": f"{type_path[0]}->{type_path[-1]}",
                                    "relation_type_name_signature": f"{nt_names[0]}->{nt_names[-1]}",
                                    "new_transitive_edge": 1,
                                    "filtered_existing_type_signature": 0,
                                    "existing_type_signature_relations": "",
                                    "existing_type_signature_relation_count": 0,
                                    "filtered_existing_subset": 0,
                                    "existing_subset_relation": "",
                                    "subset_transitive_edge_count": t_edges,
                                    "subset_base_edge_count": base_edges,
                                    **summary,
                                }
                                emit_row(row)

            if len(seq) == args.max_hops or should_stop():
                return
            for r in by_start.get(int(current_type), []):
                spec = relation_specs[r]
                next_M = relation_matrices[r] if not seq else multiply_extend(M, relation_matrices[r], args.group_binary)
                dfs(seq + (r,), int(spec.end_type), type_path + (int(spec.end_type),), next_M)

        for start_type in sorted(by_start):
            if should_stop():
                break
            # Start with empty sequence; first extension passes relation matrix directly.
            dummy = sp.csr_matrix((int(nodes["num_nodes"]), int(nodes["num_nodes"])), dtype=np.int8)
            dfs(tuple(), int(start_type), (int(start_type),), dummy)

        jf.flush()
        cf.flush()

    overview = {
        "data_dir": str(args.data_dir),
        "num_nodes": int(nodes["num_nodes"]),
        "num_original_relations": len(original_relation_names),
        "num_enumerated_relations": len(relation_names),
        "add_inverse_relations": bool(args.add_inverse_relations),
        "min_hops": int(args.min_hops),
        "max_hops": int(args.max_hops),
        "must_include_node_type": args.must_include_node_type,
        "must_include_node_type_id": required_type,
        "filter_existing_type_signature": bool(not args.no_filter_existing_signature),
        "type_signature_filter_definition": "skip if any original one-hop base relation has the same start/end node-type signature as the composed endpoint relation",
        "filter_existing_subset": bool(not args.no_filter_existing_subset),
        "subset_filter_definition": "skip if composed binary endpoint edge set is a subset of any original one-hop base relation with the same start/end type signature",
        "emitted_rows": int(emitted),
        "counters": dict(counters),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {overview_path}")
    print(f"Emitted new transitive edges after signature/subset filtering: {emitted}")


if __name__ == "__main__":
    main()
