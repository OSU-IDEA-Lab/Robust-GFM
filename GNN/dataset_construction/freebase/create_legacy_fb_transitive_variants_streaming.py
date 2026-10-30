#!/usr/bin/env python3
"""
Memory-safe legacy Freebase-style dataset variant builder from transitive summaries.

Creates:
  unchanged/
  exact_2/, exact_3/, ..., exact_K/
  range_2_3/, range_2_4/, ..., range_2_K/

Each variant contains:
  info.dat
  node.dat
  link.dat                or link.dat.gz with --gzip-link
  label.dat               merged label.dat + label.dat.test
  new_transitive_relations.csv/jsonl
  variant_overview.json

Main improvement over the earlier version:
  - Never stores all materialized transitive edges in memory.
  - Streams each composed sparse matrix directly to link.dat.
  - Stores only per-relation stats/metadata.
  - Optional gzip output for large link.dat files.

A transitive edge is materialized only when there is at least one concrete
chained path for the composition, because edges are generated from sparse
matrix composition of the chain itself.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------


def existing_file(data_dir: Path, candidates: Sequence[str]) -> Path | None:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None


def require_file(data_dir: Path, candidates: Sequence[str], logical_name: str) -> Path:
    p = existing_file(data_dir, candidates)
    if p is None:
        raise FileNotFoundError(f"Could not find {logical_name}; tried: {', '.join(str(data_dir / c) for c in candidates)}")
    return p


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


# ---------------------------------------------------------------------
# Parse legacy files
# ---------------------------------------------------------------------


def parse_info(info_path: Path):
    node_type_names: List[str] = []
    relation_rows: List[dict] = []
    label_rows: List[dict] = []
    section = None

    with info_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped == "node.dat":
                section = "nodes"; continue
            if stripped == "link.dat":
                section = "links"; continue
            if stripped in {"label.dat", "label.dat.test"}:
                section = "labels"; continue
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
                relation_rows.append({
                    "id": int(parts[0]),
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "name": parts[-1],
                })
            elif section == "labels" and len(parts) >= 3:
                label_rows.append({
                    "type": int(parts[0]),
                    "class": int(parts[1]),
                    "name": parts[-1],
                })

    relation_rows.sort(key=lambda r: r["id"])
    label_rows.sort(key=lambda r: (r["type"], r["class"]))
    return node_type_names, relation_rows, label_rows


def parse_link_arrays(link_path: Path):
    srcs: List[int] = []
    dsts: List[int] = []
    rels: List[int] = []
    weights: List[str] = []
    with link_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad link line {idx}: {line!r}")
            srcs.append(int(parts[0]))
            dsts.append(int(parts[1]))
            rels.append(int(parts[2]))
            weights.append(parts[3] if len(parts) > 3 else "1.0")
    return (
        np.asarray(srcs, dtype=np.int32),
        np.asarray(dsts, dtype=np.int32),
        np.asarray(rels, dtype=np.int32),
        weights,
    )


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def merge_labels(data_dir: Path, out_path: Path) -> int:
    paths = []
    for candidates in [
        ["label.dat", "label(1).dat"],
        ["label.dat.test", "label(1).dat.test", "label.dat(1).test"],
    ]:
        p = existing_file(data_dir, candidates)
        if p is not None:
            paths.append(p)

    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    return count


# ---------------------------------------------------------------------
# Summary parsing and relation naming
# ---------------------------------------------------------------------


def read_summary(summary_path: Path) -> List[dict]:
    if summary_path.suffix.lower() == ".jsonl":
        rows = []
        with summary_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_composition(composition: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in composition.split(" ; ") if x.strip())


def reduced_composition_name(chain: Sequence[str], type_sig: Dict[str, Tuple[int, int]], node_type_names: Sequence[str]) -> str:
    """Reduce relation names while preserving chained semantics.

    Example:
      ORGANIZATION-to-MUSIC ; MUSIC-in-BOOK -> ORGANIZATION-to-MUSIC-in-BOOK
    """
    if not chain:
        return ""

    def base_relation_name(r: str) -> str:
        return r[:-5] if r.endswith("__inv") else r

    def readable(r: str) -> str:
        base = base_relation_name(r)
        if not r.endswith("__inv"):
            return base
        s, e = type_sig[r]
        s_name, e_name = node_type_names[s], node_type_names[e]
        if base in type_sig:
            orig_s, orig_e = type_sig[base]
            orig_s_name, orig_e_name = node_type_names[orig_s], node_type_names[orig_e]
            if base.startswith(orig_s_name) and base.endswith(orig_e_name):
                connector = base[len(orig_s_name): len(base) - len(orig_e_name)]
                return f"{s_name}{connector}{e_name}__inv"
        return f"{s_name}-to-{e_name}__inv"

    out = readable(chain[0])
    for r in chain[1:]:
        nxt = readable(r)
        start_name = node_type_names[type_sig[r][0]]
        if nxt.startswith(start_name):
            out += nxt[len(start_name):]
        else:
            out += "-then-" + nxt
    return out


def relation_chain_type_path(chain: Sequence[str], type_sig: Dict[str, Tuple[int, int]]) -> Tuple[int, ...]:
    if not chain:
        return tuple()
    s0, e0 = type_sig[chain[0]]
    out = [s0, e0]
    for r in chain[1:]:
        s, e = type_sig[r]
        if out[-1] != s:
            raise ValueError(f"Type-incompatible chain {' ; '.join(chain)} at {r}: prev end {out[-1]}, next start {s}")
        out.append(e)
    return tuple(out)


def unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}__{i}" in used:
        i += 1
    out = f"{name}__{i}"
    used.add(out)
    return out


# ---------------------------------------------------------------------
# Sparse composition
# ---------------------------------------------------------------------


def binarize(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr(copy=True)
    if M.nnz:
        M.data[:] = 1
        M.eliminate_zeros()
        M.sort_indices()
    return M


def build_relation_matrices(num_nodes: int, srcs: np.ndarray, dsts: np.ndarray, rels: np.ndarray, relation_rows: Sequence[dict], add_inverse_relations: bool):
    mats: Dict[str, sp.csr_matrix] = {}
    type_sig: Dict[str, Tuple[int, int]] = {}
    for r in relation_rows:
        rid = int(r["id"])
        name = r["name"]
        mask = rels == rid
        if np.any(mask):
            data = np.ones(int(mask.sum()), dtype=np.int8)
            M = sp.coo_matrix((data, (srcs[mask], dsts[mask])), shape=(num_nodes, num_nodes), dtype=np.int8).tocsr()
            M.sum_duplicates()
            M.data[:] = 1
            M.sort_indices()
        else:
            M = sp.csr_matrix((num_nodes, num_nodes), dtype=np.int8)
        mats[name] = M
        type_sig[name] = (int(r["start"]), int(r["end"]))
        if add_inverse_relations:
            inv_name = f"{name}__inv"
            mats[inv_name] = M.transpose().tocsr()
            mats[inv_name].sort_indices()
            type_sig[inv_name] = (int(r["end"]), int(r["start"]))
    return mats, type_sig


def compose_chain(mats: Dict[str, sp.csr_matrix], chain: Sequence[str]) -> sp.csr_matrix:
    M = mats[chain[0]]
    for r in chain[1:]:
        M = (M @ mats[r]).tocsr()
        if M.nnz == 0:
            return M
        M.data[:] = 1
        M.eliminate_zeros()
        M.sort_indices()
    return binarize(M)


def iter_matrix_edges(M: sp.csr_matrix, rel_id: int, weight: str = "1.0") -> Iterator[str]:
    M = M.tocoo(copy=False)
    order = np.lexsort((M.col, M.row)) if M.nnz else []
    rows = M.row[order] if M.nnz else []
    cols = M.col[order] if M.nnz else []
    for src, dst in zip(rows, cols):
        yield f"{int(src)}\t{int(dst)}\t{int(rel_id)}\t{weight}\n"


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------


def write_info(path: Path, node_type_names: Sequence[str], relation_rows: Sequence[dict], label_rows: Sequence[dict]):
    with path.open("w", encoding="utf-8") as f:
        f.write("node.dat\n\n")
        f.write("TYPE\tMEANING\n")
        for tid, name in enumerate(node_type_names):
            f.write(f"{tid}\t\t{name}\n")
        f.write("\nAttribute Dimension: 0\n\n")
        f.write("-----------------------------------------------\n\n")
        f.write("link.dat\n\n")
        f.write("LINK\tSTART\tEND\tMEANING\n")
        for r in sorted(relation_rows, key=lambda x: x["id"]):
            f.write(f"{r['id']}\t\t{r['start']}\t\t{r['end']}\t{r['name']}\n")
        f.write("\nTargeting: Link Type (0,0,0)\n\n")
        f.write("----------------------------------------------- \n\n")
        f.write("label.dat\n\n")
        f.write("TYPE\tCLASS\tMEANING\n")
        for row in sorted(label_rows, key=lambda x: (x["type"], x["class"])):
            f.write(f"{row['type']}\t\t{row['class']}\t\t{row['name']}\n")
        f.write("\nTargeting: Label Type 0\n")


def stream_original_links(in_link: Path, out_handle) -> int:
    count = 0
    with in_link.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out_handle.write(line if line.endswith("\n") else line + "\n")
                count += 1
    return count


def write_relation_summaries(outdir: Path, summaries: Sequence[dict]):
    jsonl = outdir / "new_transitive_relations.jsonl"
    csv_path = outdir / "new_transitive_relations.csv"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in summaries:
            f.write(json.dumps(row) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "new_relation_id", "new_relation_name", "composition", "hop_count",
            "start_type", "end_type", "node_type_path", "num_new_edges",
            "unique_endpoint_pairs", "skipped_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summaries:
            writer.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Memory-safe builder for transitive legacy FB dataset variants.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True, help="Transitive summary CSV or JSONL.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--min-hops", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=None)
    ap.add_argument("--add-inverse-relations", action="store_true", help="Allow compositions containing generated __inv relations.")
    ap.add_argument("--gzip-link", action="store_true", help="Write link.dat.gz instead of link.dat to reduce storage.")
    ap.add_argument("--hardlink-static-files", action="store_true", help="Hardlink node.dat when possible instead of copying.")
    ap.add_argument("--max-new-edges-per-relation", type=int, default=0, help="If >0, skip a materialized transitive relation when its edge count exceeds this limit.")
    ap.add_argument("--keep-empty-relations", action="store_true", help="Normally empty transitive relations are skipped.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    info_path = require_file(args.data_dir, ["info.dat", "info(1).dat"], "info.dat")
    node_path = require_file(args.data_dir, ["node.dat", "node(1).dat"], "node.dat")
    link_path = require_file(args.data_dir, ["link.dat", "link(1).dat"], "link.dat")

    node_type_names, base_relation_rows, label_rows = parse_info(info_path)
    srcs, dsts, rels, _weights = parse_link_arrays(link_path)
    num_nodes = count_lines(node_path)
    mats, type_sig = build_relation_matrices(num_nodes, srcs, dsts, rels, base_relation_rows, args.add_inverse_relations)

    summary_rows = read_summary(args.summary)
    candidates_by_chain: Dict[Tuple[str, ...], dict] = {}
    for row in summary_rows:
        comp = row.get("composition", "")
        chain = split_composition(comp)
        if not chain:
            continue
        hop_count = int(row.get("hop_count") or len(chain))
        if hop_count < args.min_hops:
            continue
        if args.max_hops is not None and hop_count > args.max_hops:
            continue
        missing = [r for r in chain if r not in mats]
        if missing:
            raise ValueError(f"Missing relation(s) in composition {' ; '.join(chain)}: {missing}. Use --add-inverse-relations if needed.")
        relation_chain_type_path(chain, type_sig)
        candidates_by_chain[chain] = {"composition": " ; ".join(chain), "hop_count": hop_count}

    if not candidates_by_chain:
        raise ValueError("No candidate compositions loaded from summary.")

    max_hops = args.max_hops or max(r["hop_count"] for r in candidates_by_chain.values())
    hop_values = list(range(args.min_hops, max_hops + 1))
    variants: List[Tuple[str, set[int]]] = [("unchanged", set())]
    for h in hop_values:
        variants.append((f"exact_{h}", {h}))
    for h in range(args.min_hops + 1, max_hops + 1):
        variants.append((f"range_{args.min_hops}_{h}", set(range(args.min_hops, h + 1))))

    all_overview = {
        "data_dir": str(args.data_dir),
        "summary": str(args.summary),
        "num_original_nodes": num_nodes,
        "num_original_edges": int(len(srcs)),
        "num_original_relations": len(base_relation_rows),
        "num_summary_rows": len(summary_rows),
        "num_unique_candidate_compositions": len(candidates_by_chain),
        "gzip_link": bool(args.gzip_link),
        "variants": [],
    }

    base_used_names = {r["name"] for r in base_relation_rows}
    base_next_rid = max(r["id"] for r in base_relation_rows) + 1 if base_relation_rows else 0

    for variant_name, hop_set in variants:
        print(f"[VARIANT] starting {variant_name}", flush=True)
        vdir = args.outdir / variant_name
        vdir.mkdir(parents=True, exist_ok=True)

        # Static files.
        node_out = vdir / "node.dat"
        if args.hardlink_static_files:
            try:
                if node_out.exists():
                    node_out.unlink()
                os.link(node_path, node_out)
            except OSError:
                shutil.copyfile(node_path, node_out)
        else:
            shutil.copyfile(node_path, node_out)
        merged_label_rows = merge_labels(args.data_dir, vdir / "label.dat")

        if variant_name == "unchanged":
            write_info(vdir / "info.dat", node_type_names, base_relation_rows, label_rows)
            link_out = vdir / ("link.dat.gz" if args.gzip_link else "link.dat")
            with open_text(link_out, "wt") as out:
                total_edges = stream_original_links(link_path, out)
            variant_summary = {
                "variant": variant_name,
                "hop_filter": [],
                "num_new_relations": 0,
                "num_new_edges": 0,
                "num_total_relations": len(base_relation_rows),
                "num_total_edges": total_edges,
                "merged_label_rows": merged_label_rows,
                "link_file": link_out.name,
            }
            with (vdir / "variant_overview.json").open("w", encoding="utf-8") as f:
                json.dump(variant_summary, f, indent=2)
            all_overview["variants"].append(variant_summary)
            print(f"[VARIANT] wrote {variant_name}", flush=True)
            continue

        selected = [(chain, row) for chain, row in candidates_by_chain.items() if int(row["hop_count"]) in hop_set]
        selected.sort(key=lambda x: (int(x[1]["hop_count"]), x[1]["composition"]))

        # First pass: assign relation ids, compose only for stats and skip decisions.
        used_names = set(base_used_names)
        next_rid = base_next_rid
        materialized: List[dict] = []
        relation_rows_out = list(base_relation_rows)
        summaries: List[dict] = []

        for idx, (chain, row) in enumerate(selected, 1):
            T = compose_chain(mats, chain)
            edge_count = int(T.nnz)
            type_path = relation_chain_type_path(chain, type_sig)
            start_type, end_type = type_path[0], type_path[-1]
            rel_name = unique_name(reduced_composition_name(chain, type_sig, node_type_names), used_names)

            skipped_reason = ""
            if edge_count == 0 and not args.keep_empty_relations:
                skipped_reason = "empty"
            elif args.max_new_edges_per_relation > 0 and edge_count > args.max_new_edges_per_relation:
                skipped_reason = f"too_many_edges>{args.max_new_edges_per_relation}"

            if skipped_reason:
                summaries.append({
                    "new_relation_id": "",
                    "new_relation_name": rel_name,
                    "composition": " ; ".join(chain),
                    "hop_count": int(row["hop_count"]),
                    "start_type": node_type_names[start_type],
                    "end_type": node_type_names[end_type],
                    "node_type_path": " ; ".join(node_type_names[t] for t in type_path),
                    "num_new_edges": edge_count,
                    "unique_endpoint_pairs": edge_count,
                    "skipped_reason": skipped_reason,
                })
                continue

            rid = next_rid
            next_rid += 1
            relation_rows_out.append({"id": rid, "start": start_type, "end": end_type, "name": rel_name})
            materialized.append({"rid": rid, "chain": chain, "row": row, "edge_count": edge_count})
            summaries.append({
                "new_relation_id": rid,
                "new_relation_name": rel_name,
                "composition": " ; ".join(chain),
                "hop_count": int(row["hop_count"]),
                "start_type": node_type_names[start_type],
                "end_type": node_type_names[end_type],
                "node_type_path": " ; ".join(node_type_names[t] for t in type_path),
                "num_new_edges": edge_count,
                "unique_endpoint_pairs": edge_count,
                "skipped_reason": "",
            })
            if idx % 25 == 0:
                print(f"  planned {idx}/{len(selected)} compositions for {variant_name}", flush=True)
            del T

        write_info(vdir / "info.dat", node_type_names, relation_rows_out, label_rows)
        write_relation_summaries(vdir, summaries)

        # Second pass: stream original + new composed edges. New edges are never kept in Python lists.
        link_out = vdir / ("link.dat.gz" if args.gzip_link else "link.dat")
        new_edge_total = 0
        with open_text(link_out, "wt") as out:
            total_edges = stream_original_links(link_path, out)
            for j, item in enumerate(materialized, 1):
                T = compose_chain(mats, item["chain"])
                for line in iter_matrix_edges(T, int(item["rid"])):
                    out.write(line)
                new_edge_total += int(T.nnz)
                total_edges += int(T.nnz)
                out.flush()
                if j % 10 == 0:
                    print(f"  streamed {j}/{len(materialized)} new relations for {variant_name}", flush=True)
                del T

        variant_summary = {
            "variant": variant_name,
            "hop_filter": sorted(hop_set),
            "num_selected_summary_compositions": len(selected),
            "num_new_relations": len(materialized),
            "num_skipped_relations": len([s for s in summaries if s.get("skipped_reason")]),
            "num_new_edges": int(new_edge_total),
            "num_total_relations": len(relation_rows_out),
            "num_total_edges": int(total_edges),
            "merged_label_rows": merged_label_rows,
            "link_file": link_out.name,
        }
        with (vdir / "variant_overview.json").open("w", encoding="utf-8") as f:
            json.dump(variant_summary, f, indent=2)
        all_overview["variants"].append(variant_summary)
        print(f"[VARIANT] wrote {variant_name}: new_relations={len(materialized)}, new_edges={new_edge_total}", flush=True)

    with (args.outdir / "all_variants_overview.json").open("w", encoding="utf-8") as f:
        json.dump(all_overview, f, indent=2)
    print(f"Wrote overview: {args.outdir / 'all_variants_overview.json'}", flush=True)


if __name__ == "__main__":
    main()
