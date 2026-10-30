#!/usr/bin/env python3
"""
Batch recovery experiments for all relation compositions of exactly X hops using full prefix and suffix contexts.

For each relation sequence r1 ; r2 ; ... ; rk with k == exact_hops, this script
tries dropping each relation position i and asks whether the dropped relation can
be recovered from:

    full composition:   P @ R_i @ S
    left context:       P = r1 @ ... @ r_{i-1}
    right context:      S = r_{i+1} @ ... @ rk

Special cases:
  - if i == 0, the left context is identity on the dropped relation's source side
  - if i == k-1, the right context is identity on the dropped relation's target side

Works on WN18RR/FB15k-237-style KGE datasets or generic TSV triples.
Supports optional inverse relations via a __inv suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Triple = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass
class Dataset:
    entity_to_id: Dict[str, int]
    id_to_entity: Dict[int, str]
    relation_to_id: Dict[str, int]
    id_to_relation: Dict[int, str]
    triples: List[Triple]


def load_kge_dict_dataset(data_dir: Path, use_splits: Sequence[str]) -> Dataset:
    entity_to_id: Dict[str, int] = {}
    relation_to_id: Dict[str, int] = {}

    with (data_dir / "entities.dict").open("r", encoding="utf-8") as f:
        for line in f:
            idx, ent = line.rstrip("\n").split("\t")
            entity_to_id[ent] = int(idx)

    with (data_dir / "relations.dict").open("r", encoding="utf-8") as f:
        for line in f:
            idx, rel = line.rstrip("\n").split("\t")
            relation_to_id[rel] = int(idx)

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    triples: List[Triple] = []
    for split in use_splits:
        path = data_dir / f"{split}.txt"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))

    return Dataset(entity_to_id, id_to_entity, relation_to_id, id_to_relation, triples)


def load_generic_tsv(triples_path: Path) -> Dataset:
    entity_to_id: Dict[str, int] = {}
    relation_to_id: Dict[str, int] = {}
    triples: List[Triple] = []

    def ent_id(x: str) -> int:
        if x not in entity_to_id:
            entity_to_id[x] = len(entity_to_id)
        return entity_to_id[x]

    def rel_id(x: str) -> int:
        if x not in relation_to_id:
            relation_to_id[x] = len(relation_to_id)
        return relation_to_id[x]

    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            triples.append((ent_id(h), rel_id(r), ent_id(t)))

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    return Dataset(entity_to_id, id_to_entity, relation_to_id, id_to_relation, triples)


def relation_edges(triples: Iterable[Triple], relation_id: int) -> Set[Edge]:
    return {(h, t) for (h, r, t) in triples if r == relation_id}


def relation_edges_by_name(triples: Iterable[Triple], dataset: Dataset, relation_name: str, allow_inverse: bool = False) -> Set[Edge]:
    if relation_name in dataset.relation_to_id:
        return relation_edges(triples, dataset.relation_to_id[relation_name])
    if allow_inverse and relation_name.endswith("__inv"):
        base_name = relation_name[:-5]
        if base_name not in dataset.relation_to_id:
            raise KeyError(f"Unknown inverse base relation: {base_name}")
        base_edges = relation_edges(triples, dataset.relation_to_id[base_name])
        return {(t, h) for (h, t) in base_edges}
    raise KeyError(f"Unknown relation: {relation_name}")


def compose_edges(edges_xy: Set[Edge], edges_yz: Set[Edge]) -> Set[Edge]:
    y_to_z: Dict[int, Set[int]] = defaultdict(set)
    for y, z in edges_yz:
        y_to_z[y].add(z)
    out: Set[Edge] = set()
    for x, y in edges_xy:
        for z in y_to_z.get(y, ()): 
            out.add((x, z))
    return out


def compose_edge_chain(edge_sets: Sequence[Set[Edge]]) -> Set[Edge]:
    if not edge_sets:
        raise ValueError("edge_sets must be non-empty")
    result = set(edge_sets[0])
    for next_edges in edge_sets[1:]:
        result = compose_edges(result, next_edges)
    return result


def make_identity_from_lefts(edges: Set[Edge]) -> Set[Edge]:
    return {(x, x) for x, _ in edges}


def make_identity_from_rights(edges: Set[Edge]) -> Set[Edge]:
    return {(y, y) for _, y in edges}


def inverse_index(edges: Set[Edge]) -> Dict[int, Set[int]]:
    right_to_left: Dict[int, Set[int]] = defaultdict(set)
    for left, right in edges:
        right_to_left[right].add(left)
    return right_to_left


def recover_removed_from_contexts(edges_full: Set[Edge], edges_prefix: Set[Edge], edges_suffix: Set[Edge]) -> Set[Edge]:
    """
    Predict dropped edges (x, y) iff there exists a,z such that:
      (a, x) in prefix, (a, z) in full, and (y, z) in suffix.

    This recovers a middle relation from the full composition and the remaining
    left/right contexts. It also handles edge positions at either end by using
    identity contexts.
    """
    a_to_xs: Dict[int, Set[int]] = defaultdict(set)
    for a, x in edges_prefix:
        a_to_xs[a].add(x)

    z_to_ys: Dict[int, Set[int]] = inverse_index(edges_suffix)

    pred: Set[Edge] = set()
    for a, z in edges_full:
        xs = a_to_xs.get(a)
        ys = z_to_ys.get(z)
        if not xs or not ys:
            continue
        for x in xs:
            for y in ys:
                pred.add((x, y))
    return pred


def right_context_blocks(edges_suffix: Set[Edge]) -> Tuple[Dict[Tuple[int, ...], int], Dict[int, int], Dict[int, Set[int]]]:
    """Collapse dropped-edge right nodes into blocks by identical suffix fibers."""
    y_to_zs: Dict[int, Set[int]] = defaultdict(set)
    for y, z in edges_suffix:
        y_to_zs[y].add(z)

    sig_to_bid: Dict[Tuple[int, ...], int] = {}
    y_to_bid: Dict[int, int] = {}
    bid_to_ys: Dict[int, Set[int]] = defaultdict(set)

    for y, zs in y_to_zs.items():
        sig = tuple(sorted(zs))
        if sig not in sig_to_bid:
            sig_to_bid[sig] = len(sig_to_bid)
        bid = sig_to_bid[sig]
        y_to_bid[y] = bid
        bid_to_ys[bid].add(y)

    return sig_to_bid, y_to_bid, dict(bid_to_ys)


def recover_removed_blocks(edges_full: Set[Edge], edges_prefix: Set[Edge], edges_suffix: Set[Edge]) -> Set[Tuple[int, int]]:
    _, y_to_bid, _ = right_context_blocks(edges_suffix)
    pred = recover_removed_from_contexts(edges_full, edges_prefix, edges_suffix)
    return {(x, y_to_bid[y]) for x, y in pred if y in y_to_bid}


def gold_removed_blocks(edges_removed: Set[Edge], edges_suffix: Set[Edge]) -> Set[Tuple[int, int]]:
    _, y_to_bid, _ = right_context_blocks(edges_suffix)
    return {(x, y_to_bid[y]) for x, y in edges_removed if y in y_to_bid}


def prf1(pred: Set[Tuple[int, int]], gold: Set[Tuple[int, int]]) -> Dict[str, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def relation_stats(edges: Set[Edge]) -> Dict[str, float]:
    left_to_right: Dict[int, Set[int]] = defaultdict(set)
    right_to_left: Dict[int, Set[int]] = defaultdict(set)
    for l, r in edges:
        left_to_right[l].add(r)
        right_to_left[r].add(l)

    left_with_edge = len(left_to_right)
    right_with_edge = len(right_to_left)
    left_singleton = sum(1 for vs in left_to_right.values() if len(vs) == 1)
    right_singleton = sum(1 for vs in right_to_left.values() if len(vs) == 1)
    left_sigs = Counter(tuple(sorted(vs)) for vs in left_to_right.values())
    right_sigs = Counter(tuple(sorted(vs)) for vs in right_to_left.values())

    return {
        "edge_count": len(edges),
        "left_nodes_with_edge": left_with_edge,
        "right_nodes_with_edge": right_with_edge,
        "forward_singleton_ratio": left_singleton / left_with_edge if left_with_edge else 0.0,
        "backward_singleton_ratio": right_singleton / right_with_edge if right_with_edge else 0.0,
        "distinct_forward_image_sets": len(left_sigs),
        "distinct_backward_preimage_sets": len(right_sigs),
    }


def summarize_relation_edges(edges: Set[Edge]) -> Dict[str, float]:
    """Set-based relation summary aligned with the matrix-based block diagnostics.

    For these recovery scripts, composed relations are represented as binary edge sets,
    so total_paths == unique_endpoint_pairs == len(edges), and max multiplicity is 1.
    """
    left_to_right: Dict[int, Set[int]] = defaultdict(set)
    right_to_left: Dict[int, Set[int]] = defaultdict(set)
    for l, r in edges:
        left_to_right[l].add(r)
        right_to_left[r].add(l)

    left_with_edge = len(left_to_right)
    right_with_edge = len(right_to_left)
    left_singleton = sum(1 for vs in left_to_right.values() if len(vs) == 1)
    right_singleton = sum(1 for vs in right_to_left.values() if len(vs) == 1)
    left_sigs = Counter(tuple(sorted(vs)) for vs in left_to_right.values())
    right_sigs = Counter(tuple(sorted(vs)) for vs in right_to_left.values())

    row_class_id_by_signature = {sig: i for i, sig in enumerate(left_sigs.keys())}
    col_class_id_by_signature = {sig: i for i, sig in enumerate(right_sigs.keys())}

    row_class_to_col_classes: Dict[int, Set[int]] = defaultdict(set)
    for row_sig, row_class_id in row_class_id_by_signature.items():
        for c in row_sig:
            col_sig = tuple(sorted(right_to_left.get(c, set())))
            if col_sig in col_class_id_by_signature:
                row_class_to_col_classes[row_class_id].add(col_class_id_by_signature[col_sig])

    col_class_to_row_classes: Dict[int, Set[int]] = defaultdict(set)
    for col_sig, col_class_id in col_class_id_by_signature.items():
        for r in col_sig:
            row_sig = tuple(sorted(left_to_right.get(r, set())))
            if row_sig in row_class_id_by_signature:
                col_class_to_row_classes[col_class_id].add(row_class_id_by_signature[row_sig])

    row_block_count = len(row_class_id_by_signature)
    col_block_count = len(col_class_id_by_signature)
    row_blocks_with_single_col_block = int(sum(1 for v in row_class_to_col_classes.values() if len(v) == 1))
    col_blocks_with_single_row_block = int(sum(1 for v in col_class_to_row_classes.values() if len(v) == 1))
    forward_block_functionality_ratio = row_blocks_with_single_col_block / row_block_count if row_block_count else 0.0
    backward_block_functionality_ratio = col_blocks_with_single_row_block / col_block_count if col_block_count else 0.0
    block_count_match = int(row_block_count == col_block_count)
    block_bijection_holds = int(
        row_block_count > 0
        and row_block_count == col_block_count
        and row_blocks_with_single_col_block == row_block_count
        and col_blocks_with_single_row_block == col_block_count
    )
    block_bijection_ratio = (
        min(forward_block_functionality_ratio, backward_block_functionality_ratio)
        * (min(row_block_count, col_block_count) / max(row_block_count, col_block_count))
        if max(row_block_count, col_block_count) > 0 else 0.0
    )

    return {
        "total_paths": len(edges),
        "unique_endpoint_pairs": len(edges),
        "start_nodes_with_path": left_with_edge,
        "end_nodes_with_path": right_with_edge,
        "max_paths_for_any_pair": 1 if edges else 0,
        "one_to_one_pairs": len(edges),
        "one_to_many_start_nodes": int(sum(1 for vs in left_to_right.values() if len(vs) > 1)),
        "many_to_one_end_nodes": int(sum(1 for vs in right_to_left.values() if len(vs) > 1)),
        "forward_singleton_ratio": left_singleton / left_with_edge if left_with_edge else 0.0,
        "backward_singleton_ratio": right_singleton / right_with_edge if right_with_edge else 0.0,
        "distinct_forward_image_sets": len(left_sigs),
        "distinct_backward_preimage_sets": len(right_sigs),
        "row_block_count": row_block_count,
        "col_block_count": col_block_count,
        "row_blocks_with_single_col_block": row_blocks_with_single_col_block,
        "col_blocks_with_single_row_block": col_blocks_with_single_row_block,
        "forward_block_functionality_ratio": forward_block_functionality_ratio,
        "backward_block_functionality_ratio": backward_block_functionality_ratio,
        "block_count_match": block_count_match,
        "block_bijection_holds": block_bijection_holds,
        "block_bijection_ratio": block_bijection_ratio,
    }


def enumerate_relation_sequences(relation_names: Sequence[str], exact_hops: int) -> Iterable[Tuple[str, ...]]:
    """Enumerate relation sequences of exactly exact_hops hops."""
    for seq in product(relation_names, repeat=exact_hops):
        yield seq




def base_relation_name(rel: str) -> str:
    return rel[:-5] if rel.endswith("__inv") else rel


def is_inverse_relation_pair(r1: str, r2: str) -> bool:
    return base_relation_name(r1) == base_relation_name(r2) and r1.endswith("__inv") != r2.endswith("__inv")


def prune_reason_for_sequence(
    seq: Sequence[str],
    prune_immediate_inverse: bool = True,
    prune_consecutive_duplicates: bool = True,
    prune_all_same_relation: bool = True,
) -> str | None:
    if prune_immediate_inverse:
        for a, b in zip(seq, seq[1:]):
            if is_inverse_relation_pair(a, b):
                return "immediate_inverse_cancellation"

    if prune_consecutive_duplicates:
        for a, b in zip(seq, seq[1:]):
            if a == b:
                return "consecutive_duplicate_relation"

    if prune_all_same_relation and len(set(seq)) == 1:
        return "all_same_relation"

    return None

def edge_preview(edges: Set[Edge], id_to_left: Dict[int, str], id_to_right: Dict[int, str], limit: int = 10) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for a, b in sorted(edges)[:limit]:
        out.append((id_to_left[a], id_to_right[b]))
    return out


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test recovery for every dropped edge position in every composition exactly X hops.")
    parser.add_argument("--dataset-format", choices=["wn18rr", "fb15k237", "generic_tsv"], default="wn18rr")
    parser.add_argument("--data-dir", type=Path, help="KGE dataset directory containing entities.dict, relations.dict, train/valid/test.txt")
    parser.add_argument("--triples", type=Path, help="Generic TSV file with head<TAB>relation<TAB>tail")
    parser.add_argument("--use-splits", type=str, default="train,valid,test")
    parser.add_argument("--exact-hops", type=int, required=True, help="Enumerate all relation sequences of length == exact_hops")
    parser.add_argument("--relations", type=str, default="", help="Optional comma-separated subset of relations to use. Empty = all base relations.")
    parser.add_argument("--add-inverse-relations", action="store_true", help="Add __inv variants for all base relations.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preview-limit", type=int, default=5)
    parser.add_argument("--top-k-json", type=int, default=100, help="Write detailed JSON for the top K rows by exact_f1 then block_f1.")
    parser.add_argument("--sort-by", choices=["exact_f1", "block_f1", "exact_precision", "exact_recall", "block_precision", "block_recall", "block_bijection_ratio"], default="block_f1")
    parser.add_argument("--prune-immediate-inverse-cancel", action="store_true", help="Skip compositions containing adjacent r ; r__inv or r__inv ; r.")
    parser.add_argument("--prune-consecutive-duplicates", action="store_true", help="Skip compositions containing adjacent repeated relations like r ; r.")
    parser.add_argument("--prune-all-same-relation", action="store_true", help="Skip compositions where every hop is the same relation.")
    args = parser.parse_args()

    if args.exact_hops < 2:
        raise ValueError("--exact-hops must be >= 2")

    if args.dataset_format in {"wn18rr", "fb15k237"}:
        if args.data_dir is None:
            raise ValueError("--data-dir is required for KGE-style dataset formats")
        dataset = load_kge_dict_dataset(args.data_dir, [s.strip() for s in args.use_splits.split(",") if s.strip()])
    else:
        if args.triples is None:
            raise ValueError("--triples is required for --dataset-format generic_tsv")
        dataset = load_generic_tsv(args.triples)

    args.outdir.mkdir(parents=True, exist_ok=True)

    base_relation_names = sorted(dataset.relation_to_id.keys())
    if args.relations.strip():
        requested = [r.strip() for r in args.relations.split(",") if r.strip()]
        unknown = [r for r in requested if r not in dataset.relation_to_id]
        if unknown:
            raise KeyError(f"Unknown requested base relations: {unknown}")
        base_relation_names = requested

    relation_names = list(base_relation_names)
    if args.add_inverse_relations:
        relation_names += [f"{r}__inv" for r in base_relation_names]

    # Cache edge sets for every relation symbol used in the search.
    edge_cache: Dict[str, Set[Edge]] = {}
    for r in relation_names:
        edge_cache[r] = relation_edges_by_name(dataset.triples, dataset, r, allow_inverse=args.add_inverse_relations)

    rows: List[Dict[str, object]] = []
    detailed: List[Dict[str, object]] = []

    total_sequences = len(relation_names) ** args.exact_hops
    seq_counter = 0
    kept_sequence_counter = 0
    pruned_sequence_counter = 0
    pruned_reason_counts: Counter[str] = Counter()

    for seq in enumerate_relation_sequences(relation_names, args.exact_hops):
        seq_counter += 1
        prune_reason = prune_reason_for_sequence(
            seq,
            prune_immediate_inverse=args.prune_immediate_inverse_cancel,
            prune_consecutive_duplicates=args.prune_consecutive_duplicates,
            prune_all_same_relation=args.prune_all_same_relation,
        )
        if args.add_inverse_relations:
            if "__inv" not in ' ; '.join(seq):
                continue

        if prune_reason is not None:
            pruned_sequence_counter += 1
            pruned_reason_counts[prune_reason] += 1
            continue
        kept_sequence_counter += 1
        edge_sets = [edge_cache[r] for r in seq]
        full_edges = compose_edge_chain(edge_sets)
        full_stats = summarize_relation_edges(full_edges)
        if float(full_stats["block_bijection_holds"]) != 1:
            continue

        row = {
            "composition": " ; ".join(seq),
            "hop_count": len(seq),
            **full_stats,
        }
        rows.append(row)
        rows.sort(key=lambda r: (r["total_paths"]), reverse=True)

        if seq_counter % 1000 == 0:
            print(f"Processed {seq_counter}/{total_sequences} compositions (kept={kept_sequence_counter}, pruned={pruned_sequence_counter})")
            rows.sort(key=lambda r: (r["total_paths"]), reverse=True)
            write_jsonl(args.outdir / "transitive_summary.jsonl", rows)
            csv_path = args.outdir / "transitive_summary.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
                if rows:
                    writer.writeheader()
                    writer.writerows(rows)

            
            

    rows.sort(key=lambda r: (r["total_paths"]), reverse=True)
    write_jsonl(args.outdir / "transitive_summary.jsonl", rows)
    csv_path = args.outdir / "transitive_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print(f"Generated {len(rows)} transitive relations from {kept_sequence_counter} kept compositions out of {total_sequences} total")


if __name__ == "__main__":
    main()
