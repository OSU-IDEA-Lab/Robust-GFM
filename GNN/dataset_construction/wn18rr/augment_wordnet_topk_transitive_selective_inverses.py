#!/usr/bin/env python3
"""
Augment a WordNet/WN18RR-style KGE dataset with transitive/composed edges.

Expected input layout:
  - entities.dict
  - relations.dict
  - train.txt
  - valid.txt
  - test.txt

Expected recovery/composition CSV:
  - batch_recovery_summary.csv or similar
  - must include:
      composition
      total_paths
  - composition format:
      rel1 ; rel2 ; rel3
    where inverse relations may appear as:
      rel__inv

This script creates two dataset variants:

  outdir/no_changes/
    - entities.dict
    - relations.dict
    - data.txt
    - metapaths.json
    - augmentation_summary.json

  outdir/transitive_edges/
    - entities.dict
    - relations.dict
    - data.txt
    - metapaths.json
    - augmentation_summary.json

Behavior:
  - no_changes:
      combines train/valid/test into one data.txt
      keeps original relation vocabulary
      metapaths.json contains the selected original chains only

  - transitive_edges:
      combines train/valid/test into one data.txt
      adds new one-hop composed/transitive relation edges
      keeps original edges
      adds new composed relation names to relations.dict
      metapaths.json contains:
        * original selected chains
        * direct one-hop composed chains

Inverse handling:
  - If a composition uses rel__inv, the script uses transpose(rel) while composing.
  - In all_inverse_edges and transitive_edges, inverse edges are materialized
    only for original/base relations whose __inv form appears in the selected
    top-k metapaths. Inverse transitive relation types are not created.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp


def parse_dict_file(path: Path):
    name_to_id = {}
    id_to_name = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad dict line {line_idx} in {path}: {line!r}")
            raw_id, name = parts
            i = int(raw_id)
            name_to_id[name] = i
            id_to_name[i] = name
    return name_to_id, id_to_name


def parse_triples(path: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]):
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad triple line {line_idx} in {path}: {line!r}")
            h, r, t = parts
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def load_kge_dataset(data_dir: Path, splits: Sequence[str]):
    entity_to_id, id_to_entity = parse_dict_file(data_dir / "entities.dict")
    relation_to_id, id_to_relation = parse_dict_file(data_dir / "relations.dict")

    triples = []
    for split in splits:
        triples.extend(parse_triples(data_dir / f"{split}.txt", entity_to_id, relation_to_id))

    return {
        "entity_to_id": entity_to_id,
        "id_to_entity": id_to_entity,
        "relation_to_id": relation_to_id,
        "id_to_relation": id_to_relation,
        "triples": triples,
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
    }


def build_base_relation_matrices(num_nodes: int, triples, id_to_relation: dict[int, str]):
    by_rel = defaultdict(list)
    for h, r, t in triples:
        by_rel[r].append((h, t))

    out = {}
    for rid, rname in id_to_relation.items():
        pairs = by_rel.get(rid, [])
        if pairs:
            rows = np.asarray([h for h, _ in pairs], dtype=np.int32)
            cols = np.asarray([t for _, t in pairs], dtype=np.int32)
            data = np.ones(len(pairs), dtype=np.int64)
            M = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.int64).tocsr()
            M.sum_duplicates()
        else:
            M = sp.csr_matrix((num_nodes, num_nodes), dtype=np.int64)
        out[rname] = M
    return out


def add_inverse_relation_matrices(relation_matrices: dict[str, sp.csr_matrix]):
    out = dict(relation_matrices)
    for rname, M in list(relation_matrices.items()):
        out[f"{rname}__inv"] = M.transpose().tocsr()
    return out


def binarize_matrix(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr(copy=True)
    if M.nnz:
        M.data[:] = 1
        M.eliminate_zeros()
    return M


def compose_relations(
    relation_matrices: dict[str, sp.csr_matrix],
    relation_names: Sequence[str],
    binary_intermediate: bool = True,
) -> sp.csr_matrix:
    if not relation_names:
        raise ValueError("Cannot compose an empty relation sequence.")

    missing = [r for r in relation_names if r not in relation_matrices]
    if missing:
        raise KeyError(
            "Composition references relation(s) not found: " + ", ".join(missing)
        )

    M = relation_matrices[relation_names[0]].tocsr()
    if binary_intermediate:
        M = binarize_matrix(M)

    for rname in relation_names[1:]:
        M = (M @ relation_matrices[rname]).tocsr()
        if binary_intermediate:
            M = binarize_matrix(M)

    return M.tocsr()


def parse_relation_list(s: str):
    if s is None:
        return []
    s = s.strip()
    if not s or s == "<identity>":
        return []
    return [p.strip() for p in s.split(" ; ") if p.strip() and p.strip() != "<identity>"]


def read_topk_rows_by_total_paths(csv_path: Path, top_k: int):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        if "composition" not in reader.fieldnames:
            raise KeyError("CSV must contain a 'composition' column.")
        if "total_paths" not in reader.fieldnames:
            raise KeyError("CSV must contain a 'total_paths' column.")

        for row in reader:
            composition = parse_relation_list(row["composition"])
            if not composition:
                continue
            row["_composition_list"] = composition
            row["_total_paths_float"] = float(row["total_paths"])
            rows.append(row)

    rows.sort(key=lambda r: r["_total_paths_float"], reverse=True)

    if top_k < 0:
        raise ValueError("--top-k must be positive.")
    if top_k == 0:
        return rows

    return rows[:top_k]


def make_transitive_relation_name(composition: Sequence[str]) -> str:
    return "__transitive__" + "|||".join(composition)


def matrix_to_edge_pairs(M: sp.csr_matrix):
    M = binarize_matrix(M).tocoo()
    return set(zip(M.row.tolist(), M.col.tolist()))


def write_entities_dict(path: Path, id_to_entity: dict[int, str]):
    with path.open("w", encoding="utf-8") as f:
        for eid in sorted(id_to_entity):
            f.write(f"{eid}\t{id_to_entity[eid]}\n")


def write_relations_dict(path: Path, relation_names_in_order: Sequence[str]):
    rel_to_id = {r: i for i, r in enumerate(relation_names_in_order)}
    with path.open("w", encoding="utf-8") as f:
        for r, i in rel_to_id.items():
            f.write(f"{i}\t{r}\n")
    return rel_to_id


def write_data_txt(path: Path, edges, id_to_entity: dict[int, str]):
    with path.open("w", encoding="utf-8") as f:
        for h, rname, t in edges:
            f.write(f"{id_to_entity[h]}\t{rname}\t{id_to_entity[t]}\n")


def add_named_inverse_edges(edges):
    """Return edges plus inverse edges for every non-inverse relation in edges."""
    out = list(edges)
    for h, rname, t in edges:
        if rname.endswith("__inv"):
            continue
        out.append((t, f"{rname}__inv", h))
    return out


def add_selected_named_inverse_edges(edges, selected_base_relations):
    """Return edges plus inverse edges only for selected base relation names."""
    selected_base_relations = set(selected_base_relations)
    out = list(edges)
    for h, rname, t in edges:
        if rname.endswith("__inv"):
            continue
        if rname in selected_base_relations:
            out.append((t, f"{rname}__inv", h))
    return out


def selected_inverse_base_relations_from_chains(chains):
    """Return base relation names r for which r__inv appears in selected chains."""
    out = set()
    for chain in chains:
        for r in chain:
            if r.endswith("__inv"):
                out.add(r[:-5])
    return out


def dedupe_chains(chains):
    seen = set()
    out = []
    for chain in chains:
        tup = tuple(chain)
        if tup not in seen:
            seen.add(tup)
            out.append(chain)
    return out


def non_inverse_contiguous_subchains(chain):
    """Return all maximal and smaller contiguous subchains that contain no __inv relations.

    Example:
      [r1, r2__inv, r3] -> [[r1], [r3]]
      [r1, r2, r3] -> [[r1], [r2], [r3], [r1,r2], [r2,r3], [r1,r2,r3]]

    Empty/0-hop chains are not written to metapaths.json because MAGNN relation_chains
    are relation-id sequences. The number of zero-length gaps is reported separately.
    """
    out = []
    n = len(chain)
    for i in range(n):
        if chain[i].endswith("__inv"):
            continue
        cur = []
        for j in range(i, n):
            if chain[j].endswith("__inv"):
                break
            cur.append(chain[j])
            out.append(cur.copy())
    return out


def count_zero_hop_inverse_gaps(chain):
    """Count positions where an inverse relation prevents a non-inverse chain segment."""
    return sum(1 for r in chain if r.endswith("__inv"))


def chain_ids_for_existing_relations(chains, rel_to_id):
    out = []
    skipped = []
    for chain in chains:
        missing = [r for r in chain if r not in rel_to_id]
        if missing:
            skipped.append({"chain": chain, "missing_relations": missing})
            continue
        out.append([rel_to_id[r] for r in chain])
    return out, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Create no_changes and transitive_edges datasets for WordNet/WN18RR-style KGE data."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--recovery-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True, help="Select top-k compositions sorted by total_paths descending.")
    parser.add_argument("--splits", type=str, default="train,valid,test")
    parser.add_argument(
        "--binary-intermediate",
        action="store_true",
        default=True,
        help="Binarize after each composition hop. Default: true.",
    )
    parser.add_argument(
        "--no-binary-intermediate",
        dest="binary_intermediate",
        action="store_false",
        help="Do not binarize after each hop.",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ds = load_kge_dataset(args.data_dir, splits)

    base_relation_names = [ds["id_to_relation"][i] for i in sorted(ds["id_to_relation"])]
    base_edges_named = [(h, ds["id_to_relation"][r], t) for h, r, t in ds["triples"]]

    base_mats = build_base_relation_matrices(ds["num_entities"], ds["triples"], ds["id_to_relation"])
    relation_matrices = add_inverse_relation_matrices(base_mats)

    selected_rows = read_topk_rows_by_total_paths(args.recovery_csv, args.top_k)
    if not selected_rows:
        raise ValueError("No valid composition rows found in recovery CSV.")

    transitive_edges_by_relation = defaultdict(set)
    selected_chains = []
    direct_chains = []
    composition_to_transitive_relation = []

    for idx, row in enumerate(selected_rows, start=1):
        composition = row["_composition_list"]
        trans_rel_name = make_transitive_relation_name(composition)

        M = compose_relations(
            relation_matrices,
            composition,
            binary_intermediate=args.binary_intermediate,
        )
        edge_pairs = matrix_to_edge_pairs(M)
        transitive_edges_by_relation[trans_rel_name].update(edge_pairs)

        selected_chains.append(composition)
        direct_chains.append([trans_rel_name])
        composition_to_transitive_relation.append({
            "composition": composition,
            "transitive_relation": trans_rel_name,
            "source_total_paths": float(row["total_paths"]),
            "num_added_transitive_edges": len(edge_pairs),
            "uses_inverse_relation": any(r.endswith("__inv") for r in composition),
        })

        if idx % 25 == 0 or idx == len(selected_rows):
            print(f"Processed {idx}/{len(selected_rows)} selected top-k compositions")

    selected_chains = dedupe_chains(selected_chains)
    direct_chains = dedupe_chains(direct_chains)
    transitive_relation_names = sorted(transitive_edges_by_relation.keys())

    # Only materialize inverse edges for base relations whose __inv form is
    # explicitly used in the selected top-k metapaths.
    selected_inverse_base_relations = sorted(
        r for r in selected_inverse_base_relations_from_chains(selected_chains)
        if r in base_relation_names
    )
    selected_inverse_relation_names = [f"{r}__inv" for r in selected_inverse_base_relations]

    args.outdir.mkdir(parents=True, exist_ok=True)

    # no_changes
    no_changes_dir = args.outdir / "no_changes"
    no_changes_dir.mkdir(parents=True, exist_ok=True)

    write_entities_dict(no_changes_dir / "entities.dict", ds["id_to_entity"])
    no_changes_rel_to_id = write_relations_dict(no_changes_dir / "relations.dict", base_relation_names)
    write_data_txt(no_changes_dir / "data.txt", base_edges_named, ds["id_to_entity"])

    # no_changes does not materialize inverse edges, so its MAGNN chains must only
    # include contiguous parts of the selected compositions that contain no __inv relation.
    no_changes_non_inverse_chains = []
    no_changes_chain_derivations = []
    zero_hop_inverse_gap_count = 0
    for chain in selected_chains:
        pieces = non_inverse_contiguous_subchains(chain)
        zero_hop_inverse_gap_count += count_zero_hop_inverse_gaps(chain)
        no_changes_non_inverse_chains.extend(pieces)
        no_changes_chain_derivations.append({
            "source_composition": chain,
            "non_inverse_contiguous_chains": pieces,
            "zero_hop_inverse_gaps": count_zero_hop_inverse_gaps(chain),
        })

    no_changes_non_inverse_chains = dedupe_chains(no_changes_non_inverse_chains)
    no_changes_chain_ids, no_changes_skipped = chain_ids_for_existing_relations(
        no_changes_non_inverse_chains, no_changes_rel_to_id
    )

    with (no_changes_dir / "metapaths.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "no_changes",
            "splits_used": splits,
            "selection": {"method": "top_k_by_total_paths", "top_k": args.top_k},
            "chains": no_changes_non_inverse_chains,
            "chain_relation_ids": no_changes_chain_ids,
            "chains_with_missing_relation_ids": no_changes_skipped,
            "chain_derivations_from_selected_compositions": no_changes_chain_derivations,
            "zero_hop_inverse_gap_count": zero_hop_inverse_gap_count,
            "note": (
                "no_changes does not materialize inverse edges. Therefore chains are the "
                "contiguous non-inverse subchains of the selected compositions. Empty/0-hop "
                "segments are reported as zero_hop_inverse_gaps but are not included as "
                "MAGNN relation chains."
            ),
        }, f, indent=2)

    with (no_changes_dir / "augmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "no_changes",
            "num_edges": len(base_edges_named),
            "num_relations": len(base_relation_names),
            "num_selected_compositions": len(selected_chains),
            "num_non_inverse_metapath_chains": len(no_changes_non_inverse_chains),
            "num_added_transitive_edges": 0,
        }, f, indent=2)

    print(f"Wrote {no_changes_dir}")

    # -----------------------------
    # all_inverse_edges
    # -----------------------------
    all_inv_dir = args.outdir / "all_inverse_edges"
    all_inv_dir.mkdir(parents=True, exist_ok=True)

    # Only add inverse edges/relations for inverse relations explicitly appearing
    # in the selected top-k metapaths.
    all_inv_relation_names = base_relation_names + selected_inverse_relation_names
    all_inv_edges_named = add_selected_named_inverse_edges(
        base_edges_named,
        selected_inverse_base_relations,
    )

    write_entities_dict(all_inv_dir / "entities.dict", ds["id_to_entity"])
    all_inv_rel_to_id = write_relations_dict(all_inv_dir / "relations.dict", all_inv_relation_names)
    write_data_txt(all_inv_dir / "data.txt", all_inv_edges_named, ds["id_to_entity"])

    all_inv_chain_ids, all_inv_skipped = chain_ids_for_existing_relations(selected_chains, all_inv_rel_to_id)

    with (all_inv_dir / "metapaths.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "all_inverse_edges",
            "splits_used": splits,
            "selection": {"method": "top_k_by_total_paths", "top_k": args.top_k},
            "chains": selected_chains,
            "chain_relation_ids": all_inv_chain_ids,
            "chains_with_missing_relation_ids": all_inv_skipped,
            "note": (
                "This variant materializes inverse edges only for original/base relations "
                "whose __inv form appears in the selected top-k metapaths."
            ),
        }, f, indent=2)

    with (all_inv_dir / "augmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "all_inverse_edges",
            "num_original_edges": len(base_edges_named),
            "num_edges_with_inverse_materialization": len(all_inv_edges_named),
            "num_base_relations": len(base_relation_names),
            "num_selected_inverse_base_relations": len(selected_inverse_base_relations),
            "selected_inverse_base_relations": selected_inverse_base_relations,
            "num_base_inverse_relation_types": len(selected_inverse_relation_names),
            "num_total_relation_types": len(all_inv_relation_names),
            "num_added_inverse_edges": len(all_inv_edges_named) - len(base_edges_named),
            "num_selected_chains": len(selected_chains),
        }, f, indent=2)

    print(f"Wrote {all_inv_dir}")

    # transitive_edges
    trans_dir = args.outdir / "transitive_edges"
    trans_dir.mkdir(parents=True, exist_ok=True)

    # In the transitive variant, materialize inverse edges ONLY for base
    # relations whose __inv form appears in the selected top-k metapaths.
    # Do NOT create inverse relation types or inverse edges for newly added
    # transitive/composed relations.
    base_edges_with_selected_inverse = add_selected_named_inverse_edges(
        base_edges_named,
        selected_inverse_base_relations,
    )

    transitive_edges_named_only = []
    for rname in transitive_relation_names:
        for h, t in sorted(transitive_edges_by_relation[rname]):
            transitive_edges_named_only.append((h, rname, t))

    trans_edges_named = base_edges_with_selected_inverse + transitive_edges_named_only

    all_relation_names = (
        base_relation_names
        + selected_inverse_relation_names
        + transitive_relation_names
    )

    write_entities_dict(trans_dir / "entities.dict", ds["id_to_entity"])
    trans_rel_to_id = write_relations_dict(trans_dir / "relations.dict", all_relation_names)
    write_data_txt(trans_dir / "data.txt", trans_edges_named, ds["id_to_entity"])

    trans_chain_ids, trans_skipped = chain_ids_for_existing_relations(selected_chains, trans_rel_to_id)

    with (trans_dir / "metapaths.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "transitive_edges",
            "splits_used": splits,
            "selection": {"method": "top_k_by_total_paths", "top_k": args.top_k},
            "chains": selected_chains,
            "direct_transitive_chains": direct_chains,
            "chain_relation_ids": trans_chain_ids,
            "chains_with_missing_relation_ids": trans_skipped,
            "direct_transitive_relation_ids": [[trans_rel_to_id[chain[0]]] for chain in direct_chains],
            "composition_to_transitive_relation": composition_to_transitive_relation,
            "note": (
                "The transitive_edges variant materializes inverse edges only for original/base "
                "relations whose __inv form appears in the selected top-k metapaths. It does "
                "not create inverse relation types or inverse edges for newly added "
                "transitive/composed relations."
            ),
        }, f, indent=2)

    with (trans_dir / "augmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": "transitive_edges",
            "num_original_edges": len(base_edges_named),
            "num_augmented_edges_total": len(trans_edges_named),
            "num_edges_before_selected_base_inverse_materialization": len(base_edges_named) + len(transitive_edges_named_only),
            "num_base_relations": len(base_relation_names),
            "num_selected_inverse_base_relations": len(selected_inverse_base_relations),
            "selected_inverse_base_relations": selected_inverse_base_relations,
            "num_base_inverse_relation_types": len(selected_inverse_relation_names),
            "num_transitive_relation_types": len(transitive_relation_names),
            "num_transitive_inverse_relation_types": 0,
            "num_total_relation_types": len(all_relation_names),
            "num_added_transitive_edges_total": int(sum(len(v) for v in transitive_edges_by_relation.values())),
            "num_added_selected_base_inverse_edges_total": len(base_edges_with_selected_inverse) - len(base_edges_named),
            "added_transitive_edges_by_relation": {r: len(v) for r, v in sorted(transitive_edges_by_relation.items())},
            "num_selected_chains": len(selected_chains),
        }, f, indent=2)

    print(f"Wrote {trans_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
