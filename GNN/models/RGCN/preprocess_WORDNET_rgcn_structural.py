#!/usr/bin/env python3
"""Compile invariant WordNet RGCN semantic message programs from three raw variants.

The raw physical graph is preserved per variant. RGCN propagation consumes a
canonical semantic edge table built from:
  * all official base relations;
  * deterministic inverse views of those relations; and
  * selected three-hop dependency endpoints from the MAGNN reduced/full catalog.

For a dependency q1;q2;q3, variants without a direct shortcut reconstruct the
endpoint relation from the three-hop physical context. The universal variant
also verifies that its physical shortcut relation equals the reconstructed set.
Duplicate witnesses collapse under set semantics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from wordnet_structural_common import (
    BUDGETS,
    FORMAT_VERSION,
    INVERSE_SUFFIX,
    MODES,
    TRANSITIVE_PREFIX,
    VARIANTS,
    array_sha256,
    atomic_json_dump,
    canonical_triples,
    compose_three_hop,
    edge_table_from_triples,
    endpoint_rows,
    get_relation_view,
    get_supervision_train,
    infer_num_base_relations,
    load_selection,
    physical_hash,
    read_split_archive,
    relation_adjacency,
    relation_pair_set,
    resolve_selection,
    rotate_and_cap,
    semantic_hash,
    supervision_hash,
)


def _semantic_catalog(base_names: Sequence[str], dependencies: Sequence[dict], mode: str, include_base: bool):
    names: list[str] = []
    if include_base:
        names.extend(f"base::{name}" for name in base_names)
        names.extend(f"inverse::{name}{INVERSE_SUFFIX}" for name in base_names)
    for dep in dependencies:
        did = str(dep["dependency_id"])
        if mode == "intersection":
            names.append(f"dependency::{did}")
        elif mode == "union":
            names.extend((f"dependency::{did}::chain", f"dependency::{did}::shortcut"))
        else:
            raise ValueError(mode)
    return names


def _pairs_to_rows(pairs: np.ndarray, relation_id: int) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.column_stack(
        [pairs[:, 0], np.full(len(pairs), int(relation_id), dtype=np.int64), pairs[:, 1]]
    )


def compile_variant(
    *,
    variant: str,
    arrays: Mapping[str, np.ndarray],
    selection: Mapping,
    budget: str,
    mode: str,
    max_endpoints_per_source: int,
    sampling_seed: int,
    include_base_relations: bool,
    shortcut_audit: str,
) -> tuple[dict, dict]:
    if shortcut_audit not in {"full", "sampled", "none"}:
        raise ValueError("shortcut_audit must be full, sampled, or none")

    relation_vocab = [str(x) for x in arrays["relation_vocab"].tolist()]
    entity_vocab = [str(x) for x in arrays["entity_vocab"].tolist()]
    relation_to_id = {name: i for i, name in enumerate(relation_vocab)}
    num_entities = int(np.asarray(arrays.get("num_entities", len(entity_vocab))).item())
    num_base_relations = infer_num_base_relations(arrays)
    base_names = relation_vocab[:num_base_relations]
    physical = np.asarray(arrays[f"train_pos_{variant}"], dtype=np.int64)
    supervision_train = get_supervision_train(arrays)
    val = np.asarray(arrays["val_pos"], dtype=np.int64)
    test = np.asarray(arrays["test_pos"], dtype=np.int64)

    if np.any(supervision_train[:, 1] >= num_base_relations):
        raise ValueError("Official training supervision contains augmented relation IDs")
    if np.any(val[:, 1] >= num_base_relations) or np.any(test[:, 1] >= num_base_relations):
        raise ValueError("Validation/test contains augmented relation IDs")

    adjacency, active_names = relation_adjacency(physical, relation_vocab)
    dependencies = list(selection["selected_dependencies"])
    semantic_names = _semantic_catalog(base_names, dependencies, mode, include_base_relations)
    sem_name_to_id = {name: i for i, name in enumerate(semantic_names)}
    semantic_rows: list[np.ndarray] = []
    audit_relations: list[dict] = []

    if include_base_relations:
        for base_name in base_names:
            direct = get_relation_view(base_name, adjacency)
            pairs = endpoint_rows(direct)
            semantic_rows.append(_pairs_to_rows(pairs, sem_name_to_id[f"base::{base_name}"]))
            audit_relations.append({
                "semantic_relation": f"base::{base_name}",
                "source": "physical_base_relation",
                "num_pairs": int(len(pairs)),
            })
        for base_name in base_names:
            inv_name = f"{base_name}{INVERSE_SUFFIX}"
            inv = get_relation_view(inv_name, adjacency)
            pairs = endpoint_rows(inv)
            semantic_rows.append(_pairs_to_rows(pairs, sem_name_to_id[f"inverse::{inv_name}"]))
            direct_or_virtual = "physical_inverse_relation" if inv_name in active_names else "virtual_reverse_view"
            # A physical inverse must be the exact transpose of the base relation.
            expected = {(t, h) for h, t in relation_pair_set(get_relation_view(base_name, adjacency))}
            actual = relation_pair_set(inv)
            if actual != expected:
                raise RuntimeError(
                    f"Inverse relation mismatch for {variant}/{inv_name}: "
                    f"missing={len(expected-actual)}, extra={len(actual-expected)}"
                )
            audit_relations.append({
                "semantic_relation": f"inverse::{inv_name}",
                "source": direct_or_virtual,
                "num_pairs": int(len(pairs)),
            })

    dep_audits: list[dict] = []
    for dep_index, dep in enumerate(dependencies):
        did = str(dep["dependency_id"])
        comp = [str(x) for x in dep["composition"]]
        full_map = compose_three_hop(comp, adjacency)
        sampled_map = rotate_and_cap(
            full_map,
            cap=max_endpoints_per_source,
            seed=sampling_seed,
            dependency_id=did,
        )
        pairs = endpoint_rows(sampled_map)

        shortcut_name = str(dep.get("shortcut_relation", ""))
        direct_present = shortcut_name in active_names
        direct_full_pairs = None
        full_equal = None
        sampled_equal = None
        if direct_present and shortcut_audit != "none":
            direct_view = get_relation_view(shortcut_name, adjacency)
            direct_full_pairs = relation_pair_set(direct_view)
            composed_full_pairs = relation_pair_set(full_map)
            if shortcut_audit == "full":
                full_equal = direct_full_pairs == composed_full_pairs
                if not full_equal:
                    raise RuntimeError(
                        f"Shortcut equality failed for dependency={did}, variant={variant}: "
                        f"missing={len(composed_full_pairs-direct_full_pairs)}, "
                        f"extra={len(direct_full_pairs-composed_full_pairs)}"
                    )
            sampled_pairs = set(map(tuple, pairs.tolist()))
            sampled_equal = sampled_pairs.issubset(direct_full_pairs)
            if not sampled_equal:
                raise RuntimeError(
                    f"Sampled shortcut endpoints absent for dependency={did}, variant={variant}"
                )

        if mode == "intersection":
            semantic_rows.append(_pairs_to_rows(pairs, sem_name_to_id[f"dependency::{did}"]))
        else:
            semantic_rows.append(_pairs_to_rows(pairs, sem_name_to_id[f"dependency::{did}::chain"]))
            semantic_rows.append(_pairs_to_rows(pairs, sem_name_to_id[f"dependency::{did}::shortcut"]))

        dep_audits.append({
            "index": dep_index,
            "dependency_id": did,
            "composition": comp,
            "shortcut_relation": shortcut_name,
            "direct_shortcut_present": bool(direct_present),
            "reconstruction": "direct_verified_and_chain_composed" if direct_present else "chain_composed",
            "num_full_endpoint_pairs": int(sum(len(v) for v in full_map.values())),
            "num_sampled_endpoint_pairs": int(len(pairs)),
            "num_active_sources": int(len(sampled_map)),
            "full_shortcut_equal": full_equal,
            "sampled_pairs_in_direct_shortcut": sampled_equal,
        })
        if (dep_index + 1) % 100 == 0 or dep_index + 1 == len(dependencies):
            print(
                f"[{variant}/{budget}/{mode}] dependencies {dep_index+1}/{len(dependencies)}",
                flush=True,
            )

    all_rows = np.concatenate(semantic_rows, axis=0) if semantic_rows else np.empty((0, 3), dtype=np.int64)
    all_rows = canonical_triples(all_rows)
    semantic_edge_index, semantic_edge_type = edge_table_from_triples(all_rows)
    physical_edge_index, physical_edge_type = edge_table_from_triples(physical)

    sem_hash = semantic_hash(semantic_edge_index, semantic_edge_type, semantic_names, num_entities)
    phys_hash = physical_hash(physical, relation_vocab, num_entities)
    sup_hash = supervision_hash(
        supervision_train,
        val,
        test,
        entity_vocab,
        base_names,
    )

    payload = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "budget": budget,
        "mode": mode,
        "num_entities": num_entities,
        "num_base_relations": num_base_relations,
        "entity_vocab": entity_vocab,
        # The official task always evaluates the first ``num_base_relations``
        # relations.  ``semantic_all`` training additionally decodes every
        # compiled inverse/shortcut relation, matching the original universal
        # graph baseline's auxiliary training objective.
        "task_relation_names": base_names,
        "decoder_relation_names": semantic_names,
        "decoder_relation_names_task_only": base_names,
        "decoder_relation_names_semantic_all": semantic_names,
        "physical_relation_names": relation_vocab,
        "semantic_relation_names": semantic_names,
        "physical_edge_index": physical_edge_index,
        "physical_edge_type": physical_edge_type,
        "physical_train_pos_raw": torch.as_tensor(physical, dtype=torch.long),
        "physical_train_pos_canonical": torch.as_tensor(canonical_triples(physical), dtype=torch.long),
        "semantic_edge_index": semantic_edge_index,
        "semantic_edge_type": semantic_edge_type,
        # Backward compatibility: ``train_pos`` remains the official 11-relation
        # task supervision.  The parity runner selects ``semantic_train_pos``
        # when reproducing the universal graph objective.
        "train_pos": torch.as_tensor(supervision_train, dtype=torch.long),
        "task_train_pos": torch.as_tensor(supervision_train, dtype=torch.long),
        "semantic_train_pos": torch.as_tensor(all_rows, dtype=torch.long),
        "val_pos": torch.as_tensor(val, dtype=torch.long),
        "test_pos": torch.as_tensor(test, dtype=torch.long),
        "physical_hash": phys_hash,
        "semantic_hash": sem_hash,
        "supervision_hash": sup_hash,
        "selection_hash": array_sha256(
            np.asarray([int(str(dep["dependency_id"]), 16) & ((1 << 63) - 1) for dep in dependencies], dtype=np.int64)
        ),
        "metadata": {
            "include_base_relations": bool(include_base_relations),
            "max_endpoints_per_source": int(max_endpoints_per_source),
            "sampling_seed": int(sampling_seed),
            "shortcut_audit": shortcut_audit,
            "selection_name": selection.get("name"),
            "selection_summary": selection.get("summary", {}),
            "num_dependencies": len(dependencies),
            "num_physical_training_triples_raw": int(len(physical)),
            "num_physical_training_triples_canonical": int(len(canonical_triples(physical))),
            "num_physical_duplicate_triples": int(len(physical) - len(canonical_triples(physical))),
            "num_task_training_triples": int(len(supervision_train)),
            "num_semantic_training_triples": int(len(all_rows)),
            "num_full_dependency_endpoint_pairs": int(sum(x["num_full_endpoint_pairs"] for x in dep_audits)),
            "num_retained_dependency_endpoint_pairs": int(sum(x["num_sampled_endpoint_pairs"] for x in dep_audits)),
            "num_dropped_dependency_endpoint_pairs": int(sum(x["num_full_endpoint_pairs"] - x["num_sampled_endpoint_pairs"] for x in dep_audits)),
            "semantic_edge_order": "destination_relation_source",
            "semantic_aggregation": "deterministic_contiguous_segment_sum",
        },
    }
    audit = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "budget": budget,
        "mode": mode,
        "physical_hash": phys_hash,
        "semantic_hash": sem_hash,
        "supervision_hash": sup_hash,
        "num_physical_edges": int(physical_edge_type.numel()),
        "num_semantic_edges": int(semantic_edge_type.numel()),
        "num_semantic_relations": len(semantic_names),
        "num_dependencies": len(dependencies),
        "base_relation_audit": audit_relations,
        "dependency_audit": dep_audits,
        "metadata": payload["metadata"],
    }
    return payload, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-path", required=True)
    ap.add_argument("--catalog-root", default=str(Path(__file__).resolve().parent / "wordnet_structural_catalogs"))
    ap.add_argument("--output-root", default="data/preprocessed/RGCN_structural/WORDNET/link_prediction")
    ap.add_argument("--budget", choices=BUDGETS, default="reduced")
    ap.add_argument("--mode", choices=MODES, default="intersection")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-endpoints-per-source", type=int, default=0)
    ap.add_argument("--sampling-seed", type=int, default=1566911444)
    ap.add_argument("--include-base-relations", type=int, choices=(0, 1), default=1)
    ap.add_argument("--shortcut-audit", choices=("full", "sampled", "none"), default="full")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    variants = [x.strip() for x in args.variants.replace(" ", ",").split(",") if x.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variants: {bad}")
    if args.max_endpoints_per_source < 0:
        raise ValueError("--max-endpoints-per-source must be >= 0")

    arrays = read_split_archive(args.splits_path)
    selection_path = resolve_selection(args.catalog_root, args.budget)
    selection = load_selection(selection_path)
    output_root = Path(args.output_root).expanduser().resolve()
    records = []
    t0 = time.perf_counter()

    for variant in variants:
        out = output_root / args.budget / args.mode / variant
        graph_path = out / "graph.pt"
        if graph_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {graph_path}; pass --overwrite")
        print(f"Compiling WordNet {variant}/{args.budget}/{args.mode}", flush=True)
        payload, audit = compile_variant(
            variant=variant,
            arrays=arrays,
            selection=selection,
            budget=args.budget,
            mode=args.mode,
            max_endpoints_per_source=args.max_endpoints_per_source,
            sampling_seed=args.sampling_seed,
            include_base_relations=bool(args.include_base_relations),
            shortcut_audit=args.shortcut_audit,
        )
        out.mkdir(parents=True, exist_ok=True)
        torch.save(payload, graph_path)
        atomic_json_dump(audit, out / "audit.json")
        records.append(audit)
        print(
            f"[{variant}] physical={audit['physical_hash'][:12]} "
            f"semantic={audit['semantic_hash'][:12]} "
            f"Eraw={audit['num_physical_edges']} Esem={audit['num_semantic_edges']} "
            f"Rsem={audit['num_semantic_relations']}",
            flush=True,
        )

    semantic_hashes = {r["semantic_hash"] for r in records}
    physical_hashes = {r["physical_hash"] for r in records}
    supervision_hashes = {r["supervision_hash"] for r in records}
    if len(semantic_hashes) != 1:
        raise RuntimeError(f"Semantic hashes differ across variants: {semantic_hashes}")
    if len(physical_hashes) < 2:
        raise RuntimeError("Physical input graphs are identical")
    if len(supervision_hashes) != 1:
        raise RuntimeError(f"Supervision hashes differ: {supervision_hashes}")

    summary = {
        "status": "PASS",
        "splits_path": str(Path(args.splits_path).resolve()),
        "selection_path": str(selection_path.resolve()),
        "budget": args.budget,
        "mode": args.mode,
        "variants": variants,
        "semantic_hash": next(iter(semantic_hashes)),
        "physical_hashes": sorted(physical_hashes),
        "supervision_hash": next(iter(supervision_hashes)),
        "elapsed_sec": time.perf_counter() - t0,
        "records": records,
    }
    atomic_json_dump(summary, output_root / args.budget / args.mode / "preprocess_summary.json")
    print("WordNet preprocessing complete: physical hashes differ; semantic hashes match.")


if __name__ == "__main__":
    main()
