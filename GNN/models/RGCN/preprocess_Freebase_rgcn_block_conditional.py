#!/usr/bin/env python3
"""Compile block-local Freebase mappings into conditional RGCN relation channels.

Raw graph files are never rewritten.  Mined shortcut families are reconstructed
from their authoritative base-relation compositions, one relation family and
one source-local endpoint block at a time.  Physical shortcut edges are treated
as provenance/validation, matching the supplied MAGNN Freebase package.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from freebase_block_common import (
    RelationResolver,
    base_fingerprints,
    build_outgoing_csr,
    csr_hash,
    load_catalog,
    make_split,
    parse_info,
    read_graph,
    read_labels,
    read_nodes,
    safe_name,
    save_csr,
    sha256_array,
    sha256_file,
    sha256_json,
    transpose_csr,
    verify_full_physical_pairs,
    write_json,
)

FORMAT = "freebase_rgcn_block_conditional_v1"


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    package_root = Path(args.package_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    catalog = load_catalog(package_root, args.up_to, args.set_mode, args.scope)
    ref_node_type_names, ref_relations = parse_info(reference_dir / "info.dat")
    input_node_type_names, input_relations = parse_info(data_dir / "info.dat")
    ref_by_name = {spec.name: spec for spec in ref_relations.values()}
    input_by_name = {spec.name: spec for spec in input_relations.values()}

    ref_names, ref_types = read_nodes(reference_dir / "node.dat")
    names, node_types = read_nodes(data_dir / "node.dat")
    if names != ref_names or not np.array_equal(node_types, ref_types):
        raise ValueError("Input and unchanged reference node.dat differ")
    labels = read_labels(data_dir / "label.dat")
    ref_labels = read_labels(reference_dir / "label.dat")
    if labels != ref_labels:
        raise ValueError("Input and unchanged reference label.dat differ")

    selected = list(catalog["semantic_relations"])
    needed_base: set[str] = set()
    mapped_names: set[str] = set()
    for relation in selected:
        if relation["kind"] == "base":
            needed_base.add(str(relation["name"]))
        else:
            mapped_names.add(str(relation["name"]))
            needed_base.update(str(x) for x in relation["composition"])
    missing = sorted(needed_base - set(ref_by_name))
    if missing:
        raise KeyError(f"Catalog dependencies absent from unchanged info.dat: {missing}")

    full_dir = output_dir / ".physical_pairs" if args.verify_physical == "full" else None
    base, physical, raw_edge_count = read_graph(
        data_dir=data_dir,
        relation_schema=input_relations,
        needed_base_names=needed_base,
        mapped_names=mapped_names,
        full_physical_dir=full_dir,
    )
    resolver = RelationResolver(base)
    base_hashes = base_fingerprints(base)

    num_nodes = int(node_types.size)
    center_type = int(catalog["center_type"])
    y, target_nodes, split, global_split = make_split(
        labels,
        node_types,
        center_type,
        args.train_ratio,
        args.val_ratio,
        args.split_seed,
    )
    labels_global = np.full(num_nodes, -1, dtype=np.int64)
    labels_global[target_nodes] = y
    np.save(output_dir / "node_types.npy", node_types)
    np.save(output_dir / "target_nodes.npy", target_nodes)
    np.save(output_dir / "labels.npy", y)
    np.save(output_dir / "labels_global.npy", labels_global)
    np.savez(output_dir / "train_val_test_idx.npz", **split)
    np.savez(output_dir / "train_val_test_global_idx.npz", **global_split)

    channels_dir = output_dir / "channels"
    channels_dir.mkdir()
    relation_records: list[dict[str, Any]] = []
    channel_records: list[dict[str, Any]] = []
    physical_verification: dict[str, Any] = {}

    for relation_index, relation in enumerate(selected):
        name = str(relation["name"])
        if relation["kind"] == "base":
            spec = ref_by_name[name]
            start_type, end_type = int(spec.start), int(spec.end)
            if (start_type, end_type) != (int(relation["start_type"]), int(relation["end_type"])):
                raise ValueError(f"Base relation schema mismatch for {name}")
        else:
            start_type, end_type = int(relation["start_type"]), int(relation["end_type"])

        source_nodes = np.where(node_types == start_type)[0].astype(np.int64, copy=False)
        out_indptr, out_indices, block_stats = build_outgoing_csr(
            num_nodes=num_nodes,
            source_nodes=source_nodes,
            relation=relation,
            resolver=resolver,
        )
        in_indptr, in_indices = transpose_csr(num_nodes, out_indptr, out_indices)
        semantic_hash = block_stats["semantic_pair_sha256"]

        physical_present = name in input_by_name
        pinfo = dict(physical.get(name, {"raw_count": 0, "xor64": 0, "sum64": 0}))
        expected = relation.get("expected_unique_endpoint_pairs")
        count_ok = True
        if physical_present and expected is not None:
            count_ok = int(pinfo["raw_count"]) == int(expected)
            if args.verify_physical in {"counts", "full"} and not count_ok:
                raise ValueError(
                    f"Physical shortcut count mismatch for {name}: "
                    f"observed={pinfo['raw_count']} expected={expected}"
                )
        pcheck: dict[str, Any] = {
            "physical_relation_present": bool(physical_present),
            "raw_physical_edge_count": int(pinfo["raw_count"]),
            "registry_expected_unique_endpoint_pairs": expected,
            "count_matches_registry": bool(count_ok),
            "verification_mode": args.verify_physical,
        }
        if args.verify_physical == "full" and physical_present:
            pair_file = full_dir / f"{safe_name(name)}.pairs.i32"  # type: ignore[operator]
            full = verify_full_physical_pairs(
                pair_file,
                num_nodes=num_nodes,
                expected_indptr=out_indptr,
                expected_indices=out_indices,
            )
            pcheck.update(full)
            if not full["matches_composed_semantic_blocks"]:
                raise ValueError(f"Physical shortcut endpoints differ from composition for {name}")
        physical_verification[name] = pcheck

        forward_id = 2 * relation_index
        reverse_id = forward_id + 1
        forward_dir = channels_dir / f"{forward_id:04d}_{safe_name(name)}_forward"
        reverse_dir = channels_dir / f"{reverse_id:04d}_{safe_name(name)}_reverse"
        f_record = save_csr(forward_dir, in_indptr, in_indices)
        # Reverse incoming neighborhoods are exactly the source-major outgoing CSR.
        r_record = save_csr(reverse_dir, out_indptr, out_indices)
        channel_records.extend(
            [
                {
                    "channel_id": forward_id,
                    "semantic_relation": name,
                    "direction": 1,
                    "source_type": start_type,
                    "destination_type": end_type,
                    "directory": str(forward_dir.relative_to(output_dir)),
                    **f_record,
                },
                {
                    "channel_id": reverse_id,
                    "semantic_relation": name,
                    "direction": -1,
                    "source_type": end_type,
                    "destination_type": start_type,
                    "directory": str(reverse_dir.relative_to(output_dir)),
                    **r_record,
                },
            ]
        )
        relation_records.append(
            {
                **relation,
                "relation_index": relation_index,
                "forward_channel_id": forward_id,
                "reverse_channel_id": reverse_id,
                "model_context": (
                    {"kind": "direct_base_relation", "relations": [name]}
                    if relation["kind"] == "base"
                    else {
                        "kind": "block_local_base_composition",
                        "relations": list(relation["composition"]),
                        "context_only_internal_nodes": True,
                    }
                ),
                "physical_shortcut_available_in_input": bool(physical_present),
                "semantic_pair_sha256": semantic_hash,
                "outgoing_csr_sha256": csr_hash(out_indptr, out_indices),
                **block_stats,
            }
        )
        del out_indptr, out_indices, in_indptr, in_indices

    if full_dir is not None:
        shutil.rmtree(full_dir, ignore_errors=True)

    relation_catalog = {
        "format_version": FORMAT,
        "input_variant": args.input_variant,
        "up_to_variant": args.up_to,
        "set_mode": args.set_mode,
        "scope": args.scope,
        "center_type": center_type,
        "num_nodes": num_nodes,
        "num_relations": len(channel_records),
        "relations": relation_records,
        "channels": channel_records,
        "mapping_semantics": "independent source-local endpoint blocks; no graph-wide bijection",
        "magnn_canonical_catalog_hash": catalog.get("magnn_canonical_catalog_hash"),
    }
    write_json(output_dir / "relation_catalog.json", relation_catalog)

    model_catalog_payload = {
        "format_version": FORMAT,
        "center_type": center_type,
        "num_nodes": num_nodes,
        "num_relations": len(channel_records),
        "relations": [
            {
                "name": r["name"],
                "kind": r["kind"],
                "start_type": r["start_type"],
                "end_type": r["end_type"],
                "composition": r["composition"],
                "forward_channel_id": r["forward_channel_id"],
                "reverse_channel_id": r["reverse_channel_id"],
                "semantic_pair_sha256": r["semantic_pair_sha256"],
                "source_block_sha256": r["source_block_sha256"],
                "num_semantic_pairs": r["num_semantic_pairs"],
            }
            for r in relation_records
        ],
        "channels": [
            {
                "channel_id": c["channel_id"],
                "semantic_relation": c["semantic_relation"],
                "direction": c["direction"],
                "source_type": c["source_type"],
                "destination_type": c["destination_type"],
                "num_edges": c["num_edges"],
                "csr_sha256": c["csr_sha256"],
            }
            for c in channel_records
        ],
    }
    model_hash_payload = {
        "format_version": FORMAT,
        "catalog_sha256": sha256_json(model_catalog_payload),
        "node_types_sha256": sha256_array(node_types),
        "target_nodes_sha256": sha256_array(target_nodes),
        "labels_sha256": sha256_array(y),
        "split_sha256": sha256_json({k: sha256_array(v) for k, v in split.items()}),
        "channels": [
            {
                "channel_id": c["channel_id"],
                "semantic_relation": c["semantic_relation"],
                "direction": c["direction"],
                "num_edges": c["num_edges"],
                "csr_sha256": c["csr_sha256"],
            }
            for c in channel_records
        ],
    }
    # Input-variant/provenance fields are deliberately excluded from this hash.
    model_hash_payload["combined_model_input_hash"] = sha256_json(model_hash_payload)
    write_json(output_dir / "model_input_hashes.json", model_hash_payload)

    raw_provenance = {
        "format_version": FORMAT,
        "input_variant": args.input_variant,
        "data_dir": str(data_dir),
        "reference_dir": str(reference_dir),
        "raw_info_sha256": sha256_file(data_dir / "info.dat"),
        "raw_node_sha256": sha256_file(data_dir / "node.dat"),
        "raw_link_sha256": sha256_file(data_dir / "link.dat"),
        "raw_label_sha256": sha256_file(data_dir / "label.dat"),
        "raw_edge_count": raw_edge_count,
        "base_relation_fingerprints": base_hashes,
        "physical_shortcut_verification": physical_verification,
        "raw_graph_is_not_rewritten": True,
        "model_view_is_compiled_conditional_messages": True,
    }
    write_json(output_dir / "raw_graph_provenance.json", raw_provenance)

    overview = {
        "format_version": FORMAT,
        "input_variant": args.input_variant,
        "up_to_variant": args.up_to,
        "set_mode": args.set_mode,
        "scope": args.scope,
        "num_nodes": num_nodes,
        "num_target_nodes": int(target_nodes.size),
        "num_classes": int(y.max()) + 1,
        "num_semantic_relations": len(selected),
        "num_directed_channels": len(channel_records),
        "raw_edge_count": raw_edge_count,
        "semantic_edges_total": int(sum(c["num_edges"] for c in channel_records)),
        "train_size": int(split["train_idx"].size),
        "val_size": int(split["val_idx"].size),
        "test_size": int(split["test_idx"].size),
        "combined_model_input_hash": model_hash_payload["combined_model_input_hash"],
        "preprocessing_time_sec": time.perf_counter() - started,
    }
    write_json(output_dir / "preprocessing_overview.json", overview)
    print(json.dumps(overview, indent=2, sort_keys=True))
    return overview


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=here)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-variant", required=True)
    # Any catalog name that resolves to catalogs/<up-to>/<set-mode>/<scope>.json is
    # accepted; load_catalog() raises FileNotFoundError otherwise, so the hardcoded
    # choice list only blocked legitimate new catalogs. `physical_<variant>` catalogs
    # (build_physical_only_catalog.py) are the reason this was relaxed -- they give the
    # ORIGINALS and UNIVERSAL baselines a store the SAMPLED trainer can consume, which
    # the full-graph baseline runner cannot do for exact_3 / union_exact_2_3.
    parser.add_argument("--up-to", default="exact_2")
    parser.add_argument("--set-mode", choices=["union", "intersection"], default="union")
    parser.add_argument("--scope", choices=["center", "all_graph"], default="center")
    parser.add_argument("--split-seed", type=int, default=1566911444)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--verify-physical", choices=["none", "counts", "full"], default="counts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
