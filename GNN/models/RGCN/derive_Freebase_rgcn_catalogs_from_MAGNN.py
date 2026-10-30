#!/usr/bin/env python3
"""Derive RGCN relation-family catalogs from the supplied Freebase MAGNN package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freebase_block_common import sha256_json, write_json


def read_registries(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    source: dict[str, str] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = row["new_relation_name"]
            rows[name] = row
            source[name] = path.stem
    return rows, source


def derive(magnn_root: Path, output_root: Path) -> dict[str, Any]:
    registry, registry_source = read_registries(magnn_root / "registries")
    generated = magnn_root / "generated_preprocess" / "unchanged"
    records = []
    for up_to in ("exact_2", "exact_3", "range_2_3"):
        for set_mode in ("union", "intersection"):
            for scope in ("center", "all_graph"):
                source_path = generated / up_to / set_mode / scope / "canonical_metapath_manifest.json"
                manifest = json.loads(source_path.read_text(encoding="utf-8"))
                relations: dict[str, dict[str, Any]] = {}
                metapaths = []
                for index, metapath in enumerate(manifest["metapath_defs"]):
                    types = [int(x) for x in metapath["semantic_node_metapath"]]
                    signature = list(metapath["semantic_signature"])
                    metapaths.append(
                        {
                            "index": index,
                            "name": metapath.get("name")
                            or metapath.get("safe_name")
                            or f"M{index:03d}",
                            "semantic_node_metapath": types,
                            "semantic_signature": signature,
                            "dependency_signature": metapath.get("dependency_signature", []),
                            "source_variants": metapath.get("source_variants", []),
                        }
                    )
                    for step, token in enumerate(signature):
                        sign_text, name = token.split(":", 1)
                        sign = int(sign_text)
                        a, b = types[step], types[step + 1]
                        start, end = (a, b) if sign == 1 else (b, a)
                        mapped = name in registry
                        row = registry.get(name, {})
                        composition = (
                            [x.strip() for x in str(row["composition"]).split(" ; ") if x.strip()]
                            if mapped
                            else [name]
                        )
                        relation = relations.setdefault(
                            name,
                            {
                                "name": name,
                                "start_type": start,
                                "end_type": end,
                                "kind": "mapped" if mapped else "base",
                                "composition": composition,
                                "hop_count": int(row.get("hop_count", 1)),
                                "registry_variant": registry_source.get(name),
                                "expected_unique_endpoint_pairs": row.get("unique_endpoint_pairs"),
                                "source_variants": [],
                                "used_by_metapaths": [],
                            },
                        )
                        if (relation["start_type"], relation["end_type"]) != (start, end):
                            raise ValueError(f"Inconsistent schema for {name}")
                        for variant in metapath.get("source_variants", []):
                            if variant not in relation["source_variants"]:
                                relation["source_variants"].append(variant)
                        if index not in relation["used_by_metapaths"]:
                            relation["used_by_metapaths"].append(index)
                payload = {
                    "format_version": "freebase_rgcn_block_catalog_v1",
                    "derived_from": magnn_root.name,
                    "source_manifest": str(source_path.relative_to(magnn_root)),
                    "up_to_variant": up_to,
                    "set_mode": set_mode,
                    "scope": scope,
                    "center_type": int(manifest.get("center_type", 0)),
                    "semantic_relations": sorted(relations.values(), key=lambda x: x["name"]),
                    "source_metapaths": metapaths,
                    "magnn_canonical_catalog_hash": manifest.get("canonical_catalog_hash"),
                }
                destination = output_root / up_to / set_mode / f"{scope}.json"
                write_json(destination, payload)
                records.append(
                    {
                        "path": str(destination.relative_to(output_root)),
                        "up_to_variant": up_to,
                        "set_mode": set_mode,
                        "scope": scope,
                        "num_semantic_relations": len(relations),
                        "num_mapped_relations": sum(r["kind"] == "mapped" for r in relations.values()),
                        "num_base_relations": sum(r["kind"] == "base" for r in relations.values()),
                        "catalog_sha256": sha256_json(payload),
                    }
                )
    summary = {
        "format_version": "freebase_rgcn_catalog_derivation_summary_v1",
        "magnn_package": magnn_root.name,
        "num_catalogs": len(records),
        "catalogs": records,
    }
    write_json(output_root / "CATALOG_DERIVATION_SUMMARY.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnn-package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "catalogs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(derive(args.magnn_package_root.resolve(), args.output_root.resolve()), indent=2))
