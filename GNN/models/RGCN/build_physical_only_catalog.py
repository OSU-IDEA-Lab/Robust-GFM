#!/usr/bin/env python3
"""Emit a *physical-only* block-conditional catalog for one Freebase variant.

WHY THIS EXISTS
---------------
The invariant Freebase runner scales to the large variants because it is
mini-batch with neighbour sampling over compiled CSR channel stores. The
baseline runner (``rgcn_data_augmentation/run_FREEBASE_rgcn_augmentation.py``)
is FULL-GRAPH, so it cannot produce originals or the universal-graph
(``{model}*``) point for ``exact_3`` (447M forward edges) or
``union_exact_2_3`` (493M): one RGCNConv layer over ~900M directed edges needs
roughly 229 GB just to materialise its messages (894M x 64 floats x 4 B), which
no available card holds.

A *physical-only* catalog closes that gap without writing a second trainer. It
lists every relation that is PHYSICALLY PRESENT in the variant's own
``info.dat`` as ``kind: "base"``, and adds no composed/semantic relations. The
block-conditional compiler then produces a channel store that is simply the
variant's raw relational graph, and the existing sampled trainer runs a STOCK
RGCN over it. Same trainer, same sampling, same splits as the invariant arm --
the only difference is that no conditional/semantic relations exist, which is
exactly what "original variant" and "universal graph" mean.

IMPORTANT: pass ``--reference-dir`` = the variant itself when compiling with one
of these catalogs. The compiler validates that every base relation name appears
in the reference ``info.dat`` (``missing = needed_base - set(ref_by_name)``);
``unchanged`` only knows the 36 original relations, so it would reject the
materialised shortcut relations that ``exact_2``/``exact_3``/``union_exact_2_3``
carry. Their ``node.dat``/``label.dat`` are hardlink-identical to ``unchanged``
(verified when the union was built), so the reference's other equality checks
still hold.

Usage
-----
    python build_physical_only_catalog.py \\
        --variant-dir /path/to/dataset_variant_3hops_filtered/union_exact_2_3 \\
        --name physical_union_exact_2_3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freebase_block_common import parse_info

CENTER_TYPE = 0  # BOOK -- the node classification target type


def build(variant_dir: Path, package_root: Path, name: str, set_mode: str, scope: str) -> Path:
    node_type_names, relations = parse_info(variant_dir / "info.dat")

    semantic_relations = []
    for rid in sorted(relations):
        spec = relations[rid]
        semantic_relations.append(
            {
                "name": spec.name,
                "kind": "base",
                # A physical relation composes to itself.
                "composition": [spec.name],
                "hop_count": 1,
                "start_type": int(spec.start),
                "end_type": int(spec.end),
                # None disables the compiler's physical-count cross-check, which
                # only makes sense for COMPOSED shortcuts (it compares a derived
                # endpoint set against the registry). Here the relation IS the
                # physical edge set, so there is nothing to cross-check.
                "expected_unique_endpoint_pairs": None,
                "registry_variant": None,
                "source_variants": [variant_dir.name],
                "used_by_metapaths": [],
            }
        )

    payload = {
        "format_version": "freebase_rgcn_block_catalog_v1",
        "center_type": CENTER_TYPE,
        "scope": scope,
        "set_mode": set_mode,
        "up_to_variant": name,
        "derived_from": "physical relations of the variant's own info.dat",
        "physical_only": True,
        "source_variant_dir": str(variant_dir),
        "num_physical_relations": len(semantic_relations),
        "node_type_names": {str(k): v for k, v in sorted(node_type_names.items())},
    }
    digest = hashlib.sha256(
        json.dumps(semantic_relations, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload["physical_catalog_sha256"] = digest
    payload["semantic_relations"] = semantic_relations

    out = package_root / "catalogs" / name / set_mode / f"{scope}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[ok] {variant_dir.name}: {len(semantic_relations)} physical relations "
        f"-> {out}  (sha256 {digest[:16]}...)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--name", required=True, help="Catalog name, used as --up-to")
    ap.add_argument("--package-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--set-mode", default="union")
    ap.add_argument("--scope", default="all_graph")
    args = ap.parse_args()
    build(
        Path(args.variant_dir).resolve(),
        Path(args.package_root).resolve(),
        args.name,
        args.set_mode,
        args.scope,
    )


if __name__ == "__main__":
    main()
