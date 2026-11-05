"""Build canonical DBLP variants and export structural MOTIF link-prediction inputs.

Default: reuse INV-GNN's split builder/verifier and structural preprocessor.
With --bundle-root, export existing Paper-Conference bundles instead.
No training or model evaluation is performed by this command.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from lp_data import (ROOT, SOURCE_ROOT, CHECKSUMS, RAW_ROOT, PREPROCESSED_ROOT, BUILD_ROOT,
                     DBLP_TASK_VARIANTS, array_hash, audit_dblp_heldout,
                     check_dependencies, file_hash, forward_edges, validate_full_target_queries,
                     variants_for, verify_raw_files, write_json)

RAW_FILES = ("author_label.txt", "paper_author.txt", "paper_conf.txt", "paper_term.txt")
SOURCE_FILES = (
    "dataset_construction/dblp/paper-venue_shared_splits.py",
    "dataset_construction/dblp/verify_dblp_canonical_splits.py",
    "models/RGCN/preprocess_DBLP_rgcn_structural.py", "common/inv_rgcn/data.py",
    "common/inv_rgcn/utils.py",
)


def build_bundles(args):
    if args.work_dir.exists():
        raise FileExistsError(f"Build directory exists: {args.work_dir}; use a new --work-dir or --bundle-root")
    for name in RAW_FILES:
        if not (args.raw_dir / name).is_file():
            raise FileNotFoundError(f"Missing raw DBLP file: {args.raw_dir / name}")
    # Confirm this is the documented raw dataset before choosing canonical splits.
    raw_hashes = verify_raw_files(CHECKSUMS, args.raw_dir, "DBLP", RAW_FILES)
    args.work_dir.mkdir(parents=True)
    split_dir = args.work_dir / "DBLP_shared_splits_CANONICAL"
    split_path = split_dir / "DBLP_pc_shared_splits.npz"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "1566911444",
           "CUDA_VISIBLE_DEVICES": ""}
    # The original split script only accepts an environment data root containing
    # raw/DBLP. Stage four small files to preserve its exact code without symlinks
    # or modifications to the ignored source checkout.
    with tempfile.TemporaryDirectory(prefix="dblp-raw-", dir=args.work_dir) as temp:
        raw = Path(temp) / "raw/DBLP"
        raw.mkdir(parents=True)
        for name in RAW_FILES:
            shutil.copy2(args.raw_dir / name, raw / name)
        subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[0])], check=True,
                       env={**env, "INV_GNN_DATA": str(Path(temp).resolve()),
                            "DBLP_SPLITS_OUT": str(split_dir.resolve())})
    subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[1]),
                    "--raw", str(args.raw_dir), "--canonical", str(split_path)], check=True, env=env)
    bundles = args.work_dir / "RGCN_structural_v8_dblp"
    subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[2]),
                    "--raw-dir", str(args.raw_dir), "--shared-npz", str(split_path),
                    "--mapping-scope", "train", "--variants",
                    ",".join(variants_for(DBLP_TASK_VARIANTS["paper_conference"], args.include_universal)),
                    "--out-root", str(bundles)], check=True, env=env)
    return bundles / "DBLP/paper_conference", raw_hashes


def export_bundles(args, bundle_root, raw_hashes):
    sys.path.insert(0, str(args.source_repo / "common"))
    from inv_rgcn.data import load_graph_data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dblp-export-", dir=args.output.parent) as temp:
        stage = Path(temp) / "export"
        task = "paper_conference"
        out = stage / task
        out.mkdir(parents=True)
        records, reference = {}, None
        for variant in variants_for(DBLP_TASK_VARIANTS[task], args.include_universal):
            path = bundle_root / variant / "graph.pt"
            data = load_graph_data(path)
            meta = data.metadata
            if meta.get("dataset") != "DBLP" or meta.get("task") != task or data.variant != variant:
                raise ValueError(f"Incorrect dataset/task/variant in {path}")
            if (meta.get("mapping_scope") != "train" or meta.get("paper_conference_edges") != "train_only"
                    or not meta.get("shared_npz_sha256") or meta.get("pair_filter_model_input") is not False):
                raise ValueError("Require canonical shared splits and training-only, context-only mapping provenance")
            if data.x is not None:
                raise ValueError("This DBLP protocol expects source bundles without supplied features")
            types = data.node_type.cpu().numpy()
            pool = np.flatnonzero(types == 3).astype(np.int64)
            source_queries = {k: data.task_data[k].cpu().numpy() for k in (
                "train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
            queries = validate_full_target_queries(source_queries, types, pool)
            counts = {s: len(queries[f"{s}_pos"]) for s in ("train", "val", "test")}
            if counts != {"train": 9448, "val": 1348, "test": 2666} or data.num_nodes != 26076:
                raise ValueError(f"Unexpected canonical post-filter DBLP split counts: {counts}")
            current = (data.supervision_hash, data.semantic_hash, array_hash(types),
                       data.num_nodes, meta["shared_npz_sha256"])
            if reference is None:
                reference = current
                # Store the full conference pool, not a sampled or reshuffled set.
                np.savez_compressed(out / "queries.npz", node_type=types, candidate_pool=pool,
                                    **{f"{s}_pos": queries[f"{s}_pos"] for s in ("train", "val", "test")})
            elif current != reference:
                raise ValueError("Source supervision, node ids, canonical split, or compiled program differs across DBLP variants")
            arrays, effective, names = {}, {}, None
            for view, prefix in (("physical", "physical"), ("compiled", "semantic")):
                edges, vocab = forward_edges(getattr(data, f"{prefix}_edge_index").cpu().numpy(),
                                             getattr(data, f"{prefix}_edge_type").cpu().numpy(),
                                             getattr(data, f"{prefix}_relation_names"))
                if names is not None and vocab != names:
                    raise ValueError("Physical/compiled relation vocabularies differ")
                names = vocab
                audit_dblp_heldout(edges, vocab, queries)
                arrays[view], effective[view] = edges, array_hash(edges)
            np.savez_compressed(out / f"{variant}.npz", **arrays)
            records[variant] = {"source_bundle_sha256": file_hash(path),
                                "source_physical_hash": data.physical_hash,
                                "source_semantic_hash": data.semantic_hash,
                                "edges_sha256": effective,
                                "edges": {view: len(edges) for view, edges in arrays.items()}}
            print(f"Exported DBLP/{variant}", flush=True)
        if len({r["edges_sha256"]["compiled"] for r in records.values()}) != 1:
            raise ValueError("Compiled graphs differ across DBLP variants")
        task_meta = {
            "num_nodes": reference[3], "relation_names": names,
            "target_relation": names.index("paper_to_conference"), "variants": records,
            "supervision_hash": reference[0], "source_semantic_hash": reference[1],
            "source_shared_npz_sha256": reference[4], "split_counts": counts,
            "feature_policy": "none_structure_only",
            "protocol": "tail_only_all_20_conferences_average_ties",
            "message_semantics": "unit_weight_sets_with_internal_inverse_edges",
            "mapping_scope": "train",
            "files": {p.name: file_hash(p) for p in out.glob("*.npz")},
        }
        write_json(out / "manifest.json", task_meta)
        write_json(stage / "manifest.json", {
            "format": "motif_dblp_lp_v1", "tasks": {task: file_hash(out / "manifest.json")},
            "source_code": {name: file_hash(args.source_repo / name) for name in SOURCE_FILES},
            "adapter_code": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/lp_data.py")},
            "raw_files": raw_hashes,
        })
        shutil.move(str(stage), str(args.output))
    print(f"Portable MOTIF inputs: {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-repo", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--raw-dir", type=Path, default=RAW_ROOT / "DBLP")
    ap.add_argument("--work-dir", type=Path, default=BUILD_ROOT / "DBLP")
    ap.add_argument("--bundle-root", type=Path, help="Existing paper_conference/ directory; skips raw preprocessing")
    ap.add_argument("--output", type=Path, default=PREPROCESSED_ROOT / "DBLP")
    ap.add_argument("--include-universal", action="store_true", help="Also build/export the universal arm")
    args = ap.parse_args()
    for name in ("source_repo", "raw_dir", "work_dir", "output", "bundle_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output.exists():
        ap.error(f"Output exists: {args.output}. Choose a new --output.")
    for name in SOURCE_FILES:
        if not (args.source_repo / name).is_file():
            ap.error(f"Missing INV-GNN source file: {args.source_repo / name}")
    check_dependencies(ap, args.bundle_root is None)
    if args.bundle_root is None:
        if args.work_dir.exists():
            ap.error(f"Build directory exists: {args.work_dir}; use a new --work-dir or --bundle-root")
        for name in RAW_FILES:
            if not (args.raw_dir / name).is_file():
                ap.error(f"Missing raw DBLP file: {args.raw_dir / name}")
        bundle_root, raw_hashes = build_bundles(args)
    else:
        bundle_root, raw_hashes = args.bundle_root, {}
    for variant in variants_for(DBLP_TASK_VARIANTS["paper_conference"], args.include_universal):
        if not (bundle_root / variant / "graph.pt").is_file():
            ap.error(f"Missing {bundle_root / variant / 'graph.pt'}")
    export_bundles(args, bundle_root, raw_hashes)


if __name__ == "__main__":
    main()
