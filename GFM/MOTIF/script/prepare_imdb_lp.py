"""Build canonical IMDb variants and export structural MOTIF link-prediction inputs.

Reuses the lab's split/graph builders from Robust-GFM/GNN (lp_data.SOURCE_ROOT).
CPU preprocessing only; see docs/DATA_PREPARATION.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from lp_data import (ROOT, SOURCE_ROOT, CHECKSUMS, RAW_ROOT, PREPROCESSED_ROOT, BUILD_ROOT,
                     TARGET_NAMES, TASK_VARIANTS, array_hash,
                     audit_heldout, check_dependencies, file_hash, forward_edges,
                     validate_queries, variants_for, verify_raw_files, write_json)

SOURCE_FILES = (
    "models/RGCN/preprocess_IMDB_rgcn_structural.py",
    "dataset_construction/imdb/reconstruct_IMDB_md_ml_shared_splits.py",
    "dataset_construction/imdb/build_IMDB_my_shared_splits.py",
    "common/imdb_canonical_features.py", "common/inv_rgcn/data.py",
    "common/inv_rgcn/utils.py",
)


def build_bundles(args):
    if args.work_dir.exists():
        raise FileExistsError(f"Build directory exists: {args.work_dir}; choose a new --work-dir")
    raw_hashes = verify_raw_files(CHECKSUMS, args.csv.parent, "IMDB", (args.csv.name,),
                                 {args.csv.name: "movie_metadata.csv"})
    args.work_dir.mkdir(parents=True)
    splits = args.work_dir / "IMDB_shared_splits"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "1566911444",
           "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[1]),
                    "--csv", str(args.csv), "--out-dir", str(splits)], check=True, env=env)
    subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[2]),
                    "--csv", str(args.csv), "--inherit-from", str(splits / "IMDB_md_shared_splits.npz"),
                    "--out", str(splits / "IMDB_my_shared_splits.npz")], check=True, env=env)
    bundles = args.work_dir / "RGCN_structural_v10_year_canonfeat"
    variants = variants_for(TASK_VARIANTS["movie_year"], args.include_universal)
    subprocess.run([sys.executable, str(args.source_repo / SOURCE_FILES[0]),
                    "--raw-csv", str(args.csv), "--tasks", ",".join(args.tasks),
                    "--variants", ",".join(variants), "--feature-mode", "pooled_bow",
                    "--shared-movie-director", str(splits / "IMDB_md_shared_splits.npz"),
                    "--shared-movie-year", str(splits / "IMDB_my_shared_splits.npz"),
                    "--neg-k", "19", "--my-neg-policy", "freq_matched",
                    "--out-root", str(bundles)], check=True, env=env)
    return bundles / "IMDB", raw_hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-repo", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--csv", type=Path, default=RAW_ROOT / "IMDB/movie_metadata.csv")
    ap.add_argument("--work-dir", type=Path, default=BUILD_ROOT / "IMDB")
    ap.add_argument("--bundle-root", type=Path, help="Existing IMDB/ bundle directory; skips raw preprocessing")
    ap.add_argument("--output", type=Path, default=PREPROCESSED_ROOT / "IMDB")
    ap.add_argument("--tasks", nargs="+", choices=TASK_VARIANTS, default=list(TASK_VARIANTS))
    ap.add_argument("--include-universal", action="store_true", help="Also build/export the universal arm")
    args = ap.parse_args()
    for name in ("source_repo", "csv", "work_dir", "bundle_root", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output.exists():
        ap.error(f"Output already exists: {args.output}. Choose a new --output to avoid mixing exports.")
    source = args.source_repo.resolve()
    for name in SOURCE_FILES:
        if not (source / name).is_file():
            ap.error(f"Missing bundled source file: {source / name}")
    check_dependencies(ap, args.bundle_root is None)
    if args.bundle_root is None:
        args.bundle_root, raw_hashes = build_bundles(args)
    else:
        raw_hashes = {}
    for task in args.tasks:
        for variant in variants_for(TASK_VARIANTS[task], args.include_universal):
            path = args.bundle_root / task / variant / "graph.pt"
            if not path.is_file():
                ap.error(f"Missing {path}. See docs/DATA_PREPARATION.md.")

    sys.path.insert(0, str(source / "common"))
    from inv_rgcn.data import load_graph_data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="imdb-export-", dir=args.output.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        manifest = {"format": "motif_imdb_lp_v1", "raw_files": raw_hashes, "source_code": {
            name: file_hash(source / name) for name in SOURCE_FILES},
            "adapter_code": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/lp_data.py")},
            "tasks": {}}
        for task in args.tasks:
            out = stage / task
            out.mkdir()
            reference, records = None, {}
            for variant in variants_for(TASK_VARIANTS[task], args.include_universal):
                path = args.bundle_root / task / variant / "graph.pt"
                data = load_graph_data(path)
                if data.x is None or data.metadata.get("feature_mode") != "pooled_bow":
                    raise ValueError("Expected canonical pooled_bow features; regenerate the source bundles")
                if data.metadata.get("task") != task or data.variant != variant:
                    raise ValueError(f"Wrong task/variant in {path}")
                if not data.metadata.get("lp_target_type_pooled_train_only"):
                    raise ValueError("Missing train-only target feature provenance")
                if task == "movie_year" and data.metadata.get("movie_year_neg_policy") != "freq_matched":
                    raise ValueError("Expected frequency-matched Movie-Year negatives")
                queries = {k: data.task_data[k].cpu().numpy() for k in (
                    "train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
                types = data.node_type.cpu().numpy()
                validate_queries(queries, types, 4 if task == "movie_year" else 1)
                counts = {s: len(queries[f"{s}_pos"]) for s in ("train", "val", "test")}
                expected = {"train": 2925, "val": 418, "test": 834} if task == "movie_year" else {"train": 2926, "val": 418, "test": 836}
                if counts != expected or data.num_nodes != 15789:
                    raise ValueError(f"Noncanonical IMDb counts: {counts}, nodes={data.num_nodes}")
                current = (data.feature_hash, data.supervision_hash, data.semantic_hash,
                           array_hash(types), data.num_nodes)
                if reference is None:
                    reference = current
                    np.savez_compressed(out / "queries.npz", node_type=types, **queries)
                elif current != reference:
                    raise ValueError(f"Features, supervision, node ids, or compiled program differ: {task}/{variant}")
                arrays, effective = {}, {}
                names = None
                for view, prefix in (("physical", "physical"), ("compiled", "semantic")):
                    edges, vocab = forward_edges(
                        getattr(data, f"{prefix}_edge_index").cpu().numpy(),
                        getattr(data, f"{prefix}_edge_type").cpu().numpy(),
                        getattr(data, f"{prefix}_relation_names"))
                    if names is not None and vocab != names:
                        raise ValueError("Physical/compiled relation vocabularies differ")
                    names = vocab
                    offsets = data.metadata["offsets"]
                    audit_heldout(edges, vocab, queries, task, offsets["movie"], offsets["link"])
                    arrays[view] = edges
                    effective[view] = array_hash(edges)
                if TARGET_NAMES[task] not in names:
                    raise ValueError("Task relation missing from MOTIF vocabulary")
                np.savez_compressed(out / f"{variant}.npz", **arrays)
                records[variant] = {
                    "source_bundle_sha256": file_hash(path),
                    "source_physical_hash": data.physical_hash,
                    "source_semantic_hash": data.semantic_hash,
                    "edges_sha256": effective,
                    "edges": {k: len(v) for k, v in arrays.items()},
                }
                print(f"Exported {task}/{variant}", flush=True)
            if len({r["edges_sha256"]["compiled"] for r in records.values()}) != 1:
                raise ValueError(f"Compiled graphs differ for {task}")
            task_meta = {
                "num_nodes": reference[4], "relation_names": names,
                "target_relation": names.index(TARGET_NAMES[task]),
                "feature_hash": reference[0], "supervision_hash": reference[1],
                "source_semantic_hash": reference[2], "variants": records,
                "protocol": "tail_only_positive_plus_19_fixed_negatives_average_ties",
                "message_semantics": "unit_weight_sets_with_internal_inverse_edges",
                "feature_policy": "none_structure_only",
                "source_feature_policy": "canonical_pooled_bow_target_type_training_pairs_only_not_exported",
                "negative_policy": "freq_matched" if task == "movie_year" else "uniform",
                "split_counts": {s: len(queries[f"{s}_pos"]) for s in ("train", "val", "test")},
            }
            task_meta["files"] = {p.name: file_hash(p) for p in out.glob("*.npz")}
            write_json(out / "manifest.json", task_meta)
            manifest["tasks"][task] = file_hash(out / "manifest.json")
        write_json(stage / "manifest.json", manifest)
        shutil.move(str(stage), str(args.output))
    print(f"Portable MOTIF inputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
