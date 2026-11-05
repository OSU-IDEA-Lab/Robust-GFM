"""Report matched-variant adaptation and cross-representation transfer separately."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from export_overleaf import CLASS_COLUMNS, LEGACY_MATRIX_TABLES, LP_COLUMNS, matrix_tables
from lp_data import TASK_VARIANTS, DBLP_TASK_VARIANTS, write_json
from lp_metrics import check_alignment, compare_predictions
from report_lp import build_report, cell, csv_row, write_csv
from run_finetune_matrix import completed_cell, completed_training


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_matrix_report(root, bootstrap=1000, seed=0):
    manifest = read_json(root / "experiment.json")
    if manifest["format"] != "motif_finetune_matrix_experiment_v1":
        raise ValueError("Not a fine-tuning matrix experiment")
    contract = manifest["compute_contract"]
    dataset, task, variants = contract["dataset"], contract["task"], contract["variants"]
    families = {"imdb": TASK_VARIANTS, "dblp": DBLP_TASK_VARIANTS}
    if dataset not in families or variants != list(families[dataset][task]):
        raise ValueError("Matrix must contain the full physical variant family")
    cells, within_pairs, arrays, training_records = {}, {}, {}, {}
    shared_training = shared_evaluation = reference_arrays = None
    initial_parameters = schedule_hash = None
    for train in variants:
        training_dir = (root.parent / f"motif_finetune_{dataset}_lp_{manifest['run_tag']}" / task /
                        f"train_{train}" / f"seed{contract['seed']}")
        recorded_training = completed_training(training_dir, train, contract)
        if recorded_training != manifest["training"].get(train):
            raise ValueError(f"Training files differ from the matrix manifest: {training_dir}")
        training = read_json(training_dir / "training_summary.json")
        normalized = {key: value for key, value in training["training_contract"].items()
                      if key not in ("train_variant", "train_graph_sha256", "area_support_sha256")}
        if shared_training is None:
            shared_training = normalized
            initial_parameters = training["initial_parameters_sha256"]
            schedule_hash = training["schedule_sha256"]
        if (normalized != shared_training or training["initial_parameters_sha256"] != initial_parameters
                or training["schedule_sha256"] != schedule_hash):
            raise ValueError("Training runs differ in initialization, targets, order, budget, code or environment")
        training_records[train] = {
            **recorded_training, "selected_step": training["best_step"],
            "validation_mrr": training["best_validation_mrr"],
            "completed_steps": training["completed_steps"], "schedule_sha256": schedule_hash,
            "total_seconds": training["total_seconds"], "peak_gpu_bytes": training["peak_gpu_bytes"],
            "total_removed_training_edges": training["total_removed_training_edges"],
            **{key: training[key] for key in ("total_removed_target_edges", "total_removed_context_edges")
               if key in training},
        }
        evaluation_root = root.parent / manifest["evaluation_roots"][train]
        arrays[train] = {}
        for test in variants:
            directory = evaluation_root / task / "physical" / test
            recorded_cell = completed_cell(directory, train, test, recorded_training["checkpoint_sha256"], contract)
            if recorded_cell != manifest["evaluations"].get(train, {}).get(test):
                raise ValueError(f"Evaluation files differ from the matrix manifest: {directory}")
            meta = read_json(directory / "metrics.json")
            evaluated = meta["comparison_contract"]
            fine_tuning = evaluated["fine_tuning"]
            if (fine_tuning["training_contract"] != training["training_contract"]
                    or fine_tuning["selected_step"] != training["best_step"]
                    or fine_tuning["validation_mrr"] != training["best_validation_mrr"]
                    or fine_tuning["initial_parameters_sha256"] != initial_parameters
                    or not (meta["parameters_before"] == meta["parameters_after"] == fine_tuning["parameters_sha256"])):
                raise ValueError(f"Evaluation does not use the selected frozen training checkpoint: {directory}")
            normalized = {key: value for key, value in evaluated.items()
                          if key not in ("checkpoint_sha256", "fine_tuning")}
            if shared_evaluation is None:
                shared_evaluation = normalized
            if normalized != shared_evaluation:
                raise ValueError("Test queries, candidates, scoring settings, code or environment differ across matrix cells")
            with np.load(directory / "scores.npz") as source:
                values = {key: source[key].copy() for key in source.files}
            if dataset == "dblp" and (len(values["candidate_pool"]) != 20
                    or values["scores"].shape[1] != 20 or values["full_type_scores"].shape[1] != 20
                    or not np.array_equal(np.sort(values["candidates"], axis=1),
                                          np.broadcast_to(values["candidate_pool"], values["candidates"].shape))):
                raise ValueError("DBLP prediction and ranking metrics must use the same 20 conferences")
            if reference_arrays is None:
                reference_arrays = values
            else:
                check_alignment(reference_arrays, values)
            arrays[train][test] = values
        # Keep the original reporter's same-checkpoint requirement within each row.
        # Only this matrix reporter compares different trained models explicitly.
        row = build_report(evaluation_root, bootstrap, seed)[task]
        if row["missing_runs"] or set(row["performance"]) != {f"physical/{test}" for test in variants}:
            raise ValueError(f"Incomplete or unexpected evaluations: {evaluation_root}")
        cells[train] = {test: row["performance"][f"physical/{test}"] for test in variants}
        within_pairs[train] = row["physical_pairs"]
    diagonal_pairs = {
        f"{first} vs {second}": compare_predictions(arrays[first][first], arrays[second][second], bootstrap, seed)
        for first, second in combinations(variants, 2)
    }
    summary = {
        "format": "motif_finetune_matrix_v1", "evaluation_stage": "finetuned_matrix",
        "report_status": "preliminary", "dataset": dataset, "task": task, "variants": variants,
        "seed": contract["seed"], "split": "test", "bootstrap": bootstrap, "bootstrap_seed": seed,
        "uncertainty": "query_bootstrap_standard_error", "cells": cells,
        "diagonal_pairs": diagonal_pairs, "within_checkpoint_pairs": within_pairs,
        "shared_training_contract": shared_training, "shared_evaluation_contract": shared_evaluation,
        "training": training_records,
    }
    output = root / "tables"
    output.mkdir(exist_ok=True)
    write_json(output / "summary.json", summary)
    performance = []
    for train in variants:
        for test in variants:
            result = cells[train][test]
            performance.append(csv_row({
                "model": "MOTIF (FT)", "dataset": dataset, "task": task, "seed": contract["seed"],
                "train_variant": train, "test_variant": test,
                "regime": "C-a" if train == test else "C-b", "queries": result["num_queries"],
            }, {key: result["metrics"][key] for key, _ in LP_COLUMNS + (CLASS_COLUMNS if dataset == "dblp" else ())}))
    comparisons = []
    for regime, train, pairs in [("C-a", "independent", diagonal_pairs)] + [
            ("C-b", train, within_pairs[train]) for train in variants]:
        for pair, result in pairs.items():
            comparisons.append(csv_row({
                "model": "MOTIF (FT)", "dataset": dataset, "task": task, "seed": contract["seed"],
                "regime": regime, "train_variant": train, "comparison": pair,
                "queries": result["num_queries"], "undefined_tau_queries": result["undefined_tau_queries"],
            }, {key: value for key, value in result["metrics"].items() if key != "full_type_prediction_agreement"}))
    write_csv(output / "performance.csv", performance)
    write_csv(output / "comparisons.csv", comparisons)
    for name in LEGACY_MATRIX_TABLES:
        (output / name).unlink(missing_ok=True)
    for name, content in matrix_tables(summary, include_lp=True).items():
        (output / f"{name}.tex").write_text(content, encoding="utf-8")
    notes = [
        "# Preliminary MOTIF fine-tuning matrix", "",
        f"Task: {dataset.upper()} {task}; training seed {contract['seed']}; {contract['steps']} optimizer steps per variant.",
        "Each variant starts independently from the same released checkpoint; validation MRR selects its checkpoint.",
        "Diagonal cells are matched-variant adaptation (C-a); off-diagonal cells are cross-representation transfer (C-b).",
        (f"Values are estimates ± query-bootstrap standard error ({bootstrap} resamples), not training-seed variation."
         if bootstrap else "Values are point estimates; query-bootstrap uncertainty was not computed."),
        "All reported metrics use the original 20 candidates: the positive target plus 19 fixed negatives.",
        "The guide's training-repeat noise floor and multi-seed mean/SD remain pending; this is a single-seed pilot.", "",
    ]
    for metric, label in (("mrr", "MRR (20 candidates)"), ("hits@1", "Hits@1 (20 candidates)")):
        notes.extend([f"## {label}", "", "| Train / Test | " + " | ".join(variants) + " |",
                      "|---|" + "---:|" * len(variants)])
        notes.extend("| " + train + " | " + " | ".join(cell(cells[train][test]["metrics"][metric])
                                                       for test in variants) + " |" for train in variants)
        notes.append("")
    table_notes = (["`conference_prediction.tex` reports macro Precision/Recall/F1 for all nine cells; Accuracy and Micro-F1 equal Hits@1.",
                    "`link_prediction.tex` reports MRR and Hits for the same nine cells."] if dataset == "dblp" else
                   ["`transfer_performance.tex` contains the MRR/Hits@1/3/10 transfer matrix for all train/test cells."])
    notes.extend(table_notes + [
        "`performance.csv` contains all train/test cells and all performance metrics.",
        "`invariance.tex` combines C-a comparisons between separately trained models and C-b fixed-checkpoint comparisons, labelled by regime.",
        "All columns use the same 20 candidates per query; Delta MRR is the second test variant minus the first.",
        "`comparisons.csv` and `summary.json` retain the comparison statistics and confidence intervals.",
        "Package these saved tables with `script/export_overleaf.py`; no PDF is compiled by this reporter.", "",
    ])
    if dataset == "dblp":
        notes.extend(["DBLP exports three tables: `conference_prediction.tex` (macro Precision/Recall/F1),",
                      "`link_prediction.tex` (MRR and Hits), and `invariance.tex` (C-a/C-b labelled comparisons).",
                      "All three use the same 20-conference pool; no extra classification task or head is trained.", ""])
    (output / "report.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Wrote matrix tables: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.bootstrap == 1:
        parser.error("Bootstrap must be zero or at least two")
    build_matrix_report(args.matrix.resolve(), args.bootstrap, args.bootstrap_seed)


if __name__ == "__main__":
    main()
