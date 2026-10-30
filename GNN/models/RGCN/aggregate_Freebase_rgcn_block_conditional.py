#!/usr/bin/env python3
"""Aggregate Freebase block-conditional RGCN runs and invariance metrics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau

from freebase_block_common import write_json

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "mrr",
    "hit_at_1",
    "hit_at_3",
    "loss",
)
RESOURCES = (
    "train_time_sec",
    "inference_time_sec",
    "epochs_run",
    "best_epoch",
    "parameter_mib",
    "peak_training_gpu_mib",
    "peak_inference_gpu_mib",
    "best_checkpoint_mib",
)


def mean_std(values: Sequence[float | None]) -> dict[str, Any]:
    numeric = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    if numeric.size == 0:
        return {"mean": None, "std": None, "n": 0, "ddof": 1}
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
        "n": int(numeric.size),
        "ddof": 1,
    }


def stable_tau(a: np.ndarray, b: np.ndarray) -> float:
    value = kendalltau(a, b, variant="b", nan_policy="omit").correlation
    if value is None or np.isnan(value):
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(value)


def kendall_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Logit shape mismatch: {a.shape}, {b.shape}")
    all_tau: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    for row_a, row_b in zip(a, b):
        all_tau.append(stable_tau(row_a, row_b))
        top1.append(float(np.argmax(row_a) == np.argmax(row_b)))
        ia = np.argsort(-row_a, kind="stable")[:3]
        ib = np.argsort(-row_b, kind="stable")[:3]
        union = np.unique(np.concatenate([ia, ib]))
        top3.append(1.0 if union.size <= 1 else stable_tau(row_a[union], row_b[union]))
    diff = np.abs(a - b)
    return {
        "kendall_tau": float(np.mean(all_tau)),
        "kendall_tau_at_1": float(np.mean(top1)),
        "kendall_tau_at_3": float(np.mean(top3)),
        "prediction_agreement": float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1))),
        "max_abs_logit_difference": float(diff.max()) if diff.size else 0.0,
        "mean_abs_logit_difference": float(diff.mean()) if diff.size else 0.0,
        "exact_logits": bool(np.array_equal(a, b)),
    }


def result_path(root: Path, up_to: str, mode: str, scope: str, variant: str, seed: int) -> Path:
    return root / up_to / mode / scope / variant / f"seed_{seed}" / "result.json"


def load_run(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    logits_path = Path(result["artifacts"]["logits"])
    if not logits_path.is_absolute():
        logits_path = (path.parent / logits_path).resolve()
    arrays = np.load(logits_path)
    return {"result": result, "arrays": {k: arrays[k] for k in arrays.files}, "path": path}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.results_root).resolve()
    variants = args.variants
    seeds = args.seeds
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for variant in variants:
        for seed in seeds:
            path = result_path(root, args.up_to, args.set_mode, args.scope, variant, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            grouped[variant][seed] = load_run(path)

    reference_variant = variants[0]
    alignment: list[dict[str, Any]] = []
    for seed in seeds:
        ref = grouped[reference_variant][seed]
        ref_result, ref_arrays = ref["result"], ref["arrays"]
        for variant in variants:
            cur = grouped[variant][seed]
            checks = {
                "seed": seed,
                "reference_variant": reference_variant,
                "variant": variant,
                "model_input_hash_equal": cur["result"]["combined_model_input_hash"]
                == ref_result["combined_model_input_hash"],
                "test_idx_equal": bool(np.array_equal(cur["arrays"]["test_idx"], ref_arrays["test_idx"])),
                "test_labels_equal": bool(
                    np.array_equal(cur["arrays"]["test_labels"], ref_arrays["test_labels"])
                ),
                "test_global_ids_equal": bool(
                    np.array_equal(cur["arrays"]["test_global_ids"], ref_arrays["test_global_ids"])
                ),
            }
            if not all(value for key, value in checks.items() if key not in {"seed", "reference_variant", "variant"}):
                raise AssertionError(f"Alignment failure: {checks}")
            alignment.append(checks)

    per_run: list[dict[str, Any]] = []
    variant_stats: dict[str, Any] = {}
    for variant in variants:
        rows = []
        for seed in seeds:
            result = grouped[variant][seed]["result"]
            test = result["metrics"]["test"]
            memory = result["memory"]
            row = {
                "variant": variant,
                "seed": seed,
                **{metric: test.get(metric) for metric in METRICS},
                "train_time_sec": result.get("train_time_sec"),
                "inference_time_sec": result.get("inference_time_sec"),
                "epochs_run": result.get("epochs_run"),
                "best_epoch": result.get("best_epoch"),
                "parameter_mib": memory.get("parameter_mib"),
                "peak_training_gpu_mib": memory.get("peak_training_gpu_mib"),
                "peak_inference_gpu_mib": memory.get("peak_inference_gpu_mib"),
                "best_checkpoint_mib": memory.get("best_checkpoint_mib"),
            }
            rows.append(row)
            per_run.append(row)
        variant_stats[variant] = {
            "metrics": {metric: mean_std([row[metric] for row in rows]) for metric in METRICS},
            "resources": {key: mean_std([row[key] for row in rows]) for key in RESOURCES},
        }

    pairwise_runs: list[dict[str, Any]] = []
    pairwise_stats: dict[str, Any] = {}
    for left, right in itertools.combinations(variants, 2):
        rows = []
        for seed in seeds:
            a = grouped[left][seed]["arrays"]["test_logits"].astype(np.float64, copy=False)
            b = grouped[right][seed]["arrays"]["test_logits"].astype(np.float64, copy=False)
            row = {"comparison": f"{left} vs {right}", "left": left, "right": right, "seed": seed}
            row.update(kendall_metrics(a, b))
            rows.append(row)
            pairwise_runs.append(row)
        pairwise_stats[f"{left} vs {right}"] = {
            key: mean_std([row[key] for row in rows])
            for key in (
                "kendall_tau",
                "kendall_tau_at_1",
                "kendall_tau_at_3",
                "prediction_agreement",
                "max_abs_logit_difference",
                "mean_abs_logit_difference",
            )
        }
        pairwise_stats[f"{left} vs {right}"]["all_exact_logits"] = all(
            row["exact_logits"] for row in rows
        )

    spread = {}
    for metric in METRICS:
        means = [variant_stats[v]["metrics"][metric]["mean"] for v in variants]
        numeric = [float(x) for x in means if x is not None]
        spread[metric] = float(max(numeric) - min(numeric)) if numeric else None

    out_dir = root / args.up_to / args.set_mode / args.scope / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]))
        writer.writeheader(); writer.writerows(per_run)
    with (out_dir / "pairwise_invariance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_runs[0]))
        writer.writeheader(); writer.writerows(pairwise_runs)
    with (out_dir / "variant_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for variant in variants:
            row = {"variant": variant}
            row.update({f"{m}_mean": variant_stats[variant]["metrics"][m]["mean"] for m in METRICS})
            row.update({f"{m}_std": variant_stats[variant]["metrics"][m]["std"] for m in METRICS})
            writer.writerow(row)

    payload = {
        "format_version": "freebase_rgcn_block_aggregate_v1",
        "up_to_variant": args.up_to,
        "set_mode": args.set_mode,
        "scope": args.scope,
        "variants": variants,
        "seeds": seeds,
        "alignment": alignment,
        "variant_statistics": variant_stats,
        "pairwise_invariance": pairwise_stats,
        "max_minus_min_spread": spread,
        "definitions": {
            "kendall_tau": "test-node macro Kendall tau-b over all class logits",
            "kendall_tau_at_1": "test-node top-1 class agreement",
            "kendall_tau_at_3": "test-node macro Kendall tau-b on the union of top-3 classes",
            "std": "sample standard deviation over model seeds (ddof=1)",
        },
    }
    write_json(out_dir / "aggregate_metrics.json", payload)

    lines = []
    for variant in variants:
        stats = variant_stats[variant]["metrics"]
        lines.append(
            f"{variant} & ${stats['micro_f1']['mean']:.4f} \\pm {stats['micro_f1']['std']:.4f}$ "
            f"& ${stats['macro_f1']['mean']:.4f} \\pm {stats['macro_f1']['std']:.4f}$ \\\\"
        )
    (out_dir / "latex_rows.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--up-to", default="exact_2")
    parser.add_argument("--set-mode", choices=["union", "intersection"], default="union")
    parser.add_argument("--scope", choices=["center", "all_graph"], default="center")
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())
