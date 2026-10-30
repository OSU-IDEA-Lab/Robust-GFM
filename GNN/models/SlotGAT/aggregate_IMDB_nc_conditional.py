"""Aggregate SlotGAT IMDb original/conditional experiment outputs."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import kendalltau


METRIC_PATHS = {
    "test_accuracy": ("metrics", "test", "accuracy"),
    "test_micro_f1": ("metrics", "test", "micro_f1"),
    "test_macro_f1": ("metrics", "test", "macro_f1"),
    "test_hit_at_1": ("metrics", "test", "hit_at_1"),
    "test_hit_at_3": ("metrics", "test", "hit_at_3"),
    "test_mrr": ("metrics", "test", "mrr"),
    "training_time_seconds": ("training_time_seconds",),
    "epochs_trained": ("epochs_trained",),
    "best_epoch": ("best_epoch",),
    "cuda_peak_allocated_bytes": ("memory", "cuda_peak_allocated_bytes"),
    "cuda_peak_reserved_bytes": ("memory", "cuda_peak_reserved_bytes"),
    "process_max_rss_kb": ("memory", "process_max_rss_kb"),
}


def nested_get(payload: Mapping[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if array.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanstd(array, ddof=0)) if array.size else float("nan")


def numeric_or_nan(value: Any) -> float:
    """Convert optional profiling values to a numeric aggregation value."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*/*/seed_*/result.json")):
        payload = json.loads(path.read_text())
        if payload.get("condition") == "conditional":
            execution_mode = payload.get("execution_mode")
            shared_graph = payload.get("graph", {}).get(
                "materialized_shared_dgl_graph"
            )
            if execution_mode != "independent_raw_context" or shared_graph is not False:
                raise RuntimeError(
                    "Found stale canonical/shared conditional result at "
                    f"{path}. Remove the old conditional result directory and rerun "
                    "the genuine conditional experiment."
                )
        payload["_result_path"] = str(path.resolve())
        results.append(payload)
    if not results:
        raise FileNotFoundError(f"No result.json files found below {results_dir}")
    return results


def load_logits(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(result["test_logits_path"])
    return archive["logits"], archive["node_ids"], archive["labels"]


def mean_row_kendall(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"Logit shapes differ: {logits_a.shape} vs {logits_b.shape}")
    values = []
    for row_a, row_b in zip(logits_a, logits_b):
        if np.array_equal(row_a, row_b):
            values.append(1.0)
            continue
        tau, _ = kendalltau(row_a, row_b, variant="b", nan_policy="omit")
        if np.isfinite(tau):
            values.append(float(tau))
    return safe_mean(values)


def summarize_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["condition"], result["variant"]), []).append(result)

    rows: list[dict[str, Any]] = []
    for (condition, variant), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "condition": condition,
            "variant": variant,
            "runs": len(group),
            "seeds": ",".join(str(result["seed"]) for result in sorted(group, key=lambda x: x["seed"])),
            "raw_graph_hash": group[0]["raw_graph_hash"],
            "conditional_context_plan_hash": group[0].get(
                "conditional_context_plan_hash", ""
            ),
            "conditional_message_program_hash": group[0]["conditional_message_program_hash"],
            "materialized_shared_dgl_graph": group[0].get("graph", {}).get(
                "materialized_shared_dgl_graph"
            ),
            "execution_mode": group[0].get("execution_mode", "independent"),
            "independent_training": group[0].get("independent_training", True),
            "shared_program_representative": group[0].get(
                "shared_program_representative", ""
            ),
        }
        for name, path in METRIC_PATHS.items():
            values = [numeric_or_nan(nested_get(result, path)) for result in group]
            row[f"{name}_mean"] = safe_mean(values)
            row[f"{name}_std"] = safe_std(values)
        rows.append(row)
    return rows


def pairwise_invariance(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in results:
        by_condition_seed.setdefault((result["condition"], int(result["seed"])), []).append(result)

    rows: list[dict[str, Any]] = []
    for (condition, seed), group in sorted(by_condition_seed.items()):
        for left, right in combinations(sorted(group, key=lambda item: item["variant"]), 2):
            logits_left, nodes_left, labels_left = load_logits(left)
            logits_right, nodes_right, labels_right = load_logits(right)
            if not np.array_equal(nodes_left, nodes_right):
                raise AssertionError(
                    f"Test node IDs differ for {left['variant']} and {right['variant']} seed={seed}"
                )
            if not np.array_equal(labels_left, labels_right):
                raise AssertionError(
                    f"Test labels differ for {left['variant']} and {right['variant']} seed={seed}"
                )
            max_abs = float(np.max(np.abs(logits_left - logits_right)))
            predictions_equal = float(
                np.mean(np.argmax(logits_left, axis=1) == np.argmax(logits_right, axis=1))
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "variant_a": left["variant"],
                    "variant_b": right["variant"],
                    "raw_graph_hashes_differ": left["raw_graph_hash"] != right["raw_graph_hash"],
                    "context_plan_hashes_differ": (
                        left.get("conditional_context_plan_hash", "")
                        != right.get("conditional_context_plan_hash", "")
                    ) if condition == "conditional" else None,
                    "shared_dgl_graph_materialized": (
                        left.get("graph", {}).get("materialized_shared_dgl_graph") is True
                        or right.get("graph", {}).get("materialized_shared_dgl_graph") is True
                    ),
                    "message_program_hashes_equal": (
                        left["conditional_message_program_hash"]
                        == right["conditional_message_program_hash"]
                    ),
                    "initial_parameter_hashes_equal": (
                        left["initial_parameter_hash"] == right["initial_parameter_hash"]
                    ),
                    "first_train_logits_hashes_equal": (
                        left["first_step"].get("train_logits_hash")
                        == right["first_step"].get("train_logits_hash")
                    ),
                    "first_gradient_hashes_equal": (
                        left["first_step"].get("gradient_hash")
                        == right["first_step"].get("gradient_hash")
                    ),
                    "final_parameter_hashes_equal": (
                        left["final_parameter_hash"] == right["final_parameter_hash"]
                    ),
                    "best_epochs_equal": left["best_epoch"] == right["best_epoch"],
                    "max_abs_test_logit_difference": max_abs,
                    "prediction_agreement": predictions_equal,
                    "mean_test_logit_kendall_tau": mean_row_kendall(logits_left, logits_right),
                }
            )
    return rows


def condition_summary(
    variant_rows: list[dict[str, Any]], pairwise_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    conditions = sorted({row["condition"] for row in variant_rows})
    for condition in conditions:
        rows = [row for row in variant_rows if row["condition"] == condition]
        pair_rows = [row for row in pairwise_rows if row["condition"] == condition]
        execution_modes = sorted({str(row.get("execution_mode", "independent")) for row in rows})
        condition_payload: dict[str, Any] = {
            "variant_count": len(rows),
            "pairwise_comparison_count": len(pair_rows),
            "execution_modes": execution_modes,
            "all_runs_independently_trained": all(
                bool(row.get("independent_training", True)) for row in rows
            ),
        }
        for metric in ("test_accuracy", "test_micro_f1", "test_macro_f1", "training_time_seconds"):
            variant_means = [row[f"{metric}_mean"] for row in rows]
            variant_stds = [row[f"{metric}_std"] for row in rows]
            condition_payload[metric] = {
                "mean_across_variant_means": safe_mean(variant_means),
                "average_within_variant_std": safe_mean(variant_stds),
                "spread_across_variant_means": float(np.max(variant_means) - np.min(variant_means)),
            }
        if pair_rows:
            condition_payload["invariance"] = {
                "mean_kendall_tau": safe_mean(
                    row["mean_test_logit_kendall_tau"] for row in pair_rows
                ),
                "minimum_kendall_tau": float(
                    np.nanmin([row["mean_test_logit_kendall_tau"] for row in pair_rows])
                ),
                "maximum_absolute_logit_difference": float(
                    np.max([row["max_abs_test_logit_difference"] for row in pair_rows])
                ),
                "mean_prediction_agreement": safe_mean(
                    row["prediction_agreement"] for row in pair_rows
                ),
                "all_first_gradients_equal": all(
                    bool(row["first_gradient_hashes_equal"]) for row in pair_rows
                ),
                "all_final_parameters_equal": all(
                    bool(row["final_parameter_hashes_equal"]) for row in pair_rows
                ),
                "all_best_epochs_equal": all(bool(row["best_epochs_equal"]) for row in pair_rows),
                "all_raw_graph_pairs_differ": all(
                    bool(row["raw_graph_hashes_differ"]) for row in pair_rows
                ),
                "all_context_plan_pairs_differ": all(
                    row["context_plan_hashes_differ"] in (True, None)
                    for row in pair_rows
                ),
                "any_shared_dgl_graph_materialized": any(
                    bool(row["shared_dgl_graph_materialized"]) for row in pair_rows
                ),
            }
        output[condition] = condition_payload
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value: float, std: float, scale: float = 100.0, decimals: int = 2) -> str:
    return f"${scale * value:.{decimals}f} \\pm {scale * std:.{decimals}f}$"


def write_latex(path: Path, summary: Mapping[str, Any]) -> None:
    lines = ["% Auto-generated by aggregate_IMDB_nc_conditional.py"]
    original = summary.get("original")
    conditional = summary.get("conditional")
    if original and conditional:
        omi = original["test_micro_f1"]
        cmi = conditional["test_micro_f1"]
        oma = original["test_macro_f1"]
        cma = conditional["test_macro_f1"]
        tau_o = original.get("invariance", {}).get("mean_kendall_tau", float("nan"))
        tau_c = conditional.get("invariance", {}).get("mean_kendall_tau", float("nan"))
        lines.extend(
            [
                "% Model & Micro-F1 (Orig -> Inv) & Spread (Orig -> Inv) & Macro-F1 (Orig -> Inv) & Spread (Orig -> Inv) & Kendall-tau (Orig -> Inv)",
                (
                    "SlotGAT & "
                    f"{latex_value(omi['mean_across_variant_means'], omi['average_within_variant_std'])} "
                    f"$\\rightarrow$ {latex_value(cmi['mean_across_variant_means'], cmi['average_within_variant_std'])} & "
                    f"${100 * omi['spread_across_variant_means']:.2f} \\rightarrow {100 * cmi['spread_across_variant_means']:.2f}$ & "
                    f"{latex_value(oma['mean_across_variant_means'], oma['average_within_variant_std'])} "
                    f"$\\rightarrow$ {latex_value(cma['mean_across_variant_means'], cma['average_within_variant_std'])} & "
                    f"${100 * oma['spread_across_variant_means']:.2f} \\rightarrow {100 * cma['spread_across_variant_means']:.2f}$ & "
                    f"${tau_o:.3f} \\rightarrow {tau_c:.3f}$ \\\\"
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def aggregate_results(results_dir: str | Path) -> dict[str, Any]:
    results_dir = Path(results_dir).expanduser().resolve()
    results = load_results(results_dir)
    variant_rows = summarize_variants(results)
    pairwise_rows = pairwise_invariance(results)
    summary = condition_summary(variant_rows, pairwise_rows)

    write_csv(results_dir / "aggregate_per_variant.csv", variant_rows)
    write_csv(results_dir / "aggregate_pairwise_invariance.csv", pairwise_rows)
    (results_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_latex(results_dir / "aggregate_latex_row.tex", summary)

    print("\nSlotGAT IMDb aggregate")
    for condition, payload in summary.items():
        micro = payload["test_micro_f1"]
        macro = payload["test_macro_f1"]
        inv = payload.get("invariance", {})
        print(
            f"  {condition:11s}: Micro-F1={100 * micro['mean_across_variant_means']:.2f} "
            f"spread={100 * micro['spread_across_variant_means']:.2f}; "
            f"Macro-F1={100 * macro['mean_across_variant_means']:.2f} "
            f"spread={100 * macro['spread_across_variant_means']:.2f}; "
            f"Kendall={inv.get('mean_kendall_tau', float('nan')):.6f}; "
            f"max|dlogit|={inv.get('maximum_absolute_logit_difference', float('nan')):.3e}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SlotGAT IMDb conditional runs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "imdb_nc_conditional",
    )
    args = parser.parse_args()
    aggregate_results(args.results_dir)


if __name__ == "__main__":
    main()
