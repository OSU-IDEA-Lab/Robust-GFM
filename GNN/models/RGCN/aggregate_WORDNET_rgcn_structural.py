#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wordnet_structural_common import atomic_json_dump

try:
    from scipy.stats import kendalltau
except Exception:  # pragma: no cover
    kendalltau = None


METRICS = (
    "filtered_MRR", "raw_MRR", "Hits@1", "Hits@3", "Hits@10",
    "filtered_mean_rank", "accuracy", "precision", "recall", "macro_f1",
    "roc_auc", "average_precision",
)


def mean_std(values):
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan"), float("nan")
    ddof = 1 if finite.size > 1 else 0
    return float(np.mean(finite)), float(np.std(finite, ddof=ddof))


def safe_tau(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.array_equal(a, b):
        return 1.0
    if kendalltau is None:
        return float(np.mean(np.argsort(np.argsort(a)) == np.argsort(np.argsort(b))))
    value = kendalltau(a, b, variant="b", nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/RGCN_structural/WORDNET/link_prediction")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir) if args.output_dir else root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for path in root.glob("*/*/*/seed_*/metrics.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec["metrics_path"] = str(path)
        records.append(rec)
    if not records:
        raise SystemExit(f"No metrics.json files found under {root}")

    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["budget"], rec["mode"], rec["variant"])].append(rec)

    per_variant = []
    for (budget, mode, variant), rows in sorted(grouped.items()):
        item = {"budget": budget, "mode": mode, "variant": variant, "runs": len(rows)}
        for metric in METRICS:
            vals = [r["test_metrics"].get(metric, float("nan")) for r in rows]
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(vals)
        for field in (
            "train_time_sec", "inference_time_sec", "peak_train_gpu_bytes",
            "peak_inference_gpu_bytes", "parameter_bytes", "checkpoint_bytes",
        ):
            item[f"{field}_mean"], item[f"{field}_std"] = mean_std([r.get(field, np.nan) for r in rows])
        per_variant.append(item)

    spread_rows = []
    by_config = defaultdict(list)
    for row in per_variant:
        by_config[(row["budget"], row["mode"])].append(row)
    for (budget, mode), rows in sorted(by_config.items()):
        item = {"budget": budget, "mode": mode, "variants": len(rows)}
        for metric in METRICS:
            values = [r[f"{metric}_mean"] for r in rows]
            item[f"{metric}_spread"] = float(np.nanmax(values) - np.nanmin(values))
        spread_rows.append(item)

    invariance_rows = []
    by_seed_config = defaultdict(list)
    for rec in records:
        by_seed_config[(rec["budget"], rec["mode"], int(rec["seed"]))].append(rec)
    for (budget, mode, seed), rows in sorted(by_seed_config.items()):
        rows = sorted(rows, key=lambda r: r["variant"])
        for a, b in itertools.combinations(rows, 2):
            pa = Path(a["run_dir"]) / "test_scores.npz"
            pb = Path(b["run_dir"]) / "test_scores.npz"
            with np.load(pa, allow_pickle=False) as za, np.load(pb, allow_pickle=False) as zb:
                ids_equal = np.array_equal(za["candidate_ids"], zb["candidate_ids"])
                queries_equal = np.array_equal(za["query_triples"], zb["query_triples"])
                sides_equal = np.array_equal(za["side"], zb["side"])
                sa, sb = za["candidate_scores"], zb["candidate_scores"]
            if not (ids_equal and queries_equal and sides_equal):
                raise RuntimeError(f"Candidate alignment differs: {pa} vs {pb}")
            query_taus = [safe_tau(x, y) for x, y in zip(sa, sb)]
            invariance_rows.append({
                "budget": budget,
                "mode": mode,
                "seed": seed,
                "variant_a": a["variant"],
                "variant_b": b["variant"],
                "semantic_hash_equal": a["semantic_hash"] == b["semantic_hash"],
                "supervision_hash_equal": a["supervision_hash"] == b["supervision_hash"],
                "physical_hash_different": a["physical_hash"] != b["physical_hash"],
                "checkpoint_hash_equal": a["checkpoint_hash"] == b["checkpoint_hash"],
                "first_step_hash_equal": a["first_step_audit"] == b["first_step_audit"],
                "candidate_exact_equal": bool(np.array_equal(sa, sb)),
                "candidate_max_abs_diff": float(np.max(np.abs(sa - sb))),
                "flat_kendall_tau": safe_tau(sa, sb),
                "query_macro_kendall_tau": float(np.nanmean(query_taus)),
                "top1_agreement": float(np.mean(np.argmax(sa, axis=1) == np.argmax(sb, axis=1))),
            })

    def write_csv(path, rows):
        if not rows:
            return
        keys = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "per_variant.csv", per_variant)
    write_csv(out / "spread.csv", spread_rows)
    write_csv(out / "invariance.csv", invariance_rows)

    summary = {
        "records": records,
        "per_variant": per_variant,
        "spread": spread_rows,
        "invariance": invariance_rows,
    }
    atomic_json_dump(summary, out / "aggregate.json")

    latex = []
    for row in per_variant:
        tau_rows = [x for x in invariance_rows if x["budget"] == row["budget"] and x["mode"] == row["mode"]]
        tau = float(np.nanmean([x["query_macro_kendall_tau"] for x in tau_rows])) if tau_rows else float("nan")
        latex.append(
            f"{row['budget'].title()} & {row['mode'].title()} & {row['variant']} & "
            f"${row['filtered_MRR_mean']:.4f} \\pm {row['filtered_MRR_std']:.4f}$ & "
            f"${row['Hits@3_mean']:.4f} \\pm {row['Hits@3_std']:.4f}$ & "
            f"${tau:.4f}$ & ${row['train_time_sec_mean']:.1f}$ \\\\" 
        )
    (out / "latex_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "runs": len(records), "groups": len(per_variant)}, indent=2))


if __name__ == "__main__":
    main()
