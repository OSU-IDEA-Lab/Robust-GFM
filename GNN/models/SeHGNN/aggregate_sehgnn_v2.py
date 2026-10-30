#!/usr/bin/env python3
"""Aggregate SeHGNN v2 results: RGCN aggregator per arm_flavour directory + one summary table.

    python aggregate_sehgnn_v2.py --cell IMDB_nc        (or FREEBASE_nc)   [--results-root results/SeHGNN_v2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--results-root", default=str(_HERE / "results" / "SeHGNN_v2"))
    args = ap.parse_args()
    R = Path(args.results_root) / args.cell
    ds = args.cell.split("_")[0]
    rows = []
    for arm in sorted(d for d in os.listdir(R) if (R / d / ds).is_dir()):
        subprocess.run([sys.executable, str(_HERE.parent.parent / "common" / "aggregate_rgcn_structural.py"), "--results-root", str(R / arm),
                        "--out-dir", str(R / arm / "aggregate")], check=False, capture_output=True)
        per, extra = {}, {}
        for f in sorted(glob.glob(f"{R}/{arm}/{ds}/nc/*/seed_*/metrics.json")):
            j = json.load(open(f)); m = j["metrics"]
            per.setdefault(j["variant"], []).append(m["micro_f1"])
            extra.setdefault("macro", []).append(m["macro_f1"]); extra.setdefault("train", []).append(m["train_time_sec"])
            extra.setdefault("ep", []).append(m["epochs"]); extra.setdefault("best", []).append(m["best_epoch"])
            extra.setdefault("params", []).append(m["parameter_count"]); extra.setdefault("peak", []).append(m["peak_training_gpu_mib"] or 0)
            extra.setdefault("peak_inf", []).append(m["peak_inference_gpu_mib"] or 0); extra.setdefault("inf", []).append(m["inference_time_sec"])
        if not per:
            continue
        means = [np.mean(v) for v in per.values()]
        line = f"  {arm:28s} " + "  ".join(f"{v} {np.mean(x):.4f}(n={len(x)})" for v, x in sorted(per.items())) + \
               f"   mean {np.mean(means):.4f}  spread {max(means) - min(means):.6f}  macro {np.mean(extra['macro']):.4f}" \
               f"  ep {np.mean(extra['ep']):.0f}/best {np.mean(extra['best']):.0f}  train {np.mean(extra['train']):.0f}s  inf {np.mean(extra['inf']):.2f}s" \
               f"  params {extra['params'][0]/1e6:.2f}M  peak {np.mean(extra['peak']):.0f}/{np.mean(extra['peak_inf']):.0f} MiB"
        inv = R / arm / "aggregate" / "invariance.csv"
        if inv.exists():
            import pandas as pd
            try:
                d = pd.read_csv(inv)
            except Exception:
                d = pd.DataFrame()
            if len(d):
                line += f"  | tau mean {d.kendall_tau.mean():.4f} min {d.kendall_tau.min():.4f} ckpt_equal {bool(d.checkpoint_hash_equal.all())} max|dscore| {d.max_abs_score_diff.max():.2e}"
        joint = sorted(glob.glob(f"{R}/{arm}/{ds}/nc/joint_seed_*.json"))
        if joint:
            taus = [json.load(open(j))["mean_tau"] for j in joint]
            line += f"  | joint tau mean {np.mean(taus):.4f}"
        print(line); rows.append(line)
    (R / "summary.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
