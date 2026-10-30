#!/usr/bin/env python3
"""Summarise a four-arm result root (RGCN layout; used by PathNet and PAIN).

    python summarize_arms.py <root> <DATASET/task>  e.g.  results/imdb_nc IMDB/nc
                                                          results/lp_hetero IMDB/movie_director
                                                          results/wordnet_lp WN (WordNet layout: <arm>/full/intersection/<v>)

Per arm × variant: mean ± population std (ddof 0) over seeds of the main metric (NC micro-F1, LP MRR, WN
filtered MRR), best epoch, train time, peak train MiB. Invariance: mean over seeds of the pairwise per-row Kendall
τ-b between variants (identical rows count 1) — ``invariance_stat.row_tau``, the statistic used in docs/results —
plus whether the checkpoint hashes of a seed are identical across variants.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from invariance_stat import row_tau


def load(root: Path, sub: str):
    runs = {}
    wn = sub == "WN"
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        base = arm_dir / ("full/intersection" if wn else sub)
        if not base.exists():
            continue
        for mf in sorted(base.glob("*/seed_*/metrics.json")):
            m = json.loads(mf.read_text()); v = mf.parent.parent.name; s = mf.parent.name[5:]
            if wn:
                z = np.load(mf.parent / "test_scores.npz"); key = "filtered_MRR"
                val = m["test_metrics"][key]
                S = z[[k for k in z.files if "score" in k][0]] if any("score" in k for k in z.files) else None
                res = (m["best_epoch"], m["train_time_sec"], m["peak_train_gpu_bytes"] / 2**20)
            else:
                z = np.load(mf.parent / "scores.npz"); mm = m["metrics"]
                val = mm.get("micro_f1", mm.get("mrr")); S = z["scores"]
                res = (mm["best_epoch"], mm["train_time_sec"], mm["peak_training_gpu_mib"])
            runs.setdefault(arm_dir.name, {}).setdefault(v, {})[s] = (val, S, m["checkpoint_hash"], res)
    return runs


def main():
    root, sub = Path(sys.argv[1]), sys.argv[2]
    runs = load(root, sub)
    for arm, byv in runs.items():
        print(f"== {arm}")
        allv = []
        for v, bys in byv.items():
            vals = [r[0] for r in bys.values()]; allv += vals
            be = [r[3][0] for r in bys.values()]; tt = [r[3][1] for r in bys.values()]; pk = [r[3][2] for r in bys.values()]
            print(f"   {v:18s} n={len(vals)}  {np.mean(vals):.4f} ± {np.std(vals):.4f}   best_ep {np.mean(be):6.1f}  "
                  f"train {np.mean(tt)/60:6.1f} min  peak {np.mean(pk):8.0f} MiB")
        if len(byv) > 1:
            print(f"   {'mean over variants':18s}       {np.mean(allv):.4f} ± {np.std(allv):.4f}")
            taus, same = [], []
            seeds = set.intersection(*[set(b) for b in byv.values()])
            for s in sorted(seeds):
                for a, b in itertools.combinations(sorted(byv), 2):
                    Sa, Sb = byv[a][s][1], byv[b][s][1]
                    if Sa is not None and Sb is not None and Sa.shape == Sb.shape:
                        taus.append(row_tau(Sa, Sb))
                    same.append(byv[a][s][2] == byv[b][s][2])
            if taus:
                print(f"   pairwise tau {np.mean(taus):.4f} (min {np.min(taus):.4f}, {len(taus)} pairs)   identical checkpoints {sum(same)}/{len(same)}")


if __name__ == "__main__":
    main()
