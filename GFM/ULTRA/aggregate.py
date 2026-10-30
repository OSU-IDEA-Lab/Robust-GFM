"""Collect every comparison.json under the results root into one table: <results>/summary.{csv,md}.

One row per (dataset, checkpoint, variant) against that task's reference variant, with Kendall tau, delta MRR
(tie-aware), Hit@1 / Hit@10 disagreement, the noise floor (reference vs its repeat run) and whether the graph
was held out from the checkpoint's pretraining.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS", os.path.join(HERE, "results"))
PRETRAIN = {"ultra_3g": {"FB15k237", "WN18RR", "CoDExMedium"},
            "ultra_4g": {"FB15k237", "WN18RR", "CoDExMedium", "NELL995"},
            "ultra_50g": None}  # the 50g pretraining list is not published
ORDER = {"WN18RR": 0, "FB15k237": 1, "Hetionet": 2}


def held_out(ds, ck):
    s = PRETRAIN[ck]
    return "unknown" if s is None else ("no" if ds in s else "yes")


def main():
    rows = []
    for ds in sorted(os.listdir(RES), key=lambda d: ORDER.get(d, 9)):
        dsdir = os.path.join(RES, ds)
        if not os.path.isdir(dsdir):
            continue
        for ck in sorted(os.listdir(dsdir)):
            p = os.path.join(dsdir, ck, "comparison.json")
            if not os.path.exists(p):
                continue
            c = json.load(open(p))
            floor = [v for k, v in c["pairs"].items() if k.endswith("__repeat")]
            floor_tau = floor[0]["kendall_tau"] if floor else None
            for pair, v in c["pairs"].items():
                if pair.endswith("__repeat"):
                    continue
                ref, var = pair.split(" vs ")
                rows.append({
                    "dataset": ds, "checkpoint": ck, "held_out": held_out(ds, ck), "reference": ref, "variant": var,
                    "common_queries": v["common_queries"],
                    "mrr_reference": round(v["mrr_a"], 4),
                    "mrr_variant": round(v["mrr_b"], 4),
                    "delta_mrr": round(v["delta_mrr_tie_aware"], 4),
                    "kendall_tau": round(v["kendall_tau"], 4),
                    "noise_floor_tau": None if floor_tau is None else round(floor_tau, 4),
                    "share_hit1_differs": round(v["share_hit1_differs"], 4),
                    "share_hit10_differs": round(v["share_hit10_differs"], 4),
                })
    if not rows:
        raise SystemExit("no comparison.json found yet")
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| dataset | ckpt | held out | variant | queries | MRR ref | MRR var | dMRR | tau | floor | Hit@1 differs |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['checkpoint'].replace('ultra_', '')} | {r['held_out']} | {r['variant']} | "
                     f"{r['common_queries']} | {r['mrr_reference']:.3f} | {r['mrr_variant']:.3f} | {r['delta_mrr']:+.3f} | "
                     f"{r['kendall_tau']:.3f} | {r['noise_floor_tau']} | {r['share_hit1_differs']:.3f} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
