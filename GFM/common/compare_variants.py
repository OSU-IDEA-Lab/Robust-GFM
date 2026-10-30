"""Compare one model's outputs across representation variants of one dataset.

Reads <results>/<variant>/scores.npz (the evaluator output contract, GFM/README.md) and reports, on the
query subset answerable on BOTH variants of each pair:

  per variant      MRR, Hits@1/3/10 (tie-aware and pessimistic ranks)
  per variant pair  change in MRR, Kendall tau-b between the candidate-score vectors [true + 50 fixed
                   negatives] averaged over queries and both directions (identical vectors count as 1),
                   and the share of queries whose Hit@1 / Hit@10 differ

A '<variant>__repeat' run, if present, is the same variant evaluated twice: its tau with the original run is
the hardware noise floor (1.000 = inference is reproducible, so every tau < 1 between variants is real).
"""

import argparse
import itertools
import os

import numpy as np
from scipy.stats import kendalltau

from kg_format import write_json

HERE = os.path.dirname(os.path.abspath(__file__))


def tau_mean(a, b):
    vals = []
    for x, y in zip(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])):
        if np.array_equal(x, y):
            vals.append(1.0)
            continue
        t = kendalltau(x, y).statistic
        vals.append(0.0 if np.isnan(t) else float(t))
    return float(np.mean(vals)), vals


def metrics(r):
    r = r.reshape(-1)
    return {"mrr": float(np.mean(1.0 / r)), "hits@1": float(np.mean(r <= 1)), "hits@3": float(np.mean(r <= 3)),
            "hits@10": float(np.mean(r <= 10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory with one <variant>/scores.npz per variant")
    ap.add_argument("--reference", default=None, help="variant every other variant is compared with")
    args = ap.parse_args()

    runs = {}
    for v in sorted(os.listdir(args.results)):
        p = os.path.join(args.results, v, "scores.npz")
        if os.path.exists(p):
            runs[v] = dict(np.load(p))
    if len(runs) < 2:
        raise SystemExit(f"need at least two finished variants under {args.results}")
    out = {"variants": {}, "pairs": {}}
    for v, r in runs.items():
        out["variants"][v] = {"tie_aware": metrics(r["rank_tie"]), "pessimistic": metrics(r["rank_pess"]),
                              "queries_evaluated": int(len(r["query_index"]))}
    ref = args.reference or ("original" if "original" in runs else "no_changes" if "no_changes" in runs else None)
    pairs = [(ref, v) for v in runs if v != ref] if ref else list(itertools.combinations(runs, 2))
    for a, b in pairs:
        ra, rb = runs[a], runs[b]
        # each pair is scored on the queries answerable on BOTH variants (a reified relation removes its queries)
        common = np.intersect1d(ra["query_index"], rb["query_index"])
        ia, ib = np.searchsorted(ra["query_index"], common), np.searchsorted(rb["query_index"], common)
        sa, sb = ra["cand_scores"][ia], rb["cand_scores"][ib]
        tau, _ = tau_mean(sa, sb)
        ka, kb = ra["rank_tie"][ia], rb["rank_tie"][ib]
        ma, mb = metrics(ka), metrics(kb)
        out["pairs"][f"{a} vs {b}"] = {
            "common_queries": int(len(common)),
            "mrr_a": ma["mrr"], "mrr_b": mb["mrr"], "hits1_a": ma["hits@1"], "hits1_b": mb["hits@1"],
            "kendall_tau": tau,
            "delta_mrr_tie_aware": mb["mrr"] - ma["mrr"],
            "share_hit1_differs": float(np.mean((ka <= 1) != (kb <= 1))),
            "share_hit10_differs": float(np.mean((ka <= 10) != (kb <= 10))),
            "scores_bitwise_identical": bool(np.array_equal(sa, sb)),
        }
    write_json(os.path.join(args.results, "comparison.json"), out)

    lines = [f"# {os.path.relpath(args.results, HERE)}", "", "Per variant, on every query answerable on it:", "",
             "| variant | queries | MRR | Hits@1 | Hits@10 |", "| --- | --- | --- | --- | --- |"]
    for v, m in out["variants"].items():
        t = m["tie_aware"]
        lines.append(f"| {v} | {m['queries_evaluated']} | {t['mrr']:.4f} | {t['hits@1']:.4f} | {t['hits@10']:.4f} |")
    lines += ["", "Per pair, on the queries answerable on both (x 2 directions):", "",
              "| pair | queries | Kendall tau | delta MRR | Hit@1 differs | Hit@10 differs | identical |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, p in out["pairs"].items():
        lines.append(f"| {k} | {p['common_queries']} | {p['kendall_tau']:.4f} | {p['delta_mrr_tie_aware']:+.4f} | "
                     f"{p['share_hit1_differs']:.3f} | {p['share_hit10_differs']:.3f} | {p['scores_bitwise_identical']} |")
    with open(os.path.join(args.results, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
