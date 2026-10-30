#!/usr/bin/env python3
"""Aggregate the SlotGAT LP four-arm results (WordNet + IMDb/DBLP): metrics per arm × variant (mean ± std over
seeds), pairwise cross-variant Kendall τ per seed (originals / augmentation / invariant), τ@1/τ@3 (LP),
bitwise checks for the invariant arm (checkpoint hashes equal across variants per seed), and the
invariant-vs-universal identity check. Writes <root>/aggregate/summary.json + summary.md.

    python aggregate_lp_arms.py --family wordnet_nls   # results/wn18rr_lp
    python aggregate_lp_arms.py --family hetero_nls    # results/lp_hetero
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
ROOTS = {"wordnet": HERE.parent / "results/wn18rr_lp", "hetero": HERE.parent / "results/lp_hetero",
         "wordnet_nls": HERE.parent / "results/wn18rr_lp", "hetero_nls": HERE.parent / "results/lp_hetero"}
SEEDS = (1566911444, 20241017, 20251017)
ARMS = ("originals", "universal", "invariant", "augmentation")
CELLS = {"wordnet_nls": None, "hetero_nls": None, "wordnet": [("WORDNET", "link_prediction", ["no_changes", "all_inverse_edges", "transitive_edges"], ["universal_edges"])],
         "hetero": [("IMDB", "movie_year", ["v1", "v2", "v3", "v4"], ["universal"]), ("IMDB", "movie_director", ["v1", "v3"], ["universal"]),
                    ("DBLP", "paper_conference", ["v1", "v2", "v3"], ["universal"])]}


def run_dir(root, family, arm, ds, task, v, s):
    return root / arm / "full" / "intersection" / v / f"seed_{s}" if family == "wordnet" else root / arm / ds / task / v / f"seed_{s}"


def load(root, family, arm, ds, task, v, s):
    d = run_dir(root, family, arm, ds, task, v, s)
    if not (d / "metrics.json").exists():
        return None
    m = json.load(open(d / "metrics.json"))
    if family == "wordnet":
        z = np.load(d / "test_scores.npz")
        side = z["side"]; n = len(side)
        ranks = np.where(side == 0, z["filtered_tail_ranks"][np.arange(n) // 2], z["filtered_head_ranks"][np.arange(n) // 2]).astype(float)
        tm = m["test_metrics"]
        stats = {"mrr": tm["filtered_MRR"], "hits@1": tm["Hits@1"], "hits@3": tm["Hits@3"], "hits@10": tm["Hits@10"]}
        S = z["candidate_scores"]
        res = {"train_time_sec": m["train_time_sec"], "epochs": m["epochs_completed"], "parameter_mib": m["parameter_bytes"] / 2**20,
               "peak_training_gpu_mib": m["peak_train_gpu_bytes"] / 2**20, "peak_inference_gpu_mib": m["peak_inference_gpu_bytes"] / 2**20}
    else:
        z = np.load(d / "scores.npz"); S = z["scores"]
        mm = m["metrics"]; stats = {k: mm[k] for k in ("mrr", "hits@1", "hits@3", "hits@10")}
        res = {k: mm[k] for k in ("train_time_sec", "epochs", "parameter_mib", "peak_training_gpu_mib", "peak_inference_gpu_mib")}
    return {"stats": stats, "scores": S, "ckpt": m["checkpoint_hash"], "msg_hash": m["message_graph"]["message_hash"], "res": res}


def tau_pair(A: np.ndarray, B: np.ndarray) -> dict:
    """Kendall τ-b between two score matrices row by row (identical rows count 1), plus τ@1 / τ@3 on the hit indicators."""
    taus = []
    for a, b in zip(A, B):
        if np.array_equal(a, b): taus.append(1.0); continue
        t = kendalltau(a, b).statistic
        taus.append(1.0 if np.isnan(t) else float(t))
    def hits(S, k):
        s0 = S[:, :1]; rk = 1 + (S[:, 1:] > s0).sum(1) + 0.5 * (S[:, 1:] == s0).sum(1); return (rk <= k).astype(int)
    out = {"tau": float(np.mean(taus))}
    for k in (1, 3):
        ha, hb = hits(A, k), hits(B, k)
        out[f"tau@{k}"] = 1.0 if np.array_equal(ha, hb) else float(np.nan_to_num(kendalltau(ha, hb).statistic, nan=1.0))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", choices=list(ROOTS), required=True); args = ap.parse_args()
    CELLS["wordnet_nls"], CELLS["hetero_nls"] = CELLS["wordnet"], CELLS["hetero"]
    root = ROOTS[args.family]; family = args.family.replace("_nls", "")   # _nls roots share the layout of the ce roots
    summary, lines = {}, [f"# SlotGAT LP four arms — {family} (root `{root}`)\n"]
    for ds, task, variants, uni in CELLS[family]:
        cell = f"{ds}/{task}"; summary[cell] = {}
        lines.append(f"\n## {cell}\n")
        for arm in ARMS:
            vs = uni if arm == "universal" else variants
            runs = {(v, s): load(root, family, arm, ds, task, v, s) for v in vs for s in SEEDS}
            have = {k: r for k, r in runs.items() if r is not None}
            if not have:
                lines.append(f"* {arm}: (no runs)\n"); continue
            block = {"per_variant": {}, "tau": {}, "bitwise": {}, "resources": {}}
            lines.append(f"### {arm}  ({len(have)}/{len(runs)} runs)\n\n| variant | MRR | Hits@1 | Hits@3 | Hits@10 | seeds |\n|---|---|---|---|---|---|")
            for v in vs:
                rs = [have[(v, s)] for s in SEEDS if (v, s) in have]
                if not rs: continue
                agg = {k: (float(np.mean([r["stats"][k] for r in rs])), float(np.std([r["stats"][k] for r in rs]))) for k in rs[0]["stats"]}
                block["per_variant"][v] = agg
                lines.append(f"| {v} | " + " | ".join(f"{agg[k][0]:.4f} ± {agg[k][1]:.4f}" for k in ("mrr", "hits@1", "hits@3", "hits@10")) + f" | {len(rs)} |")
            # cross-variant τ per seed
            if arm != "universal" and len(vs) > 1:
                taus = {}
                for s in SEEDS:
                    for a, b in itertools.combinations(vs, 2):
                        if (a, s) in have and (b, s) in have:
                            taus.setdefault(f"{a} vs {b}", []).append(tau_pair(have[(a, s)]["scores"], have[(b, s)]["scores"]))
                if taus:
                    lines.append("\n| pair | τ | τ@1 | τ@3 |\n|---|---|---|---|")
                    for pair, ts in taus.items():
                        block["tau"][pair] = {k: (float(np.mean([t[k] for t in ts])), float(np.std([t[k] for t in ts]))) for k in ts[0]}
                        lines.append(f"| {pair} | " + " | ".join(f"{block['tau'][pair][k][0]:.4f} ± {block['tau'][pair][k][1]:.4f}" for k in ("tau", "tau@1", "tau@3")) + " |")
                    allt = [t["tau"] for ts in taus.values() for t in ts]
                    block["tau"]["mean_over_pairs"] = float(np.mean(allt))
                    lines.append(f"\nmean τ over pairs and seeds: **{np.mean(allt):.4f}**")
            # bitwise checks
            if arm == "invariant":
                for s in SEEDS:
                    ck = {v: have[(v, s)]["ckpt"] for v in vs if (v, s) in have}
                    mh = {v: have[(v, s)]["msg_hash"] for v in vs if (v, s) in have}
                    spread = float(max(have[(v, s)]["stats"]["mrr"] for v in ck) - min(have[(v, s)]["stats"]["mrr"] for v in ck)) if ck else None
                    block["bitwise"][str(s)] = {"checkpoints_identical": len(set(ck.values())) == 1, "message_hashes_identical": len(set(mh.values())) == 1,
                                                "mrr_spread": spread, "n": len(ck)}
                lines.append("\nbitwise (per seed): " + "; ".join(f"{s}: ckpt {'=' if b['checkpoints_identical'] else '≠'} msg {'=' if b['message_hashes_identical'] else '≠'} spread {b['mrr_spread']}" for s, b in block["bitwise"].items()))
            block["resources"] = {k: float(np.mean([r["res"][k] for r in have.values()])) for k in next(iter(have.values()))["res"]}
            lines.append("\nresources (mean): " + ", ".join(f"{k} {v:.1f}" for k, v in block["resources"].items()) + "\n")
            summary[cell][arm] = block
        # invariant vs universal identity
        inv, unv = summary[cell].get("invariant"), summary[cell].get("universal")
        if inv and unv:
            ids = []
            for s in SEEDS:
                a, b = load(root, family, "invariant", ds, task, variants[0], s), load(root, family, "universal", ds, task, uni[0], s)
                if a and b: ids.append({"seed": s, "checkpoint_equal": a["ckpt"] == b["ckpt"], "message_hash_equal": a["msg_hash"] == b["msg_hash"], "mrr_inv": a["stats"]["mrr"], "mrr_uni": b["stats"]["mrr"]})
            summary[cell]["invariant_vs_universal"] = ids
            lines.append("invariant vs universal: " + "; ".join(f"{i['seed']}: ckpt {'=' if i['checkpoint_equal'] else '≠'} msg {'=' if i['message_hash_equal'] else '≠'} ({i['mrr_inv']:.4f} vs {i['mrr_uni']:.4f})" for i in ids) + "\n")
    out = root / "aggregate"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)); (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines)); print(f"\nwrote {out}/summary.{{json,md}}")


if __name__ == "__main__":
    main()
