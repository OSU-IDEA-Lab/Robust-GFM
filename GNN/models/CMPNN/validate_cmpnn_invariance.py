#!/usr/bin/env python3
"""CMPNN invariance gauntlet, data-level steps.

For one (dataset, task) and its variants:
  1. physical hashes differ, semantic hashes equal            (raw graphs vs program)
  2. forward-half relation tables identical and duplicate-free
  3. supervision hashes equal
  7. leak audit: with the runner's easy-edge masking (query edge + alias copies), the
     fraction of TRAINING queries whose true tail is reachable in 1 or 2 hops must match
     the fraction for HELD-OUT queries — otherwise the model has a training-only shortcut
     (RGCN guide failure mode #10)

Steps 4-6 (first-step logits/gradients, checkpoint equality, τ) are produced by
``run_cmpnn_lp.py`` and compared by ``RGCN/aggregate_rgcn_structural.py``; the smoke
``scripts/smoke_determinism.py`` covers step 4 on the GPU.
Exit code 0 = all checks passed, 2 = a check failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from inv_cmpnn.bundles import LPBundle, load_bundle  # noqa: E402
from run_cmpnn_lp import BUNDLE_ROOTS  # noqa: E402


def adjacency(edges: torch.Tensor, n: int) -> sp.csr_matrix:
    """Unweighted, symmetric (CMPNN adds inverses) adjacency over all relations."""
    s, d = edges[:, 0].numpy(), edges[:, 1].numpy()
    a = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    return a


def reach_audit(b: LPBundle, graph: str) -> dict:
    """Direct / 2-hop reachability of the true tail, train (masked) vs held-out."""
    A = adjacency(b.edges(graph), b.num_nodes)
    A2 = (A @ A).tocsr()
    alias = b.alias.numpy()
    out = {}
    for split in ("train", "val", "test"):
        p = b.positives(split, graph).numpy()
        h, t = p[:, 0], p[:, 2]
        direct = np.asarray(A[h, t]).ravel() > 0
        two = np.asarray(A2[h, t]).ravel()
        if split == "train":
            # the runner masks (h,t), (t,h), (alias(h),t), (t,alias(h)), (h,alias(t)), (alias(t),h)
            # and every (h, *, r) edge for r in label_derived_relations
            direct = np.zeros_like(direct)
            a_h, a_t = alias[h], alias[t]
            via_ah = np.asarray(A[h, a_h]).ravel() * np.asarray(A[a_h, t]).ravel()
            via_at = np.asarray(A[h, a_t]).ravel() * np.asarray(A[a_t, t]).ravel()
            two = two - np.where(a_h != h, via_ah, 0) - np.where(a_t != t, via_at, 0)
            derived = b.label_derived_ids(graph)
            if derived:
                e = b.edges(graph)
                de = e[torch.isin(e[:, 2], torch.tensor(derived))]
                by_head = {}
                for hh, kk in de[:, :2].tolist():
                    by_head.setdefault(hh, []).append(kk)
                for i in range(len(h)):
                    for k in by_head.get(int(h[i]), []):
                        two[i] -= A[k, t[i]]
        out[split] = {"n": int(len(h)), "direct_frac": float(direct.mean()),
                      "two_hop_frac": float((two > 0).mean()), "two_hop_mean_paths": float(two.mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/preprocessed")
    ap.add_argument("--dataset", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--variants", required=True, help="comma-separated")
    ap.add_argument("--report", default="")
    ap.add_argument("--leak-tolerance", type=float, default=0.05,
                    help="max allowed |train - test| gap in 2-hop reachability fraction")
    args = ap.parse_args()
    variants = args.variants.split(",")
    bundles = {v: load_bundle(Path(args.data_root) / BUNDLE_ROOTS[(args.dataset, args.task)] / v) for v in variants}
    ok = True
    rep = {"dataset": args.dataset, "task": args.task, "variants": {}, "checks": {}}
    for v, b in bundles.items():
        s = b.summary(); rep["variants"][v] = s
        print(f"{v:12s} phys={s['physical_hash'][:12]} sem={s['semantic_hash'][:12]} sup={s['supervision_hash'][:12]} "
              f"rel_sem={s['relation_hash_semantic'][:8]} Ephys={s['num_physical_edges']} Esem={s['num_semantic_edges']}")
    phys = {b.physical_hash() for b in bundles.values()}
    sem = {b.semantic_hash() for b in bundles.values()}
    sup = {b.supervision_hash() for b in bundles.values()}
    rel = {b.relation_hash("semantic") for b in bundles.values()}
    ali = {b.alias_hash() for b in bundles.values()}
    non_univ = [v for v in variants if v != "universal"]
    c1 = len({bundles[v].physical_hash() for v in non_univ}) == len(non_univ)
    c2 = len(sem) == 1 and len(rel) == 1
    c3 = len(sup) == 1 and len(ali) == 1
    rep["checks"].update({"physical_hashes_distinct": c1, "semantic_program_equal": c2, "supervision_equal": c3})
    print(f"[1] physical graphs distinct across {non_univ}: {c1}")
    print(f"[2] semantic program + relation table identical: {c2}   (duplicates are rejected at load time)")
    print(f"[3] supervision + alias identical: {c3}")
    ok &= c1 and c2 and c3
    # 7. leak audit on the semantic program (once — identical across variants) and per physical graph
    audits = {"semantic": reach_audit(bundles[variants[0]], "semantic")}
    for v in variants:
        audits[f"physical/{v}"] = reach_audit(bundles[v], "physical")
    rep["reach_audit"] = audits
    for name, a in audits.items():
        gap = abs(a["train"]["two_hop_frac"] - a["test"]["two_hop_frac"])
        flag = "" if gap <= args.leak_tolerance else "   <-- TRAIN/TEST SHORTCUT GAP"
        print(f"[7] {name:22s} 2-hop reach of true tail: train(masked) {a['train']['two_hop_frac']:.3f} "
              f"val {a['val']['two_hop_frac']:.3f} test {a['test']['two_hop_frac']:.3f} | direct test {a['test']['direct_frac']:.3f}{flag}")
        if name == "semantic" and gap > args.leak_tolerance:
            ok = False
    rep["checks"]["semantic_leak_gap_ok"] = abs(audits["semantic"]["train"]["two_hop_frac"] - audits["semantic"]["test"]["two_hop_frac"]) <= args.leak_tolerance
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2, default=str))
    print("GAUNTLET (data-level):", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
