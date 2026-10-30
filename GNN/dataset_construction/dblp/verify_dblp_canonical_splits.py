#!/usr/bin/env python3
"""
Verify (and optionally regenerate) the DBLP Paper-Conference split.

Rebuilds the split with the logic of `paper-venue_shared_splits.py` (seed 1566911444,
paper-disjoint 70/10/20) and checks that it matches
`preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz`.

Usage
-----
  python verify_dblp_canonical_splits.py                  # verify canonical is reproducible

Exit codes: 0 = canonical verified, 2 = MISMATCH (canonical is not what the builder makes).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
_DATA = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parents[2] / "data")))
PRE = _DATA / "preprocessed"
CANONICAL = PRE / "DBLP_shared_splits_CANONICAL" / "DBLP_pc_shared_splits.npz"
LEGACY = PRE / "DBLP_shared_splits_LEGACY_unreproducible" / "DBLP_pc_shared_splits.npz"
RAW = _DATA / "raw" / "DBLP"

# Constants copied verbatim from MAGNN/paper-venue_shared_splits.py
SEED = 1566911444
TEST_PCT = 0.20
VAL_PCT = 0.10
MIN_CONF = 0
PAPER_KEEP_FRAC = 1.0


def rebuild(raw: Path) -> dict[str, np.ndarray]:
    """Re-derive the paper-disjoint splits from the raw DBLP tables."""
    rng = np.random.RandomState(SEED)
    al = pd.read_csv(raw / "author_label.txt", sep="\t",
                     names=["author_id", "label", "author_name"], header=None, encoding="utf-8")
    pa = pd.read_csv(raw / "paper_author.txt", sep="\t",
                     names=["paper_id", "author_id"], header=None, encoding="utf-8")
    pc = pd.read_csv(raw / "paper_conf.txt", sep="\t",
                     names=["paper_id", "conf_id"], header=None, encoding="utf-8")

    pa = pa[pa["author_id"].isin(set(al["author_id"]))].reset_index(drop=True)
    pc = pc[pc["paper_id"].isin(set(pa["paper_id"]))].reset_index(drop=True)
    big = pc["conf_id"].value_counts().loc[lambda x: x >= MIN_CONF].index
    pc = pc[pc["conf_id"].isin(big)].reset_index(drop=True)

    valid = set(pc["paper_id"])
    plist = np.array(sorted(valid))
    k = max(1, int(len(plist) * PAPER_KEEP_FRAC))
    keep = set(rng.choice(plist, size=k, replace=False))
    pc = pc[pc["paper_id"].isin(keep)].reset_index(drop=True)

    pc_uni = pc[["paper_id", "conf_id"]].drop_duplicates().to_numpy(dtype=np.int64)
    papers = np.array(sorted(keep))
    p_tr, p_tmp = train_test_split(papers, test_size=TEST_PCT + VAL_PCT,
                                   random_state=SEED, shuffle=True)
    p_va, p_te = train_test_split(p_tmp, test_size=TEST_PCT / (TEST_PCT + VAL_PCT),
                                  random_state=SEED, shuffle=True)

    def keep_for(P):
        return pc_uni[np.isin(pc_uni[:, 0], list(set(P.tolist())))]

    return {"train": keep_for(p_tr), "val": keep_for(p_va), "test": keep_for(p_te)}


def as_set(a):
    return set(map(tuple, a.tolist()))


def audit(z, label):
    """Report the scientific properties that make a split usable."""
    tr, va, te = (set(z[f"{s}_pos"][:, 0].tolist()) for s in ("train", "val", "test"))
    disjoint = not (tr & va or tr & te or va & te)
    negs = z["train_neg"].shape[0] / max(1, z["train_pos"].shape[0])
    print(f"  {label}")
    print(f"    positives      train/val/test = {z['train_pos'].shape[0]}/"
          f"{z['val_pos'].shape[0]}/{z['test_pos'].shape[0]}")
    print(f"    paper-disjoint = {disjoint}")
    print(f"    negatives      = {negs:.1f}x per positive"
          f"{'  <-- full 19-conference candidate pool' if negs > 15 else ''}"
          f"{'  <-- MALFORMED: only one distractor per positive' if negs <= 1.0 else ''}")
    print(f"    aux arrays     = {sorted(set(z.files) - {'train_pos','val_pos','test_pos','train_neg','val_neg','test_neg'})}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", type=Path, default=CANONICAL)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--compare-legacy", action="store_true",
                    help="also diff the canonical partition against the retired file")
    args = ap.parse_args()

    if not args.canonical.exists():
        print(f"FATAL: canonical split not found at {args.canonical}", file=sys.stderr)
        return 2

    print(f"Rebuilding the paper-disjoint splits from {args.raw} (seed {SEED}) ...")
    rep = rebuild(args.raw)
    z = np.load(args.canonical, allow_pickle=True)

    print("\n=== reproducibility ===")
    ok = True
    for s in ("train", "val", "test"):
        match = as_set(rep[s]) == as_set(z[f"{s}_pos"])
        ok &= match
        print(f"  {s:5s} rebuilt={len(rep[s]):6d}  file={len(z[f'{s}_pos']):6d}  exact set match: {match}")
    print(f"  ==> canonical split {'IS' if ok else 'IS NOT'} reproducible from the shipped builder")

    print("\n=== properties ===")
    audit(z, f"CANONICAL  {args.canonical}")

    if args.compare_legacy and LEGACY.exists():
        y = np.load(LEGACY, allow_pickle=True)
        audit(y, f"LEGACY     {LEGACY}")
        cu = set().union(*(as_set(z[f"{s}_pos"]) for s in ("train", "val", "test")))
        lu = set().union(*(as_set(y[f"{s}_pos"]) for s in ("train", "val", "test")))
        print("\n=== canonical vs legacy ===")
        print(f"  same pair universe: {cu == lu}  (|canonical|={len(cu)} |legacy|={len(lu)})")
        for s in ("train", "val", "test"):
            a, b = as_set(z[f"{s}_pos"]), as_set(y[f"{s}_pos"])
            print(f"  {s:5s} overlap {len(a & b):6d} / {len(a):6d}  ({100*len(a & b)/len(a):5.1f}%)")
        # Report BOTH directions: the contamination is asymmetric, and quoting only
        # one direction understates it.
        l2c = len(as_set(y["test_pos"]) & as_set(z["train_pos"]))
        c2l = len(as_set(z["test_pos"]) & as_set(y["train_pos"]))
        print(f"  legacy TEST pairs seen as canonical TRAIN : {l2c:4d} / {len(as_set(y['test_pos']))}")
        print(f"  canonical TEST pairs seen as legacy TRAIN : {c2l:4d} / {len(as_set(z['test_pos']))}")
        print("  ==> a model trained on one partition has already seen part of the other's"
              " test set; mixing them across models makes the comparison invalid")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
