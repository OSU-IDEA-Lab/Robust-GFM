#!/usr/bin/env python3
"""Gate a Freebase SlotGAT results root: metric coverage + cross-variant invariance.

Fields live under ``training.*`` and ``model.*`` in result.json, not at the top
level -- a gate that reads the top level silently sees None for everything.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seed", default="*")
    ap.add_argument("--expect-mode", default="exact_union")
    ap.add_argument("--expect-select", default="val_micro_f1")
    ap.add_argument("--epoch-ceiling", type=int, default=0,
                    help="If >0, fail when best_epoch equals a run that used the whole budget.")
    ap.add_argument("--require-invariance", action="store_true")
    args = ap.parse_args()

    pattern = f"{args.results_dir}/*/*/seed_{args.seed}/result.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"GATE FAILED: no result.json under {pattern}")
        return 1

    fail: list[str] = []
    by_seed: dict[str, dict[str, float]] = {}
    for path in paths:
        r = json.loads(open(path).read())
        tr, md, mem = r["training"], r["model"], r["memory"]
        variant, seed = r["variant"], str(r["seed"])
        micro = r["metrics"]["test"]["micro_f1"]
        peak = mem.get("training_peak_allocated_bytes")
        by_seed.setdefault(seed, {})[variant] = micro
        peak_txt = f"{peak / 2**30:.2f}GiB" if peak else "MISSING"
        print(f"  seed {seed} {variant}: micro_f1={micro:.6f} "
              f"best_epoch={tr.get('best_epoch')}/{tr.get('epochs_trained')} "
              f"mode={tr.get('message_mode')} select={tr.get('selection_metric')} "
              f"params={md.get('parameter_count'):,} peak_train={peak_txt}")
        tag = f"seed {seed} {variant}"
        if tr.get("message_mode") != args.expect_mode:
            fail.append(f"{tag}: message_mode={tr.get('message_mode')!r}")
        if tr.get("selection_metric") != args.expect_select:
            fail.append(f"{tag}: selection_metric={tr.get('selection_metric')!r}")
        if not peak:
            fail.append(f"{tag}: missing peak training GPU")
        if not md.get("parameter_count"):
            fail.append(f"{tag}: missing parameter_count")
        if args.epoch_ceiling and tr.get("best_epoch") == args.epoch_ceiling:
            fail.append(f"{tag}: best_epoch hit the {args.epoch_ceiling}-epoch ceiling")

    if args.require_invariance:
        for seed, vals in sorted(by_seed.items()):
            uniq = set(vals.values())
            print(f"  seed {seed}: invariant across {len(vals)} variants = {len(uniq) == 1}")
            if len(uniq) != 1:
                fail.append(f"seed {seed} variants disagree: {vals}")

    if fail:
        print("\nGATE FAILED:")
        for item in fail:
            print("  -", item)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
