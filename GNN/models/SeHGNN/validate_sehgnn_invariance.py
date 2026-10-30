#!/usr/bin/env python3
"""Data-level gauntlet for the invariant SeHGNN bundles.

For every flavour: physical hashes must DIFFER across variants, the channel program hash,
supervision hash and feature hash must be EQUAL, and one CPU forward/backward with a fixed seed
must give identical logits / loss / gradient hashes. Also reports whether the invariant program
equals the universal (union-graph) program of the same flavour — the recoverability caveat.
Exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from inv_sehgnn.channels import Bundle  # noqa: E402
from inv_sehgnn.inputs import batch_inputs, build_inputs  # noqa: E402
from run_sehgnn_nc import COMMON, CONTRACT, DEFAULT_ROOT, build_model  # noqa: E402


def first_step(b: Bundle, dataset: str, seed: int, backend: str):
    set_deterministic(seed)
    feats, lfeats, ds, _ = build_inputs(b, torch.device("cpu"), sparse_backend=backend)
    cfg = {**COMMON, **CONTRACT[dataset]}
    torch.manual_seed(seed)
    model = build_model(dataset, cfg, list(feats), list(lfeats), ds, b.num_classes, "M" if b.features is not None else str(b.target_type))
    tp = b.split["train_idx"][:512]
    idx = torch.as_tensor(tp, dtype=torch.long)
    fb, lb = batch_inputs(feats, lfeats, idx)
    model.train(); logits = model(idx, fb, lb, None)
    loss = F.cross_entropy(logits, torch.as_tensor(b.labels[tp])); loss.backward()
    return {"init_hash": state_dict_hash(model.state_dict()), "logits_hash": hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest(),
            "loss": float(loss), "gradient_hash": gradient_hash(model)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("IMDB", "FREEBASE"), required=True)
    ap.add_argument("--flavors", default="full")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--universal", default=None)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--sparse-backend", default="torch_sparse")
    ap.add_argument("--skip-first-step", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    variants = (args.variants or {"IMDB": "v1,v2,v3,v4", "FREEBASE": "unchanged,exact_2,exact_3"}[args.dataset]).split(",")
    universal = args.universal or {"IMDB": "universal", "FREEBASE": "union_exact_2_3"}[args.dataset]
    report, ok = {}, True
    for flavor in args.flavors.split(","):
        bundles = {}
        for v in variants:
            d = args.data_root / args.dataset / flavor / "semantic" / v
            if not (d / "manifest.json").exists():
                print(f"[{flavor}] missing bundle {d}"); ok = False; continue
            bundles[v] = Bundle(d)
        if len(bundles) < 2:
            continue
        ph = {v: b.manifest["physical_hash"] for v, b in bundles.items()}
        prog = {v: b.program_hash for v, b in bundles.items()}
        sup = {v: b.manifest["supervision_hash"] for v, b in bundles.items()}
        fh = {v: b.manifest.get("feature_hash") for v, b in bundles.items()}
        rel = {v: b.manifest.get("semantic_hash") for v, b in bundles.items()}
        checks = {"physical_hashes_differ": len(set(ph.values())) == len(ph), "program_hash_equal": len(set(prog.values())) == 1,
                  "relation_program_equal": len(set(rel.values())) == 1, "supervision_equal": len(set(sup.values())) == 1,
                  "feature_equal": len(set(fh.values())) == 1,
                  "channel_counts": {v: [len(b.keys), len(b.label_channels)] for v, b in bundles.items()}}
        ud = args.data_root / args.dataset / flavor / "physical" / universal
        if (ud / "manifest.json").exists():
            ub = Bundle(ud); checks["universal_program_equal_to_invariant"] = ub.program_hash == list(prog.values())[0]
            checks["universal_channels"] = [len(ub.keys), len(ub.label_channels)]
        if not args.skip_first_step:
            fs = {v: first_step(b, args.dataset, args.seed, args.sparse_backend) for v, b in bundles.items()}
            checks["first_step_logits_equal"] = len({x["logits_hash"] for x in fs.values()}) == 1
            checks["first_step_gradient_equal"] = len({x["gradient_hash"] for x in fs.values()}) == 1
            checks["first_step_loss"] = {v: x["loss"] for v, x in fs.items()}
        flavor_ok = all(checks[k] for k in ("physical_hashes_differ", "program_hash_equal", "supervision_equal", "feature_equal") if k in checks) and \
            checks.get("first_step_logits_equal", True) and checks.get("first_step_gradient_equal", True)
        ok &= flavor_ok
        report[flavor] = {"ok": flavor_ok, "physical": ph, "program": prog, "supervision": sup, **checks}
        print(f"[{flavor}] {'PASS' if flavor_ok else 'FAIL'}: " + ", ".join(f"{k}={v}" for k, v in checks.items() if not isinstance(v, dict)), flush=True)
    if args.out:
        json_dump(report, args.out)
    print("[gauntlet]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
