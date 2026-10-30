#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for SeHGNN NC on the per-variant PHYSICAL channel bundles.

One SeHGNN over the UNION of the variants' channel vocabularies (a channel absent from a variant
is a structural zero with channel mask 0 — the mask-aware model excludes it from the per-channel
bias, LayerNorm statistics and semantic attention; with an all-one mask the model is the stock
one), trained on every variant's own channels in seeded, shuffled super-epochs on the SAME split
(supervision hashes asserted equal), selecting on the mean validation micro-F1 across variants.
Same contract as ``run_sehgnn_nc.py``; ``--epochs`` / ``--patience`` are variant-epochs divided by
the variant count. Writes one ``metrics.json`` + ``scores.npz`` per variant (aggregator layout) and
``joint_seed_<s>.json`` with the pairwise Kendall τ over per-node test logits. Resumable per
super-epoch (``last_training_state.pt``).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402
from inv_sehgnn.channels import Bundle  # noqa: E402
from inv_sehgnn.inputs import batch_inputs, build_inputs, label_channel_keys  # noqa: E402
from run_sehgnn_nc import COMMON, CONTRACT, DEFAULT_ROOT, atomic_save, build_model, cpu_state, predict, scores, step_inputs  # noqa: E402


def row_tau(a, b):
    v = [kendalltau(x, y, nan_policy="omit")[0] for x, y in zip(a, b)]
    v = [t for t in v if not np.isnan(t)]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("IMDB", "FREEBASE"), required=True)
    ap.add_argument("--flavor", choices=("full", "full_rel"), required=True)
    ap.add_argument("--variants", default=None, help="default: v1,v2,v3,v4 (IMDB) / unchanged,exact_2,exact_3 (FREEBASE)")
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-root", default="results/SeHGNN_v2")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int)
    ap.add_argument("--input-drop", type=float); ap.add_argument("--embed-size", type=int); ap.add_argument("--hidden", type=int)
    ap.add_argument("--batch-size", type=int, help="training batch (default 10000 = full batch); smaller for the 597-channel Freebase full_rel on a 32 GB GPU")
    ap.add_argument("--projection", choices=("channel", "type"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sparse-backend", choices=("torch_sparse", "torch_csr"), default="torch_sparse")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="write the resume state every N super-epochs")
    args = ap.parse_args()
    cfg = {**COMMON, **CONTRACT[args.dataset]}
    for k in ("epochs", "patience", "input_drop", "embed_size", "hidden", "projection", "batch_size"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    variants = (args.variants or {"IMDB": "v1,v2,v3,v4", "FREEBASE": "unchanged,exact_2,exact_3"}[args.dataset]).split(",")
    out_root = Path(args.output_root) / f"{args.dataset}_nc" / f"augmentation_{args.flavor}{args.tag}" / args.dataset / "nc"
    if all((out_root / v / f"seed_{args.seed}" / "metrics.json").exists() for v in variants):
        print("[skip] all variants done"); return
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[aug] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.dataset} flavor={args.flavor} variants={variants} seed={args.seed}", flush=True)
    set_deterministic(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundles = {v: Bundle(args.data_root / args.dataset / args.flavor / "physical" / v) for v in variants}
    ref = bundles[variants[0]]
    for v, b in bundles.items():
        if b.manifest["supervision_hash"] != ref.manifest["supervision_hash"]:
            raise SystemExit(f"{v}: supervision hash differs -- not a shared-split cell")
    union = sorted({k for b in bundles.values() for k in b.keys})
    union_l = sorted({k for b in bundles.values() for k in label_channel_keys(b)})
    print(f"[aug] union vocabulary {len(union)} channels + {len(union_l)} label channels: " +
          ", ".join(f"{v}={len(b.keys)}+{len(label_channel_keys(b))}" for v, b in bundles.items()), flush=True)
    inputs = {}
    data_size = {}
    dense = ref.features is not None
    for v, b in bundles.items():
        feats, lfeats, ds, mask = build_inputs(b, device, keys=union, label_keys=union_l, sparse_backend=args.sparse_backend)
        if dense:   # park the per-variant dense channels on the CPU; one variant lives on the GPU at a time
            feats = {k: x.cpu() for k, x in feats.items()}; lfeats = {k: x.cpu() for k, x in lfeats.items()}
        inputs[v] = (feats, lfeats, mask.to(device)); data_size.update(ds)

    def active(v):
        feats, lfeats, mask = inputs[v]
        if dense:
            return {k: x.to(device, non_blocking=True) for k, x in feats.items()}, {k: x.to(device) for k, x in lfeats.items()}, mask
        return feats, lfeats, mask
    tgt_key = "M" if ref.features is not None else str(ref.target_type)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = build_model(args.dataset, cfg, union, union_l, data_size, ref.num_classes, tgt_key).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(variants); super_epochs = max(1, round(cfg["epochs"] / n)); patience = max(1, round(cfg["patience"] / n))
    print(f"[aug] budget {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, patience {patience}", flush=True)
    ys = {v: torch.as_tensor(bundles[v].labels, device=device) for v in variants}
    order_rng = np.random.RandomState(args.seed)
    best, best_state, best_super, bad, trace, first, start, train_sec = -1.0, None, 0, 0, [], None, 1, 0.0
    state_path = out_root / f"joint_seed_{args.seed}_last_training_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"])
        if torch.cuda.is_available() and st.get("rng_cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        order_rng.set_state(st["order_rng"])
        best, best_state, best_super, bad, trace, first, train_sec = st["best"], st["best_state"], st["best_super"], st["bad"], st["trace"], st["first"], st["train_sec"]
        start = st["super_epoch"] + 1
        print(f"[resume] from super-epoch {start} (best {best:.4f} @ {best_super})", flush=True)
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(start, super_epochs + 1):
        model.train(); losses = []
        for v in order_rng.permutation(variants):
            feats, lfeats, mask = active(v); tp = bundles[v].split["train_idx"]
            order = np.random.default_rng(args.seed * 1000003 + se * 31 + variants.index(v)).permutation(len(tp))
            for s in range(0, len(order), cfg["batch_size"]):
                batch = torch.as_tensor(tp[order[s:s + cfg["batch_size"]]], dtype=torch.long, device=device)
                fb, lb, projected = step_inputs(model, feats, lfeats, batch)
                logits = model(batch, fb, lb, mask, projected=projected)
                loss = F.cross_entropy(logits, ys[v][batch])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if first is None:
                    first = {"variant": str(v), "loss": float(loss), "gradient_hash": gradient_hash(model)}
                optimizer.step(); losses.append(float(loss))
        vals = {}
        del feats, lfeats
        for v in variants:
            feats, lfeats, mask = active(v); vp = bundles[v].split["val_idx"]
            vals[v] = scores(predict(model, feats, lfeats, vp, cfg["eval_batch_size"], device, mask), bundles[v].labels[vp])["micro_f1"]
        score = float(np.mean(list(vals.values())))
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)), "val_micro_per_variant": vals,
                      "val_micro_mean": score, "state_hash": state_dict_hash(model.state_dict())})
        print(f"[super {se}/{super_epochs}] loss={np.mean(losses):.4f} val_micro={score:.4f} per-variant={[round(x, 4) for x in vals.values()]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_super, bad, best_state = score, se, 0, cpu_state(model)
        else:
            bad += 1
        if se % args.checkpoint_every == 0 or bad >= patience or se == super_epochs:
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "order_rng": order_rng.get_state(),
                         "best": best, "best_state": best_state, "best_super": best_super, "bad": bad, "trace": trace, "first": first,
                         "super_epoch": se, "train_sec": train_sec + (time.perf_counter() - t0)}, state_path)
        if bad >= patience:
            print("[early stop]", flush=True); break
    train_sec += time.perf_counter() - t0; peak_train = gpu_peak(device)
    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    per, sc = {}, {}
    for v in variants:
        feats, lfeats, mask = active(v); tp = bundles[v].split["test_idx"]
        sc[v] = predict(model, feats, lfeats, tp, cfg["eval_batch_size"], device, mask); per[v] = scores(sc[v], bundles[v].labels[tp])
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    taus = {f"{a}|{c}": row_tau(sc[a], sc[c]) for a, c in itertools.combinations(variants, 2)}
    ckpt = state_dict_hash(model.state_dict()); size = model_size(model)
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace), "best_epoch": best_super * n,
              "best_super_epoch": best_super, "best_val": best, "select_on": "val_micro_f1_mean", "inference_time_sec": inf_sec,
              "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"], **size}
    for v in variants:
        b = bundles[v]; run_dir = out_root / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / "scores.npz", ids=b.targets[b.split["test_idx"]], scores=sc[v])
        json_dump({"dataset": args.dataset, "task": "nc", "variant": v, "seed": args.seed, "arm": "augmentation", "flavor": args.flavor, "model": "SeHGNN",
                   "run_dir": str(run_dir), "physical_hash": b.manifest["physical_hash"], "semantic_hash": b.program_hash,
                   "supervision_hash": b.manifest["supervision_hash"], "feature_hash": b.manifest.get("feature_hash") or "identity",
                   "config": {**cfg, "union_channels": len(union), "union_label_channels": len(union_l), "variants": variants, "super_epochs": super_epochs,
                              "sparse_backend": args.sparse_backend},
                   "contract": {"select_on": "val_micro_f1_mean", "loss": "cross_entropy",
                                "features": "canonical pooled BoW (all types)" if b.features is not None else "learned per-type embeddings (featureless)"},
                   "first_step_audit": first, "checkpoint_hash": ckpt, "metrics": {**per[v], **shared}}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / f"joint_seed_{args.seed}.pt")
    if state_path.exists():
        state_path.unlink()
    json_dump({"variants": variants, "union_channels": union, "union_label_channels": union_l, "per_variant_test": per, "pairwise_kendall_tau": taus,
               "mean_tau": float(np.mean(list(taus.values()))), "shared": shared, "checkpoint_hash": ckpt}, out_root / f"joint_seed_{args.seed}.json")
    print("[done] augmentation seed=%d: " % args.seed + " ".join(f"{v}={per[v]['micro_f1']:.4f}" for v in variants) +
          f" | mean tau={np.mean(list(taus.values())):.4f} best_super={best_super}/{len(trace)}", flush=True)


if __name__ == "__main__":
    main()
