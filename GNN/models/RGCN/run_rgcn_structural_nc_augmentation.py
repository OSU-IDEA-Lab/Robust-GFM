#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for IMDb NC on the STRUCTURAL bundles.

Exists so that all four arms of the RGCN IMDb-NC cell come from ONE model
class, ONE feature table and ONE data root:

    invariant     run_rgcn_structural.py --graph semantic   (per variant)
    originals     run_rgcn_structural.py --graph physical   (per variant)
    universal     run_rgcn_structural.py --graph physical --variant universal
    augmentation  THIS FILE: one shared ConditionalRGCN + classifier trained on every
                  variant's PHYSICAL graph in seeded, shuffled super-epochs

The older ``rgcn_data_augmentation/run_IMDB_rgcn_augmentation.py`` reads the year-free
``IMDB_rgcn_v*`` format with learned embeddings and is not comparable to the other arms.

Design mirrors ``SlotGAT/run_IMDB_nc_augmentation.py``: super-epoch = one full-batch step
per variant in a seeded random order; selection on the MEAN validation micro-F1 across
variants; ``--epochs`` / ``--patience`` are in VARIANT-epochs and are divided by the
variant count so the budget matches the per-variant arms (300 / 50 -> 75 / 12 super-epochs
for four variants). Outputs one ``metrics.json`` + ``scores.npz`` per variant under
``<output-root>/IMDB/nc/<variant>/seed_<seed>/`` (the layout ``aggregate_rgcn_structural.py``
reads, so cross-variant Kendall tau comes out of the same aggregator as the other arms),
plus ``joint_seed_<seed>.json`` with the shared-model summary.
"""
from __future__ import annotations

import argparse
import copy
import json
import itertools
import socket
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from inv_rgcn.data import load_graph_data
from inv_rgcn.metrics import classification_metrics
from inv_rgcn.utils import set_deterministic, state_dict_hash, gradient_hash, json_dump
from run_rgcn_structural import DEFAULTS, build_encoder, message_graph, cfg_value
from rgcn_resource_metrics import gpu_reset, gpu_peak, model_size


def row_kendall(a: np.ndarray, b: np.ndarray) -> float:
    taus = []
    for x, y in zip(a, b):
        if np.allclose(x, y):
            taus.append(1.0); continue
        t, _ = kendalltau(x, y, nan_policy="omit")
        taus.append(0.0 if np.isnan(t) else float(t))
    return float(np.mean(taus)) if taus else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--variants", default="v1,v2,v3,v4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--norm", choices=("mean", "sum"), default="mean")
    ap.add_argument("--emb-init", dest="emb_init", type=float, default=0.1)
    for name, typ in (("in_dim", int), ("hidden_dim", int), ("out_dim", int), ("layers", int), ("bases", int),
                      ("epochs", int), ("patience", int), ("dropout", float), ("lr", float),
                      ("weight_decay", float), ("emb_reg", float)):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    args = ap.parse_args()
    print(f"[aug] start {datetime.datetime.now().isoformat(timespec='seconds')} host={socket.gethostname()} "
          f"seed={args.seed} variants={args.variants}", flush=True)
    defaults = DEFAULTS[("IMDB", "nc")]
    cfg = {k: cfg_value(args, k, defaults) for k in defaults}
    variants = [v for v in args.variants.split(",") if v]
    set_deterministic(args.seed, strict=True)
    device = torch.device(args.device)

    bundles = {v: load_graph_data(Path(args.data_root) / "IMDB" / "nc" / v).to(device) for v in variants}
    ref = bundles[variants[0]]
    # Shared supervision + features are a precondition of the joint arm; distinct physical
    # graphs are what make it an augmentation arm rather than one run repeated.
    for v, d in bundles.items():
        if d.supervision_hash != ref.supervision_hash or d.feature_hash != ref.feature_hash:
            raise SystemExit(f"{v}: supervision/feature hash differs from {variants[0]} -- not a shared-split cell")
        if d.semantic_relation_names != ref.semantic_relation_names:
            raise SystemExit(f"{v}: relation vocabulary differs")
    if len({d.physical_hash for d in bundles.values()}) != len(bundles):
        raise SystemExit("physical graphs are not all distinct -- the joint arm would be degenerate")
    print(f"[aug] data loaded {datetime.datetime.now().isoformat(timespec='seconds')}; "
          f"physical hashes {[d.physical_hash[:8] for d in bundles.values()]}", flush=True)

    encoder = build_encoder(ref, cfg, device, emb_init=args.emb_init, norm=args.norm)
    classifier = nn.Linear(cfg["out_dim"], int(ref.metadata.get("num_classes", 3))).to(device)
    model = nn.ModuleDict({"encoder": encoder, "classifier": classifier})
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    n = len(variants)
    super_epochs = max(1, int(round(cfg["epochs"] / n)))
    patience = max(1, int(round(cfg["patience"] / n)))
    print(f"[aug] budget: {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, "
          f"patience {cfg['patience']} -> {patience} super-epochs", flush=True)
    order_rng = np.random.RandomState(args.seed)

    best = -float("inf"); best_state = None; best_super = 0; bad = 0; trace = []; first_audit = {}
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(1, super_epochs + 1):
        model.train(); losses = []
        for v in order_rng.permutation(variants):
            d = bundles[str(v)]
            opt.zero_grad(set_to_none=True)
            logits = classifier(encoder(*message_graph(d, "physical")))
            loss = F.cross_entropy(logits[d.train_mask], d.y[d.train_mask])
            loss.backward()
            if not first_audit:
                first_audit = {"first_variant": str(v), "initial_logits_max_abs": float(logits.detach().abs().max()),
                               "gradient_hash": gradient_hash(model)}
            opt.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vals = []
            for v in variants:
                d = bundles[v]
                logits = classifier(encoder(*message_graph(d, "physical")))
                vals.append(classification_metrics(logits[d.val_mask], d.y[d.val_mask])["micro_f1"])
        score = float(np.mean(vals))
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)),
                      "mean_val_micro_f1": score, "val_micro_f1": dict(zip(variants, vals)),
                      "state_hash": state_dict_hash(model.state_dict())})
        if score > best + 1e-12:
            best = score; best_super = se; bad = 0; best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: break
    train_sec = time.perf_counter() - t0
    peak_train = gpu_peak(device)
    model.load_state_dict(best_state); model.eval()

    gpu_reset(device); t_inf = time.perf_counter()
    per_variant = {}; test_logits = {}
    with torch.no_grad():
        for v in variants:
            d = bundles[v]
            logits = classifier(encoder(*message_graph(d, "physical")))
            per_variant[v] = classification_metrics(logits[d.test_mask], d.y[d.test_mask])
            test_logits[v] = (torch.where(d.test_mask)[0].cpu().numpy(), logits[d.test_mask].cpu().numpy())
    inf_sec = time.perf_counter() - t_inf
    peak_inf = gpu_peak(device)

    pairwise = []
    for a, b in itertools.combinations(variants, 2):
        pairwise.append({"variant_a": a, "variant_b": b,
                         "kendall_tau": row_kendall(test_logits[a][1], test_logits[b][1]),
                         "max_abs_score_diff": float(np.max(np.abs(test_logits[a][1] - test_logits[b][1]))),
                         "prediction_agreement": float(np.mean(test_logits[a][1].argmax(1) == test_logits[b][1].argmax(1)))})
    micro = [per_variant[v]["micro_f1"] for v in variants]
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace),
              "best_epoch": best_super * n, "best_super_epoch": best_super, "best_val": best,
              "select_on": "mean_val_micro_f1", "inference_time_sec": inf_sec,
              "peak_training_gpu_mib": peak_train["peak_gpu_mib"],
              "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"],
              "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
              **model_size(model)}
    ckpt_hash = state_dict_hash(model.state_dict())
    out_root = Path(args.output_root)
    for v in variants:
        d = bundles[v]
        run_dir = out_root / "IMDB" / "nc" / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        ids, sc = test_logits[v]
        np.savez(run_dir / "scores.npz", ids=ids, scores=sc)
        json_dump({"dataset": "IMDB", "task": "nc", "variant": v, "seed": args.seed, "arm": "augmentation",
                   "config": cfg, "contract": {"norm": args.norm, "emb_init": args.emb_init, "graph": "physical",
                                                "shared_model": True, "shared_optimizer": True,
                                                "variants_jointly_trained": variants},
                   "physical_hash": d.physical_hash, "semantic_hash": d.semantic_hash,
                   "feature_hash": d.feature_hash, "supervision_hash": d.supervision_hash,
                   "checkpoint_hash": ckpt_hash, "metrics": {**per_variant[v], **shared},
                   "first_step_audit": first_audit, "run_dir": str(run_dir)}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / "IMDB" / "nc" / f"joint_seed_{args.seed}.pt")
    summary = {"seed": args.seed, "variants": variants, "per_variant_test": per_variant,
               "test_micro_f1_mean": float(np.mean(micro)), "test_micro_f1_spread": float(max(micro) - min(micro)),
               "pairwise_invariance": pairwise,
               "mean_pairwise_kendall_tau": float(np.mean([p["kendall_tau"] for p in pairwise])),
               **shared, "checkpoint_hash": ckpt_hash}
    json_dump(summary, out_root / "IMDB" / "nc" / f"joint_seed_{args.seed}.json")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"[aug] complete {datetime.datetime.now().isoformat(timespec='seconds')} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
