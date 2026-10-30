#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for IMDb LP on the STRUCTURAL bundles.

LP twin of ``run_rgcn_structural_nc_augmentation.py``: one shared
ConditionalRGCN encoder + LinkPredictor trained on every variant's PHYSICAL graph in
seeded, shuffled super-epochs, on the SAME shared positives/negatives (the bundles'
``task_data`` is asserted identical across variants), selecting on the MEAN validation MRR
across variants. Same model class, same canonical features and the same contract flags
(``--decoder``, ``--loss``, ``--select-on val_mrr``, sampled protocol) as the invariant /
originals / universal arms produced by ``run_rgcn_structural.py --graph``.

``--epochs`` / ``--patience`` are in VARIANT-epochs and are divided by the variant count
(1000 / 200 -> 250 / 50 super-epochs for four variants, 500 / 100 for two).

Outputs one ``metrics.json`` + ``scores.npz`` per variant under
``<output-root>/IMDB/<task>/<variant>/seed_<seed>/`` (aggregator layout) and a
``joint_seed_<seed>.json`` with the shared-model summary and pairwise Kendall tau over the
per-query candidate score rows.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from inv_rgcn.data import load_graph_data
from inv_rgcn.model import LinkPredictor
from inv_rgcn.metrics import binary_metrics, ranking_metrics
from inv_rgcn.utils import set_deterministic, state_dict_hash, gradient_hash, json_dump
from run_rgcn_structural import (DEFAULTS, build_encoder, message_graph, cfg_value,
                                 flatten_negatives, lp_loss, val_mrr, score_ranking)
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
    ap.add_argument("--task", required=True, choices=("movie_director", "movie_year", "movie_link"))
    ap.add_argument("--variants", default="v1,v2,v3,v4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--norm", choices=("mean", "sum"), default="mean")
    ap.add_argument("--emb-init", dest="emb_init", type=float, default=0.1)
    ap.add_argument("--decoder", choices=("dot", "distmult"), default="dot")
    ap.add_argument("--loss", choices=("pairwise", "bce"), default="pairwise")
    ap.add_argument("--select-on", dest="select_on", choices=("val_mrr", "val_loss"), default="val_mrr")
    for name, typ in (("in_dim", int), ("hidden_dim", int), ("out_dim", int), ("layers", int), ("bases", int),
                      ("epochs", int), ("patience", int), ("dropout", float), ("lr", float),
                      ("weight_decay", float), ("emb_reg", float)):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    args = ap.parse_args()
    args.graph = "physical"   # every variant is presented through its own raw graph
    args.eval_protocol = "sampled"
    print(f"[aug-lp] start {datetime.datetime.now().isoformat(timespec='seconds')} host={socket.gethostname()} "
          f"task={args.task} seed={args.seed} variants={args.variants}", flush=True)
    defaults = DEFAULTS[("IMDB", args.task)]
    cfg = {k: cfg_value(args, k, defaults) for k in defaults}
    variants = [v for v in args.variants.split(",") if v]
    set_deterministic(args.seed, strict=True)
    device = torch.device(args.device)

    bundles = {v: load_graph_data(Path(args.data_root) / "IMDB" / args.task / v).to(device) for v in variants}
    ref = bundles[variants[0]]
    for v, d in bundles.items():
        if d.supervision_hash != ref.supervision_hash or d.feature_hash != ref.feature_hash:
            raise SystemExit(f"{v}: supervision/feature hash differs from {variants[0]} -- not a shared-split cell")
        for key in ("train_pos", "train_neg", "val_pos", "val_neg", "test_pos", "test_neg"):
            if not torch.equal(d.task_data[key].cpu(), ref.task_data[key].cpu()):
                raise SystemExit(f"{v}: task_data[{key}] differs -- positives/negatives must be shared")
    if len({d.physical_hash for d in bundles.values()}) != len(bundles):
        raise SystemExit("physical graphs are not all distinct -- the joint arm would be degenerate")
    print(f"[aug-lp] data loaded {datetime.datetime.now().isoformat(timespec='seconds')}; "
          f"physical hashes {[d.physical_hash[:8] for d in bundles.values()]}; x={'present' if ref.x is not None else 'none'}", flush=True)

    td = {k: v.to(device) if torch.is_tensor(v) else v for k, v in ref.task_data.items()}
    train_pos, train_neg = td["train_pos"], td["train_neg"]
    val_pos, val_neg = td["val_pos"], td["val_neg"]
    test_pos, test_neg = td["test_pos"], td["test_neg"]
    flat_train_neg, _ = flatten_negatives(train_pos, train_neg)
    flat_val_neg, _ = flatten_negatives(val_pos, val_neg)

    encoder = build_encoder(ref, cfg, device, emb_init=args.emb_init, norm=args.norm)
    predictor = LinkPredictor(cfg["out_dim"], mode=args.decoder).to(device)
    model = nn.ModuleDict({"encoder": encoder, "predictor": predictor})
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    n = len(variants)
    super_epochs = max(1, int(round(cfg["epochs"] / n)))
    patience = max(1, int(round(cfg["patience"] / n)))
    print(f"[aug-lp] budget: {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, "
          f"patience {cfg['patience']} -> {patience} super-epochs", flush=True)
    order_rng = np.random.RandomState(args.seed)

    best = -float("inf"); best_state = None; best_super = 0; bad = 0; trace = []; first_audit = {}
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(1, super_epochs + 1):
        model.train(); losses = []
        for v in order_rng.permutation(variants):
            d = bundles[str(v)]
            opt.zero_grad(set_to_none=True)
            h = encoder(*message_graph(d, "physical"))
            pos_log = predictor(h, train_pos); neg_log = predictor(h, flat_train_neg)
            loss = lp_loss(pos_log, neg_log, args.loss)
            if cfg["emb_reg"]:
                loss = loss + cfg["emb_reg"] * h.pow(2).mean()
            loss.backward()
            if not first_audit:
                first_audit = {"first_variant": str(v),
                               "initial_logits_max_abs": float(torch.cat([pos_log.detach(), neg_log.detach()]).abs().max()),
                               "gradient_hash": gradient_hash(model)}
            opt.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vals = []; vls = []
            for v in variants:
                h = encoder(*message_graph(bundles[v], "physical"))
                vals.append(float(val_mrr(predictor, h, val_pos, val_neg)))
                vls.append(float(lp_loss(predictor(h, val_pos), predictor(h, flat_val_neg), args.loss)))
        score = float(np.mean(vals)) if args.select_on == "val_mrr" else -float(np.mean(vls))
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)),
                      "mean_val_mrr": float(np.mean(vals)), "mean_val_loss": float(np.mean(vls)),
                      "val_mrr": dict(zip(variants, vals)), "state_hash": state_dict_hash(model.state_dict())})
        if score > best + 1e-12:
            best = score; best_super = se; bad = 0; best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: break
    train_sec = time.perf_counter() - t0
    peak_train = gpu_peak(device)
    model.load_state_dict(best_state); model.eval()

    gpu_reset(device); t_inf = time.perf_counter()
    per_variant = {}; cand = {}
    for v in variants:
        h, pos_log, neg_log = score_ranking(encoder, predictor, bundles[v], test_pos, test_neg, device, args)
        p = pos_log.cpu().numpy(); ng = neg_log.cpu().numpy()
        labels = np.concatenate([np.ones(len(p)), np.zeros(ng.size)])
        m = binary_metrics(np.concatenate([p, ng.reshape(-1)]), labels); m.update(ranking_metrics(p, ng))
        m.update({"eval_protocol": "sampled", "num_candidates": int(ng.shape[1]) + 1})
        per_variant[v] = m
        cand[v] = np.concatenate([p[:, None], ng], axis=1)
    inf_sec = time.perf_counter() - t_inf
    peak_inf = gpu_peak(device)

    pairwise = []
    for a, b in itertools.combinations(variants, 2):
        pairwise.append({"variant_a": a, "variant_b": b, "kendall_tau": row_kendall(cand[a], cand[b]),
                         "max_abs_score_diff": float(np.max(np.abs(cand[a] - cand[b])))})
    mrr = [per_variant[v]["mrr"] for v in variants]
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace),
              "best_epoch": best_super * n, "best_super_epoch": best_super, "best_val_score": best,
              "select_on": args.select_on, "inference_time_sec": inf_sec,
              "peak_training_gpu_mib": peak_train["peak_gpu_mib"],
              "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"],
              "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
              **model_size(model)}
    ckpt_hash = state_dict_hash(model.state_dict())
    out_root = Path(args.output_root)
    contract = {"norm": args.norm, "eval_protocol": "sampled", "decoder": args.decoder, "loss": args.loss,
                "select_on": args.select_on, "emb_init": args.emb_init, "graph": "physical",
                "shared_model": True, "shared_optimizer": True, "variants_jointly_trained": variants}
    for v in variants:
        d = bundles[v]
        run_dir = out_root / "IMDB" / args.task / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / "scores.npz", queries=test_pos[:, 0].cpu().numpy(),
                 positive_tails=test_pos[:, 1].cpu().numpy(), scores=cand[v])
        json_dump({"dataset": "IMDB", "task": args.task, "variant": v, "seed": args.seed, "arm": "augmentation",
                   "config": cfg, "contract": contract,
                   "physical_hash": d.physical_hash, "semantic_hash": d.semantic_hash,
                   "feature_hash": d.feature_hash, "supervision_hash": d.supervision_hash,
                   "checkpoint_hash": ckpt_hash, "metrics": {**per_variant[v], **shared},
                   "first_step_audit": first_audit, "run_dir": str(run_dir)}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / "IMDB" / args.task / f"joint_seed_{args.seed}.pt")
    summary = {"task": args.task, "seed": args.seed, "variants": variants, "per_variant_test": per_variant,
               "test_mrr_mean": float(np.mean(mrr)), "test_mrr_spread": float(max(mrr) - min(mrr)),
               "pairwise_invariance": pairwise,
               "mean_pairwise_kendall_tau": float(np.mean([p["kendall_tau"] for p in pairwise])),
               **shared, "checkpoint_hash": ckpt_hash}
    json_dump(summary, out_root / "IMDB" / args.task / f"joint_seed_{args.seed}.json")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"[aug-lp] complete {datetime.datetime.now().isoformat(timespec='seconds')} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
