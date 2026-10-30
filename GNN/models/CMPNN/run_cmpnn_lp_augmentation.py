#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for CMPNN link prediction.

CMPNN twin of ``RGCN/run_rgcn_structural_lp_augmentation.py``: ONE model trained on every
variant's PHYSICAL graph in seeded, shuffled super-epochs, on the SAME shared
positives/negatives (supervision hashes asserted equal across variants), selecting on the
MEAN validation MRR across variants. Same architecture, contract and metrics as
``run_cmpnn_lp.py``. ``--epochs`` / ``--patience`` are in VARIANT-epochs and are divided
by the variant count.

Because the physical graphs may carry different relation vocabularies (WordNet: 11 vs 22
vs 537 vs 2087 named relations), the shared model is built over the UNION vocabulary and
each variant's edges are mapped into it (a relation absent from a variant simply has no
edges there) — the same shared-vocabulary device the RGCN Freebase augmentation uses.

Outputs one ``metrics.json`` + ``scores.npz`` per variant under
``<output-root>/<dataset>/<task>/<variant>/seed_<seed>/`` (aggregator layout) and a
``joint_seed_<seed>.json`` with the shared summary and pairwise Kendall tau.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "common"))

from inv_cmpnn.bundles import load_bundle  # noqa: E402
from inv_cmpnn.model import build_graph, build_model  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402
from run_cmpnn_lp import BUNDLE_ROOTS, DEFAULTS, QueryBatches, evaluate_sampled, evaluate_wordnet, loss_fn, warm_match_index  # noqa: E402


def row_tau(a: np.ndarray, b: np.ndarray) -> float:
    vals = [kendalltau(x, y, nan_policy="omit")[0] for x, y in zip(a, b)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/preprocessed")
    ap.add_argument("--dataset", choices=("IMDB", "DBLP", "WORDNET"), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", default="v1,v2,v3,v4")
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--output-root", default="results/CMPNN_structural/augmentation")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--batch-size", type=int)
    ap.add_argument("--input-dim", type=int, default=32); ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--lr", type=float, help="default: the dataset contract")
    ap.add_argument("--neg-k", type=int, default=32); ap.add_argument("--tau-candidates", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gradient-checkpoint", action="store_true")
    ap.add_argument("--chunk-edges", type=int, help="deterministic chunked reduction slice (0 = whole mode)")
    ap.add_argument("--resume", action="store_true", help="continue from last_training_state.pt (per super-epoch)")
    ap.add_argument("--max-train-batches", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(DEFAULTS[(args.dataset, args.task)])
    for k in ("epochs", "patience", "batch_size", "lr"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["remove_one_hop"] = bool(cfg["remove_one_hop"])
    if args.chunk_edges is not None:
        cfg["chunk_edges"] = args.chunk_edges
    cfg["chunk_edges"] = cfg.get("chunk_edges") or None
    variants = [v for v in args.variants.split(",") if v]
    out_root = Path(args.output_root)
    if all((out_root / args.dataset / args.task / v / f"seed_{args.seed}" / "metrics.json").exists() for v in variants):
        print("[skip] every variant already has metrics.json"); return
    print(f"[aug] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.dataset}/{args.task} variants={variants} seed={args.seed}", flush=True)

    set_deterministic(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    bundles = {v: load_bundle(Path(args.data_root) / BUNDLE_ROOTS[(args.dataset, args.task)] / v) for v in variants}
    ref = bundles[variants[0]]
    for v, b in bundles.items():
        if b.supervision_hash() != ref.supervision_hash():
            raise SystemExit(f"{v}: supervision hash differs from {variants[0]} -- not a shared-split cell")
        if b.alias_hash() != ref.alias_hash():
            raise SystemExit(f"{v}: alias map differs")
    # union physical vocabulary
    union = []
    for b in bundles.values():
        for n in b.physical_relation_names:
            if n not in union:
                union.append(n)
    R = len(union)
    graphs = {}
    for v, b in bundles.items():
        remap = torch.tensor([union.index(n) for n in b.physical_relation_names], dtype=torch.int64)
        e = b.physical_edges.clone(); e[:, 2] = remap[e[:, 2]]
        graphs[v] = build_graph(e, b.num_nodes, R, device)
    wn = args.dataset == "WORDNET"

    def positives_union(b, split):
        p = b.positives(split, "physical").clone()
        remap = torch.tensor([union.index(n) for n in b.physical_relation_names], dtype=torch.int64)
        p[:, 1] = remap[p[:, 1]]
        return p

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = build_model(R, ref.alias.to(device), input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.layers, gradient_checkpoint=args.gradient_checkpoint,
                        aggregate_func=cfg["aggregate_func"], remove_one_hop=cfg["remove_one_hop"], chunk_edges=cfg["chunk_edges"],
                        label_derived_relations=[union.index(n) for n in ref.label_derived_relations if n in union]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    class UnionBatches(QueryBatches):
        def __init__(self, b, split, shuffle):
            super().__init__(b, "physical", split, cfg["batch_size"], args.seed, args.neg_k, wn, shuffle)
            self.pos = positives_union(b, split)

    train_b = {v: UnionBatches(bundles[v], "train", True) for v in variants}
    n = len(variants)
    super_epochs = max(1, int(round(cfg["epochs"] / n)))
    patience = max(1, int(round(cfg["patience"] / n)))
    print(f"[aug] budget: {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, patience {patience}; union vocabulary {R} relations", flush=True)

    def run_eval(v, split):
        b = bundles[v]
        if wn:
            # evaluate_wordnet reads b.positives(split, 'physical'); relation ids must be union ids
            saved = b.physical_relation_names
            b.physical_relation_names = union
            b.metadata = {**b.metadata, "physical_task_relation_ids": [union.index(saved[i]) for i in b.metadata["physical_task_relation_ids"]]}
            m, ranks, sc = evaluate_wordnet(model, graphs[v], b, "physical", split, device, cfg["batch_size"], args.tau_candidates, args.seed)
            b.physical_relation_names = saved
            return m, sc
        qb = UnionBatches(b, split, False)
        return evaluate_sampled(model, graphs[v], qb, device)

    order_rng = np.random.RandomState(args.seed)
    best, best_state, best_super, bad, trace, first_audit, resumes = -float("inf"), None, 0, 0, [], None, []
    start_se, train_sec = 1, 0.0
    state_path = out_root / args.dataset / args.task / f"last_training_state_seed_{args.seed}.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        for v in variants:                                        # BEFORE the RNG restore (see run_cmpnn_lp.warm_match_index)
            warm_match_index(model, graphs[v], train_b[v].pos, device)
        torch.set_rng_state(st["rng_cpu"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        order_rng.set_state(st["order_rng"])
        for v, ep in st["batch_epochs"].items():
            train_b[v].epoch = ep
        best, best_state, best_super, bad, trace, first_audit = st["best"], st["best_state"], st["best_super"], st["bad"], st["trace"], st["first_audit"]
        start_se, train_sec = st["super_epoch"] + 1, st["train_sec"]
        resumes = list(st.get("resumes", [])) + [start_se]
        print(f"[resume] from super-epoch {start_se}", flush=True)
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(start_se, super_epochs + 1):
        model.train(); losses = []
        for v in order_rng.permutation(variants):
            for bi, (idx, h, t, r) in enumerate(train_b[v]):
                if args.max_train_batches and bi >= args.max_train_batches:
                    break
                scores = model(graphs[v], h.to(device), t.to(device), r.to(device), all_loss=torch.tensor(0.0))
                loss = loss_fn(scores)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if first_audit is None:
                    first_audit = {"variant": str(v), "loss": float(loss), "gradient_hash": gradient_hash(model)}
                optimizer.step(); losses.append(float(loss))
        vals = [run_eval(v, "val")[0]["mrr"] for v in variants]
        score = float(np.mean(vals))
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)),
                      "val_mrr_per_variant": dict(zip(variants, vals)), "val_mrr_mean": score,
                      "state_hash": state_dict_hash(model.state_dict())})
        print(f"[super {se}/{super_epochs}] loss={np.mean(losses):.4f} val_mrr={score:.4f} per-variant={[round(x, 4) for x in vals]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_super, bad, best_state = score, se, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(), "order_rng": order_rng.get_state(),
                    "batch_epochs": {v: train_b[v].epoch for v in variants}, "best": best, "best_state": best_state,
                    "best_super": best_super, "bad": bad, "trace": trace, "first_audit": first_audit,
                    "super_epoch": se, "train_sec": train_sec + (time.perf_counter() - t0), "resumes": resumes}, state_path)
        if bad >= patience:
            print("[early stop]", flush=True); break
    train_sec += time.perf_counter() - t0; peak_train = gpu_peak(device)

    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    per_variant, score_rows = {}, {}
    for v in variants:
        m, sc = run_eval(v, "test")
        per_variant[v] = m; score_rows[v] = sc
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    taus = {f"{a}|{b}": row_tau(score_rows[a], score_rows[b]) for a, b in itertools.combinations(variants, 2)}
    size = model_size(model); ckpt = state_dict_hash(model.state_dict())
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace), "resumed_at_super_epochs": resumes,
              "best_epoch": best_super * n, "best_super_epoch": best_super, "best_val_score": best, "select_on": "val_mrr",
              "inference_time_sec": inf_sec, "peak_training_gpu_mib": peak_train["peak_gpu_mib"],
              "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"], **size}
    for v in variants:
        b = bundles[v]
        run_dir = out_root / args.dataset / args.task / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        tp = b.positives("test", "physical")
        np.savez(run_dir / "scores.npz", queries=tp[:, 0].numpy(), positive_tails=tp[:, 2].numpy(), scores=score_rows[v])
        s = b.summary()
        json_dump({"dataset": args.dataset, "task": args.task, "variant": v, "seed": args.seed, "graph": "physical", "arm": "augmentation",
                   "model": "CMPNN", "run_dir": str(run_dir), "physical_hash": s["physical_hash"], "semantic_hash": s["semantic_hash"],
                   "supervision_hash": s["supervision_hash"], "feature_hash": None, "alias_hash": s["alias_hash"],
                   "config": {**cfg, "input_dim": args.input_dim, "hidden_dim": args.hidden_dim, "layers": args.layers,
                              "union_relations": R, "variants": variants, "super_epochs": super_epochs},
                   "contract": {"loss": "neg_logsigmoid", "select_on": "val_mrr", "message_func": "distmult",
                                "aggregate_func": cfg["aggregate_func"], "dependent": False, "remove_one_hop": cfg["remove_one_hop"],
                                "eval_protocol": per_variant[v].get("eval_protocol", "sampled")},
                   "first_step_audit": first_audit, "checkpoint_hash": ckpt,
                   "metrics": {**per_variant[v], **shared}}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / args.dataset / args.task / f"joint_seed_{args.seed}.pt")
    if state_path.exists():
        state_path.unlink()
    json_dump({"variants": variants, "per_variant_test": per_variant, "pairwise_kendall_tau": taus,
               "mean_tau": float(np.mean(list(taus.values()))) if taus else None, "shared": shared, "checkpoint_hash": ckpt},
              out_root / args.dataset / args.task / f"joint_seed_{args.seed}.json")
    print(f"[done] augmentation {args.dataset}/{args.task} seed={args.seed}: " +
          " ".join(f"{v}={per_variant[v]['mrr']:.4f}" for v in variants) +
          f" | mean tau={np.mean(list(taus.values())) if taus else float('nan'):.4f} best_super={best_super}/{len(trace)}", flush=True)


if __name__ == "__main__":
    main()
