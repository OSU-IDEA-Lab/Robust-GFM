#!/usr/bin/env python3
"""PAIN × IMDb / Freebase node classification — the four arms.

    --dataset IMDB     --arm originals|universal|invariant|augmentation   [--variant v1..v4]
    --dataset FREEBASE --arm ...                                          [--variant unchanged|exact_2|exact_3]

Inputs are the artifacts written by PAIN's preprocessors under ``$PAIN_PREPROCESSED_ROOT``:

* IMDb: ``IMDB/{v1..v4,universal}_L3.pt`` (year graph, canonical features, exhaustive rooted simple paths, L = 3);
  invariant ``IMDB_invariant_minimal/invariant_v{k}_L3.pt`` (``build_imdb_invariant_minimal.py``: each variant compiled
  into the semantic program; byte-identical path programs on v1–v4).
* Freebase: ``Freebase/<v>_L3_rel4_fan8_cap256.pt`` (sampled paths; universal = ``union_exact_2_3``); invariant
  ``Freebase_invariant/invariant_<v>_...pt`` (``preprocessing.freebase_invariant``: every variant compiled from its own
  base relations into the family program; one semantic graph and one path-program hash).

The invariant artifacts are checked before training (status PASS, distinct physical hashes, one semantic graph, one
path program). Data loading and the model are PAIN's (``pain_nc.data.load_imdb_graph``, ``PainNodeClassifier`` via
``experiments.node_classification.train.build_model`` / ``set_seed``); this runner sets the training protocol shared
with the other models:

* selection + early stopping on validation micro-F1 (ties → lower val loss);
* no wall-clock limit — stop on the epoch budget and patience only;
* augmentation: super-epochs = budget / #variants with one full-batch step per variant in a seeded order, one
  optimizer, patience and lr-plateau patience ÷ #variants;
* per-epoch atomic resume state (model, optimizer, scheduler, best state, history, RNG) so a preempted run
  continues exactly; removed when the run completes;
* inference time measured separately; RGCN result layout, read by ``summarize_arms.py``.

Model settings come from ``configs/{imdb,freebase}_nc.yaml`` (hidden 128, 2-layer shared LSTM, reversed paths, sum
aggregation, no dropout, Adam lr 1e-3, ReduceLROnPlateau on val loss, grad clip 5.0, full batch); the scripts set
``--num-layers 1``.

Outputs: <out-root>/<arm>/<DATASET>/nc/<variant>/seed_<s>/{metrics.json, scores.npz (ids, scores = test logits), history.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
PAIN = HERE
sys.path.insert(0, str(PAIN))
from pain_nc import paths  # noqa: E402
from pain_nc.config import load_config  # noqa: E402
from pain_nc.data import load_imdb_graph  # noqa: E402
from experiments.node_classification.train import build_model, set_seed  # noqa: E402
from experiments.node_classification.benchmark_imdb import preflight  # noqa: E402
from experiments.node_classification.freebase_invariant import require_invariant_contract  # noqa: E402
import pain_memfix  # noqa: E402
pain_memfix.install()                      # memory-only: checkpoint saves host-side chunk inputs (see pain_memfix.py)

VARIANTS = {"IMDB": ["v1", "v2", "v3", "v4"], "FREEBASE": ["unchanged", "exact_2", "exact_3"]}
UNIVERSAL = {"IMDB": "universal", "FREEBASE": "union_exact_2_3"}
FB_TAG = "L3_rel4_fan8_cap256"


IMDB_PROGRAM = "IMDB_invariant_minimal"
FB_PROGRAM = "Freebase_invariant"


def minimal_gate(variants):
    """Check of the IMDb invariant artifacts (build_imdb_invariant_minimal.py): PASS, 4 distinct physical graphs ->
    1 semantic graph -> 1 path program."""
    m = json.loads((paths.preprocessed(IMDB_PROGRAM) / "metadata.json").read_text())
    ok = (m.get("status") == "PASS" and m.get("semantic_equals_universal") is True
          and len(set(m["physical_graph_sha256"].values())) == 4 and len(set(m["semantic_graph_sha256"].values())) == 1
          and len(set(m["selected_path_program_sha256"].values())) == 1 and set(m["semantic_graph_sha256"]) == {"v1", "v2", "v3", "v4"})
    if not ok:
        raise ValueError("IMDb invariant artifacts missing or invalid; run build_imdb_invariant_minimal.py")


def artifact(dataset: str, arm: str, variant: str) -> tuple[Path, Path]:
    if dataset == "FREEBASE":
        if arm == "invariant":
            d = paths.preprocessed(FB_PROGRAM); return d / "shared.pt", d / f"invariant_{variant}_{FB_TAG}.pt"
        d = paths.preprocessed("Freebase"); return d / "shared.pt", d / f"{variant}_{FB_TAG}.pt"
    if arm == "invariant":
        d = paths.preprocessed(IMDB_PROGRAM); return d / "shared.pt", d / f"invariant_{variant}_L3.pt"
    d = paths.preprocessed("IMDB"); return d / "shared.pt", d / f"{variant}_L3.pt"


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a)); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_hash(m) -> str:
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def nc_metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0), "micro_f1": f1_score(y, pred, average="micro"),
            "macro_f1": f1_score(y, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["IMDB", "FREEBASE"], default="IMDB")
    ap.add_argument("--arm", choices=["originals", "universal", "invariant", "augmentation"], required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=None, help="model block + optimiser settings (default: configs/{imdb,freebase}_nc.yaml)")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--path-chunk-size", type=int, default=None,
                    help="paths per checkpointed chunk (exact chunking; one value for every arm); default = the config's")
    ap.add_argument("--path-agg", choices=["sum", "mean"], default=None, help="override model.path_aggregation (upstream option; config: sum)")
    ap.add_argument("--num-layers", type=int, default=None, help="override model.num_layers (config: 5, upstream's ZINC setting; the scripts use 1)")
    ap.add_argument("--lr", type=float, default=None, help="override training.learning_rate")
    ap.add_argument("--dropout", type=float, default=None, help="override model.dropout")
    ap.add_argument("--stop-after", type=int, default=0); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    args.config = args.config or str(PAIN / ("configs/imdb_nc.yaml" if ds == "IMDB" else "configs/freebase_nc.yaml"))
    args.out_root = args.out_root or HERE / ("results/imdb_nc" if ds == "IMDB" else "results/freebase_nc")
    args.epochs = args.epochs or (300 if ds == "IMDB" else 60)
    args.patience = args.patience or (40 if ds == "IMDB" else 20)
    cfg = load_config(args.config)
    cfg["model"]["paths_on_device"] = False
    if args.path_chunk_size:
        cfg["model"]["path_chunk_size"] = int(args.path_chunk_size)
    if args.num_layers: cfg["model"]["num_layers"] = int(args.num_layers)
    if args.path_agg: cfg["model"]["path_aggregation"] = args.path_agg
    cfg["model"]["path_length"] = 3
    if args.dropout is not None: cfg["model"]["dropout"] = float(args.dropout)
    if args.lr: cfg["training"]["learning_rate"] = float(args.lr)
    tr_cfg = cfg["training"]
    set_seed(args.seed, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arm in ("originals", "invariant"):
        assert args.variant in VARIANTS[ds], args.variant
        variants = [args.variant]
    elif args.arm == "universal":
        variants = [UNIVERSAL[ds]]
    else:
        variants = list(VARIANTS[ds])
    run_dirs = {v: args.out_root / args.arm / f"{ds}/nc" / v / f"seed_{args.seed}" for v in variants}
    if all((d / "metrics.json").exists() for d in run_dirs.values()) and not args.force:
        print(f"[skip] {args.arm} {variants} seed {args.seed}"); return
    tag = f"PAIN/{ds}/nc/{args.arm}/{'+'.join(variants)}/s{args.seed}"
    state_path = args.out_root / "_resume" / args.arm / f"{'+'.join(variants)}_seed{args.seed}.pt"

    # artifacts present, contract hashes consistent, and for the invariant arm the recorded build-time check
    # (distinct physical hashes -> one semantic graph -> one path program, status PASS)
    if ds == "FREEBASE":
        if args.arm == "invariant":
            require_invariant_contract(paths.preprocessed(FB_PROGRAM),
                                       [f"Freebase_invariant{VARIANTS[ds].index(v) + 1}" for v in variants], FB_TAG)
    elif args.arm == "invariant":
        minimal_gate(variants)
    else:
        preflight(cfg, [{"v1": "IMDb1", "v2": "IMDb2", "v3": "IMDb3", "v4": "IMDb4", "universal": "IMDb_universal"}[v] for v in variants])
    graphs, meta, shared_payload = {}, {}, None
    for v in variants:
        sp, vp = artifact(ds, args.arm, v)
        if shared_payload is None:
            shared_payload = torch.load(sp, map_location="cpu", weights_only=False)
        g = load_imdb_graph(sp, vp, reverse_paths=bool(cfg["model"].get("reverse_paths", True)), shared_payload=shared_payload)
        meta[v] = {k: g.variant_meta.get(k) for k in ("physical_graph_sha256", "semantic_graph_sha256", "selected_path_program_sha256",
                                                        "shared_contract_sha256", "mapping_mode", "variant")}
        meta[v]["num_paths"] = g.num_paths; meta[v]["artifact"] = str(vp)
        graphs[v] = g.to(device, move_paths=False, shared_from=graphs[variants[0]] if graphs else None)
        print(f"[{tag}] {v}: paths {g.num_paths:,} program {str(meta[v]['selected_path_program_sha256'])[:12]} "
              f"physical {str(meta[v]['physical_graph_sha256'])[:12]}", flush=True)
    ref = graphs[variants[0]]
    for v, g in graphs.items():
        assert g.num_edge_types == ref.num_edge_types and g.num_node_types == ref.num_node_types
        assert torch.equal(g.train_mask.cpu(), ref.train_mask.cpu()) and torch.equal(g.y.cpu(), ref.y.cpu())
    y = ref.y
    tr, va, te = (torch.nonzero(m).flatten() for m in (ref.train_mask, ref.val_mask, ref.test_mask))
    supervision_hash = sha(y.cpu().numpy(), tr.cpu().numpy(), va.cpu().numpy(), te.cpu().numpy())

    model = build_model(ref, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()); param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["learning_rate"]), weight_decay=float(tr_cfg.get("weight_decay", 0.0)))
    n_var = len(variants)
    budget = args.epochs // n_var if args.arm == "augmentation" else args.epochs
    patience = max(1, args.patience // n_var) if args.arm == "augmentation" else args.patience
    lr_pat = max(1, int(tr_cfg.get("lr_patience", 20)) // n_var) if args.arm == "augmentation" else int(tr_cfg.get("lr_patience", 20))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=float(tr_cfg.get("lr_factor", 0.5)), patience=lr_pat,
                                                       min_lr=float(tr_cfg.get("min_learning_rate", 1e-5)))
    clip = float(tr_cfg.get("gradient_clip_norm", 0.0))
    init_hash = state_hash(model)
    print(f"[{tag}] params {n_params:,} init {init_hash[:12]} supervision {supervision_hash[:12]} budget {budget} patience {patience}", flush=True)

    rng = np.random.default_rng(args.seed)
    history, best, best_state, bad, start_epoch, prior_sec, first_step_audit, peak_train = [], None, None, 0, 1, 0.0, None, 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        history, best, best_state, bad = st["history"], st["best"], st["best_state"], st["bad"]
        start_epoch, prior_sec, first_step_audit, peak_train = st["epoch"] + 1, st["elapsed"], st["first_step_audit"], st["peak_train"]
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"[{tag}] resumed at epoch {start_epoch} (param hash {state_hash(model)[:12]})", flush=True)

    def evaluate(v, idx):
        model.eval()
        with torch.no_grad():
            L = model(graphs[v])
            loss = float(F.cross_entropy(L[idx], y[idx]))
            L = L.index_select(0, idx)
        return L, loss, nc_metrics(y[idx].cpu().numpy(), L.argmax(1).cpu().numpy())

    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time() - prior_sec
    epoch = start_epoch - 1
    stopped = bool(best is not None and bad >= patience)
    for epoch in range(start_epoch, budget + 1):
        if stopped: break
        order = [variants[i] for i in rng.permutation(n_var)] if n_var > 1 else variants
        losses = []
        for v in order:
            model.train(); opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(graphs[v])[tr], y[tr])
            loss.backward()
            if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if first_step_audit is None:
                first_step_audit = {"loss": float(loss), "grad_hash": sha(*[p.grad.detach().cpu().numpy() for p in model.parameters() if p.grad is not None])}
            opt.step(); losses.append(float(loss))
        if device.type == "cuda": peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))
        ev = {v: evaluate(v, va) for v in variants}
        vals = {v: e[2] | {"loss": e[1]} for v, e in ev.items()}
        score = float(np.mean([m["micro_f1"] for m in vals.values()])); vloss = float(np.mean([m["loss"] for m in vals.values()]))
        sched.step(vloss)
        rec = {"epoch": epoch, "variant_epochs": epoch * n_var, "train_loss": float(np.mean(losses)), "val": vals,
               "val_selection_micro_f1": score, "val_loss": vloss, "lr": float(opt.param_groups[0]["lr"]),
               "param_hash": state_hash(model), "elapsed_sec": time.time() - t0}
        history.append(rec)
        print(f"[{tag}] ep {epoch:4d} loss {rec['train_loss']:.4f} val micro-F1 " + " ".join(f"{v}={m['micro_f1']:.4f}" for v, m in vals.items()) +
              f" val_loss {vloss:.4f} lr {rec['lr']:.1e} | params {rec['param_hash'][:12]} | {rec['elapsed_sec']:.0f}s", flush=True)
        if best is None or score > best["val_selection_micro_f1"] or (score == best["val_selection_micro_f1"] and vloss < best["val_loss"]):
            best, bad = rec, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        stopped = bad >= patience or bool(args.stop_after and epoch >= args.stop_after)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history, "best": best,
                    "best_state": best_state, "bad": bad, "epoch": epoch, "elapsed": time.time() - t0, "first_step_audit": first_step_audit,
                    "peak_train": peak_train, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, tmp)
        tmp.replace(state_path)
        if stopped:
            print(f"[{tag}] stop at epoch {epoch} (bad {bad}/{patience})", flush=True); break
    train_time = time.time() - t0; epochs_completed = history[-1]["epoch"]

    model.load_state_dict(best_state); ckpt_hash = state_hash(model)
    for v in variants:
        d = run_dirs[v]; d.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time(); L, _, tm = evaluate(v, te); inf_time = time.time() - t1
        peak_inf = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        np.savez(d / "scores.npz", ids=te.cpu().numpy(), scores=L.float().cpu().numpy())
        metrics = {"format_version": "pain_nc_arms_v1",
                   "model": "PAIN (exhaustive rooted simple paths, L = 3)" if ds == "IMDB" else "PAIN (sampled: rel4 / fan8 / cap256, L = 3)",
                   "arm": args.arm, "dataset": ds, "task": "nc", "variant": v, "seed": args.seed, "variants_trained_on": variants,
                   "config": {"cli": vars(args) | {"out_root": str(args.out_root)}, "pain_config": cfg},
                   "metrics": {**tm, "train_time_sec": train_time, "epochs": epochs_completed * n_var, "best_epoch": best["epoch"],
                               "parameter_count": n_params, "parameter_mib": param_bytes / 2**20,
                               "peak_training_gpu_mib": peak_train / 2**20, "peak_inference_gpu_mib": peak_inf / 2**20,
                               "inference_time_sec": inf_time, "num_paths": meta[v]["num_paths"]},
                   "val_metrics": best["val"][v], "best_val_selection_micro_f1": best["val_selection_micro_f1"],
                   "artifact_meta": meta[v], "supervision_hash": supervision_hash, "init_hash": init_hash,
                   "first_step_audit": first_step_audit, "checkpoint_hash": ckpt_hash,
                   "hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "torch": torch.__version__, "run_dir": str(d)}
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str)); (d / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{tag}] {v}: TEST micro-F1 {tm['micro_f1']:.4f} macro-F1 {tm['macro_f1']:.4f} | best ep {best['epoch']} | "
              f"ckpt {ckpt_hash[:12]} | {train_time/60:.1f} min", flush=True)
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
