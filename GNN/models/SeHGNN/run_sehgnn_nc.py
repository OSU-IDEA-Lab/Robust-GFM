#!/usr/bin/env python3
"""One SeHGNN runner for node classification on the compiled channel bundles (three per-variant arms).

    --arm invariant  --variant v          bundle <data-root>/<DATASET>/<flavor>/semantic/<v>
    --arm originals  --variant v          bundle <data-root>/<DATASET>/<flavor>/physical/<v>
    --arm universal  --variant universal|union_exact_2_3   ... physical/<union graph>
    --flavor {full,full_rel}            (IMDb: full, Freebase: full_rel)

Same model (repository SeHGNN; mask-aware copy with an all-one mask) and contract for every arm:
IMDb embed/hidden 512, ≤200 epochs, patience 50, dense channels +
label channels; Freebase embed/hidden 256, ≤100 epochs, patience 30, sparse selectors over
learned per-type embeddings. Adam 5e-3 / wd 1e-4, batch 10 000, dropout .5, 2 projection + 4
task layers, early stopping and selection on VALIDATION MICRO-F1. Writes the RGCN run layout
(``<output-root>/<DATASET>_nc/<arm>_<flavor>/<DATASET>/nc/<variant>/seed_<s>/{metrics.json,
scores.npz, trace.json, checkpoint.pt}``) so ``RGCN/aggregate_rgcn_structural.py`` works unchanged.
Resumable per epoch (``last_training_state.pt``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402
from inv_sehgnn.channels import Bundle  # noqa: E402
from inv_sehgnn.inputs import batch_inputs, build_inputs  # noqa: E402
from inv_sehgnn.model import SeHGNN  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data"))) / "preprocessed" / "sehgnn_v2"
CONTRACT = {
    "IMDB": dict(embed_size=512, hidden=512, epochs=200, patience=50, projection="type"),
    "FREEBASE": dict(embed_size=256, hidden=256, epochs=100, patience=30),
}
COMMON = dict(dropout=0.5, input_drop=0.0, att_drop=0.0, n_fp_layers=2, n_task_layers=4, act="none", lr=5e-3,
              weight_decay=1e-4, batch_size=10000, eval_batch_size=1024, select_on="val_micro_f1")
MODEL_DATASET = {"IMDB": "IMDB", "FREEBASE": "Freebase"}


def build_model(dataset: str, cfg: dict, feat_keys, label_keys, data_size, num_classes: int, tgt_key: str) -> SeHGNN:
    proj_key = None
    if cfg.get("projection", "channel") == "type" and dataset == "IMDB":
        # one input projection per SOURCE NODE TYPE (last letter of the key) instead of per channel
        proj_key = {k: k[-1] for k in feat_keys}
        data_size = {k[-1]: data_size[k] for k in feat_keys}
    return SeHGNN(MODEL_DATASET[dataset], cfg["embed_size"], cfg["hidden"], num_classes, feat_keys, label_keys, tgt_key,
                  cfg["dropout"], cfg["input_drop"], cfg["att_drop"], cfg["n_fp_layers"], cfg["n_task_layers"], cfg["act"],
                  residual=False, data_size=data_size, num_heads=1, proj_key=proj_key)


def is_dense(feats) -> bool:
    return all(isinstance(v, torch.Tensor) for v in feats.values())


def step_inputs(model, feats, label_feats, idx):
    """Batch inputs for one forward. Dense (IMDb): project the WHOLE channel table with the model's
    per-channel embeddings and index projected rows -- autograd then keeps a reference to the resident
    table instead of a per-batch copy of 143 x 3008-d rows (which OOMed the V100). Sparse (Freebase):
    row-slice the selectors."""
    if is_dense(feats):
        proj = model.project(feats)
        return {k: v[idx] for k, v in proj.items()}, {k: v[idx] for k, v in label_feats.items()}, True
    fb, lb = batch_inputs(feats, label_feats, idx)
    return fb, lb, False


@torch.no_grad()
def predict(model, feats, label_feats, positions: np.ndarray, bs: int, device, mask=None):
    model.eval(); out = []
    proj = model.project(feats) if is_dense(feats) else None
    for s in range(0, len(positions), bs):
        idx = torch.as_tensor(positions[s:s + bs], dtype=torch.long, device=device)
        if proj is not None:
            fb = {k: v[idx] for k, v in proj.items()}; lb = {k: v[idx] for k, v in label_feats.items()}
            out.append(model(idx, fb, lb, mask, projected=True).float().cpu())
        else:
            fb, lb = batch_inputs(feats, label_feats, idx)
            out.append(model(idx, fb, lb, mask).float().cpu())
    return torch.cat(out).numpy()


def scores(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {"micro_f1": float(f1_score(y, pred, average="micro")), "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean())}


def cpu_state(model) -> dict:
    """Best-checkpoint copy parked on the CPU (the 143-channel IMDb model is ~0.9 GB)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp"); torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("IMDB", "FREEBASE"), required=True)
    ap.add_argument("--arm", choices=("invariant", "originals", "universal"), required=True)
    ap.add_argument("--flavor", choices=("full", "full_rel"), required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-root", default="results/SeHGNN_v2")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int)
    ap.add_argument("--input-drop", type=float); ap.add_argument("--embed-size", type=int); ap.add_argument("--hidden", type=int)
    ap.add_argument("--batch-size", type=int, help="training batch (default 10000 = full batch); smaller for the 597-channel Freebase full_rel on a 32 GB GPU")
    ap.add_argument("--projection", choices=("channel", "type"), help="IMDb input projection: per channel (stock IMDb) or per source type (stock Freebase)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sparse-backend", choices=("torch_sparse", "torch_csr"), default="torch_sparse")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="", help="suffix on the arm directory (smoke tests)")
    ap.add_argument("--stop-after-epoch", type=int, default=0, help="smoke: exit after saving the resume state of this epoch")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="write the resume state every N epochs (the 143-channel IMDb state is 2.7 GB)")
    args = ap.parse_args()
    cfg = {**COMMON, **CONTRACT[args.dataset]}
    for k in ("epochs", "patience", "input_drop", "embed_size", "hidden", "projection", "batch_size"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    graph = "semantic" if args.arm == "invariant" else "physical"
    bundle_dir = args.data_root / args.dataset / args.flavor / graph / args.variant
    run_dir = Path(args.output_root) / f"{args.dataset}_nc" / f"{args.arm}_{args.flavor}{args.tag}" / args.dataset / "nc" / args.variant / f"seed_{args.seed}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir}"); return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.dataset} arm={args.arm} flavor={args.flavor} variant={args.variant} seed={args.seed}", flush=True)
    set_deterministic(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    b = Bundle(bundle_dir)
    t_load = time.perf_counter()
    feats, label_feats, data_size, _ = build_inputs(b, device, sparse_backend=args.sparse_backend)
    tgt_key = "M" if b.features is not None else str(b.target_type)
    print(f"[data] {bundle_dir}: {len(feats)} channels + {len(label_feats)} label channels, program {b.program_hash[:12]}, "
          f"physical {str(b.manifest['physical_hash'])[:12]}, supervision {b.manifest['supervision_hash'][:12]}, targets {len(b.targets)} ({time.perf_counter()-t_load:.0f}s)", flush=True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = build_model(args.dataset, cfg, list(feats), list(label_feats), data_size, b.num_classes, tgt_key).to(device)
    init_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_pos, val_pos, test_pos = b.split["train_idx"], b.split["val_idx"], b.split["test_idx"]
    y = b.labels; y_dev = torch.as_tensor(y, device=device)

    state_path = run_dir / "last_training_state.pt"
    trace, best, best_state, best_epoch, bad, start_epoch, train_sec, first_step = [], -1.0, None, 0, 0, 0, 0.0, None
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"])
        if torch.cuda.is_available() and st.get("rng_cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        trace, best, best_state, best_epoch, bad = st["trace"], st["best"], st["best_state"], st["best_epoch"], st["bad"]
        start_epoch, train_sec, first_step = st["epoch"] + 1, st["train_sec"], st["first_step"]
        print(f"[resume] from epoch {start_epoch} (best {best:.4f} @ {best_epoch})", flush=True)

    gpu_reset(device); t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train(); losses = []
        order = np.random.default_rng(args.seed * 1000003 + epoch).permutation(len(train_pos))
        for s in range(0, len(order), cfg["batch_size"]):
            batch = torch.as_tensor(train_pos[order[s:s + cfg["batch_size"]]], dtype=torch.long, device=device)
            fb, lb, projected = step_inputs(model, feats, label_feats, batch)
            logits = model(batch, fb, lb, None, projected=projected)
            loss = F.cross_entropy(logits, y_dev[batch])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_step is None:
                first_step = {"logits_hash": hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest(),
                              "loss": float(loss), "gradient_hash": gradient_hash(model), "init_hash": init_hash}
            optimizer.step(); losses.append(float(loss))
        vl = predict(model, feats, label_feats, val_pos, cfg["eval_batch_size"], device)
        vm = scores(vl, y[val_pos]); vloss = float(F.cross_entropy(torch.from_numpy(vl), torch.from_numpy(y[val_pos])))
        sh = state_dict_hash(model.state_dict())
        trace.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": vloss, "val_micro_f1": vm["micro_f1"], "val_macro_f1": vm["macro_f1"], "state_hash": sh})
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"[epoch {epoch + 1}] loss={np.mean(losses):.4f} val_loss={vloss:.4f} val_micro={vm['micro_f1']:.4f} val_macro={vm['macro_f1']:.4f} state={sh[:10]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if vm["micro_f1"] > best:
            best, bad, best_epoch, best_state = vm["micro_f1"], 0, epoch + 1, cpu_state(model)
        else:
            bad += 1
        stopping = bad >= cfg["patience"] or epoch + 1 >= cfg["epochs"] or (args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch)
        if (epoch + 1) % args.checkpoint_every == 0 or stopping:
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "trace": trace, "best": best,
                         "best_state": best_state, "best_epoch": best_epoch, "bad": bad, "epoch": epoch,
                         "train_sec": train_sec + (time.perf_counter() - t0), "first_step": first_step}, state_path)
        if bad >= cfg["patience"]:
            print(f"[early stop] epoch {epoch + 1}, best {best:.4f} @ {best_epoch}", flush=True); break
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            print(f"[stop-after-epoch] state saved at epoch {epoch + 1}; exiting for the resume test", flush=True); return
    train_sec += time.perf_counter() - t0; peak_train = gpu_peak(device)

    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    test_logits = predict(model, feats, label_feats, test_pos, cfg["eval_batch_size"], device)
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    metrics = scores(test_logits, y[test_pos])
    val_logits = predict(model, feats, label_feats, val_pos, cfg["eval_batch_size"], device)
    metrics.update({f"val_{k}": v for k, v in scores(val_logits, y[val_pos]).items()})
    np.savez(run_dir / "scores.npz", ids=b.targets[test_pos], scores=test_logits)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if state_path.exists():
        state_path.unlink()
    metrics.update({"train_time_sec": train_sec, "epochs": len(trace), "best_epoch": best_epoch, "best_val": best,
                    "select_on": cfg["select_on"], "inference_time_sec": inf_sec,
                    "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
                    **model_size(model)})
    json_dump({"dataset": args.dataset, "task": "nc", "variant": args.variant, "seed": args.seed, "arm": args.arm, "flavor": args.flavor,
               "model": "SeHGNN", "run_dir": str(run_dir), "physical_hash": b.manifest["physical_hash"], "semantic_hash": b.program_hash,
               "graph_hash": b.manifest.get("graph_hash"), "relation_program_hash": b.manifest.get("semantic_hash"),
               "supervision_hash": b.manifest["supervision_hash"], "feature_hash": b.manifest.get("feature_hash") or "identity",
               "bundle": {"dir": str(bundle_dir), "num_channels": len(feats), "num_label_channels": len(label_feats), "K": b.manifest["K"],
                          "cap": b.manifest["cap"], "channels": sorted(feats), "label_channels": sorted(label_feats)},
               "config": {**cfg, "sparse_backend": args.sparse_backend, "device": str(device)},
               "contract": {"select_on": cfg["select_on"], "loss": "cross_entropy",
                            "features": "canonical pooled BoW (all types)" if b.features is not None else "learned per-type embeddings (featureless)"},
               "first_step_audit": first_step, "checkpoint_hash": state_dict_hash(model.state_dict()), "metrics": metrics},
              run_dir / "metrics.json")
    json_dump(trace, run_dir / "trace.json")
    print(f"[done] {args.arm}_{args.flavor}/{args.variant} seed={args.seed} test_micro={metrics['micro_f1']:.4f} macro={metrics['macro_f1']:.4f} "
          f"best_epoch={best_epoch}/{len(trace)} train={train_sec:.0f}s peak={peak_train['peak_gpu_mib']} ckpt={state_dict_hash(model.state_dict())[:12]}", flush=True)


if __name__ == "__main__":
    main()
