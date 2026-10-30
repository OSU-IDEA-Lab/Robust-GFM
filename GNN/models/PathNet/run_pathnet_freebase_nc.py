#!/usr/bin/env python3
"""PathNet × Freebase node classification — the four arms, one contract.

    --arm originals  --variant unchanged|exact_2|exact_3   walks on the variant's physical store
    --arm universal                                         walks on union_exact_2_3
    --arm invariant  --variant unchanged|exact_2|exact_3   walks on the variant's compiled block-conditional store
                                                            (RGCN's compiled program; hash-identical on all three)
    --arm augmentation                                      one model over the three physical walk sets

Walks come from ``build_freebase_walks.py`` (MERW 40 × 4 from every labelled book, per seed, on RGCN's Freebase stores);
split / labels = the stores' (``make_split``: 4,772 / 1,591 / 1,591, 7 classes). With ``--resample`` (upstream protocol,
used by ``scripts/PathNet/run_freebase_nc.sh``) a fresh walk set is drawn every epoch by
``fast_walks_targets.EpochWalkerT`` on the same walk graph (``freebase_pairs.store_pairs``) with exact BFS buckets;
the stored walk sets then provide only the targets and the recorded hashes. The exact 2-hop bitset the walker needs is
built once per graph and cached under ``preprocessed/pathnet_inv/freebase_twohop_bits/``. Featureless graph: the
input ``x`` is a learned 64-d entity table (``--input id``) or a fixed random vector per node plus a learned node-type
embedding (``--input rand_type``). Model: ``--model pathnet_homo`` (upstream ``PathNet_homo``) or ``pathnet``
(upstream ``PathNet``). Full batch over all 7,954 targets per step (every arm), cross-entropy on the train rows,
selection + early stop on val micro-F1 (``--patience 0`` = no early stop); augmentation = super-epochs over the
three variants (one step each, seeded order), budget / patience ÷ 3. The CLI defaults are not the settings used by
the scripts; see ``scripts/PathNet``.

Outputs (RGCN layout): <out-root>/<arm>/FREEBASE/nc/<variant>/seed_<s>/{metrics.json, scores.npz, history.json}
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
for p in (str(HERE), str(HERE.parent / "RGCN"), str(HERE.parent.parent / "common")):
    sys.path.insert(0, p)
from pathnet_nc import PathNetNC, PathNetHomoNC  # noqa: E402
from freebase_samples import STORES  # noqa: E402

WALKS = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parent / "data"))) / "preprocessed/pathnet_inv/freebase_walks"
R2_CACHE = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parent / "data"))) / "preprocessed/pathnet_inv/freebase_twohop_bits"
VARIANTS = ["unchanged", "exact_2", "exact_3"]


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a)); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_hash(*ms) -> str:
    h = hashlib.sha256()
    for m in ms:
        for k, v in m.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def nc_metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0), "micro_f1": f1_score(y, pred, average="micro"),
            "macro_f1": f1_score(y, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["originals", "universal", "invariant", "augmentation"], required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=HERE / "results/freebase_nc")
    ap.add_argument("--emb", type=int, default=64); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5); ap.add_argument("--walk-length", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--input", choices=["id", "rand_type"], default="id",
                    help="id = learned per-node embedding table (can memorise every training book); "
                         "rand_type = fixed (non-trainable) seeded N(0, 1/emb) per-node vector + learned node-type embedding")
    ap.add_argument("--walk-suffix", default="", help="'' = MERW walks; '_rw' = plain random-walk walks")
    ap.add_argument("--model", choices=["pathnet", "pathnet_homo"], default="pathnet",
                    help="pathnet = upstream PathNet (heterophily model); pathnet_homo = upstream PathNet_homo (homophilous-data model)")
    ap.add_argument("--resample", action="store_true",
                    help="upstream protocol: fresh walks every epoch (RNG (seed, epoch); walk graph rebuilt from the store exactly as "
                         "build_freebase_walks.py; exact BFS buckets); val/test use that epoch's walks, final test the best epoch's")
    ap.add_argument("--walk-type", choices=["merw", "rw"], default="merw", help="kernel for --resample")
    ap.add_argument("--stop-after", type=int, default=0); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arm in ("originals", "invariant"):
        assert args.variant in VARIANTS, args.variant
        variants, kind = [args.variant], ("physical" if args.arm == "originals" else "semantic")
    elif args.arm == "universal":
        variants, kind = ["union_exact_2_3"], "physical"
    else:
        variants, kind = list(VARIANTS), "physical"
    run_dirs = {v: args.out_root / args.arm / "FREEBASE/nc" / v / f"seed_{args.seed}" for v in variants}
    if all((d / "metrics.json").exists() for d in run_dirs.values()) and not args.force:
        print(f"[skip] {args.arm} {variants} seed {args.seed}"); return
    tag = f"PathNet/FREEBASE/nc/{args.arm}/{'+'.join(variants)}/s{args.seed}"

    root = STORES[(kind, variants[0])]
    labels_t = np.asarray(np.load(root / "labels.npy"), dtype=np.int64)
    sp = np.load(root / "train_val_test_idx.npz"); tr, va, te = (np.asarray(sp[k], dtype=np.int64) for k in ("train_idx", "val_idx", "test_idx"))
    num_nodes = int(np.load(root / "node_types.npy", mmap_mode="r").size)
    walks, metas = {}, {}
    for v in variants:
        z = np.load(WALKS / f"{kind}_{v}{args.walk_suffix}" / f"walks_seed{args.seed}.npz")
        metas[v] = json.loads((WALKS / f"{kind}_{v}{args.walk_suffix}" / "meta.json").read_text())
        walks[v] = (torch.from_numpy(z["walks"]).to(device), torch.from_numpy(z["dists"]).to(device))
        targets = z["targets"]
        assert np.array_equal(targets, np.load(STORES[(kind, v)] / "target_nodes.npy"))
        print(f"[{tag}] {v} {kind}: pairs_hash {metas[v]['undirected_pairs_hash'][:12]} walk_hash {metas[v]['seeds'][str(args.seed)][:12]} "
              f"store {metas[v]['store_combined_hash'][:12]}", flush=True)
    supervision_hash = sha(targets, labels_t, tr, va, te)
    y = torch.from_numpy(labels_t).to(device)
    trd, vad, ted = (torch.from_numpy(a).to(device) for a in (tr, va, te))
    n_cls = int(labels_t.max()) + 1

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    emb = torch.nn.Embedding(num_nodes, args.emb).to(device)
    torch.nn.init.normal_(emb.weight, std=0.1)
    node_type = torch.from_numpy(np.load(root / "node_types.npy").astype(np.int64)).to(device)
    type_emb = torch.nn.Embedding(int(node_type.max()) + 1, args.emb).to(device)
    if args.input == "rand_type":
        g = torch.Generator().manual_seed(12345)                       # fixed across arms / seeds / variants
        emb.weight.data = (torch.randn(num_nodes, args.emb, generator=g) / args.emb ** 0.5).to(device)
        emb.weight.requires_grad_(False)
    features = (lambda: emb.weight + type_emb(node_type)) if args.input == "rand_type" else (lambda: emb.weight)
    model = (PathNetHomoNC if args.model == "pathnet_homo" else PathNetNC)(args.emb, args.hidden, n_cls, args.walk_length, args.dropout).to(device)
    params = [p for p in list(emb.parameters()) + list(type_emb.parameters()) + list(model.parameters()) if p.requires_grad]
    n_params = sum(p.numel() for p in params); param_bytes = sum(p.numel() * p.element_size() for p in params)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    init_hash = state_hash(emb, type_emb, model)
    print(f"[{tag}] nodes {num_nodes} targets {len(targets)} classes {n_cls} params {n_params:,} init {init_hash[:12]} "
          f"supervision {supervision_hash[:12]}", flush=True)

    walkers = {}
    if args.resample:
        from fast_walks_targets import EpochWalkerT
        from freebase_pairs import store_pairs
        for v in variants:
            t1 = time.time()
            pr = store_pairs(STORES[(kind, v)]); ph = sha(pr)
            walkers[v] = EpochWalkerT(pr, num_nodes, targets, 40, args.walk_length, args.walk_type); del pr
            r2f = R2_CACHE / f"{kind}_{v}_{ph[:16]}.npz"                          # exact 2-hop bitset, a function of the graph only
            same = sorted(R2_CACHE.glob(f"*_{ph[:16]}.npz")) if R2_CACHE.exists() else []   # hash-identical graph (invariant arm)
            if not r2f.exists() and same: r2f = same[0]
            if r2f.exists():
                z2 = np.load(r2f); walkers[v].R2 = z2["R2"]
                walkers[v].tpos = np.full(num_nodes, -1, np.int64); walkers[v].tpos[targets] = np.arange(len(targets))
                assert np.array_equal(z2["targets"], targets)
            else:
                walkers[v].build_two_hop_bits(); R2_CACHE.mkdir(parents=True, exist_ok=True)
                tmp = r2f.with_suffix(f".tmp{os.getpid()}.npz"); np.savez(tmp, R2=walkers[v].R2, targets=targets); os.replace(tmp, r2f)
            print(f"[{tag}] {v}: per-epoch {args.walk_type} walker built ({time.time() - t1:.0f}s)", flush=True)

    def epoch_walks(v, epoch):
        w, dd = walkers[v].sample(args.seed, epoch)
        return torch.from_numpy(w).to(device), torch.from_numpy(dd).to(device)

    cur = dict(walks)
    def evaluate(v, idx):
        emb.eval(); model.eval()
        with torch.no_grad():
            L = model(features(), *cur[v]).index_select(0, idx)
        return L, nc_metrics(labels_t[idx.cpu().numpy()], L.argmax(1).cpu().numpy())

    rng = np.random.default_rng(args.seed)
    n_var = len(variants)
    budget = args.epochs // n_var if args.arm == "augmentation" else args.epochs
    patience = max(1, args.patience // n_var) if args.arm == "augmentation" else args.patience
    history, best, best_state, bad, peak_train, first_step_audit = [], None, None, 0, 0, None
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for epoch in range(1, budget + 1):
        order = [variants[i] for i in rng.permutation(n_var)] if n_var > 1 else variants
        losses = []
        if args.resample: cur = {v: epoch_walks(v, epoch) for v in variants}
        for v in order:
            emb.train(); model.train()
            loss = F.cross_entropy(model(features(), *cur[v]).index_select(0, trd), y[trd])
            opt.zero_grad(set_to_none=True); loss.backward()
            if first_step_audit is None:
                first_step_audit = {"loss": float(loss), "grad_hash": sha(*[p.grad.detach().cpu().numpy() for p in params if p.grad is not None])}
            opt.step(); losses.append(float(loss))
        if device.type == "cuda": peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))
        vals = {v: evaluate(v, vad)[1] for v in variants}
        score = float(np.mean([m["micro_f1"] for m in vals.values()]))
        rec = {"epoch": epoch, "variant_epochs": epoch * n_var, "train_loss": float(np.mean(losses)), "val": vals,
               "val_selection_micro_f1": score, "param_hash": state_hash(emb, type_emb, model), "elapsed_sec": time.time() - t0}
        history.append(rec)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{tag}] ep {epoch:4d} loss {rec['train_loss']:.4f} val micro-F1 " + " ".join(f"{v}={m['micro_f1']:.4f}" for v, m in vals.items()) +
                  f" | params {rec['param_hash'][:12]} | {rec['elapsed_sec']:.0f}s", flush=True)
        if best is None or score > best["val_selection_micro_f1"]:
            best, bad = rec, 0
            best_state = ({k: t.detach().clone() for k, t in emb.state_dict().items()}, {k: t.detach().clone() for k, t in model.state_dict().items()},
                          {k: t.detach().clone() for k, t in type_emb.state_dict().items()})
        else:
            bad += 1
            if args.patience > 0 and bad >= patience:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True); break
        if args.stop_after and epoch >= args.stop_after: break
    train_time = time.time() - t0; epochs_completed = epoch

    emb.load_state_dict(best_state[0]); model.load_state_dict(best_state[1]); type_emb.load_state_dict(best_state[2]); ckpt_hash = state_hash(emb, type_emb, model)
    if args.resample: cur = {v: epoch_walks(v, best["epoch"]) for v in variants}
    for v in variants:
        d = run_dirs[v]; d.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time(); L, tm = evaluate(v, ted); inf_time = time.time() - t1
        peak_inf = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        np.savez(d / "scores.npz", ids=targets[te], local_idx=te, scores=L.float().cpu().numpy())
        metrics = {"format_version": "pathnet_freebase_nc_arms_v1", "model": "PathNet (stock; MERW 40x4; learned 64-d entity table)",
                   "arm": args.arm, "dataset": "FREEBASE", "task": "nc", "graph": kind, "variant": v, "seed": args.seed,
                   "variants_trained_on": variants, "config": vars(args) | {"out_root": str(args.out_root)},
                   "metrics": {**tm, "train_time_sec": train_time, "epochs": epochs_completed * n_var, "best_epoch": best["epoch"],
                               "parameter_count": n_params, "parameter_mib": param_bytes / 2**20,
                               "peak_training_gpu_mib": peak_train / 2**20, "peak_inference_gpu_mib": peak_inf / 2**20,
                               "inference_time_sec": inf_time},
                   "val_metrics": best["val"][v], "best_val_selection_micro_f1": best["val_selection_micro_f1"],
                   "walk_meta": metas[v], "supervision_hash": supervision_hash, "init_hash": init_hash,
                   "first_step_audit": first_step_audit, "checkpoint_hash": ckpt_hash,
                   "hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "torch": torch.__version__, "run_dir": str(d)}
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2)); (d / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{tag}] {v}: TEST micro-F1 {tm['micro_f1']:.4f} macro-F1 {tm['macro_f1']:.4f} | best ep {best['epoch']} | "
              f"ckpt {ckpt_hash[:12]} | {train_time/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
