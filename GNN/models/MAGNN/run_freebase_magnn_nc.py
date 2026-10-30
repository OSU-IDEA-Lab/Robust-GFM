#!/usr/bin/env python3
"""One MAGNN runner for node classification on the v2 bundles (Freebase and IMDb; three per-variant arms).

    --dataset FREEBASE  bundles: freebase_magnn_v2/{invariant,invariant_union,stock}/<variant>/   (identity features)
    --dataset IMDB      bundles: imdb_magnn_v2/nc/{invariant,invariant_union,stock}/<variant>/mode0 (canonical BoW features)

    --arm invariant  --variant v   bundles from preprocess_freebase_magnn_v2.py --arm invariant
    --arm originals  --variant v   ... --arm stock  (the variant's own metapaths)
    --arm universal  --variant union_exact_2_3   ... --arm stock on the union graph

Same model (repository ``MAGNN_nc_mb``), same contract for every arm: hidden 64, 8 heads,
attention-vector 128, RotatE0 instance encoder, dropout .5, Adam lr 5e-3 / wd 1e-3, batch
8 target nodes, 100 sampled instances per (node, metapath) per batch, ≤100 epochs, early
stopping on VALIDATION MICRO-F1 with patience 5, seeds 1566911444 / 20241017 / 20251017.
(The public MAGNN DBLP defaults, except that model selection uses the evaluation metric.)

Instance sampling reproduces the stock ``utils/tools.parse_adjlist`` rule (undersample
frequent neighbours, ``p ∝ count^¾``) with a per-(epoch, batch) seeded generator, so two
variants with identical instance tables draw identical samples. Writes the RGCN run layout
so ``RGCN/aggregate_rgcn_structural.py`` aggregates MAGNN results unchanged:
``<output-root>/FREEBASE/nc/<variant>/seed_<seed>/{metrics.json, scores.npz, trace.json,
checkpoint.pt}``. Resumable per epoch (``last_training_state.pt``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "common"))
import dgl  # noqa: E402
from model.MAGNN_nc_mb import MAGNN_nc_mb  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402

CONTRACT = dict(hidden_dim=64, num_heads=8, attn_vec_dim=128, rnn_type="RotatE0", dropout=0.5, lr=5e-3,
                weight_decay=1e-3, batch_size=8, samples=100, epochs=100, patience=5, select_on="val_micro_f1")
DEFAULT_ROOT = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data"))) / "preprocessed/freebase_magnn_v2"


# ----------------------------------------------------------------------------- bundle
class Bundle:
    def __init__(self, d: Path, group: str = ""):
        """``group`` = "" for Freebase bundles (metapaths at the root); "mode0" for IMDb bundles."""
        self.dir = d
        self.summary = json.loads((d / "summary.json").read_text())
        self.meta = json.loads((d / "metapaths.json").read_text())
        gsum = self.summary["groups"][group] if group else self.summary
        self.node_types = np.load(d / "node_types.npy")
        self.labels = np.load(d / "labels.npy")                     # aligned with target_nodes
        self.target_nodes = np.load(d / "target_nodes.npy")         # sorted labelled target ids
        z = np.load(d / "split.npz")
        self.split = {k: z[k] for k in ("train_idx", "val_idx", "test_idx")}   # local positions
        self.num_classes = int(self.labels.max()) + 1
        self.M = gsum["num_metapaths"]
        d = d / group if group else d
        self.gsum = gsum
        self.pos_of = {int(b): i for i, b in enumerate(self.target_nodes)}
        # per metapath: rows[i][p] = int32 [k, 3] instances of target position p
        self.rows = []
        for i in range(self.M):
            inst = np.load(d / "idx" / f"{i}.npy")
            per = [[] for _ in range(len(self.target_nodes))]
            if inst.shape[0]:
                starts = np.searchsorted(inst[:, 0], self.target_nodes, side="left")
                ends = np.searchsorted(inst[:, 0], self.target_nodes, side="right")
                for p, (s, e) in enumerate(zip(starts, ends)):
                    per[p] = inst[s:e]
            w = inst.shape[1] if inst.ndim == 2 else 3
            self.rows.append([np.asarray(r, dtype=np.int64).reshape(-1, w) for r in per])
        self.etypes_list = self.gsum.get("etypes_list") or [[2 * i, 2 * i + 1] for i in range(self.M)]
        self.num_edge_type = int(self.gsum.get("num_edge_type") or 2 * self.M)
        self.metapath_names = self.gsum["metapath_names"]
        self.feature_files = sorted(self.dir.glob("features_*.npy"))
        self.num_types = int(self.node_types.max()) + 1
        self.type_counts = [int((self.node_types == t).sum()) for t in range(self.num_types)]
        self.type_offsets = [int(np.where(self.node_types == t)[0].min()) if c else 0 for t, c in enumerate(self.type_counts)]

    def program_hash(self) -> str:
        return self.summary["program_hash"]


def features_identity(b: Bundle, device):
    """Freebase: sparse identity per type. IMDb: the bundle's canonical feature table per type."""
    if b.feature_files:
        feats = [torch.as_tensor(np.load(b.dir / f"features_{t}.npy"), device=device) for t in range(b.num_types)]
        return feats, [int(f.shape[1]) for f in feats]
    feats = []
    for t in range(b.num_types):
        n = b.type_counts[t]
        idx = torch.arange(n, device=device)
        feats.append(torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n, device=device), (n, n)).coalesce())
    return feats, list(b.type_counts)


def sample_rows(rows: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """stock parse_adjlist: undersample frequent neighbours, p ∝ count^¾ / count per row."""
    if rows.shape[0] <= k:
        return rows
    nbr = rows[:, -1]
    uniq, inv, counts = np.unique(nbr, return_inverse=True, return_counts=True)
    p = (counts[inv] ** 0.75) / counts[inv]
    p = p / p.sum()
    pick = np.sort(rng.choice(rows.shape[0], size=k, replace=False, p=p))
    return rows[pick]


def parse_minibatch(b: Bundle, batch_pos: np.ndarray, k: int, rng: np.random.Generator, device):
    g_list, idx_list, mapped_list = [], [], []
    for i in range(b.M):
        edges, paths, nodes = [], [], set()
        for p in batch_pos:
            src = int(b.target_nodes[p]); nodes.add(src)
            r = sample_rows(b.rows[i][p], k, rng)
            for row in r:
                nodes.add(int(row[-1])); edges.append((src, int(row[-1])))
            paths.append(r)
        mapping = {n: j for j, n in enumerate(sorted(nodes))}
        edges = [(mapping[s], mapping[d]) for s, d in edges]
        arr = np.concatenate(paths) if paths else np.empty((0, b.rows[i][0].shape[1] if len(b.rows[i]) else 3), dtype=np.int64)
        if edges:
            order = sorted(range(len(edges)), key=lambda j: edges[j])
            g = dgl.graph(([edges[j][1] for j in order], [edges[j][0] for j in order]), num_nodes=len(mapping))
            arr = arr[order]
        else:
            g = dgl.graph(([], []), num_nodes=len(mapping))
        g_list.append(g.to(device))
        idx_list.append(torch.as_tensor(arr, dtype=torch.long, device=device))
        mapped_list.append(np.asarray([mapping[int(b.target_nodes[p])] for p in batch_pos]))
    return g_list, idx_list, mapped_list


@torch.no_grad()
def evaluate(model, b, feats, type_mask, positions, k, seed, device, bs=64):
    model.eval()
    logits_all = []
    for s in range(0, len(positions), bs):
        batch = positions[s:s + bs]
        rng = np.random.default_rng((seed * 7919 + s) % (2 ** 32))
        g, idx, mapped = parse_minibatch(b, batch, k, rng, device)
        logits, _ = model((g, feats, type_mask, idx, mapped))
        logits_all.append(logits.float().cpu())
    logits = torch.cat(logits_all).numpy()
    y = b.labels[positions]; pred = logits.argmax(1)
    return {"micro_f1": float(f1_score(y, pred, average="micro")), "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean())}, logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("FREEBASE", "IMDB"), default="FREEBASE")
    ap.add_argument("--data-root", type=Path, default=None, help="default: freebase_magnn_v2 or imdb_magnn_v2/nc")
    ap.add_argument("--arm", choices=("invariant", "invariant_union", "originals", "universal"), required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--output-root", default="results/MAGNN_freebase_v2")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--samples", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(CONTRACT)
    for kk in ("epochs", "patience", "samples"):
        if getattr(args, kk) is not None:
            cfg[kk] = getattr(args, kk)
    if args.data_root is None:
        args.data_root = DEFAULT_ROOT if args.dataset == "FREEBASE" else DEFAULT_ROOT.parent / "imdb_magnn_v2" / "nc"
    bundle_dir = args.data_root / {"invariant": "invariant", "invariant_union": "invariant_union"}.get(args.arm, "stock") / args.variant
    run_dir = Path(args.output_root) / args.arm / args.dataset / "nc" / args.variant / f"seed_{args.seed}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir}"); return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} arm={args.arm} variant={args.variant} seed={args.seed}", flush=True)
    set_deterministic(args.seed)
    # INV_GNN_DETERMINISM_STRICT=1 makes an op without a deterministic kernel RAISE (naming it) instead of
    # silently falling back with a warning (useful when checking determinism on a new GPU type).
    torch.use_deterministic_algorithms(True, warn_only=os.environ.get("INV_GNN_DETERMINISM_STRICT", "0") != "1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    b = Bundle(bundle_dir, group="mode0" if args.dataset == "IMDB" else "")
    print(f"[data] {bundle_dir}: {b.M} metapaths, program {b.program_hash()[:12]}, physical {b.summary['physical_hash'][:12]}, "
          f"supervision {b.summary['supervision_hash'][:12]}, targets {len(b.target_nodes)}", flush=True)
    feats, in_dims = features_identity(b, device)
    type_mask = b.node_types
    etypes = b.etypes_list
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = MAGNN_nc_mb(b.M, b.num_edge_type, etypes, in_dims, cfg["hidden_dim"], b.num_classes, cfg["num_heads"],
                        cfg["attn_vec_dim"], cfg["rnn_type"], cfg["dropout"]).to(device)
    init_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_pos, val_pos, test_pos = b.split["train_idx"], b.split["val_idx"], b.split["test_idx"]

    state_path = run_dir / "last_training_state.pt"
    trace, best, best_state, best_epoch, bad, start_epoch, train_sec, first_step = [], -1.0, None, 0, 0, 0, 0.0, None
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        trace, best, best_state, best_epoch, bad = st["trace"], st["best"], st["best_state"], st["best_epoch"], st["bad"]
        start_epoch, train_sec, first_step = st["epoch"] + 1, st["train_sec"], st["first_step"]
        print(f"[resume] from epoch {start_epoch}", flush=True)

    gpu_reset(device); t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train(); losses = []
        order = np.random.default_rng(args.seed * 1000003 + epoch).permutation(len(train_pos))
        for bi, s in enumerate(range(0, len(order), cfg["batch_size"])):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            batch = train_pos[order[s:s + cfg["batch_size"]]]
            rng = np.random.default_rng((args.seed * 1000003 + epoch) * 4099 + bi)
            g, idx, mapped = parse_minibatch(b, batch, cfg["samples"], rng, device)
            logits, _ = model((g, feats, type_mask, idx, mapped))
            loss = F.cross_entropy(logits, torch.as_tensor(b.labels[batch], device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_step is None:
                first_step = {"logits_hash": hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest(),
                              "loss": float(loss), "gradient_hash": gradient_hash(model), "init_hash": init_hash}
            optimizer.step(); losses.append(float(loss))
        vm, _ = evaluate(model, b, feats, type_mask, val_pos, cfg["samples"], args.seed, device)
        sh = state_dict_hash(model.state_dict())
        trace.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_micro_f1": vm["micro_f1"], "val_macro_f1": vm["macro_f1"], "state_hash": sh})
        print(f"[epoch {epoch + 1}] loss={np.mean(losses):.4f} val_micro={vm['micro_f1']:.4f} val_macro={vm['macro_f1']:.4f} state={sh[:10]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if vm["micro_f1"] > best:
            best, bad, best_epoch, best_state = vm["micro_f1"], 0, epoch + 1, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(), "trace": trace, "best": best, "best_state": best_state,
                    "best_epoch": best_epoch, "bad": bad, "epoch": epoch, "train_sec": train_sec + (time.perf_counter() - t0),
                    "first_step": first_step}, state_path)
        if bad >= cfg["patience"]:
            print("[early stop]", flush=True); break
    train_sec += time.perf_counter() - t0; peak_train = gpu_peak(device)

    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    metrics, test_logits = evaluate(model, b, feats, type_mask, test_pos, cfg["samples"], args.seed, device)
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    np.savez(run_dir / "scores.npz", ids=b.target_nodes[test_pos], scores=test_logits)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if state_path.exists():
        state_path.unlink()
    metrics.update({"train_time_sec": train_sec, "epochs": len(trace), "best_epoch": best_epoch, "best_val": best,
                    "select_on": cfg["select_on"], "inference_time_sec": inf_sec,
                    "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
                    **model_size(model)})
    json_dump({"dataset": args.dataset, "task": "nc", "variant": args.variant, "seed": args.seed, "arm": args.arm, "model": "MAGNN",
               "run_dir": str(run_dir), "physical_hash": b.summary["physical_hash"], "semantic_hash": b.program_hash(),
               "supervision_hash": b.summary["supervision_hash"], "feature_hash": b.summary.get("feature_hash") or "identity",
               "bundle": {"dir": str(bundle_dir), "num_metapaths": b.M, "metapaths": b.metapath_names, "cap": b.summary["cap"]},
               "config": cfg, "contract": {"select_on": cfg["select_on"], "loss": "cross_entropy",
                                           "features": "canonical pooled BoW" if b.feature_files else "sparse identity per type"},
               "first_step_audit": first_step, "checkpoint_hash": state_dict_hash(model.state_dict()), "metrics": metrics},
              run_dir / "metrics.json")
    json_dump(trace, run_dir / "trace.json")
    print(f"[done] {args.arm}/{args.variant} seed={args.seed} test_micro={metrics['micro_f1']:.4f} macro={metrics['macro_f1']:.4f} "
          f"best_epoch={best_epoch}/{len(trace)} train={train_sec:.0f}s peak={peak_train['peak_gpu_mib']} ckpt={state_dict_hash(model.state_dict())[:12]}", flush=True)


if __name__ == "__main__":
    main()
