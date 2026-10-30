#!/usr/bin/env python3
"""PathNet × IMDb node classification (year graph, canonical features) — the four arms, one contract.

    --arm originals|universal|invariant|augmentation   [--variant v1..v4 for originals / invariant]

Input: the RGCN structural bundles ``RGCN_structural_v10_year_canonfeat/IMDB/nc/<variant>`` (the bundles every
other model's IMDb NC cell reads): physical edges of the variant (originals), of the union graph (universal),
or the compiled semantic program (invariant: physical + ``link_to_{director,actor,year}`` projections, byte-
identical on v1–v4 — the same program RGCN / MAGNN / CMPNN use); canonical 3,008-d pooled-BoW features for all
five node types; movie split 2,926 / 418 / 836; 3 genres.

PathNet is a homogeneous walk model: the walk graph is the undirected 0/1 adjacency of the arm's edges (edge types
dropped, as upstream). Walks: MERW (``pathnet/walks.py``) — 40 walks × length 4 per movie with exact BFS
distance-to-target buckets. With ``--resample`` (upstream protocol, used by ``scripts/PathNet/run_imdb_nc.sh``) a
fresh walk set is drawn every epoch by ``fast_walks.EpochWalker`` with RNG (seed, epoch); otherwise one walk set
per run. On the invariant arm the adjacency, hence the Perron vector, transition rows and walks, are identical on
every variant. Model: ``--model pathnet_homo`` (upstream ``PathNet_homo``) or ``pathnet`` (upstream ``PathNet``);
full batch, selection + early stop on val micro-F1 (``--patience 0`` = no early stop); augmentation = super-epochs
over v1–v4 in a seeded order (one full-batch step per variant), one optimizer, budget and patience ÷ 4, selection
on mean val micro-F1. The CLI defaults are not the settings used by the scripts; see ``scripts/PathNet``.

Outputs (RGCN layout, read by summarize_arms.py):
    <out-root>/<arm>/IMDB/nc/<variant>/seed_<s>/{metrics.json, scores.npz (ids, scores = test logits), history.json}
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
for p in (str(HERE), str(HERE.parent.parent / "common")):
    if p not in sys.path:
        sys.path.insert(0, p)
from pathnet_nc import PathNetNC, PathNetHomoNC  # noqa: E402
from fast_walks import EpochWalker  # noqa: E402
from pathnet.walks import build_walk_data, build_symmetric_adj  # noqa: E402  (MERW sampler of the PathNet port)
from walk_dist import bfs_distance_buckets  # noqa: E402
from inv_rgcn.data import load_graph_data  # noqa: E402

DATA = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parent / "data"))) / "preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc"
VARIANTS = ["v1", "v2", "v3", "v4"]


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


class WalkBundle:
    def __init__(self, variant, graph, args):
        d = load_graph_data(DATA / variant)
        self.variant, self.graph = variant, graph
        self.num_nodes = int(d.num_nodes)
        self.x, self.y, self.node_type = d.x.float(), d.y.to(torch.int64), d.node_type.to(torch.int64)
        self.train_idx, self.val_idx, self.test_idx = (torch.nonzero(m).flatten() for m in (d.train_mask, d.val_mask, d.test_mask))
        ei = d.physical_edge_index if graph == "physical" else d.semantic_edge_index
        pairs = np.unique(np.sort(ei.numpy().T, axis=1), axis=0)            # undirected, canonical order, no dups
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        self.pairs = pairs
        movie_off, n_movies = int(d.metadata["offsets"]["movie"]), int(d.metadata["num_nodes_by_type"]["movie"])
        self.targets = np.arange(movie_off, movie_off + n_movies, dtype=np.int64)
        assert set(self.train_idx.tolist()) <= set(self.targets.tolist())
        t0 = time.time()
        self.walk = build_walk_data(pairs, self.num_nodes, self.targets, num_walks=args.num_walks, walk_length=args.walk_length,
                                    walk_type=args.walk_type, seed=args.seed, cache_dir=None, use_cache=False)
        if args.dist == "bfs":                                          # upstream gen_merw.cpp: BFS distance to the target
            self.walk.dists = torch.from_numpy(bfs_distance_buckets(self.walk.walks.numpy(), build_symmetric_adj(pairs, self.num_nodes)))
        self.walker = EpochWalker(pairs, self.num_nodes, self.targets, args.num_walks, args.walk_length, args.walk_type) if args.resample else None
        self.walk_sec = time.time() - t0
        self.graph_hash = sha(pairs)
        self.walk_hash = sha(self.walk.walks.numpy(), self.walk.dists.numpy())
        self.supervision_hash = sha(self.y.numpy(), self.train_idx.numpy(), self.val_idx.numpy(), self.test_idx.numpy())
        self.feature_hash = sha(self.x.numpy())
        self.stored = {"physical": str(d.physical_hash), "semantic": str(d.semantic_hash)}
        self.stats = {"undirected_pairs": int(len(pairs)), "walk_tensor": list(self.walk.walks.shape), "walk_build_sec": self.walk_sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["originals", "universal", "invariant", "augmentation"], required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=HERE / "results/imdb_nc")
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num-walks", type=int, default=40); ap.add_argument("--walk-length", type=int, default=4)
    ap.add_argument("--walk-type", choices=["merw", "rw"], default="merw")
    ap.add_argument("--input", choices=["feat", "feat_type", "rand_type", "id"], default="feat",
                    help="feat = canonical 3,008-d BoW only; feat_type = canonical BoW ⊕ one-hot node type (5 dims) — through PathNet's "
                         "linear fc0 this is a learned per-type embedding added to the projected features (PathNet is homogeneous "
                         "and otherwise cannot tell director / actor / link / year walk nodes apart; identical in every variant); "
                         "without the BoW: rand_type = fixed seeded random 64-d vector per node + learned node-type "
                         "embedding (the Freebase input); id = trainable 64-d vector per node")
    ap.add_argument("--dist", choices=["earliest", "bfs"], default="earliest",
                    help="distance-to-target bucket: bfs = upstream's BFS shortest-path distance (gen_merw.cpp); earliest = the "
                         "proxy: earliest position of the node in the target's walks")
    ap.add_argument("--model", choices=["pathnet", "pathnet_homo"], default="pathnet",
                    help="pathnet = upstream PathNet (heterophily model, softmax over walks); pathnet_homo = upstream PathNet_homo "
                         "(the model upstream runs on homophilous data: ReLU, (1+att) walk gate, no softmax over walks)")
    ap.add_argument("--resample", action="store_true",
                    help="upstream protocol: a fresh walk set every epoch (gen_epoch_merw.cpp), RNG = (seed, epoch); val/test of an epoch "
                         "use that epoch's walks, the final test re-draws the best epoch's walks")
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--stop-after", type=int, default=0); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arm in ("originals", "invariant"):
        assert args.variant in VARIANTS, args.variant
        variants, graph = [args.variant], ("physical" if args.arm == "originals" else "semantic")
    elif args.arm == "universal":
        variants, graph = ["universal"], "physical"
    else:
        variants, graph = list(VARIANTS), "physical"
    run_dirs = {v: args.out_root / args.arm / "IMDB/nc" / v / f"seed_{args.seed}" for v in variants}
    if all((d / "metrics.json").exists() for d in run_dirs.values()) and not args.force:
        print(f"[skip] {args.arm} {variants} seed {args.seed}"); return
    tag = f"PathNet/IMDB/nc/{args.arm}/{'+'.join(variants)}/s{args.seed}"

    bundles = {v: WalkBundle(v, graph, args) for v in variants}
    ref = bundles[variants[0]]
    for b in bundles.values():
        assert b.supervision_hash == ref.supervision_hash and b.feature_hash == ref.feature_hash
        assert np.array_equal(b.targets, ref.targets)
        print(f"[{tag}] {b.variant} {graph}: graph_hash {b.graph_hash[:12]} walk_hash {b.walk_hash[:12]} {b.stats}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)                  # after walk sampling: same model RNG in every arm
    x, y = ref.x.to(device), ref.y.to(device)
    if args.input == "feat_type":
        x = torch.cat([x, F.one_hot(ref.node_type, int(ref.node_type.max()) + 1).float().to(device)], dim=1)
    extra = []                                                              # trainable input parameters (rand_type / id inputs)
    if args.input in ("rand_type", "id"):
        nt = ref.node_type.to(device); dim = 64
        node_tab = torch.nn.Embedding(ref.num_nodes, dim).to(device)
        type_tab = torch.nn.Embedding(int(nt.max()) + 1, dim).to(device)
        if args.input == "rand_type":
            g = torch.Generator().manual_seed(12345)                        # fixed across arms / seeds / variants
            node_tab.weight.data = (torch.randn(ref.num_nodes, dim, generator=g) / dim ** 0.5).to(device)
            node_tab.weight.requires_grad_(False)
            extra = list(type_tab.parameters())
            feats = lambda: node_tab.weight + type_tab(nt)
        else:
            torch.nn.init.normal_(node_tab.weight, std=0.1)
            extra = list(node_tab.parameters())
            feats = lambda: node_tab.weight
    else:
        feats = lambda: x
    walks = {v: (b.walk.walks.to(device), b.walk.dists.to(device)) for v, b in bundles.items()}
    tpos = {int(t): i for i, t in enumerate(ref.targets)}
    rows = lambda idx: torch.tensor([tpos[int(i)] for i in idx], dtype=torch.long, device=device)
    tr, va, te = rows(ref.train_idx), rows(ref.val_idx), rows(ref.test_idx)
    ytr, yva, yte = y[ref.train_idx.to(device)], y[ref.val_idx.to(device)], y[ref.test_idx.to(device)]
    n_cls = int(ytr.max()) + 1

    model = (PathNetHomoNC if args.model == "pathnet_homo" else PathNetNC)(64 if args.input in ("rand_type", "id") else x.shape[1], args.hidden, n_cls, args.walk_length, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()); param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    opt = torch.optim.Adam(list(model.parameters()) + extra, lr=args.lr, weight_decay=args.wd)
    init_hash = state_hash(model)
    print(f"[{tag}] targets {len(ref.targets)} classes {n_cls} params {n_params:,} init {init_hash[:12]} supervision {ref.supervision_hash[:12]}", flush=True)

    def epoch_walks(v, epoch):
        if not args.resample: return walks[v]
        w, dd = bundles[v].walker.sample(args.seed, epoch)
        return torch.from_numpy(w).to(device), torch.from_numpy(dd).to(device)

    cur = dict(walks)
    def evaluate(v, r, yy):
        model.eval()
        with torch.no_grad():
            L = model(feats(), *cur[v]).index_select(0, r)
        return L, nc_metrics(yy.cpu().numpy(), L.argmax(1).cpu().numpy())

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
            model.train()
            loss = F.cross_entropy(model(feats(), *cur[v]).index_select(0, tr), ytr)
            opt.zero_grad(set_to_none=True); loss.backward()
            if first_step_audit is None:
                first_step_audit = {"loss": float(loss), "grad_hash": sha(*[p.grad.detach().cpu().numpy() for p in model.parameters() if p.grad is not None])}
            opt.step(); losses.append(float(loss))
        if device.type == "cuda": peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))
        vals = {v: evaluate(v, va, yva)[1] for v in variants}
        score = float(np.mean([m["micro_f1"] for m in vals.values()]))
        rec = {"epoch": epoch, "variant_epochs": epoch * n_var, "train_loss": float(np.mean(losses)), "val": vals,
               "val_selection_micro_f1": score, "param_hash": state_hash(model), "elapsed_sec": time.time() - t0}
        history.append(rec)
        if epoch % 10 == 0 or epoch == 1:
            gn = lambda mods: float(sum(p.grad.norm() ** 2 for m in mods for p in m.parameters() if p.grad is not None) ** 0.5)
            print(f"[{tag}] ep {epoch:4d} |ego| {model._diag_norms[0]:.3f} |h_agg| {model._diag_norms[1]:.4f} grad fc0 {gn([model.fc0]):.2e} "
                  f"LSTM {gn([model.LSTM]):.2e} nets {gn([model.nets]):.2e} attw {gn([model.attw]):.2e}", flush=True)
            print(f"[{tag}] ep {epoch:4d} loss {rec['train_loss']:.4f} val micro-F1 " + " ".join(f"{v}={m['micro_f1']:.4f}" for v, m in vals.items()) +
                  f" | params {rec['param_hash'][:12]} | {rec['elapsed_sec']:.0f}s", flush=True)
        if best is None or score > best["val_selection_micro_f1"]:
            best, bad = rec, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if args.patience > 0 and bad >= patience:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True); break
        if args.stop_after and epoch >= args.stop_after: break
    train_time = time.time() - t0; epochs_completed = epoch

    model.load_state_dict(best_state); ckpt_hash = state_hash(model)
    if args.resample: cur = {v: epoch_walks(v, best["epoch"]) for v in variants}
    for v in variants:
        d = run_dirs[v]; d.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time(); L, tm = evaluate(v, te, yte); inf_time = time.time() - t1
        peak_inf = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        np.savez(d / "scores.npz", ids=ref.test_idx.numpy(), scores=L.float().cpu().numpy())
        b = bundles[v]
        metrics = {"format_version": "pathnet_imdb_nc_arms_v1", "model": "PathNet (stock: ego concat; MERW 40x4)", "arm": args.arm,
                   "dataset": "IMDB", "task": "nc", "graph": graph, "variant": v, "seed": args.seed, "variants_trained_on": variants,
                   "config": vars(args) | {"out_root": str(args.out_root)},
                   "metrics": {**tm, "train_time_sec": train_time, "epochs": epochs_completed * n_var, "best_epoch": best["epoch"],
                               "parameter_count": n_params, "parameter_mib": param_bytes / 2**20,
                               "peak_training_gpu_mib": peak_train / 2**20, "peak_inference_gpu_mib": peak_inf / 2**20,
                               "inference_time_sec": inf_time, "walk_build_sec": b.walk_sec},
                   "val_metrics": best["val"][v], "best_val_selection_micro_f1": best["val_selection_micro_f1"],
                   "walk_graph": b.stats, "graph_hash": b.graph_hash, "walk_hash": b.walk_hash, "stored_hashes": b.stored,
                   "supervision_hash": ref.supervision_hash, "feature_hash": ref.feature_hash, "init_hash": init_hash,
                   "first_step_audit": first_step_audit, "checkpoint_hash": ckpt_hash,
                   "hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "torch": torch.__version__, "run_dir": str(d)}
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2)); (d / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{tag}] {v}: TEST micro-F1 {tm['micro_f1']:.4f} macro-F1 {tm['macro_f1']:.4f} | best ep {best['epoch']} | "
              f"ckpt {ckpt_hash[:12]} | {train_time/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
