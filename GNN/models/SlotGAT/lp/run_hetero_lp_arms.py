#!/usr/bin/env python3
"""SlotGAT × IMDb / DBLP link prediction — the four arms, one contract.

    --dataset IMDB --task movie_year      variants v1..v4 (+ universal)
    --dataset IMDB --task movie_director  variants v1, v3 (+ universal)
    --dataset DBLP --task paper_conference variants v1..v3 (+ universal)
    --arm originals|universal|invariant|augmentation   (as run_wordnet_lp_arms.py)

Inputs: the RGCN structural bundles (``RGCN_structural_v10_year_canonfeat/IMDB/<task>/<variant>``,
``RGCN_structural_v8_dblp/DBLP/paper_conference/<variant>``) — physical edges for originals / universal /
augmentation, the compiled semantic program for invariant; one relation vocabulary (both directions
named explicitly), the canonical IMDb feature table (3008-d) / no features on DBLP, and the shared
supervision: train/val/test (head, tail) pairs with the stored 19 frequency-matched negatives (IMDb) or
the full 20-conference candidate set (DBLP, the ``full_target_type`` protocol RGCN and CMPNN use).

Model / contract: deterministic multi-slot parity SlotGAT (lp_parity_hetero_model.py; slots = node types),
hidden 128, 2 layers × 4 heads + 1-head output, concat-of-layers + LayerNorm readout, diagonal DistMult over
the task relation, cross-entropy over (positive + 19 negatives), Adam lr 1e-3 wd 0, batches of 512 positives,
up to 400 epochs, validation every 5, patience 8. Training target edges stay in the message graph as in the
RGCN structural runner unless --mask-batch-edges (query edge, its alias copy on IMDb's Link node and the
label-derived relations incident to the head on DBLP are dropped for the batch — the CMPNN/MAGNN convention).

Outputs (RGCN result layout):
    <out-root>/<arm>/<DATASET>/<task>/<variant>/seed_<s>/{metrics.json, scores.npz, history.json, best_model.pt}
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

HERE = Path(__file__).resolve().parent
RGCN_DIR = HERE.parent.parent.parent / "common"
for p in (str(HERE), str(RGCN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
from lp_parity_model import MessagePlan  # noqa: E402
from lp_parity_hetero_model import DistMultSingle, HeteroParitySlotGATLPEncoder  # noqa: E402
from inv_rgcn.data import load_graph_data  # noqa: E402  (RGCN, read-only)

DATA = Path(os.environ.get("INV_GNN_DATA", str(HERE.parent / "data"))) / "preprocessed"
BUNDLE_ROOT = {"IMDB": DATA / "RGCN_structural_v10_year_canonfeat/IMDB", "DBLP": DATA / "RGCN_structural_v8_dblp/DBLP"}
VARIANTS = {("IMDB", "movie_year"): ["v1", "v2", "v3", "v4"], ("IMDB", "movie_director"): ["v1", "v3"],
            ("DBLP", "paper_conference"): ["v1", "v2", "v3"]}
TARGET = {("IMDB", "movie_year"): ("movie_to_year", "year_to_movie"), ("IMDB", "movie_director"): ("movie_to_director", "director_to_movie"),
          ("DBLP", "paper_conference"): ("paper_to_conference", "conference_to_paper")}
ALIAS_TARGET = {("IMDB", "movie_year"): ("link_to_year", "year_to_link"), ("IMDB", "movie_director"): ("link_to_director", "director_to_link")}
LABEL_DERIVED = {("DBLP", "paper_conference"): ("paper_to_area", "area_to_paper")}
SEEDS = (1566911444, 20241017, 20251017)


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a)); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_hash(*modules) -> str:
    h = hashlib.sha256()
    for m in modules:
        for k, v in m.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Bundle:
    def __init__(self, dataset: str, task: str, variant: str, graph: str):
        d = load_graph_data(BUNDLE_ROOT[dataset] / task / variant)
        self.dataset, self.task, self.variant, self.graph = dataset, task, variant, graph
        assert list(d.physical_relation_names) == list(d.semantic_relation_names)
        self.relation_names = list(d.physical_relation_names)
        self.num_nodes, self.node_type = int(d.num_nodes), d.node_type.to(torch.int64)
        self.num_types = int(self.node_type.max()) + 1
        self.offsets, self.counts = dict(d.metadata.get("offsets") or {}), dict(d.metadata.get("num_nodes_by_type") or {})
        self.x = d.x.float() if d.x is not None else None
        ei, et = (d.physical_edge_index, d.physical_edge_type) if graph == "physical" else (d.semantic_edge_index, d.semantic_edge_type)
        edges = torch.stack([ei[0], ei[1], et], 1).to(torch.int64)
        order = np.lexsort((edges[:, 0].numpy(), edges[:, 1].numpy(), edges[:, 2].numpy()))
        self.edges = torch.unique(edges[torch.from_numpy(order)], dim=0)      # canonical order, duplicate-free
        td = d.task_data
        t = lambda k: torch.as_tensor(np.asarray(td[k]), dtype=torch.int64)
        self.train_pos, self.val_pos, self.test_pos = t("train_pos"), t("val_pos"), t("test_pos")
        if dataset == "DBLP":
            # full 20-conference protocol: negatives = every other conference (the stored flat negatives are malformed)
            conf_type = int(self.node_type[self.train_pos[0, 1]])
            self.pool = torch.nonzero(self.node_type == conf_type).flatten()
            self.train_neg = self.val_neg = self.test_neg = None
        else:
            self.pool = None
            self.train_neg, self.val_neg, self.test_neg = t("train_neg"), t("val_neg"), t("test_neg")
        sup = [self.train_pos, self.val_pos, self.test_pos] + ([self.pool] if self.pool is not None else [self.train_neg, self.val_neg, self.test_neg])
        self.supervision_hash = sha(*[s.numpy() for s in sup])
        self.edge_hash = sha(self.edges.numpy())
        self.feature_hash = sha(self.x.numpy()) if self.x is not None else "none"
        self.stored = {"physical": str(d.physical_hash), "semantic": str(d.semantic_hash), "supervision": str(d.supervision_hash)}
        # alias map (IMDb: link i is the copy of movie i)
        self.alias = torch.arange(self.num_nodes)
        if dataset == "IMDB" and "link" in self.offsets:
            n, lo, mo = int(self.counts["link"]), int(self.offsets["link"]), int(self.offsets["movie"])
            self.alias[mo:mo + n] = torch.arange(lo, lo + n); self.alias[lo:lo + n] = torch.arange(mo, mo + n)


def candidates(b: Bundle, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(heads [Q], cand [Q, C]) with the positive in column 0."""
    pos = {"train": b.train_pos, "val": b.val_pos, "test": b.test_pos}[split]
    if b.pool is not None:
        C = b.pool.numel()
        cand = b.pool[None, :].expand(len(pos), C).clone()
        # move the positive to column 0 (swap with whatever sits there)
        col = (cand == pos[:, 1:2]).float().argmax(1)
        first = cand[:, 0].clone(); cand[torch.arange(len(pos)), col] = first; cand[:, 0] = pos[:, 1]
        return pos[:, 0], cand
    neg = {"train": b.train_neg, "val": b.val_neg, "test": b.test_neg}[split]
    return pos[:, 0], torch.cat([pos[:, 1:2], neg], 1)


def rank_metrics(scores: np.ndarray) -> dict:
    """positive in column 0; ties get the averaged rank (RGCN results convention)."""
    s0 = scores[:, :1]
    rk = 1 + (scores[:, 1:] > s0).sum(1) + 0.5 * (scores[:, 1:] == s0).sum(1)
    return {"mrr": float((1 / rk).mean()), "hits@1": float((rk <= 1).mean()), "hits@3": float((rk <= 3).mean()), "hits@10": float((rk <= 10).mean()),
            "mean_rank": float(rk.mean())}


class PlanFactory:
    """Base message array built once (edges, synthetic reverses, self loops); masked plans reuse it.
    Bundle edge i is row i of the array, so a set of bundle edge ids to drop maps to rows directly."""

    def __init__(self, b: Bundle, device):
        R = len(b.relation_names)
        e = b.edges.numpy()
        pairs = set(map(tuple, e[:, :2].tolist()))
        rev = np.asarray([(t, h, R + r) for h, t, r in e.tolist() if h != t and (t, h) not in pairs], dtype=np.int64).reshape(-1, 3)
        n = b.num_nodes
        loops = np.stack([np.arange(n), np.arange(n), np.full(n, 2 * R)], 1)
        all_e = np.concatenate([e, rev, loops], 0)
        self.all_e = torch.from_numpy(all_e).to(device)
        self.num_nodes, self.device = n, device
        self.stats = {"edges": int(len(e)), "synthetic_reverse": int(len(rev)), "self_loops": n, "messages": int(len(all_e)),
                      "num_edge_types": 2 * R + 1, "message_hash": sha(all_e)}
        self.base = self.plan()

    def plan(self, drop_rows: torch.Tensor | None = None) -> MessagePlan:
        a = self.all_e
        if drop_rows is not None and drop_rows.numel():
            keep = torch.ones(a.shape[0], dtype=torch.bool, device=a.device); keep[drop_rows] = False
            a = a[keep]
        return MessagePlan(a[:, 0], a[:, 1], a[:, 2], self.num_nodes)


def mask_edge_ids(b: Bundle, key) -> list[torch.Tensor]:
    """For every training positive i: the edge indices (into b.edges) to drop when i is in the batch."""
    names = b.relation_names
    rid = {n: i for i, n in enumerate(names)}
    e = b.edges
    by_pair = {}
    for idx, (h, t, r) in enumerate(e.tolist()):
        by_pair.setdefault((h, t, r), idx)
    ld = LABEL_DERIVED.get(key, ())
    ld_ids = {rid[n] for n in ld if n in rid}
    by_head_ld = {}
    if ld_ids:
        for idx, (h, t, r) in enumerate(e.tolist()):
            if r in ld_ids:
                by_head_ld.setdefault(h, []).append(idx); by_head_ld.setdefault(t, []).append(idx)
    fwd, bwd = TARGET[key]
    al = ALIAS_TARGET.get(key)
    out = []
    for h, t in b.train_pos.tolist():
        ids = []
        for (a, c, r) in ((h, t, fwd), (t, h, bwd)):
            if r in rid and (a, c, rid[r]) in by_pair: ids.append(by_pair[(a, c, rid[r])])
        if al is not None:
            ah = int(b.alias[h])
            for (a, c, r) in ((ah, t, al[0]), (t, ah, al[1])):
                if r in rid and (a, c, rid[r]) in by_pair: ids.append(by_pair[(a, c, rid[r])])
        ids += by_head_ld.get(h, [])
        out.append(torch.tensor(sorted(set(ids)), dtype=torch.int64))
    return out


def lp_loss(logits: torch.Tensor, mode: str, zeros: torch.Tensor) -> torch.Tensor:
    """logits [B, 1 + K] with the positive in column 0."""
    if mode == "ce":
        return F.cross_entropy(logits, zeros)
    return -(F.logsigmoid(logits[:, 0]).mean() + F.logsigmoid(-logits[:, 1:]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["IMDB", "DBLP"], required=True)
    ap.add_argument("--task", choices=["movie_year", "movie_director", "paper_conference"], required=True)
    ap.add_argument("--arm", choices=["originals", "universal", "invariant", "augmentation"], required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=HERE.parent / "results/lp_hetero")
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--edge-feats", type=int, default=64)
    ap.add_argument("--feat-drop", type=float, default=0.2); ap.add_argument("--attn-drop", type=float, default=0.2)
    ap.add_argument("--slope", type=float, default=0.05); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--mask-batch-edges", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0); ap.add_argument("--force", action="store_true")
    ap.add_argument("--loss", choices=["ce", "neg_logsigmoid"], default="ce",
                    help="ce = softmax over (positive + negatives) [study default]; neg_logsigmoid = the per-candidate binary objective "
                         "-(logsigmoid(pos).mean() + logsigmoid(-neg).mean()) that RGCN / CMPNN / MAGNN use (decision boundary at logit 0, "
                         "so the tables' precision/recall/F1 at threshold 0 are meaningful)")
    args = ap.parse_args()
    key = (args.dataset, args.task)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arm in ("originals", "invariant"):
        assert args.variant in VARIANTS[key], args.variant
        variants, graph = [args.variant], ("physical" if args.arm == "originals" else "semantic")
    elif args.arm == "universal":
        variants, graph = ["universal"], "physical"
    else:
        variants, graph = list(VARIANTS[key]), "physical"
    run_dirs = {v: args.out_root / args.arm / args.dataset / args.task / v / f"seed_{args.seed}" for v in variants}
    if all((d / "metrics.json").exists() for d in run_dirs.values()) and not args.force:
        print(f"[skip] {args.arm} {key} {variants} seed {args.seed}"); return
    tag = f"{args.dataset}/{args.task}/{args.arm}/{'+'.join(variants)}/s{args.seed}"

    bundles = {v: Bundle(args.dataset, args.task, v, graph) for v in variants}
    ref = bundles[variants[0]]
    for b in bundles.values():
        assert b.supervision_hash == ref.supervision_hash and b.feature_hash == ref.feature_hash and torch.equal(b.node_type, ref.node_type)
    plans, stats, factories = {}, {}, {}
    for v, b in bundles.items():
        factories[v] = PlanFactory(b, device); plans[v], stats[v] = factories[v].base, factories[v].stats
        print(f"[{tag}] {v} {graph}: edge_hash {b.edge_hash[:12]} (stored {b.stored[graph][:12]}) {stats[v]}", flush=True)
    R = len(ref.relation_names)
    x = ref.x.to(device) if ref.x is not None else None
    masks = {v: mask_edge_ids(b, key) for v, b in bundles.items()} if args.mask_batch_edges else None

    enc = HeteroParitySlotGATLPEncoder(node_type=ref.node_type, num_types=ref.num_types, feature_dim=None if x is None else x.shape[1],
                                       num_edge_types=2 * R + 1, hidden=args.hidden, out_dim=args.out_dim, num_layers=args.layers,
                                       num_heads=args.heads, edge_feats=args.edge_feats, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                                       slope=args.slope, alpha=args.alpha).to(device)
    dec = DistMultSingle(enc.embedding_dim).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    n_params = sum(p.numel() for p in params); param_bytes = sum(p.numel() * p.element_size() for p in params)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    init_hash = state_hash(enc, dec)
    print(f"[{tag}] nodes {ref.num_nodes} types {ref.num_types} features {None if x is None else tuple(x.shape)} edge types {2*R+1} "
          f"params {n_params:,} emb_dim {enc.embedding_dim} init {init_hash[:12]} supervision {ref.supervision_hash[:12]}", flush=True)

    tr_h, tr_c = candidates(ref, "train"); tr_h, tr_c = tr_h.to(device), tr_c.to(device)
    va_h, va_c = candidates(ref, "val"); va_h, va_c = va_h.to(device), va_c.to(device)
    te_h, te_c = candidates(ref, "test"); te_h, te_c = te_h.to(device), te_c.to(device)
    rng = np.random.default_rng(args.seed)
    n_var = len(variants)
    super_epochs = args.epochs // n_var if args.arm == "augmentation" else args.epochs
    eval_every = max(1, args.eval_every // n_var) if args.arm == "augmentation" else args.eval_every
    history, best, best_state, bad, peak_train = [], None, None, 0, 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    def eval_split(v, h, c):
        enc.eval(); dec.eval()
        with torch.no_grad():
            ent = enc(plans[v], x)
            return dec.score_candidates(ent, h, c).float().cpu().numpy()

    for epoch in range(1, super_epochs + 1):
        enc.train(); dec.train(); losses = []
        for v in variants:
            perm = rng.permutation(len(tr_h))
            for s in range(0, len(perm), args.batch_size):
                bidx = torch.from_numpy(perm[s:s + args.batch_size]).to(device)
                plan = plans[v]
                if masks is not None:
                    drop = torch.cat([masks[v][i] for i in perm[s:s + args.batch_size]]).to(device)
                    plan = factories[v].plan(drop_rows=drop)
                ent = enc(plan, x)
                logits = dec.score_candidates(ent, tr_h[bidx], tr_c[bidx])
                loss = lp_loss(logits, args.loss, torch.zeros(len(bidx), dtype=torch.long, device=device))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss))
        if device.type == "cuda": peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))
        if epoch % eval_every == 0 or epoch == super_epochs or (args.stop_after and epoch >= args.stop_after):
            vals = {v: rank_metrics(eval_split(v, va_h, va_c)) for v in variants}
            score = float(np.mean([m["mrr"] for m in vals.values()]))
            rec = {"epoch": epoch, "variant_epochs": epoch * n_var, "train_loss": float(np.mean(losses)), "val": vals,
                   "val_selection_mrr": score, "param_hash": state_hash(enc, dec), "elapsed_sec": time.time() - t0}
            history.append(rec)
            print(f"[{tag}] ep {epoch:4d} loss {rec['train_loss']:.4f} val MRR " + " ".join(f"{v}={m['mrr']:.4f}" for v, m in vals.items()) +
                  f" | params {rec['param_hash'][:12]} | {rec['elapsed_sec']:.0f}s", flush=True)
            if best is None or score > best["val_selection_mrr"]:
                best, bad = rec, 0
                best_state = ({k: t.detach().clone() for k, t in enc.state_dict().items()}, {k: t.detach().clone() for k, t in dec.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[{tag}] early stop at epoch {epoch}", flush=True); break
        if args.stop_after and epoch >= args.stop_after: break
    train_time = time.time() - t0; epochs_completed = epoch

    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1]); ckpt_hash = state_hash(enc, dec)
    for v in variants:
        d = run_dirs[v]; d.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time(); S = eval_split(v, te_h, te_c); inf_time = time.time() - t1
        peak_inf = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        tm = rank_metrics(S)
        b = bundles[v]
        payload = {"queries": b.test_pos[:, 0].numpy(), "positive_tails": b.test_pos[:, 1].numpy(), "scores": S.astype(np.float32)}
        if b.pool is not None:
            # RGCN full_target_type layout: scores in POOL order, 1-D candidate ids
            pool = b.pool.numpy(); col = {int(c): i for i, c in enumerate(pool)}
            cand = te_c.cpu().numpy(); S_pool = np.empty_like(S)
            for q in range(len(S)):
                for j, c in enumerate(cand[q]): S_pool[q, col[int(c)]] = S[q, j]
            payload["scores"] = S_pool.astype(np.float32); payload["candidate_ids"] = pool.astype(np.int64)
        np.savez(d / "scores.npz", **payload)
        torch.save({"encoder": best_state[0], "decoder": best_state[1], "args": vars(args) | {"out_root": str(args.out_root)}, "variant": v}, d / "best_model.pt")
        metrics = {"format_version": "slotgat_hetero_lp_arms_v1", "model": "SlotGAT (multi-slot parity, deterministic)", "arm": args.arm,
                   "dataset": args.dataset, "task": args.task, "graph": graph, "variant": v, "seed": args.seed, "variants_trained_on": variants,
                   "config": vars(args) | {"out_root": str(args.out_root)},
                   "metrics": {**tm, "train_time_sec": train_time, "epochs": epochs_completed * n_var, "best_epoch": best["epoch"],
                               "parameter_count": n_params, "parameter_mib": param_bytes / 2**20,
                               "peak_training_gpu_mib": peak_train / 2**20, "peak_inference_gpu_mib": peak_inf / 2**20,
                               "inference_time_sec": inf_time, "num_candidates": int(S.shape[1]),
                               "eval_protocol": "full_target_type" if b.pool is not None else "sampled"},
                   "val_metrics": best["val"][v], "best_val_selection_mrr": best["val_selection_mrr"],
                   "message_graph": stats[v], "edge_hash": b.edge_hash, "stored_hashes": b.stored, "supervision_hash": ref.supervision_hash,
                   "feature_hash": ref.feature_hash, "init_hash": init_hash, "checkpoint_hash": ckpt_hash,
                   "hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "run_dir": str(d)}
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2)); (d / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{tag}] {v}: TEST MRR {tm['mrr']:.4f} H@1 {tm['hits@1']:.4f} H@3 {tm['hits@3']:.4f} | best ep {best['epoch']} | ckpt {ckpt_hash[:12]} | {train_time/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
