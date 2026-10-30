#!/usr/bin/env python3
"""One MAGNN link-prediction runner for the v2 LP bundles (IMDb year-graph tasks, DBLP).

    --dataset IMDB --task movie_director|movie_year  --arm invariant|invariant_union|originals|universal  --variant v
    --dataset DBLP --task paper_conference           (bundles: dblp_magnn_v2/{invariant,invariant_union,stock}/<variant>)

Repository ``MAGNN_lp`` (mode0 = movie-centred metapaths, mode1 = target-type-centred
metapaths; score = <h_movie, h_target>; K parallel layers averaged, K=3 as in
``run_IMDB_magnn_lp.py``). Contract for every arm: hidden 64, 8 heads, attn-vec 128,
RotatE0, dropout .5, Adam 5e-3 / wd 1e-3, batch 64 positives × 19 bundle negatives,
100 sampled instances per (node, metapath), ≤100 epochs, early stopping on VALIDATION MRR
(patience 10), seeds 1566911444 / 20241017 / 20251017, features from the bundle (IMDb: the
canonical pooled-BoW table; DBLP: no stored features -> sparse identity per type, MAGNN's
feature type 0), ranking against the bundle's K=19 negatives (tie-aware MRR / Hits; on DBLP
the 19 negatives are every other conference, i.e. the full 20-conference ranking).

Training masks (the query-edge removal CMPNN's ``remove_easy_edges`` applies): for a training
query (h, t) the head's instances that contain its TRUE target t are dropped for the positive
AND for every negative candidate, and a candidate's instances that contain h are dropped, so a
training query looks exactly like a held-out one (whose target edge is absent from the graph).
⚠ Masking only the positive (MAGNN's LastFM ``use_masks``, the protocol of earlier
IMDb runs) leaks here: with one target per head the mask empties the head's whole target
metapath slot for positives only, and "slot empty ⇒ positive" separates the training batches
perfectly (train loss → 1e-10) without transferring to test. Instances are sampled by a
per-(epoch, batch) seeded generator, so variants with identical instance tables draw identical
samples. RGCN run layout:
``<output-root>/<arm>/<DATASET>/<task>/<variant>/seed_<seed>/{metrics.json, scores.npz,
trace.json, checkpoint.pt}``; resumable per epoch.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
import dgl  # noqa: E402
from model.MAGNN_lp import MAGNN_lp  # noqa: E402
from inv_rgcn.metrics import ranking_metrics  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402

CONTRACT = dict(hidden_dim=64, num_heads=8, attn_vec_dim=128, rnn_type="RotatE0", dropout=0.5, lr=5e-3, weight_decay=1e-3,
                batch_size=64, samples=100, epochs=100, patience=10, K=3, select_on="val_mrr")
_DATA = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data")))
DEFAULT_ROOT = {"IMDB": _DATA / "preprocessed/imdb_magnn_v2",
                "DBLP": _DATA / "preprocessed/dblp_magnn_v2"}
TASKS = {"IMDB": ("movie_director", "movie_year"), "DBLP": ("paper_conference",)}
# Query-edge masking beyond the pair itself, mirroring CMPNN's remove_easy_edges (inv_cmpnn/model.py)
# so a training query looks like a held-out one:
#  * LABEL_DERIVED: relations the compiler emits for TRAINING heads only (DBLP paper_to_area = the
#    area reached through the train-only paper_to_conference edge). Metapaths that leave the head
#    along such a relation are dropped for the query head (CMPNN masks (h, *, r)).
#  * ALIAS: a bijective copy of the head type whose train-only edges duplicate the target edge
#    (IMDb link(m) carries link_to_director for training pairs). The copy joins the mask set on the
#    tail side (CMPNN masks (alias(h), t)); the head side masks alias(t) likewise (identity here).
LABEL_DERIVED = {("DBLP", "paper_conference"): {"paper_to_area"}, ("IMDB", "movie_director"): {"link_to_director"}}
ALIAS = {("IMDB", "movie_director"): ("movie", "link"), ("IMDB", "movie_year"): ("movie", "link")}
IMDB_TYPES = {"movie": 0, "director": 1, "actor": 2, "link": 3, "year": 4}


class Group:
    """One metapath group (mode0 / mode1) of an LP bundle: rows[i][node] -> instances."""

    def __init__(self, d: Path, gsum: dict):
        self.M = gsum["num_metapaths"]; self.etypes_list = gsum["etypes_list"]; self.num_edge_type = gsum["num_edge_type"]
        self.widths = gsum["widths"]; self.names = gsum["metapath_names"]
        self.rows = []
        for i in range(self.M):
            inst = np.load(d / "idx" / f"{i}.npy").astype(np.int64)
            per = {}
            if inst.shape[0]:
                srcs, starts = np.unique(inst[:, 0], return_index=True)
                ends = np.append(starts[1:], inst.shape[0])
                for s0, a, b in zip(srcs, starts, ends):
                    per[int(s0)] = inst[a:b]
            self.rows.append(per)
        self.empty = [np.empty((0, w), dtype=np.int64) for w in self.widths]


class LPBundle:
    def __init__(self, d: Path):
        self.dir = d
        self.summary = json.loads((d / "summary.json").read_text())
        self.node_types = np.load(d / "node_types.npy")
        self.num_types = int(self.node_types.max()) + 1
        z = np.load(d / "task_data.npz")
        self.pos = {k: z[f"{k}_pos"] for k in ("train", "val", "test")}
        self.neg = {k: z[f"{k}_neg"] for k in ("train", "val", "test")}
        self.groups = [Group(d / "mode0", self.summary["groups"]["mode0"]), Group(d / "mode1", self.summary["groups"]["mode1"])]
        self.has_features = (d / "features_0.npy").exists()
        self.feats = [np.load(d / f"features_{t}.npy") for t in range(self.num_types)] if self.has_features else None
        self.type_counts = [int((self.node_types == t).sum()) for t in range(self.num_types)]
        self.type_first = [int(np.where(self.node_types == t)[0].min()) if c else 0 for t, c in enumerate(self.type_counts)]

    def alias_map(self, dataset: str, task: str) -> np.ndarray:
        """node -> its bijective copy (IMDb movie <-> link, numbered in movie order), identity elsewhere."""
        alias = np.arange(len(self.node_types))
        if (dataset, task) in ALIAS:
            a, b = (IMDB_TYPES[x] for x in ALIAS[(dataset, task)])
            assert self.type_counts[a] == self.type_counts[b], "alias types must be a bijection"
            ia = np.arange(self.type_counts[a]) + self.type_first[a]; ib = np.arange(self.type_counts[b]) + self.type_first[b]
            alias[ia] = ib; alias[ib] = ia
        return alias

    def label_derived_metapaths(self, dataset: str, task: str):
        """Per group: indices of metapaths whose first relation is label-derived (dropped for a training query's head)."""
        rels = LABEL_DERIVED.get((dataset, task), set())
        return [{i for i, n in enumerate(g.names) if n.split("::", 1)[1].split(";")[0] in rels} for g in self.groups]

    def feature_tensors(self, device):
        """Per-type feature tensors: the stored table, or a sparse identity (MAGNN feature type 0)."""
        if self.has_features:
            return [torch.as_tensor(f, device=device) for f in self.feats]
        out = []
        for n in self.type_counts:
            idx = torch.arange(n, device=device)
            out.append(torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n, device=device), (n, n)).coalesce())
        return out


def build_mode(group: Group, nodes, k: int, rng: np.random.Generator, device, mask_partner=None, drop=frozenset()):
    """Per metapath: DGL graph over the batch nodes' sampled instances (+ endpoints), index tensor, mapped batch.
    mask_partner: [n] or [n, m] node ids whose instances are dropped per batch node (the query's own edge and its
    alias copies); drop: metapath indices emptied for every batch node (label-derived relations of a training head)."""
    g_list, idx_list, mapped_list = [], [], []
    if mask_partner is not None:
        mask_partner = np.asarray(mask_partner).reshape(len(nodes), -1)
    for i in range(group.M):
        edges, paths, nset = [], [], set()
        for j, src in enumerate(nodes):
            src = int(src); nset.add(src)
            r = group.rows[i].get(src, group.empty[i])
            if mask_partner is not None and r.shape[0]:
                if i in drop:
                    r = group.empty[i]
                else:
                    r = r[~np.isin(r, mask_partner[j]).any(axis=1)]      # drop instances through the query's edge / its alias
            if r.shape[0] > k:
                r = r[np.sort(rng.choice(r.shape[0], size=k, replace=False))]
            for row in r:
                nset.add(int(row[-1])); edges.append((src, int(row[-1])))
            paths.append(r)
        mapping = {n: q for q, n in enumerate(sorted(nset))}
        arr = np.concatenate(paths) if paths else group.empty[i]
        if edges:
            e = [(mapping[s], mapping[d]) for s, d in edges]
            order = sorted(range(len(e)), key=lambda q: e[q])
            g = dgl.graph(([e[q][1] for q in order], [e[q][0] for q in order]), num_nodes=len(mapping)); arr = arr[order]
        else:
            g = dgl.graph(([], []), num_nodes=len(mapping))
        g_list.append(g.to(device)); idx_list.append(torch.as_tensor(arr, dtype=torch.long, device=device))
        mapped_list.append(np.asarray([mapping[int(n)] for n in nodes]))
    return g_list, idx_list, mapped_list


def score_pairs(model, b: LPBundle, feats, heads, tails, k, rng, device, head_mask=None, tail_mask=None):
    """Scores <h_head, h_tail> for parallel arrays heads/tails; head_mask[j] / tail_mask[j] = node(s) whose
    instances are dropped from heads[j] / tails[j] (the query's own edge, both endpoints, plus aliases)."""
    g0, i0, m0 = build_mode(b.groups[0], heads, k, rng, device, head_mask, b.drop[0] if head_mask is not None else frozenset())
    g1, i1, m1 = build_mode(b.groups[1], tails, k, rng, device, tail_mask, b.drop[1] if tail_mask is not None else frozenset())
    [hm, ht], _ = model(([g0, g1], feats, b.node_types, [i0, i1], [m0, m1]))
    return (hm * ht).sum(-1)


def query_masks(b: LPBundle, p: np.ndarray):
    """[n, 2] mask sets for training queries p = (head, true tail): head side {t, alias(t)}, tail side {h, alias(h)}."""
    return np.stack([p[:, 1], b.alias[p[:, 1]]], 1), np.stack([p[:, 0], b.alias[p[:, 0]]], 1)


@torch.no_grad()
def evaluate(model, b, feats, split, k, seed, device, bs=64):
    model.eval()
    pos, neg = b.pos[split], b.neg[split]
    out = np.zeros((len(pos), 1 + neg.shape[1]), dtype=np.float32)
    for s in range(0, len(pos), bs):
        p = pos[s:s + bs]; n = neg[s:s + bs]
        rng = np.random.default_rng((seed * 7919 + s) % (2 ** 32))
        heads = np.concatenate([p[:, 0], np.repeat(p[:, 0], n.shape[1])]); tails = np.concatenate([p[:, 1], n.ravel()])
        sc = score_pairs(model, b, feats, heads, tails, k, rng, device).float().cpu().numpy()
        out[s:s + bs, 0] = sc[:len(p)]; out[s:s + bs, 1:] = sc[len(p):].reshape(len(p), -1)
    m = ranking_metrics(out[:, 0], out[:, 1:]); m["num_candidates"] = int(out.shape[1])
    return m, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("IMDB", "DBLP"), default="IMDB")
    ap.add_argument("--data-root", type=Path, default=None, help="default: the dataset's v2 bundle root")
    ap.add_argument("--task", choices=("movie_director", "movie_year", "paper_conference"), required=True)
    ap.add_argument("--arm", choices=("invariant", "invariant_union", "originals", "universal"), required=True)
    ap.add_argument("--variant", required=True); ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--output-root", default="results/MAGNN_imdb_v2")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--samples", type=int)
    ap.add_argument("--no-mask", action="store_true", help="do not drop the training query's own edge from its instances")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=0)
    args = ap.parse_args()
    if args.task not in TASKS[args.dataset]:
        raise SystemExit(f"{args.task} is not a {args.dataset} task; choose from {TASKS[args.dataset]}")
    data_root = args.data_root or DEFAULT_ROOT[args.dataset]
    cfg = dict(CONTRACT)
    for kk in ("epochs", "patience", "samples"):
        if getattr(args, kk) is not None:
            cfg[kk] = getattr(args, kk)
    cfg["mask_query_edge"] = not args.no_mask      # head masked by its true target for every candidate, tail by the head
    sub = {"invariant": "invariant", "invariant_union": "invariant_union"}.get(args.arm, "stock")
    bundle_dir = data_root / args.task / sub / args.variant if args.dataset == "IMDB" else data_root / sub / args.variant
    run_dir = Path(args.output_root) / args.arm / args.dataset / args.task / args.variant / f"seed_{args.seed}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir}"); return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.task} arm={args.arm} variant={args.variant} seed={args.seed}", flush=True)
    set_deterministic(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    b = LPBundle(bundle_dir)
    b.alias = b.alias_map(args.dataset, args.task); b.drop = b.label_derived_metapaths(args.dataset, args.task)
    print(f"[mask] alias={'movie<->link' if (args.dataset, args.task) in ALIAS else 'identity'} label-derived metapaths dropped for training heads: "
          f"mode0 {[b.groups[0].names[i] for i in sorted(b.drop[0])]} mode1 {[b.groups[1].names[i] for i in sorted(b.drop[1])]}", flush=True)
    print(f"[data] {bundle_dir}: mode0 {b.groups[0].M} / mode1 {b.groups[1].M} metapaths, program {b.summary['program_hash'][:12]}, "
          f"physical {b.summary['physical_hash'][:12]}, supervision {b.summary['supervision_hash'][:12]}, train {len(b.pos['train'])}", flush=True)
    feats = b.feature_tensors(device)
    off = b.groups[0].num_edge_type
    etypes_lists = [b.groups[0].etypes_list, [[e + off for e in et] for et in b.groups[1].etypes_list]]
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = MAGNN_lp([b.groups[0].M, b.groups[1].M], off + b.groups[1].num_edge_type, etypes_lists, [int(f.shape[1]) for f in feats],
                     cfg["hidden_dim"], cfg["hidden_dim"], cfg["num_heads"], cfg["attn_vec_dim"], cfg["rnn_type"], cfg["dropout"],
                     num_layers=cfg["K"]).to(device)
    init_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr_pos, tr_neg = b.pos["train"], b.neg["train"]

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
        order = np.random.default_rng(args.seed * 1000003 + epoch).permutation(len(tr_pos))
        for bi, s in enumerate(range(0, len(order), cfg["batch_size"])):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            idx = order[s:s + cfg["batch_size"]]; p = tr_pos[idx]; n = tr_neg[idx]
            rng = np.random.default_rng((args.seed * 1000003 + epoch) * 4099 + bi)
            hm, tm = query_masks(b, p) if cfg["mask_query_edge"] else (None, None)
            pos_sc = score_pairs(model, b, feats, p[:, 0], p[:, 1], cfg["samples"], rng, device, hm, tm)
            nh = np.repeat(p[:, 0], n.shape[1])
            hm, tm = (np.repeat(hm, n.shape[1], axis=0), np.repeat(tm, n.shape[1], axis=0)) if cfg["mask_query_edge"] else (None, None)
            neg_sc = score_pairs(model, b, feats, nh, n.ravel(), cfg["samples"], rng, device, hm, tm)
            loss = -(F.logsigmoid(pos_sc).mean() + F.logsigmoid(-neg_sc).mean())
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_step is None:
                first_step = {"logits_hash": hashlib.sha256(torch.cat([pos_sc, neg_sc]).detach().cpu().numpy().tobytes()).hexdigest(),
                              "loss": float(loss), "gradient_hash": gradient_hash(model), "init_hash": init_hash}
            optimizer.step(); losses.append(float(loss))
        vm, _ = evaluate(model, b, feats, "val", cfg["samples"], args.seed, device)
        sh = state_dict_hash(model.state_dict())
        trace.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_mrr": vm["mrr"], "val_hits@1": vm["hits@1"], "state_hash": sh})
        print(f"[epoch {epoch + 1}] loss={np.mean(losses):.4f} val_mrr={vm['mrr']:.4f} val_h@1={vm['hits@1']:.4f} state={sh[:10]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if vm["mrr"] > best:
            best, bad, best_epoch, best_state = vm["mrr"], 0, epoch + 1, copy.deepcopy(model.state_dict())
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
    metrics, test_scores = evaluate(model, b, feats, "test", cfg["samples"], args.seed, device)
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    np.savez(run_dir / "scores.npz", queries=b.pos["test"][:, 0], positive_tails=b.pos["test"][:, 1], scores=test_scores)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if state_path.exists():
        state_path.unlink()
    metrics.update({"train_time_sec": train_sec, "epochs": len(trace), "best_epoch": best_epoch, "best_val_score": best, "select_on": "val_mrr",
                    "inference_time_sec": inf_sec, "eval_protocol": "full_target_type" if args.dataset == "DBLP" else "sampled",
                    "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
                    **model_size(model)})
    json_dump({"dataset": args.dataset, "task": args.task, "variant": args.variant, "seed": args.seed, "arm": args.arm, "model": "MAGNN",
               "run_dir": str(run_dir), "physical_hash": b.summary["physical_hash"], "semantic_hash": b.summary["program_hash"],
               "supervision_hash": b.summary["supervision_hash"], "feature_hash": b.summary["feature_hash"] if b.has_features else "identity",
               "bundle": {"dir": str(bundle_dir), "mode0": b.groups[0].names, "mode1": b.groups[1].names, "cap": b.summary["cap"]},
               "config": cfg, "contract": {"select_on": "val_mrr", "loss": "neg_logsigmoid", "score": "dot",
                                           "features": "canonical pooled BoW" if b.has_features else "sparse identity per type",
                                           "eval_protocol": "full_target_type" if args.dataset == "DBLP" else "sampled",
                                           "mask_query_edge": cfg["mask_query_edge"],
                                           "mask_alias": ALIAS.get((args.dataset, args.task)), "mask_label_derived": sorted(LABEL_DERIVED.get((args.dataset, args.task), ()))},
               "first_step_audit": first_step, "checkpoint_hash": state_dict_hash(model.state_dict()), "metrics": metrics}, run_dir / "metrics.json")
    json_dump(trace, run_dir / "trace.json")
    print(f"[done] {args.task} {args.arm}/{args.variant} seed={args.seed} test_mrr={metrics['mrr']:.4f} h@1={metrics['hits@1']:.4f} "
          f"best_epoch={best_epoch}/{len(trace)} train={train_sec:.0f}s peak={peak_train['peak_gpu_mib']} ckpt={state_dict_hash(model.state_dict())[:12]}", flush=True)


if __name__ == "__main__":
    main()
