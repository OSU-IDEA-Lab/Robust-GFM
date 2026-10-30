#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for MAGNN link prediction on the v2 STOCK bundles.

    --dataset DBLP --task paper_conference --variants v1,v2,v3
    --dataset IMDB --task movie_year       --variants v1,v2,v3,v4

One ``MAGNN_lp`` over the UNION of the variants' stock metapath vocabularies on both sides
(mode0 = head-type-centred, mode1 = target-type-centred; a metapath absent from a variant has
no instances there — the shared-slot device of ``run_freebase_magnn_nc_augmentation.py``),
trained on every variant's own PHYSICAL graph in seeded, shuffled super-epochs on the SAME
supervision (task-data hashes asserted equal), selecting on the mean validation MRR across
variants. Same contract as ``run_magnn_lp.py`` (loss, query-edge masking for every candidate,
negatives, features); ``--epochs`` / ``--patience`` are variant-epochs divided by
the variant count. Resumable per super-epoch. Writes one ``metrics.json`` + ``scores.npz`` per
variant (RGCN aggregator layout, so the pairwise Kendall τ over per-query candidate scores is
computed like every other arm) and ``joint_seed_<s>.json`` with the in-run pairwise τ.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
import dgl  # noqa: E402
from model.MAGNN_lp import MAGNN_lp  # noqa: E402
from inv_rgcn.metrics import ranking_metrics  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402
from run_magnn_lp import ALIAS, CONTRACT, DEFAULT_ROOT, LABEL_DERIVED, TASKS, Group, LPBundle, query_masks  # noqa: E402


class UnionVocab:
    """Union metapath vocabulary of one group (mode0 / mode1) across the variants' bundles."""

    def __init__(self, groups: dict):           # variant -> Group
        self.names, self.widths = [], []
        for g in groups.values():
            for n, w in zip(g.names, g.widths):
                if n not in self.names:
                    self.names.append(n); self.widths.append(w)
        self.M = len(self.names)
        self.slot_of = {v: [self.names.index(n) for n in g.names] for v, g in groups.items()}
        self.etypes_list, base = [], 0
        for w in self.widths:
            k = w - 1; et = [base + 2 * j for j in range(k // 2)]; self.etypes_list.append(et + [e + 1 for e in reversed(et)]); base += k
        self.num_edge_type = base
        self.empty = [np.empty((0, w), dtype=np.int64) for w in self.widths]


def build_mode_union(group: Group, vocab: UnionVocab, slot_of, nodes, k, rng, device, mask_partner=None, drop=frozenset()):
    """``run_magnn_lp.build_mode`` over the union slots; slots this bundle lacks get an empty graph over the batch nodes."""
    srcs = sorted({int(n) for n in nodes}); emap = {n: q for q, n in enumerate(srcs)}
    g_list = [None] * vocab.M; idx_list = [None] * vocab.M; mapped_list = [None] * vocab.M
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
                    r = r[~np.isin(r, mask_partner[j]).any(axis=1)]
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
        s = slot_of[i]
        g_list[s], idx_list[s], mapped_list[s] = g.to(device), torch.as_tensor(arr, dtype=torch.long, device=device), np.asarray([mapping[int(n)] for n in nodes])
    for s in range(vocab.M):
        if g_list[s] is None:
            g_list[s] = dgl.graph(([], []), num_nodes=len(srcs)).to(device)
            idx_list[s] = torch.as_tensor(vocab.empty[s], device=device)
            mapped_list[s] = np.asarray([emap[int(n)] for n in nodes])
    return g_list, idx_list, mapped_list


def score_pairs(model, b: LPBundle, vocabs, v, feats, heads, tails, k, rng, device, head_mask=None, tail_mask=None):
    g0, i0, m0 = build_mode_union(b.groups[0], vocabs[0], vocabs[0].slot_of[v], heads, k, rng, device, head_mask, b.drop[0] if head_mask is not None else frozenset())
    g1, i1, m1 = build_mode_union(b.groups[1], vocabs[1], vocabs[1].slot_of[v], tails, k, rng, device, tail_mask, b.drop[1] if tail_mask is not None else frozenset())
    [hm, ht], _ = model(([g0, g1], feats, b.node_types, [i0, i1], [m0, m1]))
    return (hm * ht).sum(-1)


@torch.no_grad()
def evaluate(model, b, vocabs, v, feats, split, k, seed, device, bs=64):
    model.eval()
    pos, neg = b.pos[split], b.neg[split]
    out = np.zeros((len(pos), 1 + neg.shape[1]), dtype=np.float32)
    for s in range(0, len(pos), bs):
        p = pos[s:s + bs]; n = neg[s:s + bs]
        rng = np.random.default_rng((seed * 7919 + s) % (2 ** 32))
        heads = np.concatenate([p[:, 0], np.repeat(p[:, 0], n.shape[1])]); tails = np.concatenate([p[:, 1], n.ravel()])
        sc = score_pairs(model, b, vocabs, v, feats, heads, tails, k, rng, device).float().cpu().numpy()
        out[s:s + bs, 0] = sc[:len(p)]; out[s:s + bs, 1:] = sc[len(p):].reshape(len(p), -1)
    m = ranking_metrics(out[:, 0], out[:, 1:]); m["num_candidates"] = int(out.shape[1])
    return m, out


def row_tau(a, b):
    v = [kendalltau(x, y, nan_policy="omit")[0] for x, y in zip(a, b)]
    v = [t for t in v if not np.isnan(t)]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("IMDB", "DBLP"), default="DBLP")
    ap.add_argument("--task", choices=("movie_director", "movie_year", "paper_conference"), default="paper_conference")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--variants", default="v1,v2,v3"); ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--output-root", default="results/MAGNN_dblp_v2/augmentation")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--samples", type=int)
    ap.add_argument("--no-mask", action="store_true"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--max-train-batches", type=int, default=0)
    args = ap.parse_args()
    if args.task not in TASKS[args.dataset]:
        raise SystemExit(f"{args.task} is not a {args.dataset} task")
    data_root = args.data_root or DEFAULT_ROOT[args.dataset]
    cfg = dict(CONTRACT)
    for kk in ("epochs", "patience", "samples"):
        if getattr(args, kk) is not None:
            cfg[kk] = getattr(args, kk)
    cfg["mask_query_edge"] = not args.no_mask      # the query's own edge dropped for every candidate (run_magnn_lp.py)
    variants = args.variants.split(",")
    out_root = Path(args.output_root) / args.dataset / args.task
    if all((out_root / v / f"seed_{args.seed}" / "metrics.json").exists() for v in variants):
        print("[skip] all variants done"); return
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[aug] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.dataset}/{args.task} variants={variants} seed={args.seed}", flush=True)
    set_deterministic(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundles = {v: LPBundle(data_root / args.task / "stock" / v if args.dataset == "IMDB" else data_root / "stock" / v) for v in variants}
    ref = bundles[variants[0]]
    for v, b in bundles.items():
        b.alias = b.alias_map(args.dataset, args.task); b.drop = b.label_derived_metapaths(args.dataset, args.task)
        if b.summary["supervision_hash"] != ref.summary["supervision_hash"]:
            raise SystemExit(f"{v}: supervision hash differs -- not a shared-split cell")
        if b.summary.get("feature_hash") != ref.summary.get("feature_hash"):
            raise SystemExit(f"{v}: feature hash differs")
        for k in ("train", "val", "test"):
            assert np.array_equal(b.pos[k], ref.pos[k]) and np.array_equal(b.neg[k], ref.neg[k]), f"{v}: {k} supervision differs"
    vocabs = [UnionVocab({v: b.groups[0] for v, b in bundles.items()}), UnionVocab({v: b.groups[1] for v, b in bundles.items()})]
    print(f"[aug] union vocabulary mode0 {vocabs[0].M} / mode1 {vocabs[1].M} metapaths: " +
          ", ".join(f"{v}={b.groups[0].M}+{b.groups[1].M}" for v, b in bundles.items()), flush=True)
    print(f"[aug] mode0 slots: {vocabs[0].names}\n[aug] mode1 slots: {vocabs[1].names}", flush=True)
    feats = ref.feature_tensors(device)
    off = vocabs[0].num_edge_type
    etypes_lists = [vocabs[0].etypes_list, [[e + off for e in et] for et in vocabs[1].etypes_list]]
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = MAGNN_lp([vocabs[0].M, vocabs[1].M], off + vocabs[1].num_edge_type, etypes_lists, [int(f.shape[1]) for f in feats],
                     cfg["hidden_dim"], cfg["hidden_dim"], cfg["num_heads"], cfg["attn_vec_dim"], cfg["rnn_type"], cfg["dropout"],
                     num_layers=cfg["K"]).to(device)
    init_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(variants); super_epochs = max(1, round(cfg["epochs"] / n)); patience = max(1, round(cfg["patience"] / n))
    print(f"[aug] budget {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, patience {patience}", flush=True)

    state_path = out_root / f"joint_seed_{args.seed}_training_state.pt"
    trace, best, best_state, best_super, bad, start_se, train_sec, first = [], -1.0, None, 0, 0, 1, 0.0, None
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        trace, best, best_state, best_super, bad = st["trace"], st["best"], st["best_state"], st["best_super"], st["bad"]
        start_se, train_sec, first = st["super_epoch"] + 1, st["train_sec"], st["first"]
        print(f"[resume] from super-epoch {start_se}", flush=True)
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(start_se, super_epochs + 1):
        model.train(); losses = []
        for v in np.random.default_rng(args.seed * 31 + se).permutation(variants):
            v = str(v); b = bundles[v]; tp, tn = b.pos["train"], b.neg["train"]
            order = np.random.default_rng(args.seed * 1000003 + se * 31 + variants.index(v)).permutation(len(tp))
            for bi, s in enumerate(range(0, len(order), cfg["batch_size"])):
                if args.max_train_batches and bi >= args.max_train_batches:
                    break
                idx = order[s:s + cfg["batch_size"]]; p = tp[idx]; ng = tn[idx]
                rng = np.random.default_rng((args.seed * 1000003 + se) * 4099 + bi * 7 + variants.index(v))
                hm, tm = query_masks(b, p) if cfg["mask_query_edge"] else (None, None)
                pos_sc = score_pairs(model, b, vocabs, v, feats, p[:, 0], p[:, 1], cfg["samples"], rng, device, hm, tm)
                nh = np.repeat(p[:, 0], ng.shape[1])
                hm, tm = (np.repeat(hm, ng.shape[1], axis=0), np.repeat(tm, ng.shape[1], axis=0)) if cfg["mask_query_edge"] else (None, None)
                neg_sc = score_pairs(model, b, vocabs, v, feats, nh, ng.ravel(), cfg["samples"], rng, device, hm, tm)
                loss = -(F.logsigmoid(pos_sc).mean() + F.logsigmoid(-neg_sc).mean())
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if first is None:
                    first = {"variant": v, "loss": float(loss), "gradient_hash": gradient_hash(model), "init_hash": init_hash,
                             "logits_hash": hashlib.sha256(torch.cat([pos_sc, neg_sc]).detach().cpu().numpy().tobytes()).hexdigest()}
                optimizer.step(); losses.append(float(loss))
        vals = {v: evaluate(model, bundles[v], vocabs, v, feats, "val", cfg["samples"], args.seed, device)[0]["mrr"] for v in variants}
        score = float(np.mean(list(vals.values()))); sh = state_dict_hash(model.state_dict())
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)), "val_mrr_per_variant": vals,
                      "val_mrr_mean": score, "state_hash": sh})
        print(f"[super {se}/{super_epochs}] loss={np.mean(losses):.4f} val_mrr={score:.4f} per-variant={[round(x, 4) for x in vals.values()]} "
              f"state={sh[:10]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_super, bad, best_state = score, se, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(), "trace": trace, "best": best, "best_state": best_state,
                    "best_super": best_super, "bad": bad, "super_epoch": se, "train_sec": train_sec + (time.perf_counter() - t0),
                    "first": first}, state_path)
        if bad >= patience:
            print("[early stop]", flush=True); break
    train_sec += time.perf_counter() - t0; peak_train = gpu_peak(device)
    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    per, scores = {}, {}
    for v in variants:
        per[v], scores[v] = evaluate(model, bundles[v], vocabs, v, feats, "test", cfg["samples"], args.seed, device)
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    taus = {f"{a}|{c}": row_tau(scores[a], scores[c]) for a, c in itertools.combinations(variants, 2)}
    ckpt = state_dict_hash(model.state_dict()); size = model_size(model)
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace), "best_epoch": best_super * n,
              "best_super_epoch": best_super, "best_val_score": best, "select_on": "val_mrr_mean", "inference_time_sec": inf_sec,
              "eval_protocol": "full_target_type" if args.dataset == "DBLP" else "sampled",
              "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"], **size}
    for v in variants:
        b = bundles[v]; run_dir = out_root / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / "scores.npz", queries=b.pos["test"][:, 0], positive_tails=b.pos["test"][:, 1], scores=scores[v])
        json_dump({"dataset": args.dataset, "task": args.task, "variant": v, "seed": args.seed, "arm": "augmentation", "model": "MAGNN",
                   "run_dir": str(run_dir), "physical_hash": b.summary["physical_hash"], "semantic_hash": b.summary["program_hash"],
                   "supervision_hash": b.summary["supervision_hash"], "feature_hash": b.summary["feature_hash"] if b.has_features else "identity",
                   "bundle": {"dir": str(b.dir), "mode0": b.groups[0].names, "mode1": b.groups[1].names, "cap": b.summary["cap"]},
                   "config": {**cfg, "union_mode0": vocabs[0].names, "union_mode1": vocabs[1].names, "variants": variants, "super_epochs": super_epochs},
                   "contract": {"select_on": "val_mrr_mean", "loss": "neg_logsigmoid", "score": "dot",
                                "features": "canonical pooled BoW" if b.has_features else "sparse identity per type",
                                "eval_protocol": shared["eval_protocol"], "mask_query_edge": cfg["mask_query_edge"],
                                "mask_alias": ALIAS.get((args.dataset, args.task)), "mask_label_derived": sorted(LABEL_DERIVED.get((args.dataset, args.task), ()))},
                   "first_step_audit": first, "checkpoint_hash": ckpt, "metrics": {**per[v], **shared}}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / f"joint_seed_{args.seed}.pt")
    if state_path.exists():
        state_path.unlink()
    json_dump({"variants": variants, "union_mode0": vocabs[0].names, "union_mode1": vocabs[1].names, "per_variant_test": per,
               "pairwise_kendall_tau": taus, "mean_tau": float(np.mean(list(taus.values()))), "shared": shared, "checkpoint_hash": ckpt},
              out_root / f"joint_seed_{args.seed}.json")
    print("[done] augmentation seed=%d: " % args.seed + " ".join(f"{v}={per[v]['mrr']:.4f}" for v in variants) +
          f" | mean tau={np.mean(list(taus.values())):.4f} best_super={best_super}/{len(trace)} train={train_sec:.0f}s ckpt={ckpt[:12]}", flush=True)


if __name__ == "__main__":
    main()
