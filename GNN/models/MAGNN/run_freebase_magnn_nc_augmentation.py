#!/usr/bin/env python3
"""Data-augmentation (joint-training) arm for MAGNN Freebase NC on the v2 STOCK bundles.

One ``MAGNN_nc_mb`` over the UNION of the variants' stock metapath vocabularies (a metapath
absent from a variant simply has no instances there — the same shared-slot device the
earlier harness used), trained on every variant's own graph in seeded, shuffled
super-epochs on the SAME split (supervision hashes asserted equal), selecting on the mean
validation micro-F1 across variants. Same contract as ``run_freebase_magnn_nc.py``;
``--epochs`` / ``--patience`` are variant-epochs divided by the variant count. Writes one
``metrics.json`` + ``scores.npz`` per variant (aggregator layout) and ``joint_seed_<s>.json``
with the pairwise Kendall τ over per-node test logits.
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
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "common"))
import dgl  # noqa: E402
from model.MAGNN_nc_mb import MAGNN_nc_mb  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402
from run_freebase_magnn_nc import CONTRACT, DEFAULT_ROOT, Bundle, features_identity, sample_rows  # noqa: E402


def parse_minibatch_union(b: Bundle, slot_of: list, M: int, batch_pos, k, rng, device, widths=None):
    """Like run_freebase_magnn_nc.parse_minibatch but over M union slots; slot_of[i] = union index of
    the bundle's metapath i. Slots the bundle lacks get an empty graph over the batch nodes."""
    g_list = [None] * M; idx_list = [None] * M; mapped_list = [None] * M
    local = {}
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
            g = dgl.graph(([edges[j][1] for j in order], [edges[j][0] for j in order]), num_nodes=len(mapping)); arr = arr[order]
        else:
            g = dgl.graph(([], []), num_nodes=len(mapping))
        local[slot_of[i]] = (g.to(device), torch.as_tensor(arr, dtype=torch.long, device=device),
                             np.asarray([mapping[int(b.target_nodes[p])] for p in batch_pos]))
    srcs = sorted({int(b.target_nodes[p]) for p in batch_pos}); emap = {n: j for j, n in enumerate(srcs)}
    for s in range(M):
        if s in local:
            g_list[s], idx_list[s], mapped_list[s] = local[s]
        else:
            w = widths[s] if widths else 3
            g_list[s], idx_list[s], mapped_list[s] = (dgl.graph(([], []), num_nodes=len(srcs)).to(device),
                torch.empty((0, w), dtype=torch.long, device=device), np.asarray([emap[int(b.target_nodes[p])] for p in batch_pos]))
    return g_list, idx_list, mapped_list


@torch.no_grad()
def evaluate(model, b, slot_of, M, feats, type_mask, positions, k, seed, device, bs=64):
    model.eval(); out = []
    for s in range(0, len(positions), bs):
        batch = positions[s:s + bs]
        rng = np.random.default_rng((seed * 7919 + s) % (2 ** 32))
        g, idx, mapped = parse_minibatch_union(b, slot_of, M, batch, k, rng, device, WIDTHS)
        logits, _ = model((g, feats, type_mask, idx, mapped)); out.append(logits.float().cpu())
    logits = torch.cat(out).numpy(); y = b.labels[positions]; pred = logits.argmax(1)
    return {"micro_f1": float(f1_score(y, pred, average="micro")), "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean())}, logits


WIDTHS = None


def row_tau(a, b):
    v = [kendalltau(x, y, nan_policy="omit")[0] for x, y in zip(a, b)]
    v = [t for t in v if not np.isnan(t)]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variants", default="unchanged,exact_2,exact_3")
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--output-root", default="results/MAGNN_freebase_v2/augmentation")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--samples", type=int)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--dataset", choices=("FREEBASE", "IMDB"), default="FREEBASE"); ap.add_argument("--group", default="")
    args = ap.parse_args()
    cfg = dict(CONTRACT)
    for kk in ("epochs", "patience", "samples"):
        if getattr(args, kk) is not None:
            cfg[kk] = getattr(args, kk)
    variants = args.variants.split(",")
    out_root = Path(args.output_root)
    if all((out_root / args.dataset / "nc" / v / f"seed_{args.seed}" / "metrics.json").exists() for v in variants):
        print("[skip] all variants done"); return
    print(f"[aug] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} variants={variants} seed={args.seed}", flush=True)
    set_deterministic(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundles = {v: Bundle(args.data_root / "stock" / v, group=args.group) for v in variants}
    ref = bundles[variants[0]]
    for v, b in bundles.items():
        if b.summary["supervision_hash"] != ref.summary["supervision_hash"]:
            raise SystemExit(f"{v}: supervision hash differs -- not a shared-split cell")
    union = []
    for b in bundles.values():
        for n in b.metapath_names:
            if n not in union:
                union.append(n)
    M = len(union)
    slot_of = {v: [union.index(n) for n in b.metapath_names] for v, b in bundles.items()}
    print(f"[aug] union vocabulary {M} metapaths: " + ", ".join(f"{v}={b.M}" for v, b in bundles.items()), flush=True)
    feats, in_dims = features_identity(ref, device); type_mask = ref.node_types
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    # etypes per union slot: take each metapath's width from whichever bundle has it
    width_of = {}
    for b in bundles.values():
        for n, w in zip(b.metapath_names, b.gsum.get("widths", [3] * b.M)):
            width_of[n] = w
    global WIDTHS
    WIDTHS = [width_of[n] for n in union]
    etypes_list, base = [], 0
    for n in union:
        k = width_of[n] - 1; et = [base + 2 * j for j in range(k // 2)]; etypes_list.append(et + [e + 1 for e in reversed(et)]); base += k
    model = MAGNN_nc_mb(M, base, etypes_list, in_dims, cfg["hidden_dim"], ref.num_classes,
                        cfg["num_heads"], cfg["attn_vec_dim"], cfg["rnn_type"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(variants); super_epochs = max(1, round(cfg["epochs"] / n)); patience = max(1, round(cfg["patience"] / n))
    print(f"[aug] budget {cfg['epochs']} variant-epochs -> {super_epochs} super-epochs, patience {patience}", flush=True)
    order_rng = np.random.RandomState(args.seed)
    best, best_state, best_super, bad, trace, first = -1.0, None, 0, 0, [], None
    gpu_reset(device); t0 = time.perf_counter()
    for se in range(1, super_epochs + 1):
        model.train(); losses = []
        for v in order_rng.permutation(variants):
            b = bundles[v]; tp = b.split["train_idx"]
            order = np.random.default_rng(args.seed * 1000003 + se * 31 + variants.index(v)).permutation(len(tp))
            for bi, s in enumerate(range(0, len(order), cfg["batch_size"])):
                if args.max_train_batches and bi >= args.max_train_batches:
                    break
                batch = tp[order[s:s + cfg["batch_size"]]]
                rng = np.random.default_rng((args.seed * 1000003 + se) * 4099 + bi * 7 + variants.index(v))
                g, idx, mapped = parse_minibatch_union(b, slot_of[v], M, batch, cfg["samples"], rng, device, WIDTHS)
                logits, _ = model((g, feats, type_mask, idx, mapped))
                loss = F.cross_entropy(logits, torch.as_tensor(b.labels[batch], device=device))
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if first is None:
                    first = {"variant": str(v), "loss": float(loss), "gradient_hash": gradient_hash(model)}
                optimizer.step(); losses.append(float(loss))
        vals = [evaluate(model, bundles[v], slot_of[v], M, feats, type_mask, bundles[v].split["val_idx"], cfg["samples"], args.seed, device)[0]["micro_f1"] for v in variants]
        score = float(np.mean(vals))
        trace.append({"super_epoch": se, "variant_epoch": se * n, "mean_train_loss": float(np.mean(losses)), "val_micro_per_variant": dict(zip(variants, vals)),
                      "val_micro_mean": score, "state_hash": state_dict_hash(model.state_dict())})
        print(f"[super {se}/{super_epochs}] loss={np.mean(losses):.4f} val_micro={score:.4f} per-variant={[round(x, 4) for x in vals]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_super, bad, best_state = score, se, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print("[early stop]", flush=True); break
    train_sec = time.perf_counter() - t0; peak_train = gpu_peak(device)
    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    per, scores = {}, {}
    for v in variants:
        m, sc = evaluate(model, bundles[v], slot_of[v], M, feats, type_mask, bundles[v].split["test_idx"], cfg["samples"], args.seed, device)
        per[v] = m; scores[v] = sc
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    taus = {f"{a}|{c}": row_tau(scores[a], scores[c]) for a, c in itertools.combinations(variants, 2)}
    ckpt = state_dict_hash(model.state_dict()); size = model_size(model)
    shared = {"train_time_sec": train_sec, "epochs": len(trace) * n, "super_epochs": len(trace), "best_epoch": best_super * n,
              "best_super_epoch": best_super, "best_val": best, "select_on": "val_micro_f1_mean", "inference_time_sec": inf_sec,
              "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
              "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"], **size}
    for v in variants:
        b = bundles[v]; run_dir = out_root / args.dataset / "nc" / v / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / "scores.npz", ids=b.target_nodes[b.split["test_idx"]], scores=scores[v])
        json_dump({"dataset": args.dataset, "task": "nc", "variant": v, "seed": args.seed, "arm": "augmentation", "model": "MAGNN",
                   "run_dir": str(run_dir), "physical_hash": b.summary["physical_hash"], "semantic_hash": b.summary["program_hash"],
                   "supervision_hash": b.summary["supervision_hash"], "feature_hash": "identity",
                   "config": {**cfg, "union_metapaths": M, "variants": variants, "super_epochs": super_epochs},
                   "contract": {"select_on": "val_micro_f1_mean", "loss": "cross_entropy", "features": "sparse identity per type"},
                   "first_step_audit": first, "checkpoint_hash": ckpt, "metrics": {**per[v], **shared}}, run_dir / "metrics.json")
        json_dump(trace, run_dir / "trace.json")
    torch.save(model.state_dict(), out_root / args.dataset / "nc" / f"joint_seed_{args.seed}.pt")
    json_dump({"variants": variants, "union_metapaths": union, "per_variant_test": per, "pairwise_kendall_tau": taus,
               "mean_tau": float(np.mean(list(taus.values()))), "shared": shared, "checkpoint_hash": ckpt},
              out_root / args.dataset / "nc" / f"joint_seed_{args.seed}.json")
    print("[done] augmentation seed=%d: " % args.seed + " ".join(f"{v}={per[v]['micro_f1']:.4f}" for v in variants) +
          f" | mean tau={np.mean(list(taus.values())):.4f} best_super={best_super}/{len(trace)}", flush=True)


if __name__ == "__main__":
    main()
