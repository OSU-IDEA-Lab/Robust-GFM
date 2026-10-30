#!/usr/bin/env python3
"""One CMPNN link-prediction runner for the three per-variant arms.

    --graph semantic                       invariant arm (compiled program)
    --graph physical                       originals arm (the variant's raw graph)
    --graph physical --variant universal   universal arm (CMPNN*)

Reads the RGCN structural bundles (``inv_cmpnn.bundles``), trains the repository CMPNN
with an order-independent aggregation (``inv_cmpnn.model``), selects on validation MRR,
and writes the RGCN one-runner layout so ``RGCN/aggregate_rgcn_structural.py`` aggregates
CMPNN results unchanged:

    <output-root>/<dataset>/<task>/<variant>/seed_<seed>/{metrics.json, scores.npz,
                                                         checkpoint.pt, trace.json}

Datasets / protocols
  IMDB movie_year, movie_director : bundle negatives, K=19, sampled ranking
  DBLP paper_conference           : full ranking against all 20 conferences
  WORDNET link_prediction         : filtered ranking against all entities, both
                                    directions; training negatives sampled per epoch
                                    from the seeded generator (identical across variants)

Resumable: ``last_training_state.pt`` per epoch; a finished run (metrics.json) is skipped.
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
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "common"))

from inv_cmpnn.bundles import LPBundle, load_bundle  # noqa: E402
from inv_cmpnn.model import build_graph, build_model  # noqa: E402
from inv_rgcn.metrics import ranking_metrics, full_ranking_metrics  # noqa: E402
from inv_rgcn.utils import gradient_hash, json_dump, set_deterministic, state_dict_hash  # noqa: E402
from rgcn_resource_metrics import gpu_peak, gpu_reset, model_size  # noqa: E402

DEFAULTS = {
    # the baseline runners' contracts: 32/32, 6 layers, DistMult, dependent=False;
    # IMDb/DBLP = PNA + remove_one_hop (run_CMPNN_IMDB_*.py, run_CMPNN_DBLP_pc.py),
    # WordNet = SUM aggregation, exact-edge removal only (run_CMPNN_wordnet_lp.py)
    # lr: Adam default 1e-3 in the IMDb/DBLP baseline runners, 5e-3 in the WordNet one.
    # ONE contract per dataset, shared by all four arms (invariant / originals / universal /
    # augmentation): same architecture, loss, lr, batch, epoch budget, early stopping on
    # validation MRR with the same patience, same seeds, same splits/negatives/protocol.
    ("IMDB", "movie_year"): dict(epochs=20, patience=5, batch_size=8, lr=1e-3, aggregate_func="pna", remove_one_hop=True),
    ("IMDB", "movie_director"): dict(epochs=20, patience=5, batch_size=8, lr=1e-3, aggregate_func="pna", remove_one_hop=True),
    ("DBLP", "paper_conference"): dict(epochs=20, patience=5, batch_size=8, lr=1e-3, aggregate_func="pna", remove_one_hop=True),
    ("WORDNET", "link_prediction"): dict(epochs=10, patience=3, batch_size=64, lr=5e-3, aggregate_func="sum", remove_one_hop=False, chunk_edges=400_000),
}
BUNDLE_ROOTS = {
    ("IMDB", "movie_year"): "RGCN_structural_v10_year_canonfeat/IMDB/movie_year",
    ("IMDB", "movie_director"): "RGCN_structural_v10_year_canonfeat/IMDB/movie_director",
    ("DBLP", "paper_conference"): "RGCN_structural_v8_dblp/DBLP/paper_conference",
    ("WORDNET", "link_prediction"): "RGCN_structural/WORDNET/universal_parity_full/full/intersection",
}


# ----------------------------------------------------------------------------- data
def bundle_path(args) -> Path:
    if args.bundle:
        return Path(args.bundle)
    return Path(args.data_root) / BUNDLE_ROOTS[(args.dataset, args.task)] / args.variant


class QueryBatches:
    """Deterministic (seeded) batches of (h, r, t_candidates) with the positive at column 0."""

    def __init__(self, b: LPBundle, graph: str, split: str, batch_size: int, seed: int,
                 neg_k: int, head_corruption: bool, shuffle: bool):
        self.b, self.graph, self.split = b, graph, split
        self.pos = b.positives(split, graph)                    # [N, 3] (h, r, t)
        self.neg = {"train": b.train_neg, "val": b.val_neg, "test": b.test_neg}[split]
        self.batch_size, self.seed, self.neg_k = batch_size, seed, neg_k
        self.head_corruption, self.shuffle = head_corruption, shuffle
        self.epoch = 0

    def __len__(self):
        return (self.pos.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # The counter advances at the START of an epoch so a consumer that breaks out early
        # (--max-train-batches) still sees a fresh order next epoch; resume sets it to the
        # next epoch index explicitly.
        g = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        self.epoch += 1
        n = self.pos.shape[0]
        order = torch.randperm(n, generator=g) if self.shuffle else torch.arange(n)
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            h, r, t = self.pos[idx, 0], self.pos[idx, 1], self.pos[idx, 2]
            if self.neg is not None:                            # IMDb: bundle negatives
                negs = self.neg[idx]
                t_idx = torch.cat([t[:, None], negs], 1)
                h_idx = h[:, None].expand_as(t_idx)
            elif self.b.candidate_pool is not None:            # DBLP: every other conference
                pool = self.b.candidate_pool
                negs = torch.stack([pool[pool != tt] for tt in t])
                t_idx = torch.cat([t[:, None], negs], 1)
                h_idx = h[:, None].expand_as(t_idx)
            else:                                               # WordNet training: sampled
                negs = torch.randint(0, self.b.num_nodes, (len(idx), self.neg_k), generator=g)
                t_idx = torch.cat([t[:, None], negs], 1)
                h_idx = h[:, None].expand_as(t_idx).clone()
                if self.head_corruption:                        # NBFNet: half the rows corrupt the head
                    flip = torch.rand(len(idx), generator=g) < 0.5
                    h_neg = torch.cat([h[:, None], negs], 1)
                    t_fix = t[:, None].expand_as(t_idx)
                    h_idx = torch.where(flip[:, None], h_neg, h_idx)
                    t_idx = torch.where(flip[:, None], t_fix, t_idx)
            r_idx = r[:, None].expand_as(t_idx)
            yield idx, h_idx.contiguous(), t_idx.contiguous(), r_idx.contiguous()


def warm_match_index(model, graph, pos: torch.Tensor, device) -> None:
    """Build torchdrug's ``Graph.match`` inverted index before an RNG restore.

    ``Graph.match`` builds a perfect-hash ``Dictionary`` on its FIRST call per query type,
    drawing a data-dependent number of ``torch.randint`` values from the CUDA generator. A
    straight run pays that once, inside its first forward; a resumed process would pay it
    after the RNG state was restored and shift every later ``torch.rand`` boundary draw
    (resumed runs diverged from straight runs at the first post-resume
    batch with identical parameters, inputs and RNG state hashes). Building the index here
    -- with the same query patterns ``remove_easy_edges`` uses -- keeps the restored RNG
    trajectory identical to the straight run's. Straight runs are untouched.
    """
    h, r, t = pos[:1, 0:1].to(device), pos[:1, 1:2].to(device), pos[:1, 2:3].to(device)
    with torch.no_grad():
        model.remove_easy_edges(graph, h, t, r)


def loss_fn(scores: torch.Tensor) -> torch.Tensor:
    # the baseline runners' loss: -logσ(pos) - mean(logσ(-neg))
    return -(F.logsigmoid(scores[:, 0]).mean() + F.logsigmoid(-scores[:, 1:]).mean())


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_sampled(model, graph, batches: QueryBatches, device):
    """IMDb / DBLP: positive vs its candidate tails; returns metrics + [Q, 1+K] scores."""
    model.eval()
    all_scores = np.zeros((batches.pos.shape[0], 1 + (batches.neg.shape[1] if batches.neg is not None else batches.b.candidate_pool.numel() - 1)), dtype=np.float32)
    for idx, h, t, r in batches:
        s = model(graph, h.to(device), t.to(device), r.to(device))
        all_scores[idx.numpy()] = s.float().cpu().numpy()
    m = ranking_metrics(all_scores[:, 0], all_scores[:, 1:])
    m["num_candidates"] = int(all_scores.shape[1])
    return m, all_scores


def _filter_index(b: LPBundle, graph: str):
    """Known (h, r) -> set(t) and (t, r) -> set(h) over train+val+test, for filtered ranking."""
    known_t, known_h = {}, {}
    for split in ("train", "val", "test"):
        for h, r, t in b.positives(split, graph).tolist():
            known_t.setdefault((h, r), set()).add(t)
            known_h.setdefault((t, r), set()).add(h)
    return known_t, known_h


@torch.no_grad()
def evaluate_wordnet(model, graph, b: LPBundle, gname: str, split: str, device, batch_size: int,
                     tau_candidates: int, seed: int):
    """Filtered ranking against all entities, tail AND head corruption (standard KG protocol).

    Returns metrics, per-query ranks, and a [2Q, 1+C] score matrix (positive + C fixed
    seeded candidates per direction) that the aggregator uses for cross-variant τ.
    """
    model.eval()
    pos = b.positives(split, gname)
    known_t, known_h = _filter_index(b, gname)
    N = b.num_nodes
    g = torch.Generator().manual_seed(seed)
    cand = torch.randint(0, N, (pos.shape[0], tau_candidates), generator=g)
    ranks, tau_scores = [], []
    all_nodes = torch.arange(N)
    for direction in ("tail", "head"):
        for s in range(0, pos.shape[0], batch_size):
            p = pos[s:s + batch_size]
            h, r, t = p[:, 0], p[:, 1], p[:, 2]
            if direction == "tail":
                h_idx = h[:, None].expand(-1, N); t_idx = all_nodes[None].expand(len(p), -1); r_idx = r[:, None].expand(-1, N)
                truth = t; known = known_t; key_node = h
            else:
                t_idx = t[:, None].expand(-1, N); h_idx = all_nodes[None].expand(len(p), -1); r_idx = r[:, None].expand(-1, N)
                truth = h; known = known_h; key_node = t
            # the model requires a constant head per row; head corruption is expressed as
            # the inverse query, which negative_sample_to_tail does when h varies and t is
            # constant -> both layouts are valid inputs
            sc = model(graph, h_idx.contiguous().to(device), t_idx.contiguous().to(device), r_idx.contiguous().to(device)).float().cpu()
            for i in range(len(p)):
                row = sc[i].clone()
                pos_score = row[int(truth[i])].item()
                others = known[(int(key_node[i]), int(r[i]))] - {int(truth[i])}
                if others:
                    row[list(others)] = float("-inf")
                better = int((row > pos_score).sum()); tied = int((row == pos_score).sum()) - 1
                ranks.append(1.0 + better + tied / 2.0)
                tau_scores.append(np.concatenate([[pos_score], row[cand[s + i]].numpy()]))
    ranks = np.asarray(ranks)
    m = full_ranking_metrics(ranks, ks=(1, 3, 10))
    m["num_candidates"] = int(N)
    return m, ranks, np.stack(tau_scores).astype(np.float32)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/preprocessed")
    ap.add_argument("--bundle", default="", help="explicit bundle path (overrides data-root/dataset/task/variant)")
    ap.add_argument("--dataset", choices=("IMDB", "DBLP", "WORDNET"), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--graph", choices=("semantic", "physical"), default="semantic")
    ap.add_argument("--output-root", default="results/CMPNN_structural")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--batch-size", type=int)
    ap.add_argument("--aggregate-func", choices=("pna", "sum", "mean", "max")); ap.add_argument("--remove-one-hop", type=int, choices=(0, 1))
    ap.add_argument("--chunk-edges", type=int, help="deterministic chunked reduction slice (0 = whole mode)")
    ap.add_argument("--input-dim", type=int, default=32); ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, help="default: the dataset contract (IMDb/DBLP 1e-3, WordNet 5e-3)")
    ap.add_argument("--neg-k", type=int, default=32, help="WordNet training negatives per positive")
    ap.add_argument("--tau-candidates", type=int, default=100, help="WordNet: fixed candidates kept per query for tau")
    ap.add_argument("--select-on", default="val_mrr", choices=("val_mrr", "val_loss"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stock-aggregation", action="store_true", help="use the torch_scatter path (not bitwise deterministic)")
    ap.add_argument("--no-alias-mask", action="store_true")
    ap.add_argument("--gradient-checkpoint", action="store_true", help="recompute layer messages in backward (WordNet memory)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=0, help="smoke: cap batches per epoch")
    args = ap.parse_args()
    cfg = dict(DEFAULTS[(args.dataset, args.task)])
    for k in ("epochs", "patience", "batch_size", "lr", "aggregate_func", "remove_one_hop", "chunk_edges"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["remove_one_hop"] = bool(cfg["remove_one_hop"])
    cfg["chunk_edges"] = cfg.get("chunk_edges") or None

    run_dir = Path(args.output_root) / args.dataset / args.task / args.variant / f"seed_{args.seed}"
    if (run_dir / "metrics.json").exists() and not os.environ.get("CMPNN_FORCE"):
        print(f"[skip] {run_dir} already has metrics.json"); return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] start host={os.uname().nodename} {time.strftime('%Y-%m-%dT%H:%M:%S')} {args.dataset}/{args.task}/{args.variant} seed={args.seed} graph={args.graph}", flush=True)

    set_deterministic(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    b = load_bundle(bundle_path(args))
    summary = b.summary()
    print(f"[data] {json.dumps({k: summary[k] for k in ('num_nodes','num_physical_edges','num_semantic_edges','train_pos','neg_k')})} "
          f"phys={summary['physical_hash'][:12]} sem={summary['semantic_hash'][:12]} sup={summary['supervision_hash'][:12]}", flush=True)
    edges = b.edges(args.graph)
    R = b.num_relation(args.graph)
    graph = build_graph(edges, b.num_nodes, R, device)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = build_model(R, None if args.no_alias_mask else b.alias.to(device), input_dim=args.input_dim,
                        hidden_dim=args.hidden_dim, num_layers=args.layers,
                        deterministic=not args.stock_aggregation, gradient_checkpoint=args.gradient_checkpoint,
                        aggregate_func=cfg["aggregate_func"], remove_one_hop=cfg["remove_one_hop"], chunk_edges=cfg["chunk_edges"],
                        label_derived_relations=None if args.no_alias_mask else b.label_derived_ids(args.graph)).to(device)
    init_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    wn = args.dataset == "WORDNET"
    train_b = QueryBatches(b, args.graph, "train", cfg["batch_size"], args.seed, args.neg_k, head_corruption=wn, shuffle=True)
    val_b = QueryBatches(b, args.graph, "val", cfg["batch_size"], args.seed, args.neg_k, False, False)
    test_b = QueryBatches(b, args.graph, "test", cfg["batch_size"], args.seed, args.neg_k, False, False)

    def run_eval(split):
        if wn:
            m, ranks, sc = evaluate_wordnet(model, graph, b, args.graph, split, device, cfg["batch_size"], args.tau_candidates, args.seed)
            return m, sc
        m, sc = evaluate_sampled(model, graph, val_b if split == "val" else test_b, device)
        return m, sc

    # ---- resume
    state_path = run_dir / "last_training_state.pt"
    trace, best, best_state, best_epoch, bad, start_epoch, train_sec = [], -float("inf"), None, 0, 0, 0, 0.0
    first_step, resumes = None, []
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        warm_match_index(model, graph, train_b.pos, device)      # BEFORE the RNG restore
        torch.set_rng_state(st["rng_cpu"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in st["rng_cuda"]])
        trace, best, best_state, best_epoch, bad = st["trace"], st["best"], st["best_state"], st["best_epoch"], st["bad"]
        start_epoch, train_sec, first_step, train_b.epoch = st["epoch"] + 1, st["train_sec"], st["first_step"], st["epoch"] + 1
        resumes = list(st.get("resumes", [])) + [start_epoch]     # audit trail -> metrics.json
        print(f"[resume] from epoch {start_epoch}", flush=True)

    gpu_reset(device); t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train(); losses = []
        # resume audit: RNG + first-batch fingerprints (a resumed run must reproduce these)
        fp = {"cpu_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:10],
              "cuda_rng": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()[:10] if device.type == "cuda" else None,
              "opt": hashlib.sha256(json.dumps({k: (int(v["step"]) if torch.is_tensor(v["step"]) else v["step"]) for k, v in optimizer.state_dict()["state"].items()}, sort_keys=True).encode()).hexdigest()[:10]}
        for bi, (idx, h, t, r) in enumerate(train_b):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            scores = model(graph, h.to(device), t.to(device), r.to(device), all_loss=torch.tensor(0.0))
            loss = loss_fn(scores)
            if bi == 0:
                fp.update({"first_idx": hashlib.sha256(idx.numpy().tobytes()).hexdigest()[:10], "first_t": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:10],
                           "first_loss": float(loss), "first_scores": hashlib.sha256(scores.detach().cpu().numpy().tobytes()).hexdigest()[:10]})
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_step is None:      # gauntlet step 4: first logits / loss / gradient
                first_step = {"logits_hash": hashlib.sha256(scores.detach().cpu().numpy().tobytes()).hexdigest(),
                              "loss": float(loss), "gradient_hash": gradient_hash(model), "init_hash": init_hash}
            optimizer.step(); losses.append(float(loss))
        vm, _ = run_eval("val")
        val = vm["mrr"] if args.select_on == "val_mrr" else -float(np.mean(losses))
        state_hash = state_dict_hash(model.state_dict())
        trace.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_mrr": vm["mrr"],
                      "val_hits@1": vm.get("hits@1"), "state_hash": state_hash, "epoch_start": fp})
        print(f"[epoch {epoch + 1}] loss={np.mean(losses):.4f} val_mrr={vm['mrr']:.4f} val_h@1={vm.get('hits@1', 0):.4f} state={state_hash[:10]} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if val > best:
            best, bad, best_epoch, best_state = val, 0, epoch + 1, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        train_sec_now = train_sec + (time.perf_counter() - t0)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(), "trace": trace, "best": best, "best_state": best_state,
                    "best_epoch": best_epoch, "bad": bad, "epoch": epoch, "train_sec": train_sec_now, "first_step": first_step,
                    "resumes": resumes},
                   state_path)
        if bad >= cfg["patience"]:
            print("[early stop]", flush=True); break
    train_sec += time.perf_counter() - t0
    peak_train = gpu_peak(device)

    model.load_state_dict(best_state); model.eval()
    gpu_reset(device); t_inf = time.perf_counter()
    metrics, test_scores = run_eval("test")
    inf_sec = time.perf_counter() - t_inf; peak_inf = gpu_peak(device)
    test_pos = b.positives("test", args.graph)
    np.savez(run_dir / "scores.npz", queries=test_pos[:, 0].numpy(), positive_tails=test_pos[:, 2].numpy(), scores=test_scores)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    if state_path.exists():
        state_path.unlink()   # checkpoint history is not kept once the run is complete
    size = model_size(model)
    metrics.update({"train_time_sec": train_sec, "epochs": len(trace), "best_epoch": best_epoch, "best_val_score": best, "resumed_at_epochs": resumes,
                    "select_on": args.select_on, "inference_time_sec": inf_sec,
                    "eval_protocol": "filtered_full_entity" if wn else ("full_target_type" if b.candidate_pool is not None else "sampled"),
                    "peak_training_gpu_mib": peak_train["peak_gpu_mib"], "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib": peak_inf["peak_gpu_mib"], "peak_inference_reserved_gpu_mib": peak_inf["peak_gpu_reserved_mib"],
                    **size})
    record = {
        "dataset": args.dataset, "task": args.task, "variant": args.variant, "seed": args.seed, "graph": args.graph,
        "model": "CMPNN", "run_dir": str(run_dir),
        "physical_hash": summary["physical_hash"], "semantic_hash": summary["semantic_hash"],
        "supervision_hash": summary["supervision_hash"], "relation_hash": summary["relation_hash_" + args.graph],
        "alias_hash": summary["alias_hash"], "feature_hash": None,
        "source_bundle": {"path": str(bundle_path(args)), "physical_hash": summary["source_physical_hash"], "semantic_hash": summary["source_semantic_hash"]},
        "config": {**cfg, "input_dim": args.input_dim, "hidden_dim": args.hidden_dim, "layers": args.layers,
                   "neg_k": args.neg_k if wn else summary["neg_k"], "deterministic_aggregation": not args.stock_aggregation,
                   "alias_mask": not args.no_alias_mask, "label_derived_relations": b.label_derived_relations,
                   "gradient_checkpoint": args.gradient_checkpoint},
        "contract": {"loss": "neg_logsigmoid", "select_on": args.select_on, "message_func": "distmult",
                     "aggregate_func": cfg["aggregate_func"], "dependent": False, "remove_one_hop": cfg["remove_one_hop"],
                     "eval_protocol": metrics["eval_protocol"]},
        "first_step_audit": first_step, "checkpoint_hash": state_dict_hash(model.state_dict()), "metrics": metrics,
    }
    json_dump(record, run_dir / "metrics.json")
    json_dump(trace, run_dir / "trace.json")
    print(f"[done] {args.dataset}/{args.task}/{args.variant} seed={args.seed} graph={args.graph} "
          f"test_mrr={metrics['mrr']:.4f} h@1={metrics.get('hits@1', float('nan')):.4f} best_epoch={best_epoch}/{len(trace)} "
          f"train={train_sec:.0f}s peak={peak_train['peak_gpu_mib']} MiB ckpt={record['checkpoint_hash'][:12]}", flush=True)


if __name__ == "__main__":
    main()
