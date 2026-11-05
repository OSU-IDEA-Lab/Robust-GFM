"""MOTIF on the lab's WN18RR variant family, with the CMPNN WordNet evaluation protocol.

Variants (SCR docs/results/README.md): v1 = no_changes, v2 = all_inverse_edges,
v3 = transitive_edges. Every variant shares the official split; queries and filters use the
11 base relations, while message passing runs on the variant's full training graph.

Evaluation mirrors SCR models/CMPNN/run_cmpnn_lp.py::evaluate_wordnet: filtered ranking over
all 40,943 entities in both directions (head queries go through MOTIF's inverse relations),
ties averaged, filter = official train + val + test. For Precision/Recall and Kendall-tau,
each query keeps [positive, 100 candidates] drawn with torch.Generator().manual_seed(seed)
and read from the filtered score row. The candidates depend on the seed, so they are stored
for each of the three lab seeds.

    python script/wordnet_lp.py eval --variant v1 --run-tag full
    python script/wordnet_lp.py finetune --variant v1 --seed 1566911444 --run-tag seeds
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import random
import re
import sys
import tempfile
import time

import numpy as np

from eval_lp import build_message_graph, state_hash
from finetune_imdb_lp import save_checkpoint
from lp_data import ROOT, array_hash, file_hash, write_json

SPLITS = ROOT / "rai-datasets/data_tierA/wordnet_splits.npz"
VARIANTS = {"v1": "no_changes", "v2": "all_inverse_edges", "v3": "transitive_edges"}
LAB_SEEDS = (1566911444, 20241017, 20251017)
TASK = "link_prediction"
TAU_CANDIDATES = 100


def load_variant(path, variant):
    """Graph edges as (h, t, r) with base relations first, plus base-relation queries."""
    with np.load(path) as source:
        if str(source["format_version"]) != "wordnet_lp_four_variants_v1":
            raise ValueError("Unexpected WordNet split format")
        base = source["base_relation_ids"].astype(np.int64)
        triples = source[f"train_pos_{VARIANTS[variant]}"].astype(np.int64)
        splits = {name: source[f"{name}_pos"].astype(np.int64) for name in ("val", "test")}
        splits["train"] = source["train_pos_no_changes"].astype(np.int64)
        vocabulary = source["relation_vocab"]
        num_entities = int(source["num_entities"])
    relations = np.unique(triples[:, 1])
    if not np.array_equal(relations[:len(base)], base) or not np.array_equal(base, np.arange(len(base))):
        raise ValueError("Base relations must be ids 0..10 and present in every variant")
    if any(not np.isin(split[:, 1], base).all() for split in splits.values()):
        raise ValueError("Queries must use base relations only")
    train_rows = set(map(tuple, triples.tolist()))
    if any(tuple(row) not in train_rows for row in splits["train"].tolist()):
        raise ValueError("Every variant graph must contain the official training triples")
    remapped = np.searchsorted(relations, triples[:, 1])
    edges = np.column_stack((triples[:, 0], triples[:, 2], remapped))
    return {"edges": edges, "names": [str(vocabulary[r]) for r in relations], "num_entities": num_entities,
            **splits}


def filters(data):
    known_t, known_h = {}, {}
    for split in ("train", "val", "test"):
        for h, r, t in data[split].tolist():
            known_t.setdefault((h, r), set()).add(t)
            known_h.setdefault((t, r), set()).add(h)
    return known_t, known_h


def setup(args):
    import torch
    from motif import util
    from motif.models import MOTIF

    if not torch.cuda.is_available():
        raise RuntimeError("MOTIF's Triton kernels require a CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed = getattr(args, "seed", LAB_SEEDS[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    cfg = util.load_config(str(args.config), context={"dataset": "custom", "epochs": 0, "bpe": None,
                                                      "gpus": [device.index or 0], "ckpt": str(args.checkpoint)})
    model = MOTIF(cfg.model.relation_model, cfg.model.entity_model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return torch, device, model.to(device), checkpoint.get("fine_tuning")


def score_rows(torch, model, graph, queries, side, num_entities, device):
    """[len(queries), num_entities] scores; head rows use the inverse-relation conversion."""
    batch = torch.empty((len(queries), num_entities, 3), dtype=torch.long, device=device)
    entities = torch.arange(num_entities, device=device)
    h, r, t = (torch.as_tensor(queries[:, i], device=device) for i in range(3))
    batch[:, :, 0] = h[:, None] if side == "tail" else entities[None]
    batch[:, :, 1] = entities[None] if side == "tail" else t[:, None]
    batch[:, :, 2] = r[:, None]
    return model(graph, batch).float().cpu().numpy()


def evaluate(torch, model, graph, data, split, batch_size, device, seeds=LAB_SEEDS, limit=0):
    """Filtered average-tie ranks for tail then head queries, and per-seed tau candidate rows."""
    model.eval()
    queries = data[split][:limit] if limit else data[split]
    known_t, known_h = filters(data)
    count = data["num_entities"]
    candidates = {s: torch.randint(0, count, (len(queries), TAU_CANDIDATES),
                                   generator=torch.Generator().manual_seed(s)).numpy() for s in seeds}
    ranks, rows = [], {s: [] for s in seeds}
    with torch.no_grad():
        for side in ("tail", "head"):
            for start in range(0, len(queries), batch_size):
                chunk = queries[start:start + batch_size]
                scores = score_rows(torch, model, graph, chunk, side, count, device)
                for offset, (h, r, t) in enumerate(chunk.tolist()):
                    row = scores[offset]
                    truth, others = (t, known_t[(h, r)]) if side == "tail" else (h, known_h[(t, r)])
                    positive = row[truth]
                    others = list(others - {truth})
                    if others:
                        row[others] = -np.inf
                    ranks.append(1.0 + (row > positive).sum() + ((row == positive).sum() - 1) / 2.0)
                    for s in seeds:
                        rows[s].append(np.concatenate(([positive], row[candidates[s][start + offset]])))
    ranks = np.asarray(ranks)
    metrics = {"mrr": float(np.mean(1 / ranks)),
               **{f"hits@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 10)}, "queries": len(ranks)}
    return metrics, ranks, {s: np.stack(v).astype(np.float32) for s, v in rows.items()}, candidates


def run_eval(args):
    torch, device, model, fine_tuning = setup(args)
    data = load_variant(args.splits, args.variant)
    stage = "finetuned" if fine_tuning else "zeroshot"
    if fine_tuning and fine_tuning["training_contract"]["dataset"] != "wordnet":
        raise ValueError("Fine-tuned checkpoint is not a WordNet run")
    seed_tag = f"_s{fine_tuning['training_contract']['seed']}" if fine_tuning else ""
    output = args.output / f"motif_{stage}_wordnet_lp_{args.run_tag}{seed_tag}" / TASK / "physical" / args.variant
    if output.exists():
        raise FileExistsError(f"Run exists: {output}; choose a new --run-tag")
    model.requires_grad_(False)
    before = state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph = build_message_graph(data["edges"], data["names"], data["num_entities"], device)
    lift_seconds = time.perf_counter() - started
    metrics, ranks, rows, candidates = evaluate(torch, model, graph, data, "test", args.batch_size, device,
                                                limit=args.limit)
    if state_hash(model) != before:
        raise RuntimeError("Checkpoint state changed during frozen inference")
    record = {"model": "MOTIF", "dataset": "wordnet", "task": TASK, "variant": args.variant,
              "variant_name": VARIANTS[args.variant], "evaluation_stage": stage, **metrics,
              "checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
              "splits_sha256": file_hash(args.splits), "graph_sha256": array_hash(data["edges"]),
              "graph_edges_with_inverses": graph.num_edges, "graph_relations": len(data["names"]),
              "lift_hyperedges": graph.relation_hypergraph.num_edges, "lift_seconds": lift_seconds,
              "total_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
              "protocol": "cmpnn_evaluate_wordnet_filtered_full_ranking_tail_then_head_average_ties",
              "tau_candidates": f"positive_plus_{TAU_CANDIDATES}_torch_randint_per_seed_filtered_row",
              "candidate_seeds": list(LAB_SEEDS), "limit": args.limit, "gpu": torch.cuda.get_device_name(device),
              "parameters_sha256": before, "seed": fine_tuning["training_contract"]["seed"] if fine_tuning else None}
    if fine_tuning:
        record["fine_tuning"] = {k: v for k, v in fine_tuning.items() if k != "training_contract"}
        record["training_contract"] = fine_tuning["training_contract"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motif-wn-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        np.savez_compressed(staged / "scores.npz", ranks=ranks, queries=data["test"][:args.limit or None],
                            **{f"scores_{s}": rows[s] for s in LAB_SEEDS},
                            **{f"candidates_{s}": candidates[s] for s in LAB_SEEDS})
        record["scores_sha256"] = file_hash(staged / "scores.npz")
        write_json(staged / "metrics.json", record)
        staged.rename(output)
    print(f"Saved {output}: MRR={metrics['mrr']:.4f} Hits@1={metrics['hits@1']:.4f} Hits@3={metrics['hits@3']:.4f}")


def run_finetune(args):
    torch, device, model, fine_tuning = setup(args)
    if fine_tuning:
        raise ValueError("Start each experiment from the released pretrained checkpoint")
    from motif import tasks

    data = load_variant(args.splits, args.variant)
    run_dir = args.output / f"motif_finetune_wordnet_lp_{args.run_tag}" / f"train_{args.variant}" / f"seed{args.seed}"
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}; choose a fresh --run-tag")
    native_builder = tasks.build_relation_hypergraph

    def build_lift(graph, *builder_args, **builder_kwargs):  # Same scalar-count fix as zero-shot inference.
        graph = native_builder(graph, *builder_args, **builder_kwargs)
        graph.relation_hypergraph.num_nodes = int(graph.relation_hypergraph.num_nodes)
        return graph

    tasks.build_relation_hypergraph = build_lift
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    initial_hash = state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph = build_message_graph(data["edges"], data["names"], data["num_entities"], device)
    targets = data["train"][:, [0, 2, 1]]  # (h, t, r); base relation ids are unchanged by the remap.
    contract = {
        "dataset": "wordnet", "task": TASK, "train_variant": args.variant, "seed": args.seed,
        "initial_checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
        "splits_sha256": file_hash(args.splits), "train_graph_sha256": array_hash(data["edges"]),
        "training_targets": "official_base_train_triples", "edge_removal": "exact_batch_triple_and_motif_inverse",
        "negatives": f"{args.num_negative}_strict_half_tail_half_head", "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "adversarial_temperature": args.adversarial_temperature,
        "max_grad_norm": args.max_grad_norm, "validate_every": args.validate_every, "val_limit": args.val_limit,
        "selection_metric": "filtered_validation_mrr_both_directions",
        "precision": "float32_tf32_disabled", "gpu": torch.cuda.get_device_name(device),
        "model_source": {p.relative_to(ROOT / "upstream").as_posix(): file_hash(p) for p in sorted((ROOT / "upstream/motif").rglob("*.py"))},
        "training_source": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/eval_lp.py")},
    }
    run_dir.mkdir(parents=True)
    write_json(run_dir / "training_config.json", contract)

    def validation():
        return evaluate(torch, model, graph, data, "val", args.eval_batch_size, device, seeds=(), limit=args.val_limit)[0]

    initial = validation()
    print(f"Initial validation MRR={initial['mrr']:.4f}", flush=True)
    best, best_step = float("-inf"), None
    rng = np.random.default_rng(args.seed)
    order, position = rng.permutation(len(targets)), 0
    schedule_hash = hashlib.sha256()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=("step", "loss", "grad_norm_before_clip", "validation_mrr",
                                                     "validation_hits@1", "seconds"))
        writer.writeheader()
        for step in range(1, args.steps + 1):
            if position + args.batch_size > len(order):
                order, position = rng.permutation(len(targets)), 0
            indices = order[position:position + args.batch_size]
            position += args.batch_size
            schedule_hash.update(indices.tobytes())
            model.train()
            batch = torch.as_tensor(targets[indices], device=device)
            batch = tasks.negative_sampling(graph, batch, args.num_negative, strict=True)
            optimizer.zero_grad(set_to_none=True)
            scores = model(graph, batch)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1
            losses = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
            weights = torch.ones_like(scores)
            with torch.no_grad():
                weights[:, 1:] = torch.softmax(scores[:, 1:] / args.adversarial_temperature, dim=-1)
            loss = ((losses * weights).sum(-1) / weights.sum(-1)).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "loss": float(loss.item()), "grad_norm_before_clip": float(norm.item()),
                      "seconds": time.perf_counter() - started}
            if step % args.validate_every == 0 or step == args.steps:
                result = validation()
                record.update({"validation_mrr": result["mrr"], "validation_hits@1": result["hits@1"]})
                if result["mrr"] > best:
                    best, best_step = result["mrr"], step
                    save_checkpoint(torch, run_dir / "best.pth", model, {
                        "training_contract": contract, "selected_step": step, "validation_mrr": best,
                        "initial_parameters_sha256": initial_hash, "parameters_sha256": state_hash(model)})
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}, validation MRR={result['mrr']:.4f}, "
                      f"best step={best_step}, {record['seconds']:.0f}s", flush=True)
            elif step == 1 or step % args.log_every == 0:
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}, {record['seconds']:.0f}s", flush=True)
            writer.writerow(record)
            history.flush()
    if state_hash(model) == initial_hash:
        raise RuntimeError("Model parameters did not change during fine-tuning")
    write_json(run_dir / "training_summary.json", {
        "training_contract": contract, "initial_validation": initial, "best_step": best_step,
        "best_validation_mrr": best, "completed_steps": args.steps, "schedule_sha256": schedule_hash.hexdigest(),
        "total_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "best_checkpoint": "best.pth"})
    print(f"Saved fine-tuned checkpoint: {run_dir / 'best.pth'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("eval", "finetune"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream/ckpts/motif_3g.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "upstream/config/transductive/MOTIF_inference.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--batch-size", type=int, default=8, help="eval: queries per forward; finetune: triples per step")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="eval: first N test triples (smoke tests only)")
    parser.add_argument("--seed", type=int, default=LAB_SEEDS[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-negative", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--val-limit", type=int, default=500, help="finetune: fixed first N validation triples")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("Invalid run tag")
    if min(args.batch_size, args.eval_batch_size, args.steps, args.validate_every, args.num_negative) < 1:
        parser.error("Batch sizes, steps, validation interval and negatives must be positive")
    if args.adversarial_temperature <= 0:
        parser.error("Adversarial temperature must be positive")
    for name in ("splits", "checkpoint", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    (run_eval if args.mode == "eval" else run_finetune)(args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "upstream"))
    main()
