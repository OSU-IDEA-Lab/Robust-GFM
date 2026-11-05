"""Fine-tune a released MOTIF checkpoint on one physical IMDb LP variant."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

from eval_lp import build_message_graph, load_inputs, state_hash
from lp_data import ROOT, PREPROCESSED_ROOT, TASK_VARIANTS, array_hash, file_hash, write_json
from lp_metrics import average_ranks


def load_split(args, split, limit=0):
    selected = SimpleNamespace(**{**vars(args), "dataset": "imdb", "variant": args.train_variant,
                                  "view": "physical", "split": split, "limit": limit})
    return load_inputs(selected)


def check_training_context(edges, names, target, positives, task):
    """The native target-removal path is sufficient only for direct targets here."""
    present = set(map(tuple, edges[edges[:, 2] == target, :2].tolist()))
    if present != set(map(tuple, positives.tolist())):
        raise ValueError("Physical target edges must be exactly the shared training positives")
    suffix = "year" if task == "movie_year" else "director"
    for relation, name in enumerate(names):
        if relation != target and name.endswith(f"_to_{suffix}") and np.any(edges[:, 2] == relation):
            raise ValueError(f"Target copies via {name} need an extended training removal rule")


def batch_schedule(positives, batch_size, seed, task):
    """Canonical keyed order, shuffled per pass independently of Torch dropout."""
    keys = [hashlib.sha256(f"{seed}|{task}|{h}|{t}".encode()).digest() for h, t in positives]
    canonical = np.asarray(sorted(range(len(keys)), key=keys.__getitem__))
    rng = np.random.default_rng(seed)
    while True:
        order = canonical[rng.permutation(len(canonical))]
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]


def make_batch(torch, positives, candidates, relation, device):
    batch = torch.empty((len(positives), candidates.shape[1], 3), dtype=torch.long, device=device)
    batch[:, :, 0] = torch.as_tensor(positives[:, 0], device=device)[:, None]
    batch[:, :, 1] = torch.as_tensor(candidates, device=device)
    batch[:, :, 2] = relation
    return batch


def validate(torch, model, graph, positives, candidates, target, batch_size, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            stop = start + batch_size
            batch = make_batch(torch, positives[start:stop], candidates[start:stop], target, device)
            scores.append(model(graph, batch).cpu().numpy())
    ranks = average_ranks(np.concatenate(scores))
    return {"mrr": float(np.mean(1 / ranks)), "hits@1": float(np.mean(ranks <= 1)),
            "queries": len(positives)}


def save_checkpoint(torch, path, model, metadata):
    temporary = path.with_suffix(".pth.tmp")
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "fine_tuning": metadata}, temporary)
    temporary.replace(path)


def train(args):
    import torch
    import torch_geometric
    import triton
    from motif import tasks, util
    from motif.models import MOTIF

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning with MOTIF's Triton kernels requires a CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    meta, edges, positives, candidates, _, _ = load_split(args, "train")
    _, _, val_pos, val_candidates, _, _ = load_split(args, "val", args.val_limit)
    target = meta["target_relation"]
    check_training_context(edges, meta["relation_names"], target, positives, args.task)
    cfg = util.load_config(str(args.config), context={"dataset": "custom", "epochs": 0,
                           "bpe": None, "gpus": [device.index or 0], "ckpt": str(args.checkpoint)})
    model = MOTIF(cfg.model.relation_model, cfg.model.entity_model)
    initial = torch.load(args.checkpoint, map_location="cpu")
    if "fine_tuning" in initial:
        raise ValueError("Start each experiment from the released pretrained checkpoint")
    model.load_state_dict(initial["model"], strict=True)
    model.to(device)
    # Preserve the model's frozen static positional encodings. All parameters
    # that the released architecture marks trainable participate in fine-tuning.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    initial_hash = state_hash(model)

    # Native training rebuilds Lift after removing the batch target edges.
    # Use the same scalar-count compatibility conversion as zero-shot inference.
    native_builder = tasks.build_relation_hypergraph

    def build_lift(data, *builder_args, **builder_kwargs):
        data = native_builder(data, *builder_args, **builder_kwargs)
        data.relation_hypergraph.num_nodes = int(data.relation_hypergraph.num_nodes)
        return data

    tasks.build_relation_hypergraph = build_lift
    removal = {"calls": 0, "edges": 0, "last_edges": 0}
    native_remove = model.entity_model.remove_easy_edges

    def remove_targets(data, h_index, t_index, r_index=None):
        result = native_remove(data, h_index, t_index, r_index)
        removed = data.num_edges - result.num_edges
        # There is one positive edge and one internal inverse per unique movie.
        expected = 2 * int(torch.unique(h_index[:, 0]).numel())
        if removed != expected:
            raise RuntimeError(f"Unexpected native target-edge removal: {removed}, expected {expected}")
        removal["calls"] += 1
        removal["edges"] += removed
        removal["last_edges"] = removed
        return result

    model.entity_model.remove_easy_edges = remove_targets
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph = build_message_graph(edges, meta["relation_names"], meta["num_nodes"], device)
    contract = {
        "dataset": "imdb", "task": args.task, "train_variant": args.train_variant, "train_view": "physical",
        "initial_checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
        "export_manifest_sha256": file_hash(args.data / "manifest.json"), "train_graph_sha256": array_hash(edges),
        "train_queries_sha256": array_hash(positives, candidates), "validation_queries_sha256": array_hash(val_pos, val_candidates),
        "training_targets": len(positives), "training_candidates": candidates.shape[1],
        "validation_targets": len(val_pos), "selection_metric": "validation_20_candidate_mrr_average_ties",
        "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "optimizer": "Adam", "lr": args.lr,
        "adversarial_temperature": args.adversarial_temperature, "max_grad_norm": args.max_grad_norm,
        "validate_every": args.validate_every, "val_limit": args.val_limit,
        "precision": "float32_tf32_disabled", "negative_policy": "shared_fixed_19_tail_negatives",
        "target_removal": "native_direct_target_and_internal_inverse_before_rebuilding_lift",
        "torch": torch.__version__, "pyg": torch_geometric.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "model_source": {p.relative_to(ROOT / "upstream").as_posix(): file_hash(p) for p in sorted((ROOT / "upstream/motif").rglob("*.py"))},
        "training_source": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/eval_lp.py",
                                                        ROOT / "script/lp_data.py", ROOT / "script/lp_metrics.py")},
    }
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir / "training_config.json", contract)
    before_validation = validate(torch, model, graph, val_pos, val_candidates, target, args.batch_size, device)
    print(f"Initial validation MRR={before_validation['mrr']:.4f}", flush=True)
    best, best_step = float("-inf"), None
    first_gradients = None
    schedule_hash = hashlib.sha256()
    schedule = batch_schedule(positives, args.batch_size, args.seed, args.task)
    with (args.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=("step", "loss", "grad_norm_before_clip", "batch_size",
                                                     "removed_edges", "validation_mrr", "validation_hits@1"))
        writer.writeheader()
        for step in range(1, args.steps + 1):
            model.train()
            indices = next(schedule)
            schedule_hash.update(array_hash(positives[indices], candidates[indices]).encode())
            batch = make_batch(torch, positives[indices], candidates[indices], target, device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(graph, batch)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1
            losses = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
            weights = torch.ones_like(scores)
            with torch.no_grad():
                weights[:, 1:] = (torch.softmax(scores[:, 1:] / args.adversarial_temperature, dim=-1)
                                  if args.adversarial_temperature > 0 else 1 / (scores.shape[1] - 1))
            loss = ((losses * weights).sum(-1) / weights.sum(-1)).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            if first_gradients is None:
                first_gradients = {}
                for name, module in (("relation_encoder", model.relation_model), ("entity_encoder", model.entity_model)):
                    norms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
                    first_gradients[name] = float(torch.stack(norms).norm().item()) if norms else 0.0
                if not any(first_gradients.values()):
                    raise RuntimeError("No nonzero gradients reached the trainable model")
            optimizer.step()
            record = {"step": step, "loss": float(loss.item()), "grad_norm_before_clip": float(norm.item()),
                      "batch_size": len(indices), "removed_edges": removal["last_edges"]}
            if step % args.validate_every == 0 or step == args.steps:
                record["validation"] = validate(torch, model, graph, val_pos, val_candidates,
                                                target, args.batch_size, device)
                score = record["validation"]["mrr"]
                if score > best:
                    best, best_step = score, step
                    metadata = {"training_contract": contract, "selected_step": step, "validation_mrr": score,
                                "initial_parameters_sha256": initial_hash, "parameters_sha256": state_hash(model)}
                    save_checkpoint(torch, args.run_dir / "best.pth", model, metadata)
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}, validation MRR={score:.4f}, best step={best_step}", flush=True)
            elif step == 1 or step % args.log_every == 0:
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}", flush=True)
            validation = record.pop("validation", None)
            if validation is not None:
                record.update({"validation_mrr": validation["mrr"], "validation_hits@1": validation["hits@1"]})
            writer.writerow(record)
            history.flush()
    if removal["calls"] != args.steps:
        raise RuntimeError("Training did not use native target-edge removal on every update")
    final_hash = state_hash(model)
    if final_hash == initial_hash:
        raise RuntimeError("Model parameters did not change during fine-tuning")
    torch.cuda.synchronize(device)
    write_json(args.run_dir / "training_summary.json", {
        "training_contract": contract, "initial_validation": before_validation,
        "best_step": best_step, "best_validation_mrr": best, "completed_steps": args.steps,
        "first_step_gradient_norms_after_clip": first_gradients, "initial_parameters_sha256": initial_hash,
        "final_parameters_sha256": final_hash, "schedule_sha256": schedule_hash.hexdigest(),
        "total_removed_training_edges": removal["edges"], "total_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device), "best_checkpoint": "best.pth",
    })
    print(f"Saved fine-tuned checkpoint: {args.run_dir / 'best.pth'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREPROCESSED_ROOT / "IMDB")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream/ckpts/motif_3g.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "upstream/config/transductive/MOTIF_inference.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--task", choices=TASK_VARIANTS, default="movie_year")
    parser.add_argument("--train-variant", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-limit", type=int, default=0, help="Limited validation only for the training smoke test")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--run-tag", default="pilot")
    args = parser.parse_args()
    if (min(args.steps, args.batch_size, args.validate_every, args.log_every) < 1 or args.val_limit < 0
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.max_grad_norm)
            or args.max_grad_norm <= 0 or not np.isfinite(args.adversarial_temperature)
            or args.adversarial_temperature < 0):
        parser.error("Require positive finite budgets/lr/gradient limit and nonnegative temperature/val-limit")
    if not args.device.startswith("cuda") or args.train_variant not in TASK_VARIANTS[args.task]:
        parser.error("Choose a CUDA device and a physical variant defined for this task")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("Invalid run tag")
    for name in ("data", "checkpoint", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.run_dir = (args.output / f"motif_finetune_imdb_lp_{args.run_tag}" / args.task /
                    f"train_{args.train_variant}" / f"seed{args.seed}")
    if args.run_dir.exists():
        parser.error(f"Run already exists: {args.run_dir}; choose a fresh --run-tag")
    train(args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "upstream"))
    main()
