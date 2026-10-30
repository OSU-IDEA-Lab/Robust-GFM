#!/usr/bin/env python3
"""Train one or more Freebase block-conditional RGCN runs."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from freebase_block_common import set_all_seeds, sha256_array, write_json
from freebase_conditional_rgcn import (
    BlockConditionalGraph,
    FreebaseBlockConditionalRGCN,
    build_two_layer_blocks,
)

FORMAT = "freebase_rgcn_block_conditional_result_v1"
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def torch_load(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def model_memory(model: torch.nn.Module) -> dict[str, Any]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "parameter_bytes": int(param_bytes),
        "parameter_mib": bytes_to_mib(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "buffer_mib": bytes_to_mib(buffer_bytes),
    }


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    prediction = logits.argmax(axis=1)
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        ranks[i] = int(np.where(order[i] == label)[0][0]) + 1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, prediction, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, prediction, average="micro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, prediction, average="weighted", zero_division=0
    )
    num_classes = logits.shape[1]
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(num_classes), zero_division=0
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    return {
        "loss": float(nll.mean()),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "micro_precision": float(precision_micro),
        "micro_recall": float(recall_micro),
        "micro_f1": float(f1_micro),
        "weighted_f1": float(f1_weighted),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "per_class": {
            str(i): {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(per_s[i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(num_classes)
        ).astype(int).tolist(),
    }


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    values = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def evaluate(
    *,
    model: FreebaseBlockConditionalRGCN,
    graph: BlockConditionalGraph,
    local_indices: np.ndarray,
    batch_size: int,
    fanout1: int,
    fanout2: int,
    device: torch.device,
    sampling_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_no, local_batch in enumerate(
            iter_batches(local_indices, batch_size, False, np.random.default_rng(0))
        ):
            global_batch = np.asarray(graph.target_nodes[local_batch], dtype=np.int64)
            block1, block2 = build_two_layer_blocks(
                graph,
                global_batch,
                fanout_layer1=fanout1,
                fanout_layer2=fanout2,
                sampling_seed=int(sampling_seed + batch_no),
            )
            logits = model(graph, block1, block2, device)
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            label_parts.append(np.asarray(graph.labels[local_batch], dtype=np.int64))
    logits = np.concatenate(logits_parts) if logits_parts else np.empty((0, graph.num_classes), np.float32)
    labels = np.concatenate(label_parts) if label_parts else np.empty(0, np.int64)
    return logits, classification_metrics(logits, labels)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    best_score: float,
    select_on: str,
    stale: int,
    history: list[dict[str, Any]],
    train_rng: np.random.Generator,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": "freebase_rgcn_block_resume_v1",
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        # Selection state, generalised to "higher is better" so the criterion can
        # be validation micro-F1 rather than validation loss. best_val_loss is
        # kept alongside it for reporting and for reading legacy checkpoints.
        "best_score": float(best_score),
        "select_on": str(select_on),
        "stale_evaluations": int(stale),
        "history": history,
        "train_rng_state": train_rng.bit_generator.state,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "config": dict(config),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    select_on: str = "val_loss",
) -> tuple[int, int, float, float, int, list[dict[str, Any]]]:
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_rng.bit_generator.state = checkpoint["train_rng_state"]
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    ckpt_select = str(checkpoint.get("select_on", "val_loss"))
    if ckpt_select != select_on:
        raise RuntimeError(
            f"Cannot resume: checkpoint was selected on {ckpt_select!r} but this run "
            f"uses {select_on!r}. Selection state is not transferable between criteria; "
            f"start a fresh run directory instead."
        )
    best_val_loss = float(checkpoint["best_val_loss"])
    # Legacy checkpoints (pre --select-on) carry no best_score; they were always
    # selected on validation loss, whose "higher is better" form is -loss.
    best_score = float(checkpoint.get("best_score", -best_val_loss))
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["best_epoch"]),
        best_val_loss,
        best_score,
        int(checkpoint["stale_evaluations"]),
        list(checkpoint["history"]),
    )


def run_one(args: argparse.Namespace, variant: str, seed: int) -> dict[str, Any]:
    preprocessed = (
        Path(args.preprocessed_root).resolve()
        / variant
        / args.up_to
        / args.set_mode
        / args.scope
    )
    graph = BlockConditionalGraph(preprocessed, mmap=True)
    run_dir = (
        Path(args.results_root).resolve()
        / args.up_to
        / args.set_mode
        / args.scope
        / variant
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if args.skip_completed and result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("completed"):
            print(f"Skipping completed {variant} seed={seed}")
            return result

    device = torch.device(args.device)
    set_all_seeds(seed, deterministic=args.deterministic, warn_only=args.deterministic_warn_only)
    model = FreebaseBlockConditionalRGCN(
        num_nodes=graph.num_nodes,
        num_node_types=graph.num_node_types,
        num_relations=graph.num_relations,
        num_classes=graph.num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_bases=min(args.num_bases, graph.num_relations),
        dropout=args.dropout,
        use_type_embedding=args.use_type_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rng = np.random.default_rng(seed)

    config = {
        "variant": variant,
        "seed": seed,
        "up_to": args.up_to,
        "set_mode": args.set_mode,
        "scope": args.scope,
        "preprocessed_dir": str(preprocessed),
        "combined_model_input_hash": graph.hashes["combined_model_input_hash"],
        "device": str(device),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_bases": min(args.num_bases, graph.num_relations),
        "dropout": args.dropout,
        "use_type_embedding": args.use_type_embedding,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "select_on": args.select_on,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "fanout_layer1": args.fanout_layer1,
        "fanout_layer2": args.fanout_layer2,
        "deterministic": args.deterministic,
    }
    write_json(run_dir / "config.json", config)

    start_epoch, best_epoch = 1, 0
    best_val_loss, best_score, stale, history = float("inf"), float("-inf"), 0, []
    last_ckpt = run_dir / "checkpoint_last.pt"
    best_ckpt = run_dir / "checkpoint_best.pt"
    if args.resume and last_ckpt.is_file():
        (
            start_epoch, best_epoch, best_val_loss, best_score, stale, history
        ) = restore_checkpoint(
            torch_load(last_ckpt, device), model, optimizer, train_rng, args.select_on
        )
        print(f"Resumed {variant} seed={seed} at epoch {start_epoch}")

    train_idx = graph.splits["train_idx"]
    val_idx = graph.splits["val_idx"]
    test_idx = graph.splits["test_idx"]
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    epochs_run = start_epoch - 1
    early_stopped = False

    for epoch in range(start_epoch, args.max_epochs + 1):
        epochs_run = epoch
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_no, local_batch in enumerate(iter_batches(train_idx, args.batch_size, True, train_rng)):
            global_batch = np.asarray(graph.target_nodes[local_batch], dtype=np.int64)
            sample_seed = splitmix_seed(seed, epoch, batch_no)
            block1, block2 = build_two_layer_blocks(
                graph,
                global_batch,
                fanout_layer1=args.fanout_layer1,
                fanout_layer2=args.fanout_layer2,
                sampling_seed=sample_seed,
            )
            truth = torch.as_tensor(np.asarray(graph.labels[local_batch]), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph, block1, block2, device)
            loss = F.cross_entropy(logits, truth)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(local_batch)
            seen += len(local_batch)

        if epoch % args.eval_interval != 0 and epoch != args.max_epochs:
            continue
        _, val_metrics = evaluate(
            model=model,
            graph=graph,
            local_indices=val_idx,
            batch_size=args.eval_batch_size,
            fanout1=args.fanout_layer1,
            fanout2=args.fanout_layer2,
            device=device,
            sampling_seed=seed + 1_000_000,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "validation": val_metrics,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": record["train_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_micro_f1": val_metrics["micro_f1"],
                }
            ),
            flush=True,
        )
        # Model selection. Validation loss and validation micro-F1 diverge sharply
        # on this task: the model overfits within ~2 epochs (train loss -> 1e-4) so
        # val loss bottoms almost immediately and rises, while val micro-F1 keeps
        # climbing for another ~15 epochs. Selecting on loss therefore restores a
        # checkpoint ~3 points below the best one the run ever reaches. Same failure
        # mode as the DBLP originals runner (fixed there too).
        # Score is always "higher is better"; for val_loss that is -loss.
        score = (
            float(val_metrics["micro_f1"])
            if args.select_on == "val_micro_f1"
            else -float(val_metrics["loss"])
        )
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_score=best_score,
            select_on=args.select_on,
            stale=stale,
            history=history,
            train_rng=train_rng,
            config=config,
        )
        torch.save(checkpoint, last_ckpt)
        if improved:
            torch.save(
                {
                    "format_version": "freebase_rgcn_block_best_v1",
                    "model_state_dict": checkpoint["model_state_dict"],
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_score": best_score,
                    "select_on": args.select_on,
                    "config": config,
                },
                best_ckpt,
            )
        write_json(run_dir / "history.json", history)
        if stale >= args.patience:
            early_stopped = True
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_train = bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_train_reserved = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    else:
        peak_train = peak_train_reserved = None
    training_time = time.perf_counter() - training_start
    if not best_ckpt.is_file():
        raise RuntimeError("No best checkpoint was written")
    best = torch_load(best_ckpt, device)
    model.load_state_dict(best["model_state_dict"], strict=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    split_outputs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for offset, (name, idx) in enumerate((("train", train_idx), ("val", val_idx), ("test", test_idx))):
        logits, metrics = evaluate(
            model=model,
            graph=graph,
            local_indices=idx,
            batch_size=args.eval_batch_size,
            fanout1=args.fanout_layer1,
            fanout2=args.fanout_layer2,
            device=device,
            sampling_seed=seed + 2_000_000 + offset * 100_000,
        )
        split_outputs[name] = metrics
        arrays[f"{name}_logits"] = logits
        arrays[f"{name}_labels"] = np.asarray(graph.labels[idx], dtype=np.int64)
        arrays[f"{name}_idx"] = idx
        arrays[f"{name}_global_ids"] = np.asarray(graph.target_nodes[idx], dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_infer = bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_infer_reserved = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    else:
        peak_infer = peak_infer_reserved = None
    inference_time = time.perf_counter() - inference_start
    logits_path = run_dir / "logits.npz"
    np.savez_compressed(logits_path, **arrays)

    memory = model_memory(model)
    memory.update(
        {
            "peak_training_gpu_mib": peak_train,
            "peak_training_reserved_gpu_mib": peak_train_reserved,
            "peak_inference_gpu_mib": peak_infer,
            "peak_inference_reserved_gpu_mib": peak_infer_reserved,
            "best_checkpoint_mib": bytes_to_mib(best_ckpt.stat().st_size),
            "last_checkpoint_mib": bytes_to_mib(last_ckpt.stat().st_size),
        }
    )
    result = {
        "format_version": FORMAT,
        "completed": True,
        "variant": variant,
        "seed": seed,
        "up_to_variant": args.up_to,
        "set_mode": args.set_mode,
        "scope": args.scope,
        "combined_model_input_hash": graph.hashes["combined_model_input_hash"],
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_score": best_score,
        "select_on": args.select_on,
        "early_stopped": early_stopped,
        "train_time_sec": training_time,
        "inference_time_sec": inference_time,
        "metrics": split_outputs,
        "memory": memory,
        "model_config": config,
        "artifacts": {
            "logits": str(logits_path),
            "checkpoint_best": str(best_ckpt),
            "checkpoint_last": str(last_ckpt),
            "history": str(run_dir / "history.json"),
        },
        "alignment_hashes": {
            f"{name}_{kind}_sha256": sha256_array(arrays[f"{name}_{kind}"])
            for name in ("train", "val", "test")
            for kind in ("labels", "idx", "global_ids")
        },
    }
    write_json(result_path, result)
    print(
        f"Completed {variant} seed={seed}: test Micro-F1={split_outputs['test']['micro_f1']:.6f}, "
        f"Macro-F1={split_outputs['test']['macro_f1']:.6f}"
    )
    del model, optimizer, graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def splitmix_seed(seed: int, epoch: int, batch: int) -> int:
    value = (int(seed) ^ (int(epoch) << 21) ^ (int(batch) << 42)) & ((1 << 63) - 1)
    # Keep seed in signed 63-bit range accepted throughout NumPy/Python.
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 63) - 1)
    value ^= value >> 27
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--up-to", default="exact_2")
    parser.add_argument("--set-mode", choices=["union", "intersection"], default="union")
    parser.add_argument("--scope", choices=["center", "all_graph"], default="center")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-bases", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--select-on",
        choices=["val_micro_f1", "val_loss"],
        default="val_micro_f1",
        help="Metric used for model selection and early stopping. Default val_micro_f1: "
             "validation loss bottoms within ~2 epochs on this task while validation "
             "micro-F1 keeps improving, so selecting on loss discards the better model. "
             "val_loss selects on validation loss instead.",
    )
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--fanout-layer1", type=int, default=4)
    parser.add_argument("--fanout-layer2", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-warn-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Seed-major execution keeps matched runs adjacent.
    for seed in args.seeds:
        for variant in args.variants:
            run_one(args, variant, seed)
