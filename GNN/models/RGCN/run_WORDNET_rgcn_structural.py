#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_RGCN_lp_wordnet_structural import WordNetStructuralRGCN
from wordnet_structural_common import (
    BUDGETS,
    MODES,
    VARIANTS,
    atomic_json_dump,
    atomic_torch_save,
    capture_rng_state,
    gradient_hash,
    json_sha256,
    model_parameter_bytes,
    restore_rng_state,
    set_deterministic,
    state_dict_hash,
    tensor_sha256,
)
from wordnet_structural_eval import build_filter_dicts, evaluate_ranking, sample_neg_train

DEFAULTS = dict(
    hidden_dim=200,
    num_bases=30,
    edge_dropout_other=0.4,
    root_dropout_loop=0.2,
    lr=0.01,
    weight_decay=0.01,
    neg_per_pos=1,
    batch_size=0,
    epochs=3000,
    patience=3000,
    eval_interval=10,
    eval_batch_size=128,
    binary_k=50,
    edge_chunk_size=200000,
)


def load_graph(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "variant", "num_entities", "semantic_relation_names", "decoder_relation_names",
        "semantic_edge_index", "semantic_edge_type", "train_pos", "val_pos", "test_pos",
        "physical_hash", "semantic_hash", "supervision_hash",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing preprocessed fields in {path}: {sorted(missing)}")
    return data


def semantic_train_pos(data: dict) -> torch.Tensor:
    if "semantic_train_pos" in data:
        return data["semantic_train_pos"].long()
    src, dst = data["semantic_edge_index"]
    return torch.stack((src, data["semantic_edge_type"], dst), dim=1).long()


def task_train_pos(data: dict) -> torch.Tensor:
    return data.get("task_train_pos", data["train_pos"]).long()


def objective_spec(data: dict, objective: str) -> tuple[torch.Tensor, list[str]]:
    if objective == "task_only":
        names = list(data.get("decoder_relation_names_task_only", data.get("task_relation_names", data["decoder_relation_names"])))
        return task_train_pos(data), names
    if objective == "semantic_all":
        names = list(data.get("decoder_relation_names_semantic_all", data["semantic_relation_names"]))
        return semantic_train_pos(data), names
    raise ValueError(f"Unknown training objective: {objective}")


def config_signature(args, data):
    immutable = {
        "variant": args.variant,
        "budget": args.budget,
        "mode": args.mode,
        "seed": args.seed,
        "semantic_hash": data["semantic_hash"],
        "supervision_hash": data["supervision_hash"],
        "hidden_dim": args.hidden_dim,
        "num_bases": args.num_bases,
        "edge_dropout_other": args.edge_dropout_other,
        "root_dropout_loop": args.root_dropout_loop,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "neg_per_pos": args.neg_per_pos,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "eval_batch_size": args.eval_batch_size,
        "binary_k": args.binary_k,
        "candidate_seed": args.candidate_seed,
        "edge_chunk_size": args.edge_chunk_size,
        "train_objective": args.train_objective,
    }
    return json_sha256(immutable), immutable


def build_model(args, data, device):
    _, decoder_names = objective_spec(data, args.train_objective)
    return WordNetStructuralRGCN(
        num_entities=int(data["num_entities"]),
        num_message_relations=len(data["semantic_relation_names"]),
        num_decoder_relations=len(decoder_names),
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        edge_dropout_other=args.edge_dropout_other,
        root_dropout_loop=args.root_dropout_loop,
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)


def build_optimizer(model, args):
    other = [p for n, p in model.named_parameters() if not n.startswith("rel_emb.")]
    return torch.optim.Adam(
        [
            {"params": other, "weight_decay": 0.0},
            {"params": model.rel_emb.parameters(), "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )


def checkpoint_payload(
    *, model, optimizer, epoch, best_epoch, best_metric, best_state, bad_epochs,
    history, first_step_audit, cumulative_train_sec, np_rng, signature, immutable,
    completed=False,
):
    return {
        "format_version": 3,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "bad_epochs": int(bad_epochs),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_model_state": best_state,
        "history": history,
        "first_step_audit": first_step_audit,
        "cumulative_train_sec": float(cumulative_train_sec),
        "rng_state": capture_rng_state(np_rng),
        "signature": signature,
        "immutable_config": immutable,
        "completed": bool(completed),
    }


def load_resume(run_dir, mode):
    if mode == "none":
        return None, None
    candidates = []
    if mode in {"auto", "latest"}:
        candidates.append(run_dir / "last_training_state.pt")
    if mode in {"auto", "best"}:
        candidates.append(run_dir / "best_training_state.pt")
    for path in candidates:
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False), path
    if mode in {"latest", "best"}:
        raise FileNotFoundError(f"Requested resume checkpoint not found in {run_dir}")
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Train invariant conditional RGCN on WordNet link prediction")
    ap.add_argument("--data-root", default="data/preprocessed/RGCN_structural/WORDNET/link_prediction")
    ap.add_argument("--output-root", default="results/RGCN_structural/WORDNET/link_prediction")
    ap.add_argument("--budget", choices=BUDGETS, default="reduced")
    ap.add_argument("--mode", choices=MODES, default="intersection")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    for key, typ in (
        ("hidden_dim", int), ("num_bases", int), ("edge_dropout_other", float),
        ("root_dropout_loop", float), ("lr", float), ("weight_decay", float),
        ("neg_per_pos", int), ("batch_size", int), ("epochs", int),
        ("patience", int), ("eval_interval", int), ("eval_batch_size", int),
        ("binary_k", int), ("edge_chunk_size", int),
    ):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=typ, default=DEFAULTS[key])
    ap.add_argument("--candidate-seed", type=int, default=1566911444)
    ap.add_argument(
        "--graph",
        choices=("semantic", "physical"),
        default="semantic",
        help="Graph to propagate on. 'semantic' is the compiled conditional program "
             "(the invariant arm); 'physical' is the variant's raw graph, which makes "
             "the originals and universal arms comparable to it under one architecture.",
    )
    ap.add_argument(
        "--train-objective",
        choices=("task_only", "semantic_all"),
        default="semantic_all",
        help=(
            "task_only reproduces the earlier conditional ablation; semantic_all "
            "optimizes every compiled base/inverse/shortcut triple and matches "
            "the original universal-graph RGCN objective."
        ),
    )
    ap.add_argument("--resume", choices=("none", "auto", "latest", "best"), default="auto")
    ap.add_argument("--additional-epochs", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--reset-patience-on-resume", action="store_true")
    ap.add_argument("--overwrite-run", action="store_true")
    args = ap.parse_args()

    if args.epochs <= 0 or args.eval_interval <= 0 or args.neg_per_pos <= 0:
        raise ValueError("epochs, eval_interval, and neg_per_pos must be positive")
    set_deterministic(args.seed, strict=True)
    device = torch.device(args.device)
    graph_path = Path(args.data_root) / args.budget / args.mode / args.variant / "graph.pt"
    data = load_graph(graph_path)
    if data["variant"] != args.variant:
        raise ValueError(f"Variant mismatch: {data['variant']} vs {args.variant}")

    run_dir = Path(args.output_root) / args.budget / args.mode / args.variant / f"seed_{args.seed}"
    if args.overwrite_run and run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    signature, immutable = config_signature(args, data)
    model = build_model(args, data, device)
    optimizer = build_optimizer(model, args)
    np_rng = np.random.default_rng(args.seed)

    # Which graph to propagate on. 'semantic' = the compiled conditional program
    # (the INVARIANT arm). 'physical' = the variant's own raw graph, which lets
    # this SAME runner and architecture produce the per-variant and
    # universal-graph arms, so all three are comparable. The relation vocabulary
    # is identical either way (2087 names), so the model is unchanged and only
    # the edge set differs.
    _graph = getattr(args, "graph", "semantic") or "semantic"
    edge_index = data[f"{_graph}_edge_index"].to(device)
    edge_type = data[f"{_graph}_edge_type"].to(device)
    selected_train, decoder_names = objective_spec(data, args.train_objective)
    train_np = selected_train.cpu().numpy().astype(np.int64)
    task_train_np = task_train_pos(data).cpu().numpy().astype(np.int64)
    val_np = data["val_pos"].cpu().numpy().astype(np.int64)
    test_np = data["test_pos"].cpu().numpy().astype(np.int64)
    # Evaluation queries use the 11 official task relations.  Augmented semantic
    # triples have distinct relation IDs and therefore do not change these masks.
    tail_filters, head_filters = build_filter_dicts(task_train_np, val_np, test_np)
    print(
        f"Training objective={args.train_objective} positives={len(train_np)} "
        f"decoder_relations={len(decoder_names)} message_edges={edge_type.numel()}",
        flush=True,
    )

    state, state_path = load_resume(run_dir, args.resume)
    start_epoch = 1
    best_epoch = 0
    best_metric = -float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    first_step_audit = {}
    cumulative_train_sec = 0.0
    if state is not None:
        if state.get("signature") != signature:
            raise ValueError(
                f"Resume signature mismatch for {state_path}.\n"
                f"checkpoint={state.get('immutable_config')}\ncurrent={immutable}"
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_state = state.get("best_model_state")
        best_epoch = int(state.get("best_epoch", 0))
        best_metric = float(state.get("best_metric", -float("inf")))
        bad_epochs = 0 if args.reset_patience_on_resume else int(state.get("bad_epochs", 0))
        history = list(state.get("history", []))
        first_step_audit = dict(state.get("first_step_audit", {}))
        cumulative_train_sec = float(state.get("cumulative_train_sec", 0.0))
        restore_rng_state(state["rng_state"], np_rng)
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from {state_path} at epoch {start_epoch}")

    target_epochs = args.epochs
    if args.additional_epochs > 0:
        target_epochs = max(target_epochs, start_epoch - 1 + args.additional_epochs)
    batch_size = args.batch_size if args.batch_size > 0 else len(train_np)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    last_evaluated_epoch = 0

    def validate(epoch: int, avg_loss: float):
        nonlocal best_epoch, best_metric, best_state, bad_epochs, last_evaluated_epoch
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(edge_index, edge_type, training=False)
        metrics, _ = evaluate_ranking(
            model,
            embeddings,
            val_np,
            tail_filters=tail_filters,
            head_filters=head_filters,
            num_entities=int(data["num_entities"]),
            device=device,
            eval_batch_size=args.eval_batch_size,
            binary_k=args.binary_k,
            candidate_seed=args.candidate_seed,
            save_candidates=False,
        )
        current = metrics["filtered_MRR"]
        improved = current > best_metric
        if improved:
            best_metric = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            atomic_torch_save(best_state, run_dir / "best_model.pt")
            best_payload = checkpoint_payload(
                model=model, optimizer=optimizer, epoch=epoch, best_epoch=best_epoch,
                best_metric=best_metric, best_state=best_state, bad_epochs=bad_epochs,
                history=history, first_step_audit=first_step_audit,
                cumulative_train_sec=cumulative_train_sec + (time.perf_counter() - wall_start),
                np_rng=np_rng, signature=signature, immutable=immutable,
            )
            atomic_torch_save(best_payload, run_dir / "best_training_state.pt")
        else:
            bad_epochs += args.eval_interval
        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "val": metrics,
            "improved": bool(improved),
            "best_filtered_MRR": float(best_metric),
            "bad_epochs": int(bad_epochs),
            "state_hash": state_dict_hash(model.state_dict()),
        })
        last_evaluated_epoch = epoch
        print(
            f"Epoch {epoch:4d} | loss {avg_loss:.6f} | "
            f"val filtered_MRR {current:.6f} | raw_MRR {metrics['raw_MRR']:.6f} | "
            f"H@1 {metrics['Hits@1']:.6f} H@3 {metrics['Hits@3']:.6f} H@10 {metrics['Hits@10']:.6f} "
            f"{'BEST' if improved else ''}",
            flush=True,
        )
        return metrics

    final_epoch = start_epoch - 1
    last_avg_loss = float("nan")
    for epoch in range(start_epoch, target_epochs + 1):
        final_epoch = epoch
        model.train()
        permutation = np_rng.permutation(len(train_np))
        shuffled = train_np[permutation]
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            pos_np = shuffled[start : start + batch_size]
            neg_np = sample_neg_train(pos_np, int(data["num_entities"]), args.neg_per_pos, np_rng)
            pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)

            if epoch == 1 and start == 0 and not first_step_audit:
                first_step_audit["initial_model_hash"] = state_dict_hash(model.state_dict())
                first_step_audit["positive_batch_hash"] = tensor_sha256(pos.cpu())
                first_step_audit["negative_batch_hash"] = tensor_sha256(neg.cpu())

            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode(edge_index, edge_type, training=True)
            pos_scores = model.score_triples(embeddings, pos)
            neg_scores = model.score_triples(embeddings, neg)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()

            if epoch == 1 and start == 0 and "gradient_hash" not in first_step_audit:
                first_step_audit.update({
                    "encoded_hash": tensor_sha256(embeddings.detach().cpu()),
                    "score_hash": tensor_sha256(scores.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "gradient_hash": gradient_hash(model),
                })
            optimizer.step()
            if epoch == 1 and start == 0 and "post_step_model_hash" not in first_step_audit:
                first_step_audit["post_step_model_hash"] = state_dict_hash(model.state_dict())

            total_loss += float(loss.detach().cpu())
            batches += 1
        last_avg_loss = total_loss / max(1, batches)

        if epoch % args.eval_interval == 0:
            validate(epoch, last_avg_loss)

        if epoch % args.checkpoint_every == 0:
            payload = checkpoint_payload(
                model=model, optimizer=optimizer, epoch=epoch, best_epoch=best_epoch,
                best_metric=best_metric, best_state=best_state, bad_epochs=bad_epochs,
                history=history, first_step_audit=first_step_audit,
                cumulative_train_sec=cumulative_train_sec + (time.perf_counter() - wall_start),
                np_rng=np_rng, signature=signature, immutable=immutable,
            )
            atomic_torch_save(payload, run_dir / "last_training_state.pt")
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; patience={args.patience}")
            break

    if final_epoch > 0 and last_evaluated_epoch != final_epoch:
        validate(final_epoch, last_avg_loss)

    cumulative_train_sec += time.perf_counter() - wall_start
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = final_epoch
        best_metric = float("nan")
        atomic_torch_save(best_state, run_dir / "best_model.pt")
    model.load_state_dict(best_state)

    peak_train = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(edge_index, edge_type, training=False)
    val_metrics, _ = evaluate_ranking(
        model, embeddings, val_np,
        tail_filters=tail_filters, head_filters=head_filters,
        num_entities=int(data["num_entities"]), device=device,
        eval_batch_size=args.eval_batch_size, binary_k=args.binary_k,
        candidate_seed=args.candidate_seed, save_candidates=False,
    )
    test_metrics, score_artifacts = evaluate_ranking(
        model, embeddings, test_np,
        tail_filters=tail_filters, head_filters=head_filters,
        num_entities=int(data["num_entities"]), device=device,
        eval_batch_size=args.eval_batch_size, binary_k=args.binary_k,
        candidate_seed=args.candidate_seed, save_candidates=True,
    )
    inference_sec = time.perf_counter() - infer_start
    peak_inference = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    np.savez_compressed(run_dir / "test_scores.npz", **score_artifacts)

    best_model_path = run_dir / "best_model.pt"
    result = {
        "format_version": "wordnet_rgcn_structural_result_v1",
        "budget": args.budget,
        "mode": args.mode,
        "variant": args.variant,
        "seed": args.seed,
        "graph_path": str(graph_path),
        "physical_hash": data["physical_hash"],
        "semantic_hash": data["semantic_hash"],
        "supervision_hash": data["supervision_hash"],
        "config": {**immutable, "epochs": args.epochs, "patience": args.patience},
        "train_objective": args.train_objective,
        "num_training_positives": int(len(train_np)),
        "num_task_training_positives": int(len(task_train_np)),
        "num_decoder_relations": int(len(decoder_names)),
        "num_message_relations": int(len(data["semantic_relation_names"])),
        "epochs_completed": int(final_epoch),
        "best_epoch": int(best_epoch),
        "best_val_filtered_MRR": float(best_metric),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time_sec": float(cumulative_train_sec),
        "inference_time_sec": float(inference_sec),
        "peak_train_gpu_bytes": peak_train,
        "peak_inference_gpu_bytes": peak_inference,
        "parameter_bytes": model_parameter_bytes(model),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_bytes": int(best_model_path.stat().st_size),
        "checkpoint_hash": state_dict_hash(model.state_dict()),
        "first_step_audit": first_step_audit,
        "resume_source": str(state_path) if state_path else None,
        "run_dir": str(run_dir),
    }
    atomic_json_dump(result, run_dir / "metrics.json")
    atomic_json_dump(history, run_dir / "history.json")
    final_payload = checkpoint_payload(
        model=model, optimizer=optimizer, epoch=final_epoch, best_epoch=best_epoch,
        best_metric=best_metric, best_state=best_state, bad_epochs=bad_epochs,
        history=history, first_step_audit=first_step_audit,
        cumulative_train_sec=cumulative_train_sec, np_rng=np_rng,
        signature=signature, immutable=immutable, completed=True,
    )
    atomic_torch_save(final_payload, run_dir / "last_training_state.pt")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
