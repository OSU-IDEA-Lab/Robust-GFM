#!/usr/bin/env python3
"""Freebase NC data-augmentation arm, on the SAMPLED trainer.

Why this exists
---------------
The Freebase cell has originals and a universal point and an invariant arm, but
no data-augmentation arm, because `rgcn_data_augmentation/run_FREEBASE_rgcn_augmentation.py`
trains FULL-GRAPH: it calls `model(edge_index, edge_type)` over the entire edge
set. That does not scale here -- it asked for a 41.9 GiB
allocation on a 31.7 GiB V100 while training `exact_2` alone (~94M directed
edges), and an 80 GB H100 does not clear it either. `exact_3` and
`union_exact_2_3` are ~10x larger again.

This runner supplies the missing arm by doing joint training inside the
neighbour-sampling machinery that the originals/universal and invariant arms
already use, so memory is bounded by batch x fanout rather than by edge count.

The contract
------------
* ONE shared model over several variants. A super-epoch visits every variant
  exactly once, in a fixed order, with one optimizer step per batch -- the same
  definition used by the IMDb-NC and WordNet augmentation arms.
* SHARED RELATION VOCABULARY. The variants' physical stores have different
  relation sets (unchanged 36, exact_2 42, exact_3 58). A single shared model
  needs one weight slot per relation *identity*, so the vocabulary is the union
  over the selected variants, keyed by (semantic_relation, direction), and every
  store is loaded through it (`BlockConditionalGraph(relation_vocabulary=...)`).
  Without this the same relation would land in different weight slots in
  different variants and "one shared model" would be a fiction.
* SHARED SUPERVISION. Labels and splits must be identical across variants; this
  is asserted, not assumed.
* Selection on validation micro-F1 averaged over variants, matching the fix
  applied to the per-variant runner (validation loss bottoms
  within ~2 epochs here while micro-F1 keeps climbing).

Outputs per-variant test metrics plus the cross-variant Kendall-tau, which is
the number the augmentation arm exists to produce: one model, several
representations, how consistent are its outputs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from freebase_conditional_rgcn import (  # noqa: E402
    BlockConditionalGraph,
    FreebaseBlockConditionalRGCN,
    build_two_layer_blocks,
)
from freebase_block_common import load_json, write_json  # noqa: E402
from aggregate_Freebase_rgcn_block_conditional import kendall_metrics  # noqa: E402
from rgcn_resource_metrics import gpu_reset, gpu_peak, model_size  # noqa: E402

FORMAT = "freebase_rgcn_augmentation_sampled_v1"
STATE_FILE = "latest_training_state.pt"


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = state["numpy_local"]
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["torch_cuda"]])


def _resume_signature(args, variants: list[str], seed: int, vocab_size: int) -> dict[str, Any]:
    """Everything that must be identical for a saved state to continue THIS run."""
    keys = ("up_to_template", "set_mode", "scope", "embedding_dim", "hidden_dim", "num_bases", "dropout",
            "use_type_embedding", "lr", "weight_decay", "super_epochs", "patience", "min_delta",
            "batch_size", "eval_batch_size", "fanout_layer1", "fanout_layer2", "grad_clip")
    return {"format": FORMAT, "seed": seed, "variants": list(variants), "vocab": vocab_size,
            **{k: getattr(args, k) for k in keys}}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def store_dir(root: Path, variant: str, up_to_template: str, set_mode: str, scope: str) -> Path:
    return root / variant / up_to_template.format(variant=variant) / set_mode / scope


def build_relation_vocabulary(store_paths: dict[str, Path]) -> dict[tuple[str, int], int]:
    """Union of every selected variant's (relation, direction) channels.

    Sorted so the mapping is deterministic and independent of variant order --
    two runs with the same variant SET produce the same slots regardless of the
    order they were listed on the command line.
    """
    keys: set[tuple[str, int]] = set()
    for path in store_paths.values():
        catalog = load_json(path / "relation_catalog.json")
        for record in catalog["channels"]:
            keys.add((str(record["semantic_relation"]), int(record["direction"])))
    return {key: index for index, key in enumerate(sorted(keys))}


def evaluate(model, graph, indices, *, batch_size, fanout1, fanout2, device, sampling_seed):
    model.eval()
    logits_parts, label_parts = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            targets = np.asarray(graph.target_nodes[batch], dtype=np.int64)
            block1, block2 = build_two_layer_blocks(
                graph,
                targets,
                fanout_layer1=fanout1,
                fanout_layer2=fanout2,
                sampling_seed=sampling_seed + start,
            )
            out = model(graph, block1, block2, device=device)
            logits_parts.append(out.detach().cpu().numpy())
            label_parts.append(np.asarray(graph.labels[batch], dtype=np.int64))
    logits = np.concatenate(logits_parts) if logits_parts else np.empty((0, graph.num_classes), np.float32)
    labels = np.concatenate(label_parts) if label_parts else np.empty(0, np.int64)
    predictions = logits.argmax(axis=1)
    micro = float((predictions == labels).mean()) if len(labels) else 0.0
    classes = sorted(set(labels.tolist()))
    f1s = []
    for c in classes:
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return logits, {"micro_f1": micro, "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def run_seed(args, variants: list[str], seed: int, out_root: Path) -> dict[str, Any]:
    set_determinism(seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    root = Path(args.preprocessed_root).resolve()

    paths = {v: store_dir(root, v, args.up_to_template, args.set_mode, args.scope) for v in variants}
    for v, p in paths.items():
        if not (p / "model_input_hashes.json").is_file():
            raise SystemExit(f"FATAL: {v} is not compiled at {p}")

    vocabulary = build_relation_vocabulary(paths)
    print(f"[vocab] {len(vocabulary)} shared relation slots over {len(variants)} variants", flush=True)

    graphs = {v: BlockConditionalGraph(p, relation_vocabulary=vocabulary) for v, p in paths.items()}

    # Supervision must be identical across variants, or "one shared model over
    # several representations of the same data" is not what is being trained.
    reference = graphs[variants[0]]
    for v in variants[1:]:
        g = graphs[v]
        for key in ("train_idx", "val_idx", "test_idx"):
            if not np.array_equal(reference.splits[key], g.splits[key]):
                raise SystemExit(f"FATAL: {v} has different {key} from {variants[0]}")
        if not np.array_equal(np.asarray(reference.labels), np.asarray(g.labels)):
            raise SystemExit(f"FATAL: {v} has different labels from {variants[0]}")
        if not np.array_equal(np.asarray(reference.target_nodes), np.asarray(g.target_nodes)):
            raise SystemExit(f"FATAL: {v} has different target nodes from {variants[0]}")
    print(f"[ok] supervision identical across {variants}", flush=True)

    model = FreebaseBlockConditionalRGCN(
        num_nodes=reference.num_nodes,
        num_node_types=reference.num_node_types,
        num_relations=len(vocabulary),
        num_classes=reference.num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_bases=min(args.num_bases, len(vocabulary)),
        dropout=args.dropout,
        use_type_embedding=args.use_type_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = reference.splits["train_idx"]
    val_idx = reference.splits["val_idx"]
    test_idx = reference.splits["test_idx"]

    run_dir = out_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale, history = float("-inf"), 0, 0, []
    best_state = None
    rng = np.random.default_rng(seed)
    super_epochs_ran = 0
    optimizer_steps = 0
    train_seconds_before = 0.0          # accumulated over earlier (interrupted) processes
    start_super_epoch = 1
    stopped_early = False
    signature = _resume_signature(args, variants, seed, len(vocabulary))
    state_path = run_dir / STATE_FILE
    if args.resume and state_path.is_file():
        # Exact restart from the last completed super-epoch: model, optimizer, best model,
        # early-stopping counters, history and every RNG stream. Block sampling is seeded
        # explicitly per (seed, super_epoch, batch), so it needs no state of its own.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("signature") != signature:
            raise SystemExit(f"FATAL: resume signature mismatch for {state_path}\n"
                             f"  saved={state.get('signature')}\n  current={signature}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_state = state["best_state"]
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        history = list(state["history"])
        super_epochs_ran, optimizer_steps = state["super_epochs_ran"], state["optimizer_steps"]
        train_seconds_before = float(state["train_seconds"])
        stopped_early = bool(state.get("stopped_early", False))
        _restore_rng_state(state["rng"], rng)
        start_super_epoch = super_epochs_ran + 1
        print(f"[resume] {state_path}: continuing at super-epoch {start_super_epoch} "
              f"(best {best_score:.6f} @ {best_epoch}, stale {stale}, {train_seconds_before:.0f}s so far)", flush=True)
    gpu_reset(device)
    started = time.perf_counter()

    def save_state(completed_super_epoch: int, early: bool) -> None:
        if not args.resume:
            return
        _atomic_torch_save({
            "signature": signature,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "best_state": None if best_state is None else {k: v.detach().cpu() for k, v in best_state.items()},
            "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
            "history": history, "super_epochs_ran": completed_super_epoch, "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds_before + (time.perf_counter() - started),
            "stopped_early": early,
            "rng": _rng_state(rng),
        }, state_path)

    for super_epoch in range(start_super_epoch, args.super_epochs + 1):
        if stopped_early:
            break
        super_epochs_ran = super_epoch
        model.train()
        order = rng.permutation(len(train_idx))
        losses = []
        # One super-epoch = one pass over the training nodes for EVERY variant.
        for variant in variants:
            graph = graphs[variant]
            for start in range(0, len(order), args.batch_size):
                batch = train_idx[order[start : start + args.batch_size]]
                targets = np.asarray(graph.target_nodes[batch], dtype=np.int64)
                block1, block2 = build_two_layer_blocks(
                    graph,
                    targets,
                    fanout_layer1=args.fanout_layer1,
                    fanout_layer2=args.fanout_layer2,
                    sampling_seed=seed + super_epoch * 7919 + start,
                )
                out = model(graph, block1, block2, device=device)
                labels = torch.as_tensor(
                    np.asarray(graph.labels[batch], dtype=np.int64), device=out.device
                )
                loss = F.cross_entropy(out, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))

        val_scores = {}
        for variant in variants:
            _, metrics = evaluate(
                model, graphs[variant], val_idx,
                batch_size=args.eval_batch_size, fanout1=args.fanout_layer1,
                fanout2=args.fanout_layer2, device=device,
                sampling_seed=seed + 1_000_000,
            )
            val_scores[variant] = metrics["micro_f1"]
        mean_val = float(np.mean(list(val_scores.values())))
        record = {
            "super_epoch": super_epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_micro_f1_by_variant": val_scores,
            "mean_val_micro_f1": mean_val,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if mean_val > best_score + args.min_delta:
            best_score, best_epoch, stale = mean_val, super_epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        write_json(run_dir / "history.json", history)
        stopped_early = stale >= args.patience
        save_state(super_epoch, stopped_early)
        if stopped_early:
            break
        if args.stop_after and super_epoch - start_super_epoch + 1 >= args.stop_after:
            # smoke-test hook: behave like a job that was cut here (state saved, no summary)
            print(f"[stop-after] leaving after {args.stop_after} super-epoch(s) in this process", flush=True)
            raise SystemExit(0)

    train_seconds = train_seconds_before + (time.perf_counter() - started)
    peak_train = gpu_peak(device)
    if best_state is not None:
        model.load_state_dict(best_state)
    gpu_reset(device)
    infer_started = time.perf_counter()

    per_variant, logits_by_variant = {}, {}
    for variant in variants:
        logits, metrics = evaluate(
            model, graphs[variant], test_idx,
            batch_size=args.eval_batch_size, fanout1=args.fanout_layer1,
            fanout2=args.fanout_layer2, device=device,
            sampling_seed=seed + 2_000_000,
        )
        per_variant[variant] = metrics
        logits_by_variant[variant] = logits
        print(f"  test {variant}: micro-F1 {metrics['micro_f1']:.6f}", flush=True)

    pairwise = {}
    names = list(variants)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a} vs {b}"] = kendall_metrics(
                logits_by_variant[a].astype(np.float64), logits_by_variant[b].astype(np.float64)
            )

    inference_seconds = time.perf_counter() - infer_started
    peak_infer = gpu_peak(device)
    means = [m["micro_f1"] for m in per_variant.values()]
    summary = {
        "format_version": FORMAT,
        "seed": seed,
        "variants": variants,
        "shared_relation_slots": len(vocabulary),
        "selection_metric": "mean_val_micro_f1",
        "best_super_epoch": best_epoch,
        "best_mean_val_micro_f1": best_score,
        "super_epochs_ran": super_epochs_ran,
        "variant_epochs_ran": super_epochs_ran * len(variants),
        "optimizer_steps": optimizer_steps,
        "train_seconds": train_seconds,
        # params + peak GPU beside the task metrics.
        "inference_time_sec": inference_seconds,
        "peak_training_gpu_mib": peak_train["peak_gpu_mib"],
        "peak_training_reserved_gpu_mib": peak_train["peak_gpu_reserved_mib"],
        "peak_inference_gpu_mib": peak_infer["peak_gpu_mib"],
        "peak_inference_reserved_gpu_mib": peak_infer["peak_gpu_reserved_mib"],
        **model_size(model),
        "per_variant_test_metrics": per_variant,
        "micro_f1_variant_spread": float(max(means) - min(means)) if means else None,
        "pairwise_invariance": pairwise,
        "mean_pairwise_kendall_tau": float(
            np.mean([p["kendall_tau"] for p in pairwise.values()])
        ) if pairwise else None,
        "history": history,
        "resumed": bool(args.resume and start_super_epoch > 1),
    }
    np.savez_compressed(run_dir / "test_logits.npz", **logits_by_variant)
    write_json(run_dir / "summary.json", summary)      # written last: its presence means "done"
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1566911444, 20241017, 20251017])
    ap.add_argument("--up-to-template", default="physical_{variant}",
                    help="Store subdirectory per variant; '{variant}' is substituted. "
                         "Default matches the physical-only stores.")
    ap.add_argument("--set-mode", default="union")
    ap.add_argument("--scope", default="all_graph")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-bases", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--use-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--super-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--fanout-layer1", type=int, default=4)
    ap.add_argument("--fanout-layer2", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--resume", action="store_true",
                    help="save seed_*/latest_training_state.pt after every super-epoch and continue from it "
                         "if present (exact restart: model, optimizer, best model, patience, RNG streams)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="(smoke tests) exit after this many super-epochs in this process, as if the job were cut")
    ap.add_argument("--skip-completed", action="store_true",
                    help="skip a seed whose summary.json already exists")
    args = ap.parse_args()

    out_root = Path(args.results_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        done = out_root / f"seed_{seed}" / "summary.json"
        if args.skip_completed and done.is_file():
            print(f"[skip] seed {seed}: {done} exists", flush=True)
            summaries.append(load_json(done))
            continue
        summaries.append(run_seed(args, list(args.variants), seed, out_root))

    per_variant: dict[str, list[float]] = {}
    for s in summaries:
        for v, m in s["per_variant_test_metrics"].items():
            per_variant.setdefault(v, []).append(m["micro_f1"])
    taus = [s["mean_pairwise_kendall_tau"] for s in summaries if s["mean_pairwise_kendall_tau"] is not None]
    means = [float(np.mean(v)) for v in per_variant.values()]
    aggregate = {
        "format_version": FORMAT,
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "per_variant": {
            v: {
                "micro_f1_mean": float(np.mean(vals)),
                "micro_f1_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            for v, vals in per_variant.items()
        },
        "micro_f1_variant_spread": float(max(means) - min(means)) if means else None,
        "mean_pairwise_kendall_tau": float(np.mean(taus)) if taus else None,
        "std_pairwise_kendall_tau": float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0,
    }
    write_json(out_root / "aggregate.json", aggregate)

    print("\n===== Freebase NC data augmentation (sampled, one shared model) =====")
    for v, st in aggregate["per_variant"].items():
        print(f"  {v:16s} micro-F1 {st['micro_f1_mean']:.4f} +/- {st['micro_f1_std']:.4f} (n={st['n']})")
    print(f"  variant spread {aggregate['micro_f1_variant_spread']:.4f}")
    if aggregate["mean_pairwise_kendall_tau"] is not None:
        print(f"  mean pairwise Kendall tau {aggregate['mean_pairwise_kendall_tau']:.4f}")
    print(f"  written: {out_root/'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
