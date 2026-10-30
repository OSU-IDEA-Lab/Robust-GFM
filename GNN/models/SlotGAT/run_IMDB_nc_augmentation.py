#!/usr/bin/env python3
"""Data-augmentation (joint-training) baseline for SlotGAT on IMDb NC.

This is the augmentation arm of the IMDb cell.

Design mirrors the RGCN augmentation runner:

* **One shared model and one shared optimizer** see every variant. There is no
  per-variant model and no output fusion.
* A **super-epoch** is a shuffled visit to every variant, one optimizer step per
  variant. Variant order is drawn from a seeded generator so runs are
  reproducible.
* The **channel vocabulary is global and fixed before construction** -- all
  variants already share ``conditional_bias.CHANNEL_NAMES``, so a variant simply
  leaves the channels it lacks empty. Parameter count is therefore identical to
  the single-variant arms and identical across variants.
* Each variant is presented through its **physical** context plan, i.e. its own
  edges. Joint training over physical variants is exactly what the "data
  augmentation" baseline means; using the semantic closure here would make every
  variant the same graph and the baseline meaningless.
* **Supervision is shared and asserted**: labels, splits and target nodes must be
  identical across variants or the run aborts.

**Budget equalisation.** A super-epoch performs one update per variant, so N
super-epochs over V variants is N*V variant-epochs. Run at the single-variant
budget divided by V (``--epochs`` is in *variant-epochs*, and is converted), so
this arm sees the same number of updates as the arms it is compared against.

Kendall-tau is computed by feeding each variant into the shared model and
comparing the resulting test logits, which is how the augmentation row's
invariance number is defined.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

import run_IMDB_nc_conditional as base
from run_IMDB_nc_conditional import (
    DEFAULT_SEEDS,
    NUM_CLASSES,
    build_conditional_model,
    json_dump,
    load_datasets,
    make_data_recorder,
    safe_cuda_peak_memory_stats,
    safe_reset_cuda_peak_memory_stats,
    set_deterministic,
    split_metrics,
    tensorize_features,
)


def assert_shared_supervision(bundles: dict[int, Any]) -> dict[str, str]:
    reference = None
    for variant, bundle in sorted(bundles.items()):
        signature = (bundle.labels_hash, bundle.splits_hash)
        if reference is None:
            reference = (variant, signature)
        elif signature != reference[1]:
            raise SystemExit(
                f"Variant {variant} does not share supervision with variant "
                f"{reference[0]}: {signature} vs {reference[1]}"
            )
    assert reference is not None
    return {"labels_hash": reference[1][0], "splits_hash": reference[1][1]}


def mean_row_kendall(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for row_a, row_b in zip(left, right):
        tau, _ = kendalltau(row_a, row_b, variant="b", nan_policy="omit")
        if not np.isnan(tau):
            values.append(float(tau))
    return float(np.mean(values)) if values else float("nan")


def train_joint(args: argparse.Namespace, bundles: dict[int, Any], seed: int,
                device: torch.device) -> dict[str, Any]:
    set_deterministic(seed, args.deterministic)
    memory_reset = False
    if device.type == "cuda":
        torch.cuda.empty_cache()
        memory_reset = safe_reset_cuda_peak_memory_stats()

    variants = sorted(bundles)
    reference = bundles[variants[0]]
    features, in_dims = tensorize_features(reference.features_raw, args.feats_type, device)
    labels = torch.as_tensor(reference.labels_np, dtype=torch.int64, device=device)
    train_idx = torch.as_tensor(reference.split_indices["train"], dtype=torch.int64, device=device)
    val_idx = reference.split_indices["val"]

    # One shared model over the global channel vocabulary.
    model, _forward, graph_meta = build_conditional_model(
        args,
        features_list=features,
        in_dims=in_dims,
        program=reference.physical_program,
        device=device,
        data_recorder=make_data_recorder(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Budget equalisation: --epochs is in VARIANT-epochs.
    super_epochs = max(1, int(round(args.epochs / len(variants))))
    patience = max(1, int(round(args.patience / len(variants))))
    print(
        f"[augmentation] seed={seed} variants={len(variants)} "
        f"super_epochs={super_epochs} (= {super_epochs * len(variants)} variant-epochs, "
        f"matched to --epochs {args.epochs}) patience={patience} super-epochs"
    )

    order_rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_super_epoch = 0
    bad = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for super_epoch in range(1, super_epochs + 1):
        model.train()
        visit = order_rng.permutation(variants)
        losses = []
        for variant in visit:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, bundles[int(variant)].physical_program)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            scores = []
            for variant in variants:
                logits, _ = model(features, bundles[variant].physical_program)
                scores.append(split_metrics(logits, labels, val_idx)["micro_f1"])
        # Selection on the MEAN validation micro-F1 across variants: the joint
        # model has to be good on all of them, and this is the evaluation metric.
        score = float(np.mean(scores))
        history.append(
            {
                "super_epoch": super_epoch,
                "mean_train_loss": float(np.mean(losses)),
                "mean_val_micro_f1": score,
            }
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_super_epoch = super_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("Joint trainer never captured a model state")
    model.load_state_dict(best_state, strict=True)
    elapsed = time.perf_counter() - start
    # Separate training / inference peak windows (metric contract).
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        training_peak_allocated, training_peak_reserved = safe_cuda_peak_memory_stats()
        inference_reset = safe_reset_cuda_peak_memory_stats()
    else:
        training_peak_allocated, training_peak_reserved, inference_reset = None, None, False

    infer_start = time.perf_counter()
    model.eval()
    per_variant: dict[str, Any] = {}
    test_logits: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for variant in variants:
            logits, _ = model(features, bundles[variant].physical_program)
            per_variant[bundles[variant].variant_name] = {
                split: split_metrics(logits, labels, idx)
                for split, idx in bundles[variant].split_indices.items()
            }
            idx = torch.as_tensor(
                bundles[variant].split_indices["test"], dtype=torch.int64, device=device
            )
            test_logits[variant] = logits[idx].detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - infer_start

    pairwise = []
    for a, b in itertools.combinations(variants, 2):
        pairwise.append(
            {
                "left": bundles[a].variant_name,
                "right": bundles[b].variant_name,
                "mean_test_logit_kendall_tau": mean_row_kendall(test_logits[a], test_logits[b]),
                "max_abs_logit_difference": float(
                    np.max(np.abs(test_logits[a] - test_logits[b]))
                ),
                "prediction_agreement": float(
                    np.mean(
                        np.argmax(test_logits[a], axis=1) == np.argmax(test_logits[b], axis=1)
                    )
                ),
            }
        )

    peak_allocated, peak_reserved = (
        safe_cuda_peak_memory_stats() if device.type == "cuda" else (None, None)
    )
    micro = [per_variant[bundles[v].variant_name]["test"]["micro_f1"] for v in variants]
    taus = [row["mean_test_logit_kendall_tau"] for row in pairwise]
    parameter_count = int(sum(p.numel() for p in model.parameters()))

    return {
        "format_version": "slotgat_imdb_nc_augmentation_v1",
        "dataset": "IMDb node classification",
        "model": "SlotGAT",
        "model_impl": getattr(args, "model_impl", "genuine"), "ablate": sorted(getattr(args, "ablate", []) or []),
        "condition": "augmentation",
        "seed": seed,
        "variants": [bundles[v].variant_name for v in variants],
        "selection_metric": "mean_val_micro_f1",
        "best_selection_value": best_score,
        "best_super_epoch": best_super_epoch,
        "super_epochs_run": len(history),
        "variant_epochs_run": len(history) * len(variants),
        "variant_epoch_budget": args.epochs,
        "shared_model": True,
        "shared_optimizer": True,
        "output_fusion": False,
        "per_variant_metrics": per_variant,
        "test_micro_f1_mean": float(np.mean(micro)),
        "test_micro_f1_std": float(np.std(micro)),
        "test_micro_f1_spread": float(max(micro) - min(micro)),
        "pairwise_invariance": pairwise,
        "mean_pairwise_kendall_tau": float(np.mean(taus)) if taus else float("nan"),
        "min_pairwise_kendall_tau": float(np.min(taus)) if taus else float("nan"),
        "training_time_seconds": elapsed,
        "inference_time_seconds": inference_seconds,
        "num_parameters": parameter_count,
        "parameter_mib": parameter_count * 4 / (1024 ** 2),
        "memory": {
            "cuda_peak_allocated_bytes": training_peak_allocated,
            "cuda_peak_reserved_bytes": training_peak_reserved,
            "cuda_peak_stats_reset": bool(memory_reset),
            "training_peak_allocated_bytes": training_peak_allocated,
            "training_peak_reserved_bytes": training_peak_reserved,
            "inference_peak_allocated_bytes": peak_allocated,
            "inference_peak_reserved_bytes": peak_reserved,
            "inference_peak_stats_reset": bool(inference_reset),
        },
        "history": history,
        "graph": graph_meta,
    }, test_logits, {bundles[v].variant_name: bundles[v] for v in variants}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "imdb_nc_augmentation")
    parser.add_argument("--feature-mode", default="pooled_bow", choices=["pooled_bow", "identity"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non-strict-context", action="store_true")
    parser.add_argument("--feats-type", type=int, default=0, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=300,
                        help="Budget in VARIANT-epochs; divided by the variant count to "
                             "give super-epochs, so this arm matches the single-variant arms.")
    parser.add_argument("--patience", type=int, default=40,
                        help="Also in variant-epochs; converted the same way.")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--aggregator", default="SA", choices=["SA"])
    parser.add_argument("--SAattDim", type=int, default=3)
    parser.add_argument("--model-impl", dest="model_impl", default="genuine", choices=["genuine", "parity"],
                        help="'parity' = model/parity_slotgat.py, the verified port of the upstream slotGAT")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ablate", nargs="*", default=[], choices=list(base.PARITY_ABLATIONS))
    parser.add_argument(
        "--data-suffix", dest="data_suffix", default="",
        help="Appended to each IMDB_var<N> directory name (e.g. '_year'), exactly as in "
             "run_IMDB_nc_conditional.py, so the augmentation arm reads the same build "
             "as the other three arms.")
    args = parser.parse_args()
    args.seeds = [int(v) for v in str(args.seeds).replace(",", " ").split()]
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)

    import os
    os.chdir(Path(__file__).resolve().parent)
    if getattr(args, "data_suffix", ""):
        # VARIANT_NAMES is the module-level table load_datasets reads; patch it in place
        # (the conditional runner does the same) so every bundle comes from the suffixed build.
        for _k in list(base.VARIANT_NAMES):
            if not base.VARIANT_NAMES[_k].endswith(args.data_suffix):
                base.VARIANT_NAMES[_k] = base.VARIANT_NAMES[_k] + args.data_suffix
        print(f"[data-suffix] reading variants {sorted(base.VARIANT_NAMES.values())}", flush=True)
    bundles = load_datasets(
        args.variants, strict=not args.non_strict_context, feature_mode=args.feature_mode
    )
    supervision = assert_shared_supervision(bundles)
    print(f"Shared supervision asserted: {supervision}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        result, test_logits, _named = train_joint(args, bundles, seed, device)
        run_dir = args.results_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for variant, logits in test_logits.items():
            bundle = bundles[variant]
            test_idx = bundle.split_indices["test"]
            np.savez_compressed(
                run_dir / f"{bundle.variant_name}_test_logits.npz",
                logits=logits,
                node_ids=test_idx,
                labels=bundle.labels_np[test_idx],
            )
        result["supervision"] = supervision
        json_dump(run_dir / "result.json", result)
        summaries.append(result)
        print(
            f"[augmentation] seed={seed} micro-F1 {result['test_micro_f1_mean']:.4f}"
            f" +/- {result['test_micro_f1_std']:.4f} spread {result['test_micro_f1_spread']:.4f}"
            f" tau {result['mean_pairwise_kendall_tau']:.4f}"
            f" super_epochs {result['super_epochs_run']} ({result['variant_epochs_run']} variant-epochs)"
        )

    aggregate = {
        "condition": "augmentation",
        "seeds": args.seeds,
        "test_micro_f1_mean": float(np.mean([s["test_micro_f1_mean"] for s in summaries])),
        "test_micro_f1_std": float(np.std([s["test_micro_f1_mean"] for s in summaries])),
        "mean_pairwise_kendall_tau": float(
            np.mean([s["mean_pairwise_kendall_tau"] for s in summaries])
        ),
        "min_pairwise_kendall_tau": float(
            np.min([s["min_pairwise_kendall_tau"] for s in summaries])
        ),
        "per_seed": [
            {
                "seed": s["seed"],
                "test_micro_f1_mean": s["test_micro_f1_mean"],
                "test_micro_f1_spread": s["test_micro_f1_spread"],
                "mean_pairwise_kendall_tau": s["mean_pairwise_kendall_tau"],
                "variant_epochs_run": s["variant_epochs_run"],
            }
            for s in summaries
        ],
    }
    json_dump(args.results_dir / "aggregate_summary.json", aggregate)
    print(
        f"\nAUGMENTATION micro-F1 {aggregate['test_micro_f1_mean']:.4f} "
        f"+/- {aggregate['test_micro_f1_std']:.4f}  "
        f"mean tau {aggregate['mean_pairwise_kendall_tau']:.4f}"
    )


if __name__ == "__main__":
    main()
