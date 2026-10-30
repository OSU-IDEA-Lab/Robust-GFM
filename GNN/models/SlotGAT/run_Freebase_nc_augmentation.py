#!/usr/bin/env python3
"""Data-augmentation (joint-training) baseline for SlotGAT on Freebase NC.

Fourth arm of the Freebase cell, mirroring ``run_IMDB_nc_augmentation.py``:

* **One shared model and one shared optimizer** see every variant; no per-variant
  model, no output fusion.
* A **super-epoch** is a seeded shuffled visit to every variant, one optimizer
  step per variant.
* The **channel vocabulary is global and fixed** -- every variant's context plan
  is built over the same 85-channel catalog, so parameter count is identical to
  the single-variant arms and identical across variants.
* Each variant is presented through its **own physical message schedule**, built
  by ``prepare_Freebase_union_message_equivalent.py --any-context-kind``. Joint
  training over physical variants is what the augmentation baseline means; using
  the shared union schedule would make every variant the same graph.
* **Supervision is asserted shared** across variants before training.

**Budget equalisation.** ``--epochs`` is in *variant*-epochs and is divided by the
variant count to give super-epochs, so this arm sees the same number of updates
as the arms it is compared against.

**Stopping vs selection are decoupled**, for the reason documented in
``run_Freebase_nc_block_conditional.py``: Freebase's majority class is 48.8% of
the test set, so micro-F1 plateaus there while validation loss keeps falling.
Patience runs off mean validation loss; the checkpoint is chosen on mean
validation micro-F1.
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

import run_Freebase_nc_block_conditional as native
from freebase_block_conditional import FreebaseBlockContextPlan, write_json
from freebase_union_messages import UnionMessageSchedule
from model.genuine_union_message_slotgat import ExactUnionMessageConditionalSlotGAT


def mean_row_kendall(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for row_a, row_b in zip(left, right):
        tau, _ = kendalltau(row_a, row_b, variant="b", nan_policy="omit")
        if not np.isnan(tau):
            values.append(float(tau))
    return float(np.mean(values)) if values else float("nan")


def build_shared_model(args, plan, dataset, schedule, device):
    return ExactUnionMessageConditionalSlotGAT(
        layout=plan.layout,
        in_dims=dataset.in_dims,
        schedule=schedule,
        num_classes=native.NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        edge_feats=args.edge_feats,
        dropout_feat=args.dropout_feat,
        dropout_attn=args.dropout_attn,
        slope=args.slope,
        aggregator=args.aggregator,
        sa_attention_dim=args.SAattDim,
        require_identity_sparse=not args.allow_nonidentity_sparse,
        message_chunk_edges=args.union_message_chunk_edges,
        checkpoint_chunks=not args.disable_union_checkpoint,
        parity=getattr(args, "model_impl", "genuine") == "parity",
    ).to(device)


def train_joint(args, plans, schedules, dataset, seed, device) -> dict[str, Any]:
    native.set_deterministic(seed, args.deterministic)
    memory_reset, _ = native.safe_cuda_reset_peak() if hasattr(native, "safe_cuda_reset_peak") else (False, None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
            memory_reset = True
        except Exception:
            memory_reset = False

    variants = [plan.variant for plan in plans]
    by_variant = {plan.variant: plan for plan in plans}

    # The shared model is built once. Every variant's schedule shares the same
    # channel catalog and node layout, so one parameter set serves all of them;
    # only the message tensors swap per step.
    reference = plans[0]
    model = build_shared_model(args, reference, dataset, schedules[reference.variant], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ``ExactUnionMessageConditionalSlotGAT._runtime`` caches by DEVICE ONLY, so
    # simply reassigning ``model.schedule`` would keep serving the first
    # variant's messages -- every variant would silently train on one graph and
    # the arm would look invariant for the wrong reason. Build each variant's
    # runtime once, up front, and install the right one before each forward.
    runtimes: dict[str, Any] = {}
    for plan in plans:
        model.schedule = schedules[plan.variant]
        model._runtime_cache = {}
        runtimes[plan.variant] = model._runtime(device)
    sizes = {v: int(rt.source.numel()) for v, rt in runtimes.items()}
    print(f"  per-variant runtime message counts: {sizes}")
    if len(set(sizes.values())) == 1 and len(sizes) > 1:
        # Same size is not proof of a bug, but identical tensors are.
        first = next(iter(runtimes.values()))
        if all(torch.equal(rt.source, first.source) for rt in runtimes.values()):
            raise SystemExit(
                "All variant runtimes are identical -- the schedule swap is not taking "
                "effect, so joint training would see a single graph"
            )

    def forward(variant: str):
        # Swap BOTH the schedule and its prebuilt runtime. Parameters are
        # untouched, so this remains one shared model over all variants.
        model.schedule = schedules[variant]
        model._runtime_cache = {str(device): runtimes[variant]}
        return model(dataset.features, by_variant[variant])

    super_epochs = max(1, int(round(args.epochs / len(variants))))
    patience = max(1, int(round(args.patience / len(variants))))
    print(
        f"[augmentation] seed={seed} variants={len(variants)} super_epochs={super_epochs} "
        f"(= {super_epochs * len(variants)} variant-epochs, matched to --epochs {args.epochs}) "
        f"patience={patience} super-epochs",
        flush=True,
    )

    order_rng = np.random.RandomState(seed)
    best_select = -float("inf")
    best_stop = float("inf")
    best_state = None
    best_super_epoch = 0
    bad = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    # RESUME: per-super-epoch atomic snapshot (model, optimizer, selection
    # state, history, the variant-order RNG and every torch/numpy/python RNG stream) so a
    # requeued job continues exactly where it stopped. Mirrors train_one_run in the
    # headline runner.
    latest_state_path = args.results_dir / f"seed_{seed}" / "latest_training_state.pt"
    start_super_epoch = 1
    prior_elapsed = 0.0
    prior_peak_alloc = 0
    prior_peak_reserved = 0
    resumed_from = None
    if getattr(args, "resume", False) and latest_state_path.exists():
        snap = torch.load(latest_state_path, map_location="cpu", weights_only=False)
        if snap.get("seed") != int(seed) or snap.get("variants") != list(variants):
            raise RuntimeError(f"{latest_state_path} is for a different seed/variant set; refusing to resume")
        model.load_state_dict(snap["model_state"], strict=True); model.to(device)
        optimizer.load_state_dict(snap["optimizer_state"])
        best_select, best_stop = snap["best_select"], snap["best_stop"]
        best_state, best_super_epoch, bad = snap["best_state"], snap["best_super_epoch"], snap["bad"]
        history = list(snap["history"])
        order_rng.set_state(snap["order_rng_state"])
        native._restore_rng_state(snap["rng"])
        prior_elapsed = float(snap["training_seconds_elapsed"])
        prior_peak_alloc = int(snap.get("peak_allocated_bytes") or 0)
        prior_peak_reserved = int(snap.get("peak_reserved_bytes") or 0)
        resumed_from = int(snap["super_epoch"])
        start_super_epoch = resumed_from + 1
        start = time.perf_counter() - prior_elapsed
        print(f"    [resume] seed={seed}: continuing after super-epoch {resumed_from} "
              f"(best={best_super_epoch}, noimp={bad}, {prior_elapsed:.0f}s so far)", flush=True)
        if bad >= patience or resumed_from >= super_epochs:
            start_super_epoch = super_epochs + 1

    for super_epoch in range(start_super_epoch, super_epochs + 1):
        model.train()
        losses = []
        for variant in order_rng.permutation(variants):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward(str(variant))
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(
                logp.index_select(0, dataset.train_idx),
                dataset.labels.index_select(0, dataset.train_idx),
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses, val_f1 = [], []
        with torch.no_grad():
            for variant in variants:
                logits, _ = forward(variant)
                logp = F.log_softmax(logits, dim=1)
                val_losses.append(float(F.nll_loss(
                    logp.index_select(0, dataset.val_idx),
                    dataset.labels.index_select(0, dataset.val_idx),
                ).item()))
                val_f1.append(float(native.classification_metrics(
                    logits, dataset.labels, dataset.val_idx)["micro_f1"]))
        mean_val_loss = float(np.mean(val_losses))
        mean_val_f1 = float(np.mean(val_f1))
        history.append({
            "super_epoch": super_epoch,
            "mean_train_loss": float(np.mean(losses)),
            "mean_val_loss": mean_val_loss,
            "mean_val_micro_f1": mean_val_f1,
        })
        if mean_val_loss < best_stop - args.min_delta:
            best_stop = mean_val_loss
            bad = 0
        else:
            bad += 1
        if mean_val_f1 > best_select + args.min_delta:
            best_select = mean_val_f1
            best_super_epoch = super_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.print_every > 0 and (super_epoch == 1 or super_epoch % args.print_every == 0):
            print(f"    super_epoch={super_epoch:03d} train={np.mean(losses):.6f} "
                  f"val_loss={mean_val_loss:.6f} val_microF1={mean_val_f1:.4f} "
                  f"best={best_super_epoch:03d} noimp={bad}", flush=True)
        if getattr(args, "save_latest_state", True):
            _pa = _pr = 0
            if device.type == "cuda":
                try:
                    _pa, _pr = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
                except Exception:
                    pass
            latest_state_path.parent.mkdir(parents=True, exist_ok=True)
            native._atomic_torch_save({
                "super_epoch": super_epoch, "seed": int(seed), "variants": list(variants),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "best_select": best_select, "best_stop": best_stop, "best_state": best_state,
                "best_super_epoch": best_super_epoch, "bad": bad, "history": history,
                "order_rng_state": order_rng.get_state(), "rng": native._rng_state(),
                "training_seconds_elapsed": time.perf_counter() - start,
                "peak_allocated_bytes": max(prior_peak_alloc, _pa), "peak_reserved_bytes": max(prior_peak_reserved, _pr),
            }, latest_state_path)
        if bad >= patience:
            break

    if best_state is None:
        raise RuntimeError("Joint trainer never captured a model state")
    model.load_state_dict(best_state, strict=True)
    elapsed = time.perf_counter() - start
    # Training window closes here; inference gets its own peak window (metric contract).
    train_peak_alloc = train_peak_reserved = None
    inference_reset = False
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
            train_peak_alloc = max(prior_peak_alloc, int(torch.cuda.max_memory_allocated()))
            train_peak_reserved = max(prior_peak_reserved, int(torch.cuda.max_memory_reserved()))
            torch.cuda.reset_peak_memory_stats(device); inference_reset = True
        except Exception:
            pass

    infer_start = time.perf_counter()
    model.eval()
    per_variant: dict[str, Any] = {}
    test_logits: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for variant in variants:
            logits, _ = forward(variant)
            per_variant[variant] = {
                "train": native.classification_metrics(logits, dataset.labels, dataset.train_idx),
                "val": native.classification_metrics(logits, dataset.labels, dataset.val_idx),
                "test": native.classification_metrics(logits, dataset.labels, dataset.test_idx),
            }
            test_logits[variant] = logits.index_select(0, dataset.test_idx).detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - infer_start

    pairwise = []
    for a, b in itertools.combinations(variants, 2):
        pairwise.append({
            "left": a, "right": b,
            "mean_test_logit_kendall_tau": mean_row_kendall(test_logits[a], test_logits[b]),
            "max_abs_logit_difference": float(np.max(np.abs(test_logits[a] - test_logits[b]))),
            "prediction_agreement": float(np.mean(
                np.argmax(test_logits[a], axis=1) == np.argmax(test_logits[b], axis=1))),
        })

    peak_alloc = peak_reserved = None
    if device.type == "cuda":
        try:
            peak_alloc = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
    micro = [per_variant[v]["test"]["micro_f1"] for v in variants]
    taus = [row["mean_test_logit_kendall_tau"] for row in pairwise]
    params = int(sum(p.numel() for p in model.parameters()))

    return {
        "format_version": "slotgat_freebase_nc_augmentation_v1",
        "dataset": "Freebase node classification",
        "model": "SlotGAT",
        "model_impl": getattr(args, "model_impl", "genuine"),
        "condition": "augmentation",
        "seed": seed,
        "variants": variants,
        "message_mode": "physical_per_variant",
        "selection_metric": "mean_val_micro_f1",
        "stopping_metric": "mean_val_loss",
        "best_selection_value": best_select,
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
        "training_time_seconds": elapsed,
        "inference_time_seconds": inference_seconds,
        "num_parameters": params,
        "parameter_mib": params * 4 / (1024 ** 2),
        "resumed_from_super_epoch": resumed_from,
        "memory": {
            "training_peak_allocated_bytes": train_peak_alloc,
            "training_peak_reserved_bytes": train_peak_reserved,
            "training_peak_stats_reset": bool(memory_reset),
            "inference_peak_allocated_bytes": peak_alloc,
            "inference_peak_reserved_bytes": peak_reserved,
            "inference_peak_stats_reset": bool(inference_reset),
        },
        "history": history,
        "schedules": {v: schedules[v].message_program_hash for v in variants},
    }, test_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocessed-root", type=Path,
                   default=native.INV_GNN_DATA / "preprocessed/freebase_slotgat_block")
    p.add_argument("--physical-message-root", type=Path,
                   default=native.INV_GNN_DATA / "preprocessed/freebase_slotgat_physical_messages")
    p.add_argument("--variants", nargs="+", default=["unchanged", "exact_2"])
    p.add_argument("--seeds", nargs="+", default=["1566911444,20241017,20251017"])
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results" / "freebase_nc_augmentation")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--epochs", type=int, default=200,
                   help="Budget in VARIANT-epochs; divided by the variant count.")
    p.add_argument("--patience", type=int, default=40, help="Also in variant-epochs.")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--edge-feats", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--dropout-feat", type=float, default=0.5)
    p.add_argument("--dropout-attn", type=float, default=0.2)
    p.add_argument("--slope", type=float, default=0.05)
    p.add_argument("--aggregator", default="SA", choices=["SA"])
    p.add_argument("--SAattDim", type=int, default=3)
    p.add_argument("--model-impl", dest="model_impl", default="genuine", choices=["genuine", "parity"])
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--allow-nonidentity-sparse", action="store_true")
    p.add_argument("--union-message-chunk-edges", type=int, default=250_000)
    p.add_argument("--disable-union-checkpoint", action="store_true")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--resume", action="store_true",
                   help="Continue from seed_<seed>/latest_training_state.pt if present.")
    p.add_argument("--overwrite", action="store_true", help="Redo seeds that already have result.json.")
    p.add_argument("--no-save-latest-state", dest="save_latest_state", action="store_false")
    args = p.parse_args()
    args.seeds = native.parse_seeds(args.seeds)
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
    print(f"Freebase augmentation (joint training) | device={device} | variants={args.variants}")

    plans = [FreebaseBlockContextPlan.load(args.preprocessed_root / v) for v in args.variants]

    # Shared supervision must be asserted, not assumed.
    ref = plans[0]
    schedules = {}
    for plan in plans:
        root = args.physical_message_root / plan.variant
        schedule = UnionMessageSchedule.load(root)
        if schedule.exact2_raw_link_hash != plan.raw_link_hash:
            raise SystemExit(f"{plan.variant}: schedule built from a different link.dat than its plan")
        schedule.validate_against_plan(plan)
        schedules[plan.variant] = schedule
        print(f"  [{plan.variant}] {schedule.message_count:,} messages, "
              f"hash={schedule.message_program_hash[:16]}")
    if len({s.message_program_hash for s in schedules.values()}) != len(schedules):
        raise SystemExit("Physical schedules are not all distinct; the arm would be degenerate")

    reference_data_dir = Path(ref.source_dir)
    dataset = native.load_dataset(reference_data_dir, ref.layout, device)
    print(f"Supervision: labels={dataset.label_hash[:16]} splits={dataset.split_hash[:16]}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        run_dir = args.results_dir / f"seed_{seed}"
        if (run_dir / "result.json").exists() and not args.overwrite:
            print(f"Skipping completed seed: {run_dir}")
            summaries.append(json.load(open(run_dir / "result.json")))
            continue
        result, test_logits = train_joint(args, plans, schedules, dataset, seed, device)
        run_dir.mkdir(parents=True, exist_ok=True)
        for variant, logits in test_logits.items():
            np.savez_compressed(run_dir / f"{variant}_test_logits.npz", logits=logits)
        result["supervision"] = {"labels_hash": dataset.label_hash, "splits_hash": dataset.split_hash}
        write_json(run_dir / "result.json", result)
        summaries.append(result)
        print(f"[augmentation] seed={seed} micro-F1 {result['test_micro_f1_mean']:.4f} "
              f"spread {result['test_micro_f1_spread']:.4f} "
              f"tau {result['mean_pairwise_kendall_tau']:.4f} "
              f"({result['variant_epochs_run']} variant-epochs)", flush=True)

    aggregate = {
        "condition": "augmentation",
        "seeds": args.seeds,
        "test_micro_f1_mean": float(np.mean([s["test_micro_f1_mean"] for s in summaries])),
        "test_micro_f1_std": float(np.std([s["test_micro_f1_mean"] for s in summaries])),
        "mean_pairwise_kendall_tau": float(np.mean([s["mean_pairwise_kendall_tau"] for s in summaries])),
        "per_seed": [{k: s[k] for k in
                      ("seed", "test_micro_f1_mean", "test_micro_f1_spread",
                       "mean_pairwise_kendall_tau", "variant_epochs_run")} for s in summaries],
    }
    write_json(args.results_dir / "aggregate_summary.json", aggregate)
    print(f"\nAUGMENTATION micro-F1 {aggregate['test_micro_f1_mean']:.4f} "
          f"± {aggregate['test_micro_f1_std']:.4f}  tau {aggregate['mean_pairwise_kendall_tau']:.4f}")


if __name__ == "__main__":
    main()
