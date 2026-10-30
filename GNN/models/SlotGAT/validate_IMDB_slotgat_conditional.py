"""Validate SlotGAT conditional structural and model-level invariance.

This validator distinguishes four different questions:

1. Are the raw input graphs genuinely different?
2. Are the raw conditional witness plans also genuinely different?
3. Do those different contexts induce the same runtime propagation semantics?
4. With one canonical initialization, are evaluation logits equal/close?
5. With an independently reset dropout RNG, are first-step gradients equal/close?

The genuine model uses deterministic PyTorch degree-bucket reductions rather than
a shared DGL graph. The audit records both bitwise and tolerance-based equality
and is always saved before a failed assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from run_IMDB_nc_conditional import (
    DEFAULT_VARIANTS,
    build_conditional_model,
    safe_cuda_synchronize,
    hash_array,
    hash_gradients,
    hash_state_dict,
    load_datasets,
    make_data_recorder,
    set_deterministic,
    tensorize_features,
    validate_shared_inputs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FORWARD_SEED_OFFSET = 1_000_003


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def max_state_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise AssertionError(
            "Model state keys differ; "
            f"missing_from_left={missing_left}, missing_from_right={missing_right}"
        )
    maximum = 0.0
    for key in left:
        a = left[key].detach().cpu().to(torch.float64)
        b = right[key].detach().cpu().to(torch.float64)
        if a.shape != b.shape:
            raise AssertionError(
                f"State tensor shape differs for {key}: {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        if a.numel():
            maximum = max(maximum, float(torch.max(torch.abs(a - b)).item()))
    return maximum


def state_scale(state: Mapping[str, torch.Tensor]) -> float:
    maximum = 0.0
    for tensor in state.values():
        value = tensor.detach().cpu().to(torch.float64)
        if value.numel():
            maximum = max(maximum, float(torch.max(torch.abs(value)).item()))
    return maximum


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise AssertionError(
            f"Tensor shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if left.numel() == 0:
        return 0.0
    return float(
        torch.max(
            torch.abs(
                left.detach().to(torch.float64)
                - right.detach().to(torch.float64)
            )
        ).item()
    )


def tensor_scale(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(tensor.detach().to(torch.float64))).item())


def within_tolerance(max_abs: float, reference_scale: float, atol: float, rtol: float) -> bool:
    return bool(max_abs <= float(atol) + float(rtol) * float(reference_scale))


def hash_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tuple(tensor.shape), dtype="<i8").tobytes())
    if tensor.is_sparse:
        value = tensor.detach().coalesce().cpu()
        digest.update(value.indices().contiguous().numpy().tobytes(order="C"))
        digest.update(value.values().contiguous().numpy().tobytes(order="C"))
    else:
        value = tensor.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def hash_tensor_list(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        part = hash_tensor(value).encode("ascii")
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        safe_cuda_synchronize()


def gradient_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def run_model_check(
    args: argparse.Namespace,
    bundles: dict[int, Any],
    device: torch.device,
) -> dict[str, Any]:
    eval_records: dict[str, Any] = {}
    train_records: dict[str, Any] = {}

    reference_initial_state: dict[str, torch.Tensor] | None = None
    reference_initial_hash: str | None = None
    reference_feature_hash: str | None = None
    reference_eval_logits: torch.Tensor | None = None
    reference_grad_state: dict[str, torch.Tensor] | None = None

    for variant_index, variant in enumerate(args.variants):
        bundle = bundles[variant]

        # Tensorize each variant independently, then verify the actual tensors
        # supplied to the model, not only the source scipy/numpy objects.
        features, in_dims = tensorize_features(
            bundle.features_raw, args.feats_type, device
        )
        tensor_feature_hash = hash_tensor_list(features)
        if reference_feature_hash is None:
            reference_feature_hash = tensor_feature_hash

        labels = torch.as_tensor(
            bundle.labels_np, dtype=torch.int64, device=device
        )
        train_idx = torch.as_tensor(
            bundle.split_indices["train"], dtype=torch.int64, device=device
        )

        # Build a model normally to audit whether same-seed construction is
        # already identical.  Then load one canonical state into every model so
        # the structural forward check cannot be confounded by initialization.
        set_deterministic(args.seed, args.deterministic)
        model, forward_fn, graph_meta = build_conditional_model(
            args,
            features_list=features,
            in_dims=in_dims,
            program=bundle.program,
            device=device,
            data_recorder=make_data_recorder(),
        )
        constructor_initial_hash = hash_state_dict(model.state_dict())

        if reference_initial_state is None:
            reference_initial_state = clone_state_dict(model.state_dict())
            reference_initial_hash = constructor_initial_hash
        else:
            model.load_state_dict(reference_initial_state, strict=True)

        loaded_initial_hash = hash_state_dict(model.state_dict())
        loaded_initial_max_abs = max_state_difference(
            reference_initial_state, model.state_dict()
        )

        model.eval()
        with torch.no_grad():
            logits_eval_1, _ = forward_fn(features)
            synchronize(device)
            # Repeating the exact same forward on the same model diagnoses
            # nondeterministic kernels independently of cross-variant effects.
            logits_eval_2, _ = forward_fn(features)
            synchronize(device)

        logits_eval_1_cpu = logits_eval_1.detach().cpu()
        logits_eval_2_cpu = logits_eval_2.detach().cpu()
        repeat_max_abs = tensor_max_abs(logits_eval_1_cpu, logits_eval_2_cpu)
        repeat_close = within_tolerance(
            repeat_max_abs,
            tensor_scale(logits_eval_1_cpu),
            args.atol,
            args.rtol,
        )

        if reference_eval_logits is None:
            reference_eval_logits = logits_eval_1_cpu.clone()
            eval_max_abs = 0.0
        else:
            eval_max_abs = tensor_max_abs(
                reference_eval_logits, logits_eval_1_cpu
            )
        eval_close = within_tolerance(
            eval_max_abs,
            tensor_scale(reference_eval_logits),
            args.atol,
            args.rtol,
        )

        eval_records[bundle.variant_name] = {
            "raw_feature_hash": bundle.feature_hash,
            "tensor_feature_hash": tensor_feature_hash,
            "tensor_features_match_first_variant": (
                tensor_feature_hash == reference_feature_hash
            ),
            "constructor_initial_parameter_hash": constructor_initial_hash,
            "constructor_initial_matches_first_variant": (
                constructor_initial_hash == reference_initial_hash
            ),
            "loaded_initial_parameter_hash": loaded_initial_hash,
            "loaded_initial_max_abs_difference": loaded_initial_max_abs,
            "eval_logits_hash": hash_array(logits_eval_1_cpu.numpy()),
            "eval_logits_bitwise_match_first_variant": (
                reference_eval_logits is not None
                and torch.equal(reference_eval_logits, logits_eval_1_cpu)
            ),
            "eval_logits_close_to_first_variant": eval_close,
            "max_abs_difference_from_first_variant": eval_max_abs,
            "same_model_repeat_logits_hash": hash_array(
                logits_eval_2_cpu.numpy()
            ),
            "same_model_repeat_bitwise_equal": torch.equal(
                logits_eval_1_cpu, logits_eval_2_cpu
            ),
            "same_model_repeat_close": repeat_close,
            "same_model_repeat_max_abs_difference": repeat_max_abs,
            "raw_graph_hash": bundle.program.raw_graph_hash,
            "context_plan_hash": bundle.program.context_plan_hash,
            "semantic_propagation_hash": bundle.program.semantic_propagation_hash,
            "message_program_hash": bundle.program.semantic_propagation_hash,
            "model_facing_input": graph_meta["model_facing_input"],
            "materialized_shared_dgl_graph": graph_meta["materialized_shared_dgl_graph"],
            "raw_context_edge_count": graph_meta["raw_context_edge_count"],
            "conditional_witness_count": graph_meta["conditional_witness_count"],
            "semantic_message_count": graph_meta["semantic_message_count"],
            "message_graph_edges": graph_meta["message_graph_edges"],
            "message_edge_types": graph_meta["message_edge_types"],
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Rebuild and load the same canonical parameters.  Crucially, reset the
        # RNG *after* construction/loading so dropout is independent of any
        # constructor-side random-number consumption.
        set_deterministic(args.seed, args.deterministic)
        model, forward_fn, _ = build_conditional_model(
            args,
            features_list=features,
            in_dims=in_dims,
            program=bundle.program,
            device=device,
            data_recorder=make_data_recorder(),
        )
        model.load_state_dict(reference_initial_state, strict=True)
        set_deterministic(
            args.seed + TRAIN_FORWARD_SEED_OFFSET,
            args.deterministic,
        )

        model.train()
        logits_train, _ = forward_fn(features)
        synchronize(device)
        logp = F.log_softmax(logits_train, dim=1)
        loss = F.nll_loss(logp[train_idx], labels[train_idx])
        model.zero_grad(set_to_none=True)
        loss.backward()
        synchronize(device)

        current_grad_state = gradient_state(model)
        if reference_grad_state is None:
            reference_grad_state = current_grad_state
            grad_max_abs = 0.0
        else:
            grad_max_abs = max_state_difference(
                reference_grad_state, current_grad_state
            )
        grad_close = within_tolerance(
            grad_max_abs,
            state_scale(reference_grad_state),
            args.gradient_atol,
            args.gradient_rtol,
        )

        train_records[bundle.variant_name] = {
            "training_logits_hash": hash_array(
                logits_train.detach().cpu().numpy()
            ),
            "training_loss": float(loss.item()),
            "gradient_hash": hash_gradients(model),
            "gradient_bitwise_match_first_variant": (
                variant_index == 0
                or all(
                    torch.equal(reference_grad_state[name], current_grad_state[name])
                    for name in reference_grad_state
                )
            ),
            "gradient_close_to_first_variant": grad_close,
            "max_abs_gradient_difference_from_first_variant": grad_max_abs,
            "dropout_rng_seed": args.seed + TRAIN_FORWARD_SEED_OFFSET,
        }

        del model, features, labels, train_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()

    constructor_init_hashes = {
        record["constructor_initial_parameter_hash"]
        for record in eval_records.values()
    }
    loaded_init_hashes = {
        record["loaded_initial_parameter_hash"]
        for record in eval_records.values()
    }
    eval_hashes = {
        record["eval_logits_hash"] for record in eval_records.values()
    }
    grad_hashes = {
        record["gradient_hash"] for record in train_records.values()
    }

    constructor_initial_parameters_identical = len(constructor_init_hashes) == 1
    loaded_initial_parameters_identical = len(loaded_init_hashes) == 1
    eval_logits_bitwise_identical = len(eval_hashes) == 1
    first_gradients_bitwise_identical = len(grad_hashes) == 1
    eval_logits_close = all(
        record["eval_logits_close_to_first_variant"]
        for record in eval_records.values()
    )
    first_gradients_close = all(
        record["gradient_close_to_first_variant"]
        for record in train_records.values()
    )
    repeated_forwards_close = all(
        record["same_model_repeat_close"]
        for record in eval_records.values()
    )
    tensor_features_identical = len(
        {record["tensor_feature_hash"] for record in eval_records.values()}
    ) == 1

    if args.require_bitwise:
        passed = bool(
            tensor_features_identical
            and loaded_initial_parameters_identical
            and eval_logits_bitwise_identical
            and first_gradients_bitwise_identical
        )
    else:
        passed = bool(
            tensor_features_identical
            and loaded_initial_parameters_identical
            and eval_logits_close
            and first_gradients_close
            and repeated_forwards_close
        )

    return {
        "seed": args.seed,
        "training_forward_seed": args.seed + TRAIN_FORWARD_SEED_OFFSET,
        "device": str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "require_bitwise": args.require_bitwise,
        "atol": args.atol,
        "rtol": args.rtol,
        "gradient_atol": args.gradient_atol,
        "gradient_rtol": args.gradient_rtol,
        "eval": eval_records,
        "first_training_step": train_records,
        "raw_graphs_are_distinct": len({bundle.program.raw_graph_hash for bundle in bundles.values()}) == len(bundles),
        "context_plans_are_distinct": len({bundle.program.context_plan_hash for bundle in bundles.values()}) == len(bundles),
        "semantic_propagation_is_identical": len({bundle.program.semantic_propagation_hash for bundle in bundles.values()}) == 1,
        "materialized_shared_dgl_graph": False,
        "independent_model_instances": True,
        "tensor_features_identical": tensor_features_identical,
        "constructor_initial_parameters_identical": constructor_initial_parameters_identical,
        "loaded_initial_parameters_identical": loaded_initial_parameters_identical,
        "eval_logits_bitwise_identical": eval_logits_bitwise_identical,
        "eval_logits_close": eval_logits_close,
        "first_gradients_bitwise_identical": first_gradients_bitwise_identical,
        "first_gradients_close": first_gradients_close,
        "repeated_forwards_close": repeated_forwards_close,
        "maximum_eval_logit_difference": max(
            record["max_abs_difference_from_first_variant"]
            for record in eval_records.values()
        ),
        "maximum_same_model_repeat_difference": max(
            record["same_model_repeat_max_abs_difference"]
            for record in eval_records.values()
        ),
        "maximum_gradient_difference": max(
            record["max_abs_gradient_difference_from_first_variant"]
            for record in train_records.values()
        ),
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate invariant SlotGAT structural bias"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=int,
        default=list(DEFAULT_VARIANTS),
        choices=[1, 2, 3, 4],
    )
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument(
        "--data-suffix", dest="data_suffix", default="",
        help="Appended to each IMDB_var<N> directory name, matching "
             "run_IMDB_nc_conditional.py --data-suffix (e.g. '_year').")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            SCRIPT_DIR
            / "results"
            / "imdb_nc_conditional"
            / "validation_audit.json"
        ),
    )
    parser.add_argument("--model-check", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-feature-mismatch", action="store_true")
    parser.add_argument("--non-strict-context", action="store_true")
    parser.add_argument(
        "--feature-mode",
        default="pooled_bow",
        choices=["pooled_bow", "identity"],
        help="Must match the value passed to run_IMDB_nc_conditional.py.",
    )
    parser.add_argument(
        "--require-bitwise",
        action="store_true",
        help=(
            "Require identical hashes rather than numerical closeness.  This is "
            "appropriate for the CPU reference and for the genuine deterministic "
            "degree-bucket CUDA implementation."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--gradient-atol", type=float, default=1e-6)
    parser.add_argument("--gradient-rtol", type=float, default=1e-6)

    # Must match the experiment runner for --model-check.
    parser.add_argument("--feats-type", type=int, default=0, choices=[0, 1])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--aggregator", default="SA")
    parser.add_argument("--SAattDim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_suffix:
        import run_IMDB_nc_conditional as _rc
        for _k in list(_rc.VARIANT_NAMES):
            _rc.VARIANT_NAMES[_k] = _rc.VARIANT_NAMES[_k] + args.data_suffix
        print(f"[data-suffix] validating {sorted(_rc.VARIANT_NAMES.values())}", flush=True)
    os.chdir(SCRIPT_DIR)
    bundles = load_datasets(
        args.variants,
        strict=not args.non_strict_context,
        feature_mode=getattr(args, "feature_mode", "identity"),
    )
    preflight = validate_shared_inputs(
        bundles, allow_feature_mismatch=args.allow_feature_mismatch
    )
    payload: dict[str, Any] = {"structural_preflight": preflight}
    print("Structural validation passed.")
    for variant in args.variants:
        bundle = bundles[variant]
        print(
            f"  {bundle.variant_name}: "
            f"raw={bundle.program.raw_graph_hash[:12]} "
            f"context={bundle.program.context_plan_hash[:12]} "
            f"semantic={bundle.program.semantic_propagation_hash[:12]} "
            f"messages={len(bundle.program.emit_semantic_messages())}"
        )

    failed = False
    if args.model_check:
        device = torch.device(
            f"cuda:{args.gpu}"
            if not args.cpu and torch.cuda.is_available()
            else "cpu"
        )
        payload["model_check"] = run_model_check(args, bundles, device)
        check = payload["model_check"]
        print("Model audit:")
        print(
            "  raw graph hashes distinct:                  "
            f"{check['raw_graphs_are_distinct']}"
        )
        print(
            "  conditional plans distinct:                "
            f"{check['context_plans_are_distinct']}"
        )
        print(
            "  shared DGL graph materialized:              "
            f"{check['materialized_shared_dgl_graph']}"
        )
        print(
            "  constructor initial parameters identical: "
            f"{check['constructor_initial_parameters_identical']}"
        )
        print(
            "  loaded canonical parameters identical:     "
            f"{check['loaded_initial_parameters_identical']}"
        )
        print(
            "  tensorized features identical:             "
            f"{check['tensor_features_identical']}"
        )
        print(
            "  eval logits bitwise identical:             "
            f"{check['eval_logits_bitwise_identical']}"
        )
        print(
            "  eval logits close:                         "
            f"{check['eval_logits_close']} "
            f"(max abs={check['maximum_eval_logit_difference']:.3e})"
        )
        print(
            "  repeated same-model forwards close:        "
            f"{check['repeated_forwards_close']} "
            f"(max abs={check['maximum_same_model_repeat_difference']:.3e})"
        )
        print(
            "  first gradients bitwise identical:         "
            f"{check['first_gradients_bitwise_identical']}"
        )
        print(
            "  first gradients close:                     "
            f"{check['first_gradients_close']} "
            f"(max abs={check['maximum_gradient_difference']:.3e})"
        )
        failed = not bool(check["passed"])

    # Always write the audit before raising so a failed check is diagnosable.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"Audit: {args.output.resolve()}")

    if failed:
        raise AssertionError(
            "Structural equality passed, but the corrected model audit failed. "
            "Inspect validation_audit.json.  Run once with --cpu "
            "--require-bitwise to distinguish CUDA/DGL reduction noise from an "
            "input or parameter mismatch."
        )

    if args.model_check:
        mode = "bitwise" if args.require_bitwise else "tolerance-based"
        print(f"Model validation passed ({mode}).")


if __name__ == "__main__":
    main()
