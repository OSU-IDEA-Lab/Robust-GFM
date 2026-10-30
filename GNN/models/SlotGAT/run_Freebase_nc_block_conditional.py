#!/usr/bin/env python3
"""Train genuine recoverability-conditional SlotGAT on Freebase node classification.

The raw variants are independently preprocessed into distinct physical context
plans.  Every model run is independent; no checkpoint, logit, or optimizer state
is shared across variants.  The recoverability-conditional layer consumes each plan and
activates the same factorized semantic path through either base-composition or
physical-shortcut context.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
INV_GNN_DATA = Path(os.environ.get("INV_GNN_DATA", str(SCRIPT_DIR / "data")))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "model") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "model"))

from freebase_block_conditional import FreebaseBlockContextPlan, sha256_array, write_json
from genuine_block_conditional_slotgat import GenuineBlockConditionalSlotGAT
from freebase_union_messages import UnionMessageSchedule  # noqa: E402
from genuine_union_message_slotgat import (  # noqa: E402
    ExactUnionMessageConditionalSlotGAT,
)

NUM_CLASSES = 8
BOOK_TYPE = 0
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


@dataclass
class DatasetBundle:
    features: list[torch.Tensor]
    in_dims: list[int]
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    feature_hash: str
    label_hash: str
    split_hash: str


def parse_seeds(values: Sequence[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                result.append(int(token))
    if not result:
        raise ValueError("At least one seed is required")
    return result


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def scipy_to_torch(matrix: Any) -> torch.Tensor:
    if isinstance(matrix, np.ndarray):
        return torch.as_tensor(matrix, dtype=torch.float32)
    if not sp.issparse(matrix):
        raise TypeError(f"Unsupported feature type: {type(matrix)!r}")
    coo = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.int64
    )
    values = torch.as_tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def tensor_hash(tensor: torch.Tensor) -> str:
    if tensor.layout != torch.strided:
        tensor = tensor.to_sparse_coo().coalesce()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.indices().detach().cpu().numpy().tobytes())
        digest.update(tensor.values().detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_array(array)


def feature_list_hash(features: Sequence[torch.Tensor]) -> str:
    return hashlib.sha256(
        "|".join(tensor_hash(feature) for feature in features).encode()
    ).hexdigest()


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gradient_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        if parameter.grad is None:
            digest.update(b"NONE")
        else:
            gradient = parameter.grad.detach().cpu().contiguous()
            digest.update(gradient.numpy().tobytes())
    return digest.hexdigest()


def _read_node_types(node_path: Path) -> np.ndarray:
    """Return node types indexed by global node ID.

    HGB-style ``node.dat`` files do not guarantee that rows are written in
    numeric node-ID order.  The Freebase preprocessor already verifies that the
    ID set is exactly ``0..N-1`` and that IDs are contiguous by type, so the
    dataset loader must reconstruct the dense array by the explicit ID in the
    first column rather than treating file order as node order.
    """
    id_to_type: dict[int, int] = {}
    maximum_id = -1
    with Path(node_path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed node.dat line {line_number} in {node_path}: {raw!r}"
                )
            node_id = int(parts[0])
            node_type = int(parts[2])
            if node_id < 0:
                raise ValueError(
                    f"Negative node ID {node_id} at {node_path}:{line_number}"
                )
            if node_id in id_to_type:
                raise ValueError(
                    f"Duplicate node ID {node_id} at {node_path}:{line_number}"
                )
            id_to_type[node_id] = node_type
            maximum_id = max(maximum_id, node_id)

    if not id_to_type:
        raise ValueError(f"No nodes were found in {node_path}")

    expected_count = maximum_id + 1
    if len(id_to_type) != expected_count:
        missing = sorted(set(range(expected_count)) - set(id_to_type))
        preview = missing[:10]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            "Freebase node IDs must form the dense set 0..N-1; "
            f"missing IDs in {node_path}: {preview}{suffix}"
        )

    node_types = np.empty(expected_count, dtype=np.int64)
    for node_id, node_type in id_to_type.items():
        node_types[node_id] = node_type
    return node_types


def _read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    node_ids: list[int] = []
    labels: list[int] = []
    path = Path(path)
    if not path.exists():
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"Malformed label line {line_number} in {path}: {raw!r}"
                )
            node_ids.append(int(parts[0]))
            labels.append(int(parts[3]))
    return (
        np.asarray(node_ids, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def _sparse_identity(size: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(int(size), dtype=torch.int64)
    matrix = torch.sparse_coo_tensor(
        torch.stack((index, index), dim=0),
        torch.ones(int(size), dtype=torch.float32),
        (int(size), int(size)),
    ).coalesce()
    return matrix.to(device)


def load_dataset(
    data_dir: Path,
    layout: Any,
    device: torch.device,
) -> DatasetBundle:
    """Load Freebase features, labels, and splits from an explicit directory.

    The previous implementation called ``load_data(prefix="Freebase_unchanged")``
    and therefore ignored the raw directories recorded by preprocessing.  This
    loader consumes the unchanged plan's real ``source_dir`` directly and never
    reads ``link.dat``; the conditional context plan already stores every base
    message needed by the model.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    required = [data_dir / "node.dat", data_dir / "label.dat"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Freebase feature/label directory is incomplete: " + ", ".join(missing)
        )

    node_types = _read_node_types(data_dir / "node.dat")
    observed_counts = tuple(
        int(np.count_nonzero(node_types == node_type))
        for node_type in range(len(layout.counts))
    )
    if observed_counts != tuple(int(value) for value in layout.counts):
        raise ValueError(
            "node.dat type counts do not match the preprocessed context plan: "
            f"observed={observed_counts}, expected={tuple(layout.counts)}"
        )
    expected_types = np.concatenate(
        [
            np.full(int(count), node_type, dtype=np.int64)
            for node_type, count in enumerate(layout.counts)
        ]
    )
    if not np.array_equal(node_types, expected_types):
        raise ValueError(
            "Freebase nodes are not stored in the type-contiguous order required "
            "by SlotGAT and recorded in the context plan"
        )

    # The legacy Freebase files have no node attributes.  Match the existing
    # SlotGAT/MAGNN preprocessing by using one sparse identity matrix per type.
    features = [
        _sparse_identity(int(count), device)
        for count in layout.counts
    ]
    in_dims = [int(count) for count in layout.counts]

    train_nodes, train_values = _read_labels(data_dir / "label.dat")
    test_nodes, test_values = _read_labels(data_dir / "label.dat.test")
    labels_np = np.full(int(layout.total), -1, dtype=np.int64)
    for nodes, values, split_name in (
        (train_nodes, train_values, "label.dat"),
        (test_nodes, test_values, "label.dat.test"),
    ):
        if nodes.size and (
            int(nodes.min()) < 0 or int(nodes.max()) >= int(layout.total)
        ):
            raise ValueError(f"{split_name} contains a node ID outside node.dat")
        labels_np[nodes] = values

    target_start = int(layout.offsets[BOOK_TYPE])
    target_stop = target_start + int(layout.counts[BOOK_TYPE])
    for name, nodes in (("train", train_nodes), ("test", test_nodes)):
        if nodes.size and not bool(
            np.all((nodes >= target_start) & (nodes < target_stop))
        ):
            raise ValueError(
                f"{name} labels contain non-BOOK nodes; expected IDs in "
                f"[{target_start}, {target_stop})"
            )

    split_seed = 1566911444
    if test_nodes.size:
        train_idx = np.asarray(sorted(set(train_nodes.tolist())), dtype=np.int64)
        test_idx = np.asarray(sorted(set(test_nodes.tolist())), dtype=np.int64)
        if train_idx.size >= 5:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=max(1, int(round(0.1 * train_idx.size))),
                random_state=split_seed,
            )
            train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
            val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
        else:
            val_idx = np.asarray([], dtype=np.int64)
    else:
        labeled = np.asarray(
            sorted(set(train_nodes.tolist())), dtype=np.int64
        )
        if labeled.size >= 5:
            stratify = labels_np[labeled]
            stratify_arg = stratify if np.unique(stratify).size > 1 else None
            train_idx, temporary = train_test_split(
                labeled,
                test_size=0.4,
                random_state=split_seed,
                stratify=stratify_arg,
            )
            temporary = np.asarray(temporary, dtype=np.int64)
            temp_stratify = labels_np[temporary]
            temp_stratify_arg = (
                temp_stratify if np.unique(temp_stratify).size > 1 else None
            )
            val_idx, test_idx = train_test_split(
                temporary,
                test_size=0.5,
                random_state=split_seed,
                stratify=temp_stratify_arg,
            )
            train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
            val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
            test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))
        else:
            train_idx = labeled
            val_idx = np.asarray([], dtype=np.int64)
            test_idx = np.asarray([], dtype=np.int64)

    all_split_indices = np.concatenate((train_idx, val_idx, test_idx))
    if all_split_indices.size == 0:
        raise ValueError(f"No labeled Freebase nodes were found in {data_dir}")
    if np.any(labels_np[all_split_indices] < 0):
        raise ValueError("A train/validation/test index has no class label")

    labels = torch.as_tensor(labels_np, dtype=torch.int64, device=device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.int64, device=device)
    val_tensor = torch.as_tensor(val_idx, dtype=torch.int64, device=device)
    test_tensor = torch.as_tensor(test_idx, dtype=torch.int64, device=device)
    split_hash = hashlib.sha256(
        b"".join(
            index.astype("<i8", copy=False).tobytes()
            for index in (train_idx, val_idx, test_idx)
        )
    ).hexdigest()
    return DatasetBundle(
        features=features,
        in_dims=in_dims,
        labels=labels,
        train_idx=train_tensor,
        val_idx=val_tensor,
        test_idx=test_tensor,
        feature_hash=feature_list_hash(features),
        label_hash=tensor_hash(labels),
        split_hash=split_hash,
    )


def ranking_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0}
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    matches = order == labels[:, None]
    ranks = matches.to(torch.int64).argmax(dim=1) + 1
    return {
        "hit_at_1": float((ranks <= 1).float().mean().item()),
        "hit_at_3": float((ranks <= min(3, logits.shape[1])).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, float]:
    split_logits = logits.index_select(0, indices)
    split_labels = labels.index_select(0, indices)
    predictions = split_logits.argmax(dim=1)
    true_np = split_labels.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    result = {
        "accuracy": float(accuracy_score(true_np, pred_np)),
        "precision_macro": float(
            precision_score(true_np, pred_np, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(true_np, pred_np, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(true_np, pred_np, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(true_np, pred_np, average="macro", zero_division=0)
        ),
    }
    result.update(ranking_metrics(split_logits, split_labels))
    return result


def safe_cuda_reset() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, None
    try:
        torch.cuda.reset_peak_memory_stats()
        return True, None
    except (RuntimeError, TypeError) as exc:
        return False, str(exc)


def safe_cuda_peaks() -> tuple[int | None, int | None, str | None]:
    if not torch.cuda.is_available():
        return None, None, None
    try:
        return (
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
            None,
        )
    except (RuntimeError, TypeError) as exc:
        return None, None, str(exc)


def safe_cuda_sync() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except (RuntimeError, TypeError):
            pass


def build_model(
    args: argparse.Namespace,
    plan: FreebaseBlockContextPlan,
    dataset: DatasetBundle,
):
    """Construct the propagation operator selected by ``--message-mode``.

    ``exact_union`` (default) is the CORRECTED operator. The original factorized
    implementation emitted one path aggregate per reachable destination, which
    changes source multiplicity, source-specific attention scores and the
    destination softmax denominator -- so it is NOT message-equivalent to
    SlotGAT on the physical union graph, and a union-graph baseline is exactly
    what this cell is compared against. The direct destination-CSR schedule
    reproduces the native graph builder's ordered-pair policy instead.

    ``factorized`` is retained for the ablation.

    Invariance is unaffected either way: it comes from the compiled per-variant
    context plan, and the schedule is validated against that plan on every
    forward (``schedule.validate_against_plan``).
    """
    if args.message_mode in ("physical", "exact_union"):
        if args.message_mode == "physical":
            # MATCHED CONTROL ("originals"): the same model class, geometry and
            # parameter count as the invariant arm, propagating on THIS variant's
            # own edges. The message set is then the only manipulated variable,
            # which is what the effectiveness comparison needs.
            schedule = args.physical_message_schedules[plan.variant]
            if schedule.exact2_raw_link_hash != plan.raw_link_hash:
                raise ValueError(
                    f"Physical schedule for {plan.variant!r} was built from a different "
                    f"link.dat than its context plan describes "
                    f"({schedule.exact2_raw_link_hash[:16]} vs {plan.raw_link_hash[:16]})"
                )
        else:
            schedule = args.union_message_schedule
        schedule.validate_against_plan(plan)
        return ExactUnionMessageConditionalSlotGAT(
            layout=plan.layout,
            in_dims=dataset.in_dims,
            schedule=schedule,
            num_classes=NUM_CLASSES,
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
        )
    if args.message_mode != "factorized":
        raise ValueError(f"Unknown message mode: {args.message_mode!r}")
    return GenuineBlockConditionalSlotGAT(
        layout=plan.layout,
        in_dims=dataset.in_dims,
        num_channels=plan.num_channels,
        num_classes=NUM_CLASSES,
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
    )


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one_run(
    *,
    args: argparse.Namespace,
    seed: int,
    variant: str,
    plan: FreebaseBlockContextPlan,
    dataset: DatasetBundle,
    result_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    set_deterministic(seed, args.deterministic)
    model = build_model(args, plan, dataset).to(device)
    initial_parameter_hash = state_dict_hash(model.state_dict())
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    training_seed = int(seed) + 1_000_003
    set_deterministic(training_seed, args.deterministic)

    memory_reset, memory_reset_error = safe_cuda_reset()
    safe_cuda_sync()
    start = time.perf_counter()
    best_val_loss = float("inf")
    best_selection = float("inf")
    best_stop_value = float("inf")
    best_val_micro_f1 = float("nan")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    first_train_logit_hash = None
    first_gradient_hash = None
    history: list[dict[str, float | int]] = []

    # RESUME. Each run is hours long, so every epoch writes an atomic
    # ``<checkpoint>_latest_state.pt`` carrying model + optimizer + selection state +
    # history + every RNG stream. With --resume a killed run continues from the last
    # completed epoch; RNG restoration keeps the continued trajectory identical to an
    # uninterrupted one (deterministic kernels), so cross-variant invariance survives a
    # requeue even if the two variants were interrupted at different epochs.
    latest_state_path = checkpoint_path.with_name(checkpoint_path.stem + "_latest_state.pt")
    start_epoch = 1
    prior_elapsed = 0.0
    prior_peak_allocated = 0
    prior_peak_reserved = 0
    resumed_from = None
    if getattr(args, "resume", False) and latest_state_path.exists():
        snap = torch.load(latest_state_path, map_location="cpu", weights_only=False)
        if snap.get("context_plan_hash") != plan.context_plan_hash or snap.get("seed") != int(seed):
            raise RuntimeError(f"{latest_state_path} belongs to a different plan/seed; refusing to resume")
        model.load_state_dict(snap["model_state"], strict=True)
        model.to(device)
        optimizer.load_state_dict(snap["optimizer_state"])
        best_val_loss = snap["best_val_loss"]; best_selection = snap["best_selection"]
        best_stop_value = snap["best_stop_value"]; best_val_micro_f1 = snap["best_val_micro_f1"]
        best_epoch = snap["best_epoch"]; best_state = snap["best_state"]
        epochs_without_improvement = snap["epochs_without_improvement"]
        epochs_trained = snap["epoch"]; history = list(snap["history"])
        first_train_logit_hash = snap["first_train_logit_hash"]; first_gradient_hash = snap["first_gradient_hash"]
        prior_elapsed = float(snap["training_seconds_elapsed"])
        prior_peak_allocated = int(snap.get("peak_allocated_bytes") or 0)
        prior_peak_reserved = int(snap.get("peak_reserved_bytes") or 0)
        _restore_rng_state(snap["rng"])
        start_epoch = int(snap["epoch"]) + 1
        resumed_from = int(snap["epoch"])
        start = time.perf_counter() - prior_elapsed
        print(f"    [resume] {variant} seed={seed}: continuing after epoch {resumed_from} "
              f"(best={best_epoch}, noimp={epochs_without_improvement}, {prior_elapsed:.0f}s so far)", flush=True)
        if epochs_without_improvement >= args.patience or resumed_from >= args.epochs:
            start_epoch = args.epochs + 1   # training had already finished; go straight to evaluation

    for epoch in range(start_epoch, args.epochs + 1):
        epochs_trained = epoch
        model.train()
        logits, _ = model(dataset.features, plan)
        train_logp = F.log_softmax(logits, dim=1)
        train_loss = F.nll_loss(
            train_logp.index_select(0, dataset.train_idx),
            dataset.labels.index_select(0, dataset.train_idx),
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        if epoch == 1:
            first_train_logit_hash = tensor_hash(logits)
            first_gradient_hash = gradient_hash(model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(dataset.features, plan)
            validation_logp = F.log_softmax(validation_logits, dim=1)
            validation_loss = F.nll_loss(
                validation_logp.index_select(0, dataset.val_idx),
                dataset.labels.index_select(0, dataset.val_idx),
            )
        val_value = float(validation_loss.item())
        validation_micro_f1 = float(
            classification_metrics(validation_logits, dataset.labels, dataset.val_idx)["micro_f1"]
        )
        # CHECKPOINT SELECTION and EARLY STOPPING are deliberately decoupled.
        #
        # Selecting the checkpoint on validation loss tends to pick the last
        # epoch, and on RGCN Freebase selecting on micro-F1 instead moves
        # micro-F1 .6612 -> .6885.
        #
        # But driving PATIENCE off micro-F1 is just as wrong here: Freebase's majority class is 48.84%
        # of the test set, micro-F1 sits at exactly that value while the model
        # still predicts the majority class, so it never strictly improves, the
        # patience counter runs down, and the run stops at epoch 41 with
        # best_epoch=1 -- while validation loss was still falling healthily
        # (2.079 -> 1.479). Micro-F1 is a step function with a long plateau on an
        # imbalanced dataset; validation loss is smooth.
        #
        # So: stop on the smooth metric, select on the evaluation metric.
        selection_value = (
            -validation_micro_f1 if args.select_on == "val_micro_f1" else val_value
        )
        stop_value = val_value if args.stop_on == "val_loss" else -validation_micro_f1
        if stop_value < best_stop_value - args.min_delta:
            best_stop_value = stop_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        improved = selection_value < best_selection - args.min_delta
        if improved:
            best_selection = selection_value
            best_val_micro_f1 = validation_micro_f1
            best_val_loss = val_value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "seed": seed,
                    "variant": variant,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "context_plan_hash": plan.context_plan_hash,
                    "semantic_program_hash": plan.semantic_program_hash,
                },
                checkpoint_path,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": val_value,
                "val_micro_f1": validation_micro_f1,
            }
        )
        if args.print_every > 0 and (epoch == 1 or epoch % args.print_every == 0):
            print(
                f"    epoch={epoch:03d} train_loss={train_loss.item():.6f} "
                f"val_loss={val_value:.6f} val_microF1={validation_micro_f1:.4f} "
                f"best={best_epoch:03d} noimp={epochs_without_improvement}",
                flush=True,
            )
        if getattr(args, "resume", False) or getattr(args, "save_latest_state", True):
            _pa, _pr, _ = safe_cuda_peaks()
            _atomic_torch_save(
                {
                    "epoch": epoch, "seed": int(seed), "variant": variant,
                    "context_plan_hash": plan.context_plan_hash,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss, "best_selection": best_selection,
                    "best_stop_value": best_stop_value, "best_val_micro_f1": best_val_micro_f1,
                    "best_epoch": best_epoch, "best_state": best_state,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history": history,
                    "first_train_logit_hash": first_train_logit_hash,
                    "first_gradient_hash": first_gradient_hash,
                    "training_seconds_elapsed": time.perf_counter() - start,
                    "peak_allocated_bytes": max(prior_peak_allocated, _pa or 0),
                    "peak_reserved_bytes": max(prior_peak_reserved, _pr or 0),
                    "rng": _rng_state(),
                },
                latest_state_path,
            )
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint")
    safe_cuda_sync()
    training_elapsed = time.perf_counter() - start
    training_peak_allocated, training_peak_reserved, training_peak_error = safe_cuda_peaks()
    if prior_peak_allocated or prior_peak_reserved:   # peak across all resumed segments
        training_peak_allocated = max(prior_peak_allocated, training_peak_allocated or 0)
        training_peak_reserved = max(prior_peak_reserved, training_peak_reserved or 0)

    model.load_state_dict(best_state, strict=True)
    model.to(device)
    model.eval()
    inference_reset, inference_reset_error = safe_cuda_reset()
    safe_cuda_sync()
    inference_start = time.perf_counter()
    with torch.no_grad():
        final_logits, slot_weights = model(dataset.features, plan)
    safe_cuda_sync()
    inference_elapsed = time.perf_counter() - inference_start
    inference_peak_allocated, inference_peak_reserved, inference_peak_error = safe_cuda_peaks()

    metrics = {
        split: classification_metrics(final_logits, dataset.labels, indices)
        for split, indices in (
            ("train", dataset.train_idx),
            ("val", dataset.val_idx),
            ("test", dataset.test_idx),
        )
    }
    final_state_hash = state_dict_hash(model.state_dict())
    test_logits = final_logits.index_select(0, dataset.test_idx).detach().cpu().numpy()
    np.savez_compressed(
        result_dir / "test_logits.npz",
        logits=test_logits,
        labels=dataset.labels.index_select(0, dataset.test_idx).detach().cpu().numpy(),
        test_idx=dataset.test_idx.detach().cpu().numpy(),
    )
    write_json(result_dir / "history.json", history)

    result = {
        "dataset": "Freebase",
        "task": "node_classification",
        "target_node_type": "BOOK",
        "condition": "recoverability_conditional",
        "variant": variant,
        "seed": int(seed),
        "execution_mode": "independent_raw_recoverability_context",
        "independent_training": True,
        "metrics": metrics,
        "training": {
            "epochs_trained": int(epochs_trained),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "selection_metric": args.select_on,
            "stopping_metric": args.stop_on,
            "best_val_micro_f1": float(best_val_micro_f1),
            "message_mode": args.message_mode,
            "model_impl": getattr(args, "model_impl", "genuine"),
            "time_sec": float(training_elapsed),
            "inference_time_sec": float(inference_elapsed),
            "patience": int(args.patience),
            "max_epochs": int(args.epochs),
            "resumed_from_epoch": resumed_from,
        },
        "model": {
            "parameter_count": parameter_count,
            "initial_parameter_hash": initial_parameter_hash,
            "first_train_logit_hash": first_train_logit_hash,
            "first_gradient_hash": first_gradient_hash,
            "final_parameter_hash": final_state_hash,
            "hidden_dim": int(args.hidden_dim),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "edge_feats": int(args.edge_feats),
            "aggregator": args.aggregator,
            "SAattDim": int(args.SAattDim),
        },
        "optimizer": {
            "name": "Adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
        },
        "memory": {
            "training_peak_allocated_bytes": training_peak_allocated,
            "training_peak_reserved_bytes": training_peak_reserved,
            "training_peak_stats_reset": memory_reset,
            "training_peak_reset_error": memory_reset_error,
            "training_peak_read_error": training_peak_error,
            "inference_peak_allocated_bytes": inference_peak_allocated,
            "inference_peak_reserved_bytes": inference_peak_reserved,
            "inference_peak_stats_reset": inference_reset,
            "inference_peak_reset_error": inference_reset_error,
            "inference_peak_read_error": inference_peak_error,
            "process_max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "data": {
            "feature_hash": dataset.feature_hash,
            "label_hash": dataset.label_hash,
            "split_hash": dataset.split_hash,
            "num_train": int(dataset.train_idx.numel()),
            "num_val": int(dataset.val_idx.numel()),
            "num_test": int(dataset.test_idx.numel()),
        },
        "context": plan.audit_dict(),
        # The model returns a diagnostics dictionary.  Preserve the historical
        # result field as the actual slot-attention tensor rather than trying
        # to call ``detach`` on the dictionary itself.
        "slot_attention_weights": (
            slot_weights["slot_attention"].detach().cpu().tolist()
        ),
        "checkpoint": str(checkpoint_path),
        "test_logits": str(result_dir / "test_logits.npz"),
        "deterministic": bool(args.deterministic),
    }
    write_json(result_dir / "result.json", result)
    print(
        f"[recoverability_conditional] {variant} seed={seed} "
        f"Micro-F1={metrics['test']['micro_f1'] * 100:.2f} "
        f"Macro-F1={metrics['test']['macro_f1'] * 100:.2f} "
        f"epoch={best_epoch} time={training_elapsed:.2f}s",
        flush=True,
    )
    del model, optimizer, final_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def preflight(plans: Sequence[FreebaseBlockContextPlan]) -> dict[str, Any]:
    raw_hashes = {plan.raw_graph_hash for plan in plans}
    context_hashes = {plan.context_plan_hash for plan in plans}
    semantic_hashes = {plan.semantic_program_hash for plan in plans}
    base_hashes = {plan.base_graph_hash for plan in plans}
    layouts = {json.dumps(plan.layout.to_json(), sort_keys=True) for plan in plans}
    audit = {
        "passed": (
            len(raw_hashes) == len(plans)
            and len(context_hashes) == len(plans)
            and len(semantic_hashes) == 1
            and len(base_hashes) == 1
            and len(layouts) == 1
        ),
        "raw_graph_hashes_distinct": len(raw_hashes) == len(plans),
        "context_plan_hashes_distinct": len(context_hashes) == len(plans),
        "semantic_program_hashes_equal": len(semantic_hashes) == 1,
        "base_graph_hashes_equal": len(base_hashes) == 1,
        "layouts_equal": len(layouts) == 1,
        "shared_graph_materialized": False,
        "variants": [plan.audit_dict() for plan in plans],
    }
    if not audit["passed"]:
        raise AssertionError(f"Freebase recoverability-conditional preflight failed: {audit}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genuine recoverability-conditional SlotGAT on Freebase NC"
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=INV_GNN_DATA / "preprocessed/freebase_slotgat_block",
    )
    parser.add_argument("--variants", nargs="+", default=["unchanged", "exact_2"])
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing node.dat/label.dat for features and splits. "
            "Defaults to source_dir stored in the unchanged context plan."
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", default=[",".join(str(seed) for seed in DEFAULT_SEEDS)]
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", "--epoch", type=int, default=100)
    parser.add_argument(
        "--message-mode",
        default="exact_union",
        choices=["exact_union", "physical", "factorized"],
        help=(
            "exact_union (default): direct destination-CSR schedule that is "
            "message-equivalent to SlotGAT on the physical union graph. "
            "factorized: the superseded per-destination path aggregate, kept for ablation."
        ),
    )
    parser.add_argument(
        "--union-message-root",
        type=Path,
        default=INV_GNN_DATA / "preprocessed/freebase_slotgat_union_messages/exact_2",
        help="Schedule built by prepare_Freebase_union_message_equivalent.py.",
    )
    parser.add_argument(
        "--physical-message-root",
        type=Path,
        default=INV_GNN_DATA / "preprocessed/freebase_slotgat_physical_messages",
        help=(
            "Root holding one per-variant schedule subdirectory, built with "
            "prepare_Freebase_union_message_equivalent.py --any-context-kind. "
            "Used by --message-mode physical (the matched control)."
        ),
    )
    parser.add_argument("--union-message-chunk-edges", type=int, default=1_000_000)
    parser.add_argument("--disable-union-checkpoint", action="store_true")
    parser.add_argument(
        "--stop-on",
        default="val_loss",
        choices=["val_loss", "val_micro_f1"],
        help=(
            "Early-stopping metric, deliberately separate from --select-on. "
            "Freebase's majority class is 48.8%% of the test set, so micro-F1 "
            "plateaus there for many epochs while validation loss keeps falling; "
            "running patience off micro-F1 can stop a run early with "
            "best_epoch=1."
        ),
    )
    parser.add_argument(
        "--select-on",
        default="val_micro_f1",
        choices=["val_micro_f1", "val_loss"],
        help="Checkpoint-selection criterion; defaults to validation micro-F1.",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run from its per-epoch <checkpoint>_latest_state.pt "
             "(model, optimizer, selection state, history, RNG). Completed runs are skipped "
             "either way unless --overwrite.")
    parser.add_argument(
        "--no-save-latest-state", dest="save_latest_state", action="store_false",
        help="Skip the per-epoch resume snapshot (it is ~2x the checkpoint size).")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--aggregator", default="SA", choices=["SA"])
    parser.add_argument("--SAattDim", type=int, default=3)
    parser.add_argument("--model-impl", dest="model_impl", default="genuine", choices=["genuine", "parity"],
                        help="'parity' ports the upstream slotGAT details that closed the IMDb architecture "
                             "gap (readout L2 normalisation, init, edge path, residuals) into the "
                             "union-message model")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--allow-nonidentity-sparse", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "freebase_nc_block_conditional",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "checkpoint" / "freebase_nc_block_conditional",
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = parse_seeds(args.seeds)
    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else f"cuda:{int(args.gpu)}"
    )
    print("Freebase node classification with recoverability-conditional SlotGAT")
    print(f"Device: {device}")
    print(f"Variants: {args.variants}")
    print(f"Seeds: {args.seeds}")

    plans = [
        FreebaseBlockContextPlan.load(args.preprocessed_root / variant)
        for variant in args.variants
    ]
    for plan in plans:
        if plan.variant not in args.variants:
            raise ValueError(
                f"Plan at {args.preprocessed_root / plan.variant} has unexpected variant name"
            )
    preflight_audit = preflight(plans)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load the shared direct-union message schedule once and validate it against
    # every variant's plan before any training starts. The schedule is the same
    # canonical message program for all variants; the RAW CONTEXT stays
    # variant-specific, which is what the invariance claim rests on.
    args.union_message_schedule = None
    args.physical_message_schedules = {}
    if args.message_mode == "physical":
        for plan in plans:
            root = args.physical_message_root / plan.variant
            schedule = UnionMessageSchedule.load(root)
            if schedule.exact2_raw_link_hash != plan.raw_link_hash:
                raise SystemExit(
                    f"Physical schedule at {root} was built from a different link.dat "
                    f"than the {plan.variant!r} context plan describes"
                )
            schedule.validate_against_plan(plan)
            args.physical_message_schedules[plan.variant] = schedule
            print(
                f"Physical schedule [{plan.variant}]: {schedule.message_count:,} messages, "
                f"program_hash={schedule.message_program_hash[:16]}"
            )
        preflight_audit["physical_message_schedules"] = {
            variant: schedule.audit_dict()
            for variant, schedule in args.physical_message_schedules.items()
        }
        hashes = {s.message_program_hash for s in args.physical_message_schedules.values()}
        if len(hashes) != len(args.physical_message_schedules):
            raise SystemExit(
                "Physical schedules are not all distinct -- the control arm would be "
                "invariant by construction, which defeats its purpose"
            )
    if args.message_mode == "exact_union":
        schedule = UnionMessageSchedule.load(args.union_message_root)
        for plan in plans:
            schedule.validate_against_plan(plan)
        args.union_message_schedule = schedule
        preflight_audit["union_message_schedule"] = schedule.audit_dict()
        print(
            f"Union message schedule: {schedule.message_count:,} messages over "
            f"{schedule.num_channels} channels, policy={schedule.construction_policy}, "
            f"program_hash={schedule.message_program_hash[:16]} "
            f"(validated against all {len(plans)} variant plans)"
        )
    preflight_audit["message_mode"] = args.message_mode
    preflight_audit["selection_metric"] = args.select_on
    write_json(args.results_dir / "preflight_audit.json", preflight_audit)
    operator_note = (
        "direct union-equivalent message schedule"
        if args.message_mode == "exact_union"
        else "factorized path aggregate (SUPERSEDED, ablation only)"
    )
    print(f"Preflight passed: raw/context plans differ; operator = {operator_note}.")

    reference_plan = next(
        (plan for plan in plans if plan.variant == "unchanged"), plans[0]
    )
    reference_data_dir = (
        args.reference_data_dir
        if args.reference_data_dir is not None
        else Path(reference_plan.source_dir)
    )
    dataset = load_dataset(reference_data_dir, reference_plan.layout, device)
    print(f"Feature/label source: {Path(reference_data_dir).resolve()}")
    if len(dataset.features) != len(plans[0].layout.counts):
        raise ValueError("Feature type count does not match the context plan")
    if tuple(feature.shape[0] for feature in dataset.features) != plans[0].layout.counts:
        raise ValueError("Feature row counts do not match the context plan")

    manifest = {
        "variants": args.variants,
        "seeds": args.seeds,
        "device": str(device),
        "preprocessed_root": str(args.preprocessed_root),
        "execution_mode": "independent_raw_recoverability_context",
        "hyperparameters": vars(args) | {"seeds": args.seeds},
    }
    # ``args`` carries the loaded UnionMessageSchedule (numpy arrays), which is
    # not JSON-serialisable. Record its identity instead of the object.
    schedule_obj = manifest["hyperparameters"].pop("union_message_schedule", None)
    manifest["hyperparameters"].pop("physical_message_schedules", None)
    manifest["hyperparameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["hyperparameters"].items()
    }
    manifest["union_message_schedule"] = (
        schedule_obj.audit_dict() if schedule_obj is not None else None
    )
    write_json(args.results_dir / "run_manifest.json", manifest)

    plan_by_variant = {plan.variant: plan for plan in plans}
    for seed in args.seeds:
        for variant in args.variants:
            result_dir = args.results_dir / "recoverability_conditional" / variant / f"seed_{seed}"
            checkpoint_path = (
                args.checkpoint_dir
                / "recoverability_conditional"
                / variant
                / f"seed_{seed}.pt"
            )
            if (result_dir / "result.json").exists() and not args.overwrite:
                print(f"Skipping completed run: {result_dir}")
                continue
            train_one_run(
                args=args,
                seed=seed,
                variant=variant,
                plan=plan_by_variant[variant],
                dataset=dataset,
                result_dir=result_dir,
                checkpoint_path=checkpoint_path,
                device=device,
            )

    if args.aggregate:
        from aggregate_Freebase_nc_block_conditional import aggregate_results

        aggregate_results(args.results_dir, expected_variants=args.variants)


if __name__ == "__main__":
    main()
