"""Run original and conditional-invariant SlotGAT on four IMDb variants.

This runner keeps the four HGB input graphs physically different.  For the
``conditional`` condition, each raw graph is interpreted as a context graph by
``conditional_bias.py`` and the architecture receives that distinct raw context plan and evaluates
direct-versus-Link conditional rules inside every layer.  The loss, optimizer, SlotGAT core, features, labels,
and train/validation/test splits are unchanged.

Examples
--------
python preprocess_IMDB.py
python validate_IMDB_slotgat_conditional.py
python run_IMDB_nc_conditional.py --conditions original conditional --aggregate
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from conditional_bias import (  # noqa: E402
    ConditionalContextPlan,
    assert_context_plans_invariant,
    compile_from_data_loader,
    compile_physical_from_data_loader,
)
from model import slotGAT  # noqa: E402
from model.genuine_conditional_slotgat import GenuineConditionalSlotGAT  # noqa: E402
from model.parity_slotgat import ABLATIONS as PARITY_ABLATIONS, ParitySlotGAT  # noqa: E402
from utils.data import load_data  # noqa: E402
from utils.tools import func_args_parse, set_seed, single_feat_net  # noqa: E402


RESULT_FORMAT = "slotgat_imdb_nc_genuine_conditional_v2"
NUM_CLASSES = 3
DEFAULT_VARIANTS = (1, 2, 3, 4)
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
TRAIN_FORWARD_SEED_OFFSET = 1_000_003
VARIANT_NAMES = {i: f"IMDB_var{i}" for i in range(1, 6)}
# Node-type ids inside the HGB IMDb layout (mirrors conditional_bias.py).
MOVIE_TYPE = 0
LINK_TYPE = 3
# IMDB_var5 is the union/universal graph, not a transformation variant.
UNIVERSAL_VARIANT = 5


@dataclass
class DatasetBundle:
    variant: int
    variant_name: str
    features_raw: list[Any]
    labels_np: np.ndarray
    split_indices: dict[str, np.ndarray]
    dl: Any
    program: ConditionalContextPlan
    physical_program: ConditionalContextPlan
    feature_hash: str
    labels_hash: str
    splits_hash: str


class EarlyStopper:
    """Early stopping on either a loss (minimise) or a metric (maximise).

    ``mode='max'`` is the default for selection on validation Micro-F1, which is
    the evaluation metric. Selecting on validation loss while evaluating
    Micro-F1 can pick the wrong checkpoint.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"Unknown early-stopping mode: {mode!r}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.sign = 1.0 if mode == "min" else -1.0
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    @property
    def best_value(self) -> float:
        """The tracked quantity in its natural orientation."""
        return self.best if self.mode == "min" else -self.best

    def update(self, raw_value: float, epoch: int, model: torch.nn.Module) -> bool:
        value = self.sign * float(raw_value)
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.bad_epochs = 0
            self.best_epoch = int(epoch)
            self.best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def parse_int_csv(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def safe_reset_cuda_peak_memory_stats() -> bool:
    """Reset CUDA allocator peaks without ever blocking training.

    Some cluster PyTorch builds raise ``RuntimeError: Invalid device argument``
    even for the current logical CUDA index, especially before the allocator has
    created a complete device context. Peak-memory reporting is diagnostic only,
    so an unsupported reset is recorded and training continues.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Omitting the argument uses PyTorch's current logical CUDA device and
        # avoids physical-vs-logical index mismatches under CUDA_VISIBLE_DEVICES.
        torch.cuda.reset_peak_memory_stats()
        return True
    except (RuntimeError, TypeError) as exc:
        print(
            "Warning: CUDA peak-memory reset is unavailable in this PyTorch "
            f"build; continuing without reset ({exc}).",
            flush=True,
        )
        return False


def safe_cuda_peak_memory_stats() -> tuple[int | None, int | None]:
    """Return allocator peak bytes, or ``(None, None)`` when unsupported."""
    if not torch.cuda.is_available():
        return None, None
    try:
        return (
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
        )
    except (RuntimeError, TypeError) as exc:
        print(
            "Warning: CUDA peak-memory counters are unavailable in this "
            f"PyTorch build ({exc}).",
            flush=True,
        )
        return None, None


def safe_cuda_synchronize() -> bool:
    """Synchronize the current CUDA device when the build supports it."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        return True
    except (RuntimeError, TypeError) as exc:
        print(
            "Warning: CUDA synchronization is unavailable; continuing "
            f"without an explicit synchronize ({exc}).",
            flush=True,
        )
        return False


def set_deterministic(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _hash_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def hash_array(value: Any) -> str:
    if sp.issparse(value):
        csr = value.tocsr(copy=True)
        csr.sort_indices()
        return _hash_bytes(
            [
                np.asarray(csr.shape, dtype="<i8").tobytes(),
                np.asarray(csr.indptr, dtype="<i8").tobytes(),
                np.asarray(csr.indices, dtype="<i8").tobytes(),
                np.asarray(csr.data).tobytes(),
            ]
        )
    arr = np.ascontiguousarray(np.asarray(value))
    return _hash_bytes(
        [np.asarray(arr.shape, dtype="<i8").tobytes(), str(arr.dtype).encode(), arr.tobytes()]
    )


def hash_feature_list(features: Sequence[Any]) -> str:
    return _hash_bytes([hash_array(feature).encode() for feature in features])


def hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def hash_gradients(model: torch.nn.Module) -> str:
    grads: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad
    return hash_state_dict(grads)


def sparse_to_tensor(matrix: Any, device: torch.device) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        return matrix.to(device)
    if isinstance(matrix, np.ndarray):
        return torch.as_tensor(matrix, dtype=torch.float32, device=device)
    coo = matrix.tocoo(copy=False)
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.int64, device=device
    )
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def tensorize_features(
    features_raw: Sequence[Any], feats_type: int, device: torch.device
) -> tuple[list[torch.Tensor], list[int]]:
    features = [sparse_to_tensor(feature, device) for feature in features_raw]
    if feats_type == 0:
        return features, [int(feature.shape[1]) for feature in features]
    if feats_type == 1:
        output = [features[0]]
        in_dims = [int(features[0].shape[1])]
        for feature in features[1:]:
            output.append(torch.zeros((feature.shape[0], 10), device=device))
            in_dims.append(10)
        return output, in_dims
    raise ValueError(f"Unsupported feature mode: {feats_type}")


def build_original_graph(
    *, features_list: Sequence[torch.Tensor], dl: Any, device: torch.device
) -> tuple[Any, torch.Tensor, int]:
    """Reproduce the current repository's local raw-graph SlotGAT bias."""

    edge_to_relation: dict[tuple[int, int], int] = {}
    for relation, matrix in sorted(dl.links["data"].items(), key=lambda kv: int(kv[0])):
        coo = matrix.tocoo(copy=False)
        for src, dst in zip(coo.row.tolist(), coo.col.tolist()):
            edge_to_relation[(int(src), int(dst))] = int(relation)

    self_relation = max(edge_to_relation.values(), default=-1) + 1
    for node in range(int(dl.nodes["total"])):
        edge_to_relation.setdefault((node, node), self_relation)

    relation_values = sorted(set(edge_to_relation.values()))
    relation_to_dense = {relation: i for i, relation in enumerate(relation_values)}
    records = sorted(
        (
            dst,
            relation_to_dense[relation],
            src,
        )
        for (src, dst), relation in edge_to_relation.items()
    )
    src = torch.as_tensor([record[2] for record in records], dtype=torch.int64, device=device)
    dst = torch.as_tensor([record[0] for record in records], dtype=torch.int64, device=device)
    e_feat = torch.as_tensor([record[1] for record in records], dtype=torch.int64, device=device)
    graph = dgl.graph((src, dst), num_nodes=int(dl.nodes["total"]), device=device)

    num_ntypes = len(features_list)
    graph.num_ntypes = num_ntypes
    graph.node_idx_by_ntype = []
    graph.node_ntype_indexer = torch.zeros(
        graph.num_nodes(), num_ntypes, dtype=torch.float32, device=device
    )
    offset = 0
    for node_type, feature in enumerate(features_list):
        ids = list(range(offset, offset + int(feature.shape[0])))
        graph.node_idx_by_ntype.append(ids)
        if ids:
            graph.node_ntype_indexer[
                torch.as_tensor(ids, dtype=torch.int64, device=device), node_type
            ] = 1.0
        offset += int(feature.shape[0])
    graph.edge_type_indexer = F.one_hot(
        e_feat, num_classes=len(relation_values)
    ).to(dtype=torch.float32)
    return graph, e_feat, len(relation_values)


def make_data_recorder() -> dict[str, Any]:
    return {"meta": {"getSAAttentionScore": "False"}, "data": {}, "status": "None"}


def build_original_model(
    args: argparse.Namespace,
    *,
    features_list: Sequence[torch.Tensor],
    in_dims: Sequence[int],
    dl: Any,
    device: torch.device,
    data_recorder: dict[str, Any],
) -> tuple[torch.nn.Module, Callable[[Sequence[torch.Tensor]], tuple[Any, Any]], dict[str, Any]]:
    graph, edge_types, num_edge_types = build_original_graph(
        features_list=features_list, dl=dl, device=device
    )
    heads = [args.num_heads] * args.num_layers + [1]
    fargs, fkargs = func_args_parse(
        graph,
        args.edge_feats,
        num_edge_types,
        list(in_dims),
        args.hidden_dim,
        NUM_CLASSES,
        args.num_layers,
        heads,
        F.elu,
        args.dropout_feat,
        args.dropout_attn,
        args.slope,
        True,
        0.05,
        num_ntype=len(features_list),
        eindexer=None,
        aggregator=args.aggregator,
        SAattDim=args.SAattDim,
        dataRecorder=data_recorder,
        vis_data_saver=None,
    )
    model = single_feat_net(slotGAT, *fargs, **fkargs).to(device)

    def forward_fn(current_features: Sequence[torch.Tensor]):
        return model(current_features, edge_types)

    graph_meta = {
        "message_graph_edges": int(graph.num_edges()),
        "message_edge_types": int(num_edge_types),
        "message_program_hash": None,
    }
    return model, forward_fn, graph_meta


def build_conditional_model(
    args: argparse.Namespace,
    *,
    features_list: Sequence[torch.Tensor],
    in_dims: Sequence[int],
    program: ConditionalContextPlan,
    device: torch.device,
    data_recorder: dict[str, Any],
) -> tuple[torch.nn.Module, Callable[[Sequence[torch.Tensor]], tuple[Any, Any]], dict[str, Any]]:
    """Build the genuine raw-context conditional SlotGAT.

    ``program`` is a variant-specific context plan and remains an explicit
    forward input.  No semantic DGL graph is created and no plan is shared
    across variants.
    """
    del data_recorder  # retained in the signature for runner compatibility
    impl = getattr(args, "model_impl", "genuine")
    if impl == "parity":
        # Faithful port of the upstream slotGAT (model/parity_slotgat.py) on the plan
        # interface; --ablate switches individual upstream details back to the
        # re-implementation's choices.
        model = ParitySlotGAT(
            layout=program.layout, in_dims=in_dims, num_classes=NUM_CLASSES, hidden_dim=args.hidden_dim,
            num_heads=args.num_heads, num_layers=args.num_layers, edge_feats=args.edge_feats,
            dropout_feat=args.dropout_feat, dropout_attn=args.dropout_attn, slope=args.slope,
            aggregator=args.aggregator, sa_attention_dim=args.SAattDim, alpha=float(getattr(args, "alpha", 0.05)),
            ablate=tuple(getattr(args, "ablate", ()) or ()),
        ).to(device)
    else:
        model = GenuineConditionalSlotGAT(
        layout=program.layout,
        in_dims=in_dims,
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
        ).to(device)

    def forward_fn(current_features: Sequence[torch.Tensor]):
        return model(current_features, program)

    graph_meta = {
        "model_facing_input": "distinct_raw_conditional_context_plan",
        "raw_context_hash": program.raw_graph_hash,
        "context_plan_hash": program.context_plan_hash,
        "semantic_propagation_hash": program.semantic_propagation_hash,
        "message_program_hash": program.semantic_propagation_hash,
        "raw_context_edge_count": len(program.raw_edges),
        "conditional_witness_count": len(program.witnesses),
        "semantic_message_count": len(program.emit_semantic_messages()),
        "message_graph_edges": None,
        "message_edge_types": int(program.num_edge_types),
        "materialized_shared_dgl_graph": False,
        "shared_program_execution": False,
        "independent_training": True,
        "sparse_features_densified_once": True,
        "deterministic_degree_bucket_reduction": True,
    }
    return model, forward_fn, graph_meta


def build_pooled_bow_features(dl: Any, features: Sequence[Any]) -> list[Any]:
    """Give non-movie nodes the mean of their neighbouring movies' BoW vectors.

    Matches ``preprocess_IMDB_rgcn_conditional.py:245-261`` so SlotGAT and RGCN
    read the same node inputs on this cell. The pooling neighbourhoods are taken
    from the UNION of every relation incident to a movie, which is
    variant-independent by construction: Movie<->Link is a bijection present in
    every variant, so a role node reachable directly or through the Link node
    pools over exactly the same movie set either way. Invariance is therefore
    preserved; this is verified in ``validate_IMDB_slotgat_conditional.py``.
    """
    raw_counts = dl.nodes["count"]
    # ``dl.nodes['count']`` is a MAPPING keyed by node type, so iterating it
    # directly yields the keys, not the counts.
    counts = (
        [int(raw_counts[i]) for i in range(len(raw_counts))]
        if isinstance(raw_counts, Mapping)
        else [int(value) for value in raw_counts]
    )
    offsets = [0]
    for value in counts[:-1]:
        offsets.append(offsets[-1] + value)
    # Derived from the node table rather than hardcoded, so the same code reads both
    # the four-type build and the year-bearing five-type build.
    num_types = len(counts)
    peer_types = tuple(t for t in range(1, num_types) if counts[t] > 0)
    movie_features = np.asarray(
        features[MOVIE_TYPE].todense() if sp.issparse(features[MOVIE_TYPE]) else features[MOVIE_TYPE],
        dtype=np.float32,
    )
    feature_dim = movie_features.shape[1]

    # movie -> link is a bijection; role nodes attach to the movie or its link.
    movie_of_link: dict[int, int] = {}
    for matrix in dl.links["data"].values():
        coo = matrix.tocoo(copy=False)
        for src, dst in zip(coo.row.tolist(), coo.col.tolist()):
            src_type = next(t for t in range(num_types - 1, -1, -1) if src >= offsets[t])
            dst_type = next(t for t in range(num_types - 1, -1, -1) if dst >= offsets[t])
            if src_type == MOVIE_TYPE and dst_type == LINK_TYPE:
                movie_of_link[dst] = src
            elif src_type == LINK_TYPE and dst_type == MOVIE_TYPE:
                movie_of_link[src] = dst

    sums = {t: np.zeros((counts[t], feature_dim), dtype=np.float64) for t in peer_types}
    hits = {t: np.zeros(counts[t], dtype=np.int64) for t in peer_types}
    seen: set[tuple[int, int]] = set()
    for matrix in dl.links["data"].values():
        coo = matrix.tocoo(copy=False)
        for src, dst in zip(coo.row.tolist(), coo.col.tolist()):
            src_type = next(t for t in range(num_types - 1, -1, -1) if src >= offsets[t])
            dst_type = next(t for t in range(num_types - 1, -1, -1) if dst >= offsets[t])
            for anchor, anchor_type, peer, peer_type in (
                (src, src_type, dst, dst_type),
                (dst, dst_type, src, src_type),
            ):
                if peer_type not in peer_types:
                    continue
                if anchor_type == MOVIE_TYPE:
                    movie = anchor
                elif anchor_type == LINK_TYPE:
                    movie = movie_of_link.get(anchor)
                else:
                    continue
                if movie is None:
                    continue
                local = peer - offsets[peer_type]
                key = (peer, movie)
                if key in seen:
                    continue
                seen.add(key)
                sums[peer_type][local] += movie_features[movie]
                hits[peer_type][local] += 1

    pooled = list(features)
    for node_type in peer_types:
        divisor = np.maximum(hits[node_type], 1)[:, None]
        pooled[node_type] = sp.csr_matrix(
            (sums[node_type] / divisor).astype(np.float32)
        )
    return pooled


def load_datasets(
    variants: Sequence[int], strict: bool, feature_mode: str = "identity"
) -> dict[int, DatasetBundle]:
    bundles: dict[int, DatasetBundle] = {}
    for variant in variants:
        variant_name = VARIANT_NAMES[variant]
        features, _adjacency, labels, splits, dl = load_data(
            prefix=variant_name, multi_labels=False
        )
        if feature_mode == "pooled_bow":
            features = build_pooled_bow_features(dl, features)
        elif feature_mode != "identity":
            raise ValueError(f"Unknown feature mode: {feature_mode!r}")
        split_indices = {
            "train": np.sort(np.asarray(splits["train_idx"], dtype=np.int64)),
            "val": np.sort(np.asarray(splits["val_idx"], dtype=np.int64)),
            "test": np.sort(np.asarray(splits["test_idx"], dtype=np.int64)),
        }
        labels_np = np.asarray(labels, dtype=np.int64)
        split_hash = _hash_bytes(
            [split_indices[key].astype("<i8").tobytes() for key in ("train", "val", "test")]
        )
        program = compile_from_data_loader(dl, strict=strict)
        physical_program = compile_physical_from_data_loader(dl, strict=strict)
        bundles[variant] = DatasetBundle(
            variant=variant,
            variant_name=variant_name,
            features_raw=list(features),
            labels_np=labels_np,
            split_indices=split_indices,
            dl=dl,
            program=program,
            physical_program=physical_program,
            feature_hash=hash_feature_list(features),
            labels_hash=hash_array(labels_np),
            splits_hash=split_hash,
        )
    return bundles


def validate_shared_inputs(
    bundles: Mapping[int, DatasetBundle], *, allow_feature_mismatch: bool
) -> dict[str, Any]:
    programs = {bundle.variant_name: bundle.program for bundle in bundles.values()}
    assert_context_plans_invariant(
        programs,
        require_distinct_raw_graphs=True,
        require_distinct_context_plans=True,
    )

    labels_hashes = {bundle.labels_hash for bundle in bundles.values()}
    split_hashes = {bundle.splits_hash for bundle in bundles.values()}
    feature_hashes = {bundle.feature_hash for bundle in bundles.values()}
    if len(labels_hashes) != 1:
        raise AssertionError("Labels differ across IMDb variants")
    if len(split_hashes) != 1:
        raise AssertionError("Train/validation/test splits differ across IMDb variants")
    if len(feature_hashes) != 1 and not allow_feature_mismatch:
        raise AssertionError(
            "Input features differ across variants.  Exact training invariance requires shared "
            "features; rerun preprocessing together or pass --allow-feature-mismatch only for "
            "a diagnostic non-exact experiment."
        )

    return {
        "format_version": "slotgat_imdb_genuine_conditional_preflight_v2",
        "variants": {
            bundle.variant_name: {
                "raw_graph_hash": bundle.program.raw_graph_hash,
                "context_plan_hash": bundle.program.context_plan_hash,
                "semantic_propagation_hash": bundle.program.semantic_propagation_hash,
                "message_program_hash": bundle.program.semantic_propagation_hash,
                "feature_hash": bundle.feature_hash,
                "labels_hash": bundle.labels_hash,
                "splits_hash": bundle.splits_hash,
                "program": bundle.program.to_manifest(),
            }
            for bundle in bundles.values()
        },
        "raw_graphs_are_distinct": len(
            {bundle.program.raw_graph_hash for bundle in bundles.values()}
        )
        == len(bundles),
        "context_plans_are_distinct": len(
            {bundle.program.context_plan_hash for bundle in bundles.values()}
        ) == len(bundles),
        "semantic_propagation_is_identical": len(
            {bundle.program.semantic_propagation_hash for bundle in bundles.values()}
        ) == 1,
        "message_programs_are_identical": len(
            {bundle.program.semantic_propagation_hash for bundle in bundles.values()}
        ) == 1,
        "materialized_shared_dgl_graph": False,
        "features_are_identical": len(feature_hashes) == 1,
        "labels_are_identical": True,
        "splits_are_identical": True,
    }


def split_metrics(logits: torch.Tensor, labels: torch.Tensor, indices: np.ndarray) -> dict[str, float]:
    index_t = torch.as_tensor(indices, dtype=torch.int64, device=logits.device)
    split_logits = logits[index_t]
    split_labels = labels[index_t]
    predictions = split_logits.argmax(dim=1)
    true_np = split_labels.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    sorted_indices = torch.argsort(split_logits, dim=1, descending=True)
    ranks = (
        sorted_indices == split_labels.unsqueeze(1)
    ).nonzero(as_tuple=False)[:, 1].to(torch.float32) + 1.0
    k = min(3, split_logits.shape[1])
    hit3 = (
        torch.topk(split_logits, k=k, dim=1).indices == split_labels.unsqueeze(1)
    ).any(dim=1).to(torch.float32).mean().item()
    return {
        "accuracy": float(accuracy_score(true_np, pred_np)),
        "precision_macro": float(precision_score(true_np, pred_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_np, pred_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true_np, pred_np, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true_np, pred_np, average="macro", zero_division=0)),
        "hit_at_1": float((predictions == split_labels).to(torch.float32).mean().item()),
        "hit_at_3": float(hit3),
        "mrr": float((1.0 / ranks).mean().item()),
    }


def train_one_run(
    args: argparse.Namespace,
    *,
    condition: str,
    bundle: DatasetBundle,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic(seed, args.deterministic)
    memory_peak_reset = False
    if device.type == "cuda":
        torch.cuda.empty_cache()
        memory_peak_reset = safe_reset_cuda_peak_memory_stats()

    features, in_dims = tensorize_features(bundle.features_raw, args.feats_type, device)
    labels = torch.as_tensor(bundle.labels_np, dtype=torch.int64, device=device)
    data_recorder = make_data_recorder()

    if condition == "original":
        model, forward_fn, graph_meta = build_original_model(
            args,
            features_list=features,
            in_dims=in_dims,
            dl=bundle.dl,
            device=device,
            data_recorder=data_recorder,
        )
    elif condition == "conditional":
        model, forward_fn, graph_meta = build_conditional_model(
            args,
            features_list=features,
            in_dims=in_dims,
            program=bundle.program,
            device=device,
            data_recorder=data_recorder,
        )
    elif condition == "physical":
        # Matched control: identical model class, geometry and parameter count
        # to the invariant arm; only the message set differs.
        model, forward_fn, graph_meta = build_conditional_model(
            args,
            features_list=features,
            in_dims=in_dims,
            program=bundle.physical_program,
            device=device,
            data_recorder=data_recorder,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    stopper = EarlyStopper(
        args.patience,
        args.min_delta,
        mode="min" if args.select_on == "val_loss" else "max",
    )
    train_idx_t = torch.as_tensor(
        bundle.split_indices["train"], dtype=torch.int64, device=device
    )
    val_idx_t = torch.as_tensor(
        bundle.split_indices["val"], dtype=torch.int64, device=device
    )

    run_dir = args.results_dir / condition / bundle.variant_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / condition / bundle.variant_name / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    initial_parameter_hash = hash_state_dict(model.state_dict())

    # Decouple the stochastic training stream from model construction.  Some
    # third-party modules may consume random numbers in constructors or graph
    # setup.  Resetting here guarantees that corresponding variants use the
    # same dropout masks at every epoch when their message programs match.
    training_rng_seed = int(seed) + TRAIN_FORWARD_SEED_OFFSET
    set_deterministic(training_rng_seed, args.deterministic)

    first_step: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch
        model.train()
        data_recorder["status"] = "Training"
        logits, _ = forward_fn(features)
        data_recorder["status"] = "None"
        log_probabilities = F.log_softmax(logits, dim=1)
        train_loss = F.nll_loss(log_probabilities[train_idx_t], labels[train_idx_t])
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()

        if epoch == 1:
            first_step = {
                "train_logits_hash": hash_array(logits.detach().cpu().numpy()),
                "train_loss": float(train_loss.item()),
                "gradient_hash": hash_gradients(model),
            }

        optimizer.step()
        model.eval()
        with torch.no_grad():
            data_recorder["status"] = "Validation"
            validation_logits, _ = forward_fn(features)
            data_recorder["status"] = "None"
            validation_log_probabilities = F.log_softmax(validation_logits, dim=1)
            validation_loss = F.nll_loss(
                validation_log_probabilities[val_idx_t], labels[val_idx_t]
            )

        validation_micro_f1 = float(
            split_metrics(validation_logits, labels, bundle.split_indices["val"])["micro_f1"]
        )
        if epoch == 1 or epoch % args.log_interval == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss.item()),
                    "validation_loss": float(validation_loss.item()),
                    "validation_micro_f1": validation_micro_f1,
                }
            )
        selection_value = (
            float(validation_loss.item())
            if args.select_on == "val_loss"
            else validation_micro_f1
        )
        if stopper.update(selection_value, epoch, model):
            break

    if stopper.best_state is None:
        raise RuntimeError("Early stopper never captured a model state")
    model.load_state_dict(stopper.best_state, strict=True)
    torch.save(
        {
            "format_version": RESULT_FORMAT,
            "condition": condition,
            "variant": bundle.variant_name,
            "seed": seed,
            "epoch": stopper.best_epoch,
            "state_dict": stopper.best_state,
            "raw_graph_hash": bundle.program.raw_graph_hash,
            "context_plan_hash": bundle.program.context_plan_hash,
            "semantic_propagation_hash": bundle.program.semantic_propagation_hash,
            "message_program_hash": graph_meta["message_program_hash"],
            "materialized_shared_dgl_graph": False if condition == "conditional" else None,
        },
        checkpoint_path,
    )

    # Training window closes here; peak TRAINING and peak INFERENCE GPU memory are
    # recorded separately (as run_Freebase_nc_block_conditional.py already does).
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        training_peak_allocated, training_peak_reserved = safe_cuda_peak_memory_stats()
        inference_peak_reset = safe_reset_cuda_peak_memory_stats()
    else:
        training_peak_allocated, training_peak_reserved, inference_peak_reset = None, None, False
    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        data_recorder["status"] = "FinalTesting"
        final_logits, _ = forward_fn(features)
        data_recorder["status"] = "None"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    split_results = {
        split: split_metrics(final_logits, labels, indices)
        for split, indices in bundle.split_indices.items()
    }
    test_idx = bundle.split_indices["test"]
    test_idx_t = torch.as_tensor(test_idx, dtype=torch.int64, device=device)
    logits_path = run_dir / "test_logits.npz"
    np.savez_compressed(
        logits_path,
        logits=final_logits[test_idx_t].detach().cpu().numpy(),
        node_ids=test_idx,
        labels=bundle.labels_np[test_idx],
    )

    if device.type == "cuda":
        inference_peak_allocated, inference_peak_reserved = safe_cuda_peak_memory_stats()
    else:
        inference_peak_allocated, inference_peak_reserved = None, None
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    final_state_hash = hash_state_dict(model.state_dict())
    result = {
        "format_version": RESULT_FORMAT,
        "dataset": "IMDb node classification",
        "model": "SlotGAT",
        "model_impl": "upstream_dgl" if condition == "original" else getattr(args, "model_impl", "genuine"),
        "ablate": [] if condition == "original" else sorted(getattr(args, "ablate", []) or []),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "condition": condition,
        "variant_id": bundle.variant,
        "variant": bundle.variant_name,
        "seed": seed,
        "raw_graph_hash": bundle.program.raw_graph_hash,
        "conditional_context_plan_hash": bundle.program.context_plan_hash,
        "conditional_message_program_hash": bundle.program.semantic_propagation_hash,
        "active_message_program_hash": graph_meta["message_program_hash"],
        "execution_mode": (
            "independent_raw_context" if condition == "conditional" else "independent_original"
        ),
        "independent_training": True,
        "feature_hash": bundle.feature_hash,
        "labels_hash": bundle.labels_hash,
        "splits_hash": bundle.splits_hash,
        "initial_parameter_hash": initial_parameter_hash,
        "final_parameter_hash": final_state_hash,
        "first_step": first_step,
        "best_epoch": stopper.best_epoch,
        "epochs_trained": epochs_trained,
        "selection_metric": args.select_on,
        "best_selection_value": stopper.best_value,
        "best_validation_loss": (
            stopper.best_value if args.select_on == "val_loss" else float("nan")
        ),
        "metrics": split_results,
        "training_time_seconds": elapsed,
        "inference_time_seconds": inference_seconds,
        "memory": {
            # also kept under the older key names for backward compatibility
            "cuda_peak_allocated_bytes": training_peak_allocated,
            "cuda_peak_reserved_bytes": training_peak_reserved,
            "cuda_peak_stats_reset": bool(memory_peak_reset),
            "training_peak_allocated_bytes": training_peak_allocated,
            "training_peak_reserved_bytes": training_peak_reserved,
            "training_peak_stats_reset": bool(memory_peak_reset),
            "inference_peak_allocated_bytes": inference_peak_allocated,
            "inference_peak_reserved_bytes": inference_peak_reserved,
            "inference_peak_stats_reset": bool(inference_peak_reset),
            "process_max_rss_kb": max_rss_kb,
        },
        "graph": graph_meta,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "test_logits_path": str(logits_path.resolve()),
        "history": history,
        "hyperparameters": {
            "feats_type": args.feats_type,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "edge_feats": args.edge_feats,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout_feat": args.dropout_feat,
            "dropout_attn": args.dropout_attn,
            "slope": args.slope,
            "aggregator": args.aggregator,
            "SAattDim": args.SAattDim,
            "deterministic": args.deterministic,
            "training_rng_seed": training_rng_seed,
        },
    }
    json_dump(run_dir / "result.json", result)

    print(
        f"[{condition:11s}] {bundle.variant_name} seed={seed} "
        f"Micro-F1={100 * split_results['test']['micro_f1']:.2f} "
        f"Macro-F1={100 * split_results['test']['macro_f1']:.2f} "
        f"epoch={stopper.best_epoch} time={elapsed:.2f}s"
    )

    del model, features, labels, final_logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original and conditional-invariant SlotGAT on IMDb node classification"
    )
    # Variant 5 is the union / universal graph (IMDB_var5); it shares the labels
    # and split of var1-4, so it is the SlotGAT* arm. It is not a transformation
    # variant, so keep it in its own results root -- never aggregate a
    # cross-variant Kendall tau against it.
    parser.add_argument("--variants", nargs="+", type=int, default=list(DEFAULT_VARIANTS), choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["physical", "conditional"],
        choices=["original", "physical", "conditional"],
        help=(
            "original  = upstream slotGAT on the raw graph (baseline); "
            "physical  = the SAME model class and parameter count as the invariant arm, "
            "propagating on this variant's own edges (the matched control); "
            "conditional = the invariant arm."
        ),
    )
    parser.add_argument(
        "--select-on",
        default="val_micro_f1",
        choices=["val_micro_f1", "val_loss"],
        help="Checkpoint-selection criterion. Defaults to validation micro-F1.",
    )
    parser.add_argument(
        "--feature-mode",
        default="pooled_bow",
        choices=["pooled_bow", "identity"],
        help=(
            "pooled_bow = non-movie nodes get the mean of their movies' bag-of-words "
            "vectors, matching preprocess_IMDB_rgcn_conditional.py, so SlotGAT and RGCN "
            "read the same inputs; identity = one-hot per non-movie node (HGB default)."
        ),
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results" / "imdb_nc_conditional")
    parser.add_argument("--checkpoint-dir", type=Path, default=SCRIPT_DIR / "checkpoint" / "imdb_nc_conditional")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-feature-mismatch", action="store_true")
    parser.add_argument("--non-strict-context", action="store_true")
    parser.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)

    # Paper-consistent SlotGAT/IMDb settings.
    parser.add_argument("--feats-type", type=int, default=0, choices=[0, 1])
    parser.add_argument("--epochs", "--epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
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
    parser.add_argument("--aggregator", default="SA", choices=["SA", "average", "last_fc", "max", "onedimconv"])
    parser.add_argument("--SAattDim", type=int, default=3)
    parser.add_argument("--model-impl", dest="model_impl", default="genuine", choices=["genuine", "parity"],
                        help="conditional/physical arms: 'genuine' = GenuineConditionalSlotGAT; "
                             "'parity' = ParitySlotGAT, the verified port of the upstream slotGAT")
    parser.add_argument("--alpha", type=float, default=0.05, help="parity: attention residual weight (upstream 0.05)")
    parser.add_argument("--ablate", nargs="*", default=[], choices=list(PARITY_ABLATIONS),
                        help="parity: switch these upstream details back to the re-implementation's choice")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--data-suffix", dest="data_suffix", default="",
        help="Appended to each IMDB_var<N> directory name (e.g. '_year' for the "
             "year-bearing rebuild), so an alternative build can be run without "
             "overwriting the original. Built by preprocess_IMDB.py --suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "data_suffix", ""):
        for _k in list(VARIANT_NAMES):
            VARIANT_NAMES[_k] = VARIANT_NAMES[_k] + args.data_suffix
        print(f"[data-suffix] reading variants {sorted(VARIANT_NAMES.values())}", flush=True)
    args.seeds = parse_int_csv(args.seeds)
    args.results_dir = args.results_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")

    device = torch.device(
        f"cuda:{args.gpu}" if not args.cpu and torch.cuda.is_available() else "cpu"
    )
    print("IMDb node classification and SlotGAT conditional structural bias")
    print(f"Device: {device}")
    print(f"Variants: {args.variants}")
    print(f"Conditions: {args.conditions}")
    print(f"Seeds: {args.seeds}")

    # HGB load_data resolves prefixes relative to the SlotGAT working tree.
    os.chdir(SCRIPT_DIR)
    bundles = load_datasets(
        args.variants,
        strict=not args.non_strict_context,
        feature_mode=args.feature_mode,
    )
    # IMDB_var5 is the universal graph, so it must not take part in the
    # cross-variant invariance preflight (its raw graph is a superset, not a
    # transformation). Run it on its own, in its own results root.
    transformation_variants = [v for v in args.variants if v != UNIVERSAL_VARIANT]
    if UNIVERSAL_VARIANT in args.variants and transformation_variants:
        raise SystemExit(
            "Run the universal graph (variant 5) in a separate invocation with its "
            "own --results-dir; mixing it with the transformation variants would "
            "pollute the cross-variant Kendall tau."
        )
    preflight = (
        validate_shared_inputs(
            bundles, allow_feature_mismatch=args.allow_feature_mismatch
        )
        if transformation_variants
        else {"universal_graph_only": True, "variant": VARIANT_NAMES[UNIVERSAL_VARIANT]}
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_dump(args.results_dir / "preflight_audit.json", preflight)
    print(
        "Preflight passed: raw graphs and conditional witness plans differ; "
        "runtime propagation semantics match."
    )

    manifest_results: list[str] = []
    # Seed-major execution. Every variant is trained independently and receives
    # its own raw context plan as a model-forward input.
    for seed in args.seeds:
        for condition in args.conditions:
            for variant in args.variants:
                train_one_run(
                    args,
                    condition=condition,
                    bundle=bundles[variant],
                    seed=seed,
                    device=device,
                )
                manifest_results.append(
                    str(
                        args.results_dir
                        / condition
                        / bundles[variant].variant_name
                        / f"seed_{seed}"
                        / "result.json"
                    )
                )

    manifest = {
        "format_version": RESULT_FORMAT,
        "variants": args.variants,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "conditional_execution": "independent_raw_context",
        "model_impl": getattr(args, "model_impl", "genuine"),
        "ablate": sorted(getattr(args, "ablate", []) or []),
        "materialized_shared_dgl_graph": False,
        "results": manifest_results,
        "preflight": str(args.results_dir / "preflight_audit.json"),
    }
    json_dump(args.results_dir / "run_manifest.json", manifest)

    if args.aggregate:
        from aggregate_IMDB_nc_conditional import aggregate_results

        aggregate_results(args.results_dir)


if __name__ == "__main__":
    main()
