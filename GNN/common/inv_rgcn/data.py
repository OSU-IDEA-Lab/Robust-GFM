from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import hashlib
import json

import torch


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    return str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()


def _hash_value(h: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        h.update(b"TENSOR")
        h.update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        h.update(b"MAPPING")
        for key in sorted(value, key=lambda x: str(x)):
            h.update(str(key).encode())
            _hash_value(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(b"SEQUENCE")
        for item in value:
            _hash_value(h, item)
    elif value is None:
        h.update(b"NONE")
    else:
        h.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def stable_object_hash(value: Any) -> str:
    h = hashlib.sha256()
    _hash_value(h, value)
    return h.hexdigest()


def stable_tensor_hash(*tensors: torch.Tensor, metadata: Optional[Mapping[str, Any]] = None) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(_tensor_bytes(tensor))
    if metadata is not None:
        _hash_value(h, metadata)
    return h.hexdigest()


def _move_value(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_value(v, device) for v in value)
    return value


def canonicalize_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort and deduplicate relation edges under set semantics.

    The sort key is (relation, destination, source). Duplicate semantic messages are
    intentionally collapsed. This prevents representation-dependent witness counts
    from leaking through an otherwise equivalent conditional context.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_type.ndim != 1 or edge_type.numel() != edge_index.shape[1]:
        raise ValueError("edge_type must have shape [E]")
    if edge_weight is None:
        edge_weight = torch.ones(edge_type.numel(), dtype=torch.float32)
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_type.numel():
        raise ValueError("edge_weight must have shape [E]")
    if edge_type.numel() == 0:
        return edge_index.long(), edge_type.long(), edge_weight.float()

    rows = torch.stack(
        [edge_type.long(), edge_index[1].long(), edge_index[0].long()], dim=1
    )
    unique_rows = torch.unique(rows, dim=0, sorted=True)
    out_type = unique_rows[:, 0]
    out_index = torch.stack([unique_rows[:, 2], unique_rows[:, 1]], dim=0)
    # Set semantics gives every canonical message weight 1. Normalization occurs in-layer.
    out_weight = torch.ones(out_type.numel(), dtype=torch.float32)
    return out_index, out_type, out_weight


@dataclass
class ConditionalGraphData:
    """A different physical graph plus an invariant semantic message program."""

    variant: str
    num_nodes: int
    node_type: torch.Tensor
    physical_edge_index: torch.Tensor
    physical_edge_type: torch.Tensor
    physical_relation_names: list[str]
    semantic_edge_index: torch.Tensor
    semantic_edge_type: torch.Tensor
    semantic_edge_weight: torch.Tensor
    semantic_relation_names: list[str]
    x: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    train_mask: Optional[torch.Tensor] = None
    val_mask: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    task_data: Dict[str, Any] = field(default_factory=dict)
    context_audit: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.node_type.shape != (self.num_nodes,):
            raise ValueError("node_type must have shape [num_nodes]")
        for name, idx, typ in (
            ("physical", self.physical_edge_index, self.physical_edge_type),
            ("semantic", self.semantic_edge_index, self.semantic_edge_type),
        ):
            if idx.ndim != 2 or idx.shape[0] != 2:
                raise ValueError(f"{name}_edge_index must have shape [2, E]")
            if typ.ndim != 1 or typ.numel() != idx.shape[1]:
                raise ValueError(f"{name}_edge_type must have shape [E]")
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.num_nodes):
                raise ValueError(f"{name} edge index outside node range")
        if self.semantic_edge_weight.shape != self.semantic_edge_type.shape:
            raise ValueError("semantic_edge_weight shape mismatch")
        if self.x is not None and self.x.shape[0] != self.num_nodes:
            raise ValueError("x first dimension must equal num_nodes")
        if self.y is not None and self.y.shape[0] != self.num_nodes:
            raise ValueError("y first dimension must equal num_nodes")
        for mask in (self.train_mask, self.val_mask, self.test_mask):
            if mask is not None and mask.shape != (self.num_nodes,):
                raise ValueError("masks must have shape [num_nodes]")

    @property
    def physical_hash(self) -> str:
        return stable_tensor_hash(
            self.physical_edge_index,
            self.physical_edge_type,
            self.node_type,
            metadata={"relations": self.physical_relation_names, "num_nodes": self.num_nodes},
        )

    @property
    def semantic_hash(self) -> str:
        return stable_tensor_hash(
            self.semantic_edge_index,
            self.semantic_edge_type,
            self.semantic_edge_weight,
            metadata={"relations": self.semantic_relation_names, "num_nodes": self.num_nodes},
        )

    @property
    def feature_hash(self) -> Optional[str]:
        if self.x is None:
            return None
        return stable_tensor_hash(self.x)

    @property
    def supervision_hash(self) -> str:
        return stable_object_hash({
            "y": self.y,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
            "task_data": self.task_data,
        })

    def to(self, device: torch.device | str) -> "ConditionalGraphData":
        kwargs: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            kwargs[key] = _move_value(value, device)
        return ConditionalGraphData(**kwargs)


def save_graph_data(data: ConditionalGraphData, out_dir: str | Path) -> None:
    data.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(data, out / "graph.pt")
    audit = {
        "variant": data.variant,
        "num_nodes": data.num_nodes,
        "physical_hash": data.physical_hash,
        "semantic_hash": data.semantic_hash,
        "feature_hash": data.feature_hash,
        "supervision_hash": data.supervision_hash,
        "num_physical_edges": int(data.physical_edge_type.numel()),
        "num_semantic_edges": int(data.semantic_edge_type.numel()),
        "physical_relation_names": data.physical_relation_names,
        "semantic_relation_names": data.semantic_relation_names,
        "context_audit": data.context_audit,
        "metadata": data.metadata,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=str))


def load_graph_data(path: str | Path, map_location: str | torch.device = "cpu") -> ConditionalGraphData:
    p = Path(path)
    if p.is_dir():
        p = p / "graph.pt"
    data = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(data, ConditionalGraphData):
        raise TypeError(f"Expected ConditionalGraphData in {p}, got {type(data)!r}")
    data.validate()
    return data
