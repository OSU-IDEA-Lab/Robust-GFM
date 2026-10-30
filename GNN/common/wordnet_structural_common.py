from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

VARIANTS = ("no_changes", "all_inverse_edges", "transitive_edges", "universal_edges")
BUDGETS = ("reduced", "full")
MODES = ("intersection", "union")
INVERSE_SUFFIX = "__inv"
TRANSITIVE_PREFIX = "__transitive__"
FORMAT_VERSION = "wordnet_rgcn_conditional_universal_parity_v2"


def stable_int(*parts: object) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little", signed=False)


def array_sha256(arr: np.ndarray) -> str:
    x = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(np.asarray(tuple(x.shape), dtype=np.int64).tobytes())
    h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def canonical_triples(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    if len(x) == 0:
        return x
    # destination/tail, relation, source/head.  Destination-major ordering
    # makes all incoming messages contiguous for deterministic segment sums.
    order = np.lexsort((x[:, 0], x[:, 1], x[:, 2]))
    x = x[order]
    keep = np.ones(len(x), dtype=bool)
    keep[1:] = np.any(x[1:] != x[:-1], axis=1)
    return x[keep]


def edge_table_from_triples(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = canonical_triples(rows)
    edge_index = torch.as_tensor(np.stack([x[:, 0], x[:, 2]], axis=0), dtype=torch.long)
    edge_type = torch.as_tensor(x[:, 1], dtype=torch.long)
    return edge_index, edge_type


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_selection(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selected_dependencies" not in payload:
        raise ValueError(f"Selection lacks selected_dependencies: {path}")
    return payload


def resolve_selection(catalog_root: str | Path, budget: str) -> Path:
    if budget not in BUDGETS:
        raise ValueError(f"Unknown budget {budget!r}; expected {BUDGETS}")
    path = Path(catalog_root) / f"{budget}_selection.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_split_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def infer_num_base_relations(arrays: Mapping[str, np.ndarray]) -> int:
    if "num_base_relations" in arrays:
        return int(np.asarray(arrays["num_base_relations"]).item())
    supervision = arrays.get("train_pos_supervision", arrays.get("train_pos_no_changes"))
    if supervision is None or len(supervision) == 0:
        raise ValueError("Cannot infer num_base_relations")
    return int(np.max(np.asarray(supervision)[:, 1])) + 1


def get_supervision_train(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "train_pos_supervision" in arrays:
        return np.asarray(arrays["train_pos_supervision"], dtype=np.int64)
    if "train_pos_no_changes" in arrays:
        return np.asarray(arrays["train_pos_no_changes"], dtype=np.int64)
    raise KeyError("Split archive lacks train_pos_supervision/train_pos_no_changes")


def relation_adjacency(
    physical: np.ndarray,
    relation_vocab: Sequence[str],
) -> tuple[dict[str, dict[int, tuple[int, ...]]], set[str]]:
    by_name: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    active: set[str] = set()
    for h, r, t in np.asarray(physical, dtype=np.int64):
        name = str(relation_vocab[int(r)])
        active.add(name)
        by_name[name][int(h)].append(int(t))
    out: dict[str, dict[int, tuple[int, ...]]] = {}
    for name, rows in by_name.items():
        out[name] = {h: tuple(sorted(set(ts))) for h, ts in rows.items()}
    return out, active


def get_relation_view(
    name: str,
    adjacency: Mapping[str, Mapping[int, Sequence[int]]],
) -> dict[int, tuple[int, ...]]:
    if name in adjacency:
        return {int(h): tuple(map(int, ts)) for h, ts in adjacency[name].items()}
    if name.endswith(INVERSE_SUFFIX):
        base = name[: -len(INVERSE_SUFFIX)]
        if base not in adjacency:
            raise KeyError(f"Cannot reconstruct inverse {name!r}; base relation {base!r} absent")
        rev: dict[int, list[int]] = defaultdict(list)
        for h, tails in adjacency[base].items():
            for t in tails:
                rev[int(t)].append(int(h))
        return {h: tuple(sorted(set(ts))) for h, ts in rev.items()}
    raise KeyError(f"Relation view {name!r} absent from physical graph")


def relation_pair_set(view: Mapping[int, Sequence[int]]) -> set[tuple[int, int]]:
    return {(int(h), int(t)) for h, ts in view.items() for t in ts}


def compose_three_hop(
    composition: Sequence[str],
    adjacency: Mapping[str, Mapping[int, Sequence[int]]],
) -> dict[int, tuple[int, ...]]:
    if len(composition) != 3:
        raise ValueError(f"Expected a three-hop composition, got {composition}")
    a1 = get_relation_view(str(composition[0]), adjacency)
    a2 = get_relation_view(str(composition[1]), adjacency)
    a3 = get_relation_view(str(composition[2]), adjacency)
    out: dict[int, tuple[int, ...]] = {}
    for h in sorted(a1):
        endpoints: set[int] = set()
        for x in a1[h]:
            for y in a2.get(int(x), ()):
                endpoints.update(map(int, a3.get(int(y), ())))
        if endpoints:
            out[int(h)] = tuple(sorted(endpoints))
    return out


def rotate_and_cap(
    endpoint_map: Mapping[int, Sequence[int]],
    *,
    cap: int,
    seed: int,
    dependency_id: str,
) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for h in sorted(endpoint_map):
        values = list(sorted(set(map(int, endpoint_map[h]))))
        if values:
            offset = stable_int(seed, dependency_id, h, "canonical_direct") % len(values)
            values = values[offset:] + values[:offset]
            if cap > 0:
                values = values[:cap]
            out[int(h)] = tuple(values)
    return out


def endpoint_rows(endpoint_map: Mapping[int, Sequence[int]]) -> np.ndarray:
    rows = [(int(h), int(t)) for h in sorted(endpoint_map) for t in endpoint_map[h]]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def semantic_hash(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_names: Sequence[str],
    num_entities: int,
) -> str:
    h = hashlib.sha256()
    h.update(tensor_sha256(edge_index).encode())
    h.update(tensor_sha256(edge_type).encode())
    h.update(json_sha256(list(relation_names)).encode())
    h.update(str(int(num_entities)).encode())
    return h.hexdigest()


def physical_hash(
    physical: np.ndarray,
    relation_vocab: Sequence[str],
    num_entities: int,
) -> str:
    x = canonical_triples(physical)
    h = hashlib.sha256()
    h.update(array_sha256(x).encode())
    h.update(json_sha256(list(map(str, relation_vocab))).encode())
    h.update(str(int(num_entities)).encode())
    return h.hexdigest()


def supervision_hash(train: np.ndarray, val: np.ndarray, test: np.ndarray, entity_vocab, relation_vocab) -> str:
    h = hashlib.sha256()
    for arr in (train, val, test):
        h.update(array_sha256(np.asarray(arr, dtype=np.int64)).encode())
    h.update(json_sha256(list(map(str, entity_vocab))).encode())
    h.update(json_sha256(list(map(str, relation_vocab))).encode())
    return h.hexdigest()


def set_deterministic(seed: int, strict: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(tensor_sha256(state[name]).encode())
    return h.hexdigest()


def gradient_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        if p.grad is None:
            h.update(b"NONE")
        else:
            h.update(tensor_sha256(p.grad).encode())
    return h.hexdigest()


def capture_rng_state(np_rng: np.random.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": np_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any], np_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    np_rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def model_parameter_bytes(model: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.replace(" ", ",").split(",") if x.strip()]
