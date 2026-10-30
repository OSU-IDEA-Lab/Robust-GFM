from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import os
import random

import numpy as np
import torch


def set_deterministic(seed: int, strict: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if strict:
        # A single CPU reduction thread avoids BLAS/OpenMP accumulation-order drift
        # in bitwise validation. GPU training is unaffected apart from host-side ops.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            torch.set_num_threads(int(os.environ.get("INV_GNN_NUM_THREADS", "1")))
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        value = state[key].detach().cpu().contiguous()
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def gradient_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        if p.grad is None:
            h.update(b"NONE")
        else:
            g = p.grad.detach().cpu().contiguous()
            h.update(g.numpy().tobytes())
    return h.hexdigest()


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def json_dump(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
