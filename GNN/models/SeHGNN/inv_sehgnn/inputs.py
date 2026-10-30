"""Turn a compiled channel bundle into SeHGNN's model inputs.

IMDb (features on every type): dense channel features ``R_k @ X_{end}`` and dense label channels
``R_c @ onehot(train labels)`` — SeHGNN's stock dense path (one projection matrix per channel).
Freebase (featureless): each channel stays a sparse selector ``R_k [targets × n_end]`` applied to a
learned per-type embedding inside the model (stock sparse path); ``--sparse-backend`` picks
``torch_sparse.SparseTensor`` (stock) or torch CSR + ``torch.sparse.mm``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch

from .channels import Bundle


class SparseRows:
    """A [targets × n_end] CSR selector with row slicing on the requested backend."""

    def __init__(self, R: sp.csr_matrix, backend: str, device):
        self.backend = backend; self.device = device; self.shape = R.shape
        if backend == "torch_sparse":
            from torch_sparse import SparseTensor
            coo = R.tocoo()
            self.t = SparseTensor(row=torch.from_numpy(coo.row.astype(np.int64)), col=torch.from_numpy(coo.col.astype(np.int64)),
                                  value=torch.from_numpy(coo.data.astype(np.float32)), sparse_sizes=R.shape, is_sorted=False).coalesce().to(device)
        elif backend == "torch_csr":
            self.t = torch.sparse_csr_tensor(torch.from_numpy(R.indptr.astype(np.int64)), torch.from_numpy(R.indices.astype(np.int64)),
                                             torch.from_numpy(R.data.astype(np.float32)), size=R.shape).to(device)
        else:
            raise ValueError(backend)

    def rows(self, idx: torch.Tensor):
        if self.backend == "torch_sparse":
            return self.t.index_select(0, idx.to(self.device))
        # CSR row gather: build the sub-matrix from crow/col/values (deterministic, no atomics)
        crow, col, val = self.t.crow_indices(), self.t.col_indices(), self.t.values()
        idx = idx.to(self.device)
        starts, ends = crow[idx], crow[idx + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return torch.sparse_csr_tensor(torch.zeros(len(idx) + 1, dtype=torch.int64, device=self.device),
                                           torch.zeros(0, dtype=torch.int64, device=self.device), torch.zeros(0, device=self.device),
                                           size=(len(idx), self.shape[1]))
        rep = torch.repeat_interleave(torch.arange(len(idx), device=self.device), lens)
        offs = torch.arange(total, device=self.device) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        src = torch.repeat_interleave(starts, lens) + offs
        new_crow = torch.zeros(len(idx) + 1, dtype=torch.int64, device=self.device); new_crow[1:] = torch.cumsum(lens, 0)
        del rep
        return torch.sparse_csr_tensor(new_crow, col[src], val[src], size=(len(idx), self.shape[1]))


def build_inputs(b: Bundle, device, *, keys: Iterable[str] | None = None, label_keys: Iterable[str] | None = None,
                 sparse_backend: str = "torch_sparse", feature_dtype=torch.float32):
    """Return (feats, label_feats, data_size, mask). ``keys`` / ``label_keys`` may be a union vocabulary
    (augmentation): channels the bundle lacks get structural zeros and mask 0."""
    keys = list(keys) if keys is not None else list(b.keys)
    label_keys = list(label_keys) if label_keys is not None else list(label_channel_keys(b))
    present = set(b.keys); lpresent = set(label_channel_keys(b))
    channels = b.channels()
    feats, label_feats, data_size = {}, {}, {}
    mask = torch.tensor([1.0 if k in present else 0.0 for k in sorted(keys)] + [1.0 if k in lpresent else 0.0 for k in sorted(label_keys)])
    T = len(b.targets)
    if b.features is not None:                                                       # dense path (IMDb)
        # channel feature = R_k @ X_end as a dense float32 GEMM on the target device (cuBLAS is
        # deterministic under CUBLAS_WORKSPACE_CONFIG; identical R and X bytes give identical outputs)
        X = {t: torch.from_numpy(b.features[t].astype(np.float32)).to(device) for t in b.features}
        def R_dense(i):   # transient: one dense [targets x n_end] selector at a time on the device
            return torch.from_numpy(channels[b.keys[i]].toarray().astype(np.float32)).to(device)
        for k in keys:
            if k in present:
                feats[k] = (R_dense(b.keys.index(k)) @ X[b.end_type[k]]).to(feature_dtype)
            else:
                dim = X[end_type_from_key(k, b)].shape[1]
                feats[k] = torch.zeros(T, dim, device=device, dtype=feature_dtype)
            data_size[k] = int(feats[k].shape[1])
        y1 = torch.zeros((T, b.num_classes), dtype=torch.float32, device=device)
        tr = torch.as_tensor(b.split["train_idx"], device=device); y1[tr, torch.as_tensor(b.labels[b.split["train_idx"]], device=device)] = 1.0
        for k in label_keys:
            if k in lpresent:
                # SeHGNN label propagation: a training node's own label is never its own input feature
                # (SeHGNN removes the diagonal of every label-propagation matrix; the 0-hop 'Y'
                # channel is therefore dropped altogether).
                R = R_dense(b.label_channels[k]); R.fill_diagonal_(0.0)
                label_feats[k] = (R @ y1).to(feature_dtype)
            else:
                label_feats[k] = torch.zeros(T, b.num_classes, device=device, dtype=feature_dtype)
    else:                                                                            # sparse path (Freebase)
        counts = b.manifest["type_counts"]
        for k in keys:
            t = b.end_type[k] if k in present else int(k[-1])
            if k in present:
                R = channels[k]
            else:
                R = sp.csr_matrix((T, counts[t]), dtype=np.float32)
            feats[k] = SparseRows(R, sparse_backend, device)
            data_size[str(t)] = int(counts[t])
        if label_keys:
            raise ValueError("label channels are not defined for the featureless sparse path")
    return feats, label_feats, data_size, mask


def label_channel_keys(b: Bundle) -> list[str]:
    """Label channels of a bundle minus the 0-hop self-label channel (see build_inputs)."""
    return [k for k, i in b.label_channels.items() if len(b.manifest["channels"][i]["node_types"]) > 1]


def end_type_from_key(k: str, b: Bundle) -> int:
    letters = {"M": 0, "D": 1, "A": 2, "L": 3, "Y": 4}
    return letters[k[-1]] if k[-1] in letters else int(k[-1])


def batch_inputs(feats, label_feats, idx: torch.Tensor):
    fb = {k: (v.rows(idx) if isinstance(v, SparseRows) else v[idx]) for k, v in feats.items()}
    lb = {k: v[idx] for k, v in label_feats.items()}
    return fb, lb
