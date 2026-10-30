from __future__ import annotations

from typing import Iterable, Sequence
import math
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import kendalltau


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    truth = y.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
    }


def binary_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= threshold).astype(np.int64)
    out = {
        "auc": float(roc_auc_score(labels, prob)) if len(np.unique(labels)) > 1 else float("nan"),
        "ap": float(average_precision_score(labels, prob)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
    }
    return out


def ranking_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray, ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    if neg_scores.ndim != 2:
        raise ValueError("neg_scores must have shape [num_queries, num_negatives]")
    # Ties use the AVERAGED rank (1 + #strictly_better + #tied/2). This side previously used the PESSIMISTIC rank while the baselines used an optimistic stable sort, so the invariant arm was scored under a stricter rule than the models it is compared against. All three conventions agree exactly when scores are distinct.
    better = (neg_scores > pos_scores[:, None]).sum(axis=1)
    tied = (neg_scores == pos_scores[:, None]).sum(axis=1)
    ranks = 1.0 + better + tied / 2.0
    out = {f"hits@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    return out


def full_ranking_metrics(
    ranks: np.ndarray, ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Metrics from precomputed 1-based ranks (full candidate-set protocol)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {f"hits@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["mean_rank"] = float(np.mean(ranks))
    return out


def average_row_kendall(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if scores_a.shape != scores_b.shape:
        raise ValueError("score arrays must have equal shape")
    vals = []
    for a, b in zip(scores_a, scores_b):
        tau, _ = kendalltau(a, b, nan_policy="omit")
        if not np.isnan(tau):
            vals.append(float(tau))
    return float(np.mean(vals)) if vals else float("nan")
