"""Fixed-candidate link prediction, full-type classification and paired invariance."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def average_ranks(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not len(scores) or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite [queries, positive + negatives] scores")
    better = (scores[:, 1:] > scores[:, :1]).sum(1)
    tied = (scores[:, 1:] == scores[:, :1]).sum(1)
    return 1.0 + better + tied / 2.0


def top1_tail(scores, candidates):
    """Resolve score ties by the smallest GLOBAL entity id, never positive position."""
    scores = np.asarray(scores)
    candidates = np.broadcast_to(candidates, scores.shape)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite candidate scores")
    tied = scores == scores.max(1, keepdims=True)
    return np.where(tied, candidates, np.iinfo(np.int64).max).min(1)


def classification_metrics(y_true, y_pred, classes):
    """Macro averages over the declared full class set; zero_division=0."""
    classes = np.asarray(classes)
    if len(np.unique(classes)) != len(classes) or np.any(classes[:-1] >= classes[1:]):
        raise ValueError("Class ids must be unique and sorted")
    if not len(y_true) or not np.isin(y_true, classes).all() or not np.isin(y_pred, classes).all():
        raise ValueError("Empty or out-of-class predictions/labels")
    true = np.searchsorted(classes, y_true)
    pred = np.searchsorted(classes, y_pred)
    count = len(classes)
    correct = true == pred
    tp = np.bincount(true[correct], minlength=count)
    support = np.bincount(true, minlength=count)
    predicted = np.bincount(pred, minlength=count)
    precision = np.divide(tp, predicted, out=np.zeros(count), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(count), where=(support + predicted) != 0)
    accuracy = float(correct.mean())
    return {"accuracy": accuracy, "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()), "micro_f1": accuracy, "macro_f1": float(f1.mean())}


def point_metrics(scores, positives, pool, full_scores):
    ranks = average_ranks(scores)
    result = {"mrr": float(np.mean(1 / ranks)),
              **{f"hits@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 10)}}
    predictions = top1_tail(full_scores, pool)
    result.update(classification_metrics(positives[:, 1], predictions, pool))
    return result


def binary_metrics(scores, candidates=None, positives=None):
    """Lab METRICS.md 3.3: every (query, candidate) pair is a binary instance predicted
    positive at sigmoid(logit) >= 0.5; Precision/Recall are pooled over all pairs.
    Without candidate ids (WordNet's sampled rows), column 0 is the only positive."""
    scores = np.asarray(scores)
    if candidates is None:
        labels = np.zeros(scores.shape, dtype=bool)
        labels[:, 0] = True
    else:
        labels = np.asarray(candidates) == np.asarray(positives)[:, 1:2]
    if not np.array_equal(labels.sum(1), np.ones(len(labels))) or not labels[:, 0].all():
        raise ValueError("Each query needs exactly one positive candidate, in column 0")
    predicted = scores >= 0
    tp = int((predicted & labels).sum())
    fp, fn = int((predicted & ~labels).sum()), int((~predicted & labels).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def mean_tau_b(a, b):
    """Lab METRICS.md 4.1: per-query tau_b averaged over queries where it is defined;
    identical vectors count as 1. Returns (mean, number of excluded queries)."""
    values, undefined = [], 0
    for x, y in zip(a, b):
        if np.array_equal(x, y):
            values.append(1.0)
            continue
        value = float(kendalltau(x, y, variant="b").statistic) if np.ptp(x) and np.ptp(y) else np.nan
        if np.isfinite(value):
            values.append(value)
        else:
            undefined += 1
    if not values:
        raise ValueError("Kendall tau-b is undefined for every query")
    return float(np.mean(values)), undefined


def estimate(point, draws):
    """Bootstrap SD estimates the standard error of the fixed-test-set statistic."""
    if not draws:
        return {"estimate": float(point), "bootstrap_se": None, "ci95": None}
    draws = np.asarray(draws, dtype=float)
    return {"estimate": float(point), "bootstrap_se": float(draws.std(ddof=1)),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def summarize_performance(arrays, bootstrap=1000, seed=0):
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("Bootstrap count must be zero or at least two")
    scores, pos, pool, full = (arrays[k] for k in (
        "scores", "positives", "candidate_pool", "full_type_scores"))
    point = point_metrics(scores, pos, pool, full)
    # Full-type predictions are fixed; bootstrapping resamples query rows only.
    predictions = top1_tail(full, pool)
    ranks = average_ranks(scores)
    samples = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        idx = rng.integers(len(pos), size=len(pos))
        values = {"mrr": float(np.mean(1 / ranks[idx])),
                  **{f"hits@{k}": float(np.mean(ranks[idx] <= k)) for k in (1, 3, 10)}}
        values.update(classification_metrics(pos[idx, 1], predictions[idx], pool))
        for key, value in values.items():
            samples[key].append(value)
    return {key: estimate(value, samples[key]) for key, value in point.items()}


def check_alignment(a, b):
    for key in ("query_index", "positives", "candidates", "candidate_pool"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"Query/candidate alignment differs: {key}")
    for key in ("scores", "full_type_scores"):
        if a[key].shape != b[key].shape:
            raise ValueError(f"Score dimensions differ: {key}")


def compare_predictions(a, b, bootstrap=1000, seed=0):
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("Bootstrap count must be zero or at least two")
    check_alignment(a, b)
    ar, br = average_ranks(a["scores"]), average_ranks(b["scores"])
    agreement = top1_tail(a["scores"], a["candidates"]) == top1_tail(b["scores"], b["candidates"])
    full_agreement = top1_tail(a["full_type_scores"], a["candidate_pool"]) == top1_tail(b["full_type_scores"], b["candidate_pool"])
    tau, undefined = [], 0
    for x, y in zip(a["scores"], b["scores"]):
        if np.array_equal(x, y):
            value = 1.0  # Evaluation-guide convention, including identical constants.
        elif np.ptp(x) == 0 or np.ptp(y) == 0:
            value = 0.0
            undefined += 1
        else:
            value = float(kendalltau(x, y, variant="b").statistic)
            if not np.isfinite(value):
                value = 0.0
                undefined += 1
        tau.append(value)
    vectors = {"prediction_agreement": agreement.astype(float), "tau_b": np.asarray(tau),
               "full_type_prediction_agreement": full_agreement.astype(float),
               "delta_mrr": 1 / br - 1 / ar,
               "hit1_flip_rate": ((ar <= 1) != (br <= 1)).astype(float),
               "hit10_flip_rate": ((ar <= 10) != (br <= 10)).astype(float)}
    samples = {key: [] for key in vectors}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        idx = rng.integers(len(ar), size=len(ar))
        for key, vector in vectors.items():
            samples[key].append(float(vector[idx].mean()))
    return {"num_queries": len(ar), "undefined_tau_queries": undefined,
            "bitwise_identical": bool(np.array_equal(a["scores"], b["scores"])),
            "full_type_bitwise_identical": bool(np.array_equal(a["full_type_scores"], b["full_type_scores"])),
            "max_abs_score_difference": float(np.max(np.abs(a["scores"] - b["scores"]))),
            "metrics": {key: estimate(vector.mean(), samples[key]) for key, vector in vectors.items()}}
