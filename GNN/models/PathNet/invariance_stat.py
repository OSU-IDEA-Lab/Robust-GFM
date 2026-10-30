"""Invariance statistic used by summarize_arms.py.

``row_tau`` is the invariance statistic used for every model in docs/results:
mean over rows (test nodes / queries) of Kendall tau-b between the two runs' score vectors,
identical vectors counting 1.
"""
import numpy as np
from scipy.stats import kendalltau


def row_tau(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for x, y in zip(a, b):
        if np.array_equal(x, y): vals.append(1.0); continue
        t = kendalltau(x, y, nan_policy="omit")[0]
        if not np.isnan(t): vals.append(float(t))
    return float(np.mean(vals)) if vals else float("nan")
