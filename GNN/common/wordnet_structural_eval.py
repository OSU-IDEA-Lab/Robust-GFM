from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import torch

from wordnet_structural_common import stable_int

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = average_precision_score = f1_score = None
    precision_score = recall_score = roc_auc_score = None


def build_filter_dicts(*triple_arrays: np.ndarray):
    tails = defaultdict(set)
    heads = defaultdict(set)
    for arr in triple_arrays:
        for h, r, t in np.asarray(arr, dtype=np.int64):
            tails[(int(h), int(r))].add(int(t))
            heads[(int(r), int(t))].add(int(h))
    return {k: frozenset(v) for k, v in tails.items()}, {k: frozenset(v) for k, v in heads.items()}


def sample_neg_train(pos: np.ndarray, num_entities: int, k: int, rng: np.random.Generator):
    pos = np.asarray(pos, dtype=np.int64)
    n = len(pos)
    total = n * int(k)
    h = np.repeat(pos[:, 0], k)
    r = np.repeat(pos[:, 1], k)
    t = np.repeat(pos[:, 2], k)
    values = rng.integers(0, num_entities, size=total, dtype=np.int64)
    corrupt_head = rng.random(total) < 0.5
    nh = np.where(corrupt_head, values, h)
    nt = np.where(corrupt_head, t, values)
    return np.stack([nh, r, nt], axis=1).astype(np.int64)


def _fixed_candidates(
    *,
    positive: int,
    forbidden: Iterable[int],
    num_entities: int,
    k: int,
    seed_parts: tuple,
) -> np.ndarray:
    forbidden_set = set(map(int, forbidden))
    forbidden_set.discard(int(positive))
    rng = np.random.default_rng(stable_int(*seed_parts))
    chosen: list[int] = []
    used = {int(positive)}
    attempts = 0
    while len(chosen) < k:
        need = max(64, (k - len(chosen)) * 4)
        for value in rng.integers(0, num_entities, size=need, dtype=np.int64):
            v = int(value)
            if v in used or v in forbidden_set:
                continue
            used.add(v)
            chosen.append(v)
            if len(chosen) == k:
                break
        attempts += 1
        if attempts > 1000:
            # Deterministic exhaustive fallback.
            for v in range(num_entities):
                if v not in used and v not in forbidden_set:
                    used.add(v)
                    chosen.append(v)
                    if len(chosen) == k:
                        break
            if len(chosen) < k:
                raise RuntimeError("Not enough filtered negative candidates")
    return np.asarray([int(positive), *chosen], dtype=np.int64)


def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = (logits > 0).astype(np.int64)
    if accuracy_score is None:
        acc = float(np.mean(pred == labels))
        return {"accuracy": acc}
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(labels, logits))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["average_precision"] = float(average_precision_score(labels, logits))
    except Exception:
        out["average_precision"] = float("nan")
    return out


def evaluate_ranking(
    model,
    entity_embs: torch.Tensor,
    positives: np.ndarray,
    *,
    tail_filters: Mapping,
    head_filters: Mapping,
    num_entities: int,
    device: torch.device,
    eval_batch_size: int = 128,
    binary_k: int = 50,
    candidate_seed: int = 1566911444,
    save_candidates: bool = False,
):
    positives = np.asarray(positives, dtype=np.int64).reshape(-1, 3)
    raw_tail, filt_tail, raw_head, filt_head = [], [], [], []
    binary_logits, binary_labels = [], []
    candidate_scores, candidate_ids, query_triples, sides = [], [], [], []

    pos_t = torch.as_tensor(positives, dtype=torch.long, device=device)
    for start in range(0, len(positives), eval_batch_size):
        batch = pos_t[start : start + eval_batch_size]
        with torch.no_grad():
            tail_scores = model.score_tails(entity_embs, batch[:, 0], batch[:, 1]).detach().cpu()
            head_scores = model.score_heads(entity_embs, batch[:, 1], batch[:, 2]).detach().cpu()
        for local in range(len(batch)):
            idx = start + local
            h, r, t = map(int, positives[idx])
            st = tail_scores[local]
            sh = head_scores[local]
            true_t = float(st[t])
            true_h = float(sh[h])
            raw_tail.append(int((st >= true_t).sum().item()))
            raw_head.append(int((sh >= true_h).sum().item()))

            stf = st.clone()
            for other in tail_filters.get((h, r), ()):
                if int(other) != t:
                    stf[int(other)] = float("-inf")
            shf = sh.clone()
            for other in head_filters.get((r, t), ()):
                if int(other) != h:
                    shf[int(other)] = float("-inf")
            filt_tail.append(int((stf >= true_t).sum().item()))
            filt_head.append(int((shf >= true_h).sum().item()))

            tails = _fixed_candidates(
                positive=t,
                forbidden=tail_filters.get((h, r), ()),
                num_entities=num_entities,
                k=binary_k,
                seed_parts=(candidate_seed, "tail", idx, h, r, t),
            )
            t_scores = st[tails].numpy().astype(np.float32)
            binary_logits.extend(t_scores.tolist())
            binary_labels.extend([1] + [0] * binary_k)
            if save_candidates:
                candidate_scores.append(t_scores)
                candidate_ids.append(tails)
                query_triples.append((h, r, t))
                sides.append(0)

            if save_candidates:
                heads = _fixed_candidates(
                    positive=h,
                    forbidden=head_filters.get((r, t), ()),
                    num_entities=num_entities,
                    k=binary_k,
                    seed_parts=(candidate_seed, "head", idx, h, r, t),
                )
                candidate_scores.append(sh[heads].numpy().astype(np.float32))
                candidate_ids.append(heads)
                query_triples.append((h, r, t))
                sides.append(1)

    raw = np.asarray(raw_tail + raw_head, dtype=np.float64)
    filt = np.asarray(filt_tail + filt_head, dtype=np.float64)
    metrics = {
        "raw_MRR": float(np.mean(1.0 / raw)),
        "filtered_MRR": float(np.mean(1.0 / filt)),
        "Hits@1": float(np.mean(filt <= 1)),
        "Hits@3": float(np.mean(filt <= 3)),
        "Hits@10": float(np.mean(filt <= 10)),
        "filtered_mean_rank": float(np.mean(filt)),
        "raw_mean_rank": float(np.mean(raw)),
    }
    metrics.update(_binary_metrics(np.asarray(binary_logits), np.asarray(binary_labels)))
    artifacts = {
        "raw_tail_ranks": np.asarray(raw_tail, dtype=np.int32),
        "filtered_tail_ranks": np.asarray(filt_tail, dtype=np.int32),
        "raw_head_ranks": np.asarray(raw_head, dtype=np.int32),
        "filtered_head_ranks": np.asarray(filt_head, dtype=np.int32),
    }
    if save_candidates:
        artifacts.update({
            "candidate_scores": np.asarray(candidate_scores, dtype=np.float32),
            "candidate_ids": np.asarray(candidate_ids, dtype=np.int64),
            "query_triples": np.asarray(query_triples, dtype=np.int64),
            "side": np.asarray(sides, dtype=np.int8),
        })
    return metrics, artifacts
