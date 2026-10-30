#!/usr/bin/env python3
"""WordNet invariant arm with inverse / shortcut relations as CONTEXT ONLY.

Drives the unchanged ``run_cmpnn_lp.py`` but restricts the WordNet *semantic* program to
the 11 base relations (``base::*``): the ``inverse::`` relations are what CMPNN's own
``undirected(add_inverse=True)`` reconstructs, and every ``dependency::`` shortcut is a
3-hop composition of base relations that a 6-layer path model already reaches. Per the
compiler's audit the base-relation edge set is identical in all four variants (hash
``73bf23174f47`` — and equal to the ``no_changes`` physical graph), so the arm stays
exactly invariant; what changes is that the shortcut edges, which are derived from the
training triples, no longer leak the masked query (WordNet leak
audit: 96.9 % of training queries vs 35.3 % of held-out queries carry a
shortcut consistent with the true tail).

    PYTHONPATH=. python run_cmpnn_lp_wordnet_context.py --dataset WORDNET --task link_prediction \
        --variant no_changes --graph semantic --output-root <root>/invariant ...

``--graph physical`` is passed through untouched (originals / universal are not affected).
"""
from __future__ import annotations

import dataclasses
import sys

import torch

import run_cmpnn_lp as runner  # noqa: E402
from inv_cmpnn import bundles  # noqa: E402

BASE_PREFIX = "base::"


def load_bundle_base_only(path):
    b = bundles.load_bundle(path)
    if b.dataset != "WORDNET":
        return b
    names = list(b.semantic_relation_names)
    base_ids = [i for i, n in enumerate(names) if n.startswith(BASE_PREFIX)]
    n_task = int(b.metadata["num_task_relations"])
    if base_ids != list(range(n_task)):
        raise RuntimeError(f"base:: relations are not the first {n_task} semantic ids: {base_ids[:5]}...")
    sem = b.semantic_edges
    keep = sem[:, 2] < n_task
    sem_b = sem[keep]
    sem_b = sem_b[bundles.canonical_edge_order(sem_b[:, 0], sem_b[:, 1], sem_b[:, 2])].contiguous()
    for t in (b.train_pos, b.val_pos, b.test_pos):
        if int(t[:, 1].max()) >= n_task:
            raise RuntimeError("supervision references a non-base semantic relation")
    meta = dict(b.metadata)
    meta["semantic_task_relation_ids"] = list(range(n_task))
    meta["task_relation_index_of_semantic_id"] = list(range(n_task))
    meta["semantic_program"] = "base_only_context_shortcuts"
    meta["semantic_full_edges"] = int(sem.shape[0])
    nb = dataclasses.replace(b, semantic_edges=sem_b, semantic_relation_names=names[:n_task], metadata=meta)
    print(f"[context] WordNet semantic program restricted to {n_task} base relations: "
          f"{sem.shape[0]} -> {sem_b.shape[0]} edges (inverse:: / dependency:: are context only); "
          f"semantic hash {nb.semantic_hash()[:12]}", flush=True)
    return nb


def main():
    runner.load_bundle = load_bundle_base_only
    print(f"[run_cmpnn_lp_wordnet_context] wrapping {runner.__file__}", flush=True)
    runner.main()


if __name__ == "__main__":
    main()
