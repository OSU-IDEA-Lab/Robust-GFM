#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_RGCN_lp_wordnet_structural import WordNetStructuralRGCN
from run_WORDNET_rgcn_structural import DEFAULTS, build_model, build_optimizer, objective_spec
from wordnet_structural_common import (
    BUDGETS,
    MODES,
    VARIANTS,
    atomic_json_dump,
    gradient_hash,
    set_deterministic,
    state_dict_hash,
    tensor_sha256,
)
from wordnet_structural_eval import sample_neg_train


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def one_step(data, args, device):
    set_deterministic(args.seed, strict=True)
    rng = np.random.default_rng(args.seed)
    model = build_model(args, data, device)
    # build_optimizer expects these attributes only.
    optimizer = build_optimizer(model, args)
    edge_index = data["semantic_edge_index"].to(device)
    edge_type = data["semantic_edge_type"].to(device)
    selected_train, _ = objective_spec(data, args.train_objective)
    pos_np = selected_train.cpu().numpy().astype(np.int64)
    permutation = rng.permutation(len(pos_np))
    pos_np = pos_np[permutation]
    batch_size = args.batch_size if args.batch_size > 0 else len(pos_np)
    pos_np = pos_np[:batch_size]
    neg_np = sample_neg_train(pos_np, int(data["num_entities"]), args.neg_per_pos, rng)
    pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
    neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)

    initial = state_dict_hash(model.state_dict())
    optimizer.zero_grad(set_to_none=True)
    embeddings = model.encode(edge_index, edge_type, training=True)
    pos_scores = model.score_triples(embeddings, pos)
    neg_scores = model.score_triples(embeddings, neg)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    grad = gradient_hash(model)
    optimizer.step()
    return {
        "initial_model_hash": initial,
        "positive_batch_hash": tensor_sha256(pos.cpu()),
        "negative_batch_hash": tensor_sha256(neg.cpu()),
        "encoded_hash": tensor_sha256(embeddings.detach().cpu()),
        "score_hash": tensor_sha256(scores.detach().cpu()),
        "scores": scores.detach().cpu(),
        "loss": float(loss.detach().cpu()),
        "gradient_hash": grad,
        "post_step_model_hash": state_dict_hash(model.state_dict()),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate exact WordNet conditional-RGCN invariance")
    ap.add_argument("--data-root", default="data/preprocessed/RGCN_structural/WORDNET/link_prediction")
    ap.add_argument("--budget", choices=BUDGETS, default="reduced")
    ap.add_argument("--mode", choices=MODES, default="intersection")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", default="")
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    ap.add_argument("--num-bases", type=int, default=DEFAULTS["num_bases"])
    ap.add_argument("--edge-dropout-other", type=float, default=DEFAULTS["edge_dropout_other"])
    ap.add_argument("--root-dropout-loop", type=float, default=DEFAULTS["root_dropout_loop"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--neg-per-pos", type=int, default=DEFAULTS["neg_per_pos"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--edge-chunk-size", type=int, default=DEFAULTS["edge_chunk_size"])
    ap.add_argument("--train-objective", choices=("task_only", "semantic_all"), default="semantic_all")
    args = ap.parse_args()

    variants = [x.strip() for x in args.variants.replace(" ", ",").split(",") if x.strip()]
    device = torch.device(args.device)
    records, items = [], []
    for variant in variants:
        path = Path(args.data_root) / args.budget / args.mode / variant / "graph.pt"
        data = load(path)
        items.append((variant, data))
        records.append({
            "variant": variant,
            "physical_hash": data["physical_hash"],
            "semantic_hash": data["semantic_hash"],
            "supervision_hash": data["supervision_hash"],
            "num_physical_edges": int(data["physical_edge_type"].numel()),
            "num_semantic_edges": int(data["semantic_edge_type"].numel()),
            "num_semantic_relations": len(data["semantic_relation_names"]),
        })

    if len({r["physical_hash"] for r in records}) < 2:
        raise SystemExit("FAIL: raw physical graphs are identical")
    if len({r["semantic_hash"] for r in records}) != 1:
        raise SystemExit(f"FAIL: semantic hashes differ: {records}")
    if len({r["supervision_hash"] for r in records}) != 1:
        raise SystemExit(f"FAIL: supervision hashes differ: {records}")

    steps = [(variant, one_step(data, args, device)) for variant, data in items]
    base_variant, base = steps[0]
    pair_checks = []
    for variant, current in steps[1:]:
        max_diff = float((base["scores"] - current["scores"]).abs().max())
        check = {
            "variant_a": base_variant,
            "variant_b": variant,
            "max_abs_score_diff": max_diff,
            "loss_diff": abs(base["loss"] - current["loss"]),
            "initial_model_hash_equal": base["initial_model_hash"] == current["initial_model_hash"],
            "positive_batch_hash_equal": base["positive_batch_hash"] == current["positive_batch_hash"],
            "negative_batch_hash_equal": base["negative_batch_hash"] == current["negative_batch_hash"],
            "encoded_hash_equal": base["encoded_hash"] == current["encoded_hash"],
            "score_hash_equal": base["score_hash"] == current["score_hash"],
            "gradient_hash_equal": base["gradient_hash"] == current["gradient_hash"],
            "post_step_model_hash_equal": base["post_step_model_hash"] == current["post_step_model_hash"],
        }
        required = [v for k, v in check.items() if k.endswith("_equal")]
        if max_diff > args.tolerance or check["loss_diff"] > args.tolerance or not all(required):
            raise SystemExit(f"FAIL one-step invariance: {check}")
        pair_checks.append(check)

    report = {
        "status": "PASS",
        "budget": args.budget,
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "train_objective": args.train_objective,
        "structural_records": records,
        "pair_checks": pair_checks,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        atomic_json_dump(report, args.report)


if __name__ == "__main__":
    main()
