#!/usr/bin/env python3
"""Audit structural and one-step neural invariance across Freebase variants."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from freebase_block_common import load_json, set_all_seeds, write_json
from freebase_conditional_rgcn import (
    BlockConditionalGraph,
    FreebaseBlockConditionalRGCN,
    build_two_layer_blocks,
)


def max_tensor_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def compare_blocks(a, b) -> dict[str, bool]:
    return {
        "src_nodes_equal": bool(np.array_equal(a.src_nodes, b.src_nodes)),
        "dst_nodes_equal": bool(np.array_equal(a.dst_nodes, b.dst_nodes)),
        "src_index_equal": bool(torch.equal(a.src_index, b.src_index)),
        "dst_index_equal": bool(torch.equal(a.dst_index, b.dst_index)),
        "edge_type_equal": bool(torch.equal(a.edge_type, b.edge_type)),
        "dst_in_src_equal": bool(torch.equal(a.dst_in_src, b.dst_in_src)),
        "relation_ptr_equal": bool(torch.equal(a.relation_ptr, b.relation_ptr)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    roots = [Path(x).resolve() for x in args.preprocessed_dirs]
    if len(roots) < 2:
        raise ValueError("At least two preprocessed directories are required")
    graphs = [BlockConditionalGraph(root, mmap=True) for root in roots]
    hashes = [load_json(root / "model_input_hashes.json") for root in roots]
    provenance = [load_json(root / "raw_graph_provenance.json") for root in roots]
    catalogs = [load_json(root / "relation_catalog.json") for root in roots]

    combined = [x["combined_model_input_hash"] for x in hashes]
    raw_link_hashes = [x["raw_link_sha256"] for x in provenance]
    if len(set(combined)) != 1:
        raise AssertionError(f"Model-input hashes differ: {combined}")
    if len(set(raw_link_hashes)) < 2:
        raise AssertionError("Raw graph link hashes are not distinct")

    reference_base = provenance[0]["base_relation_fingerprints"]
    for index, current in enumerate(provenance[1:], 1):
        if current["base_relation_fingerprints"] != reference_base:
            raise AssertionError(f"Base relation fingerprints differ for input {index}")

    relation_audit: list[dict[str, Any]] = []
    ref_relations = catalogs[0]["relations"]
    for position, reference in enumerate(ref_relations):
        row = {
            "relation": reference["name"],
            "semantic_pair_sha256": reference["semantic_pair_sha256"],
            "source_block_sha256": reference["source_block_sha256"],
            "counts": {},
            "passed": True,
        }
        for root, catalog in zip(roots, catalogs):
            current = catalog["relations"][position]
            checks = {
                "name": current["name"] == reference["name"],
                "pair_hash": current["semantic_pair_sha256"] == reference["semantic_pair_sha256"],
                "block_hash": current["source_block_sha256"] == reference["source_block_sha256"],
                "pair_count": current["num_semantic_pairs"] == reference["num_semantic_pairs"],
            }
            row["counts"][root.name] = int(current["num_semantic_pairs"])
            row["passed"] = bool(row["passed"] and all(checks.values()))
        if not row["passed"]:
            raise AssertionError(f"Block-local semantic relation mismatch: {row}")
        relation_audit.append(row)

    # Use the same aligned target batch and the same deterministic sampled blocks.
    batch_local = np.asarray(graphs[0].splits["train_idx"][: args.audit_batch_size], dtype=np.int64)
    if batch_local.size == 0:
        raise ValueError("Empty training split")
    target_global = np.asarray(graphs[0].target_nodes[batch_local], dtype=np.int64)
    labels = torch.as_tensor(np.asarray(graphs[0].labels[batch_local]), dtype=torch.long)

    set_all_seeds(args.seed, deterministic=True)
    template = FreebaseBlockConditionalRGCN(
        num_nodes=graphs[0].num_nodes,
        num_node_types=graphs[0].num_node_types,
        num_relations=graphs[0].num_relations,
        num_classes=graphs[0].num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_bases=min(args.num_bases, graphs[0].num_relations),
        dropout=0.0,
    )
    initial_state = copy.deepcopy(template.state_dict())

    reference_payload: dict[str, Any] | None = None
    neural_rows: list[dict[str, Any]] = []
    for root, graph in zip(roots, graphs):
        block1, block2 = build_two_layer_blocks(
            graph,
            target_global,
            fanout_layer1=args.fanout_layer1,
            fanout_layer2=args.fanout_layer2,
            sampling_seed=args.seed,
        )
        model = FreebaseBlockConditionalRGCN(
            num_nodes=graph.num_nodes,
            num_node_types=graph.num_node_types,
            num_relations=graph.num_relations,
            num_classes=graph.num_classes,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_bases=min(args.num_bases, graph.num_relations),
            dropout=0.0,
        )
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph, block1, block2, torch.device("cpu"))
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        optimizer.step()
        updated = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        current = {
            "block1": block1,
            "block2": block2,
            "logits": logits.detach().clone(),
            "loss": loss.detach().clone(),
            "grads": grads,
            "updated": updated,
        }
        row: dict[str, Any] = {"preprocessed_dir": str(root)}
        if reference_payload is None:
            row.update(
                {
                    "is_reference": True,
                    "max_logit_difference": 0.0,
                    "loss_difference": 0.0,
                    "max_gradient_difference": 0.0,
                    "max_updated_parameter_difference": 0.0,
                }
            )
            reference_payload = current
        else:
            b1 = compare_blocks(reference_payload["block1"], block1)
            b2 = compare_blocks(reference_payload["block2"], block2)
            if not all(b1.values()) or not all(b2.values()):
                raise AssertionError(f"Sampled blocks differ for {root}: {b1}, {b2}")
            max_grad = max(
                max_tensor_difference(reference_payload["grads"][name], grads[name])
                for name in reference_payload["grads"]
            )
            max_update = max(
                max_tensor_difference(reference_payload["updated"][name], updated[name])
                for name in reference_payload["updated"]
            )
            row.update(
                {
                    "is_reference": False,
                    "block1_checks": b1,
                    "block2_checks": b2,
                    "max_logit_difference": max_tensor_difference(reference_payload["logits"], logits),
                    "loss_difference": max_tensor_difference(reference_payload["loss"], loss),
                    "max_gradient_difference": max_grad,
                    "max_updated_parameter_difference": max_update,
                }
            )
            for key in (
                "max_logit_difference",
                "loss_difference",
                "max_gradient_difference",
                "max_updated_parameter_difference",
            ):
                if row[key] > args.tolerance:
                    raise AssertionError(f"{key}={row[key]} exceeds {args.tolerance} for {root}")
        neural_rows.append(row)

    output = {
        "format_version": "freebase_rgcn_block_invariance_audit_v1",
        "passed": True,
        "preprocessed_dirs": [str(x) for x in roots],
        "raw_graph_hashes_are_distinct": True,
        "combined_model_input_hash": combined[0],
        "base_relation_fingerprints_equal": True,
        "block_local_relation_audit": relation_audit,
        "neural_one_step_audit": neural_rows,
        "tolerance": args.tolerance,
    }
    if args.output:
        write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preprocessed_dirs", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument("--audit-batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--fanout-layer1", type=int, default=4)
    parser.add_argument("--fanout-layer2", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
