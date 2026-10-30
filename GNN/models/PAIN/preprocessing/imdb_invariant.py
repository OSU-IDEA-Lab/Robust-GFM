"""Compile all four IMDb physical variants into invariant PAIN programs.

Each source graph remains physically different.  The compiler independently
matches its Movie-Link conditional contexts, projects the licensed semantic
edges, and only then enumerates every rooted simple PAIN path with 0..3 edges.
The command fails unless IMDb1-4 produce byte-identical semantic graphs and
path programs.

Run from the repository root:

    python -m preprocessing.imdb_invariant
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from pain_nc import paths
from pain_nc.data import PATH_FIELDS
from pain_nc.experiment import atomic_torch_save, atomic_write_json
from preprocessing.imdb_node_classification import (
    DEFAULT_SCHEMA,
    SCHEMAS,
    PATH_LENGTH,
    edge_type_names,
    base_variant_edges,
    build_contract,
    compile_invariant_edges,
    enumerate_pain_paths,
    graph_tensors,
    save_shared,
    validate_contract,
    validate_paths,
)


INVARIANT_VARIANTS = ("v1", "v2", "v3", "v4")


def tensor_hash(tensors: Iterable[torch.Tensor]) -> str:
    """Hash tensor values together with dtype and shape."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def shared_contract_hash(contract) -> str:
    tensors = [
        contract.features,
        torch.from_numpy(contract.labels),
        torch.from_numpy(contract.node_types),
    ]
    tensors.extend(
        torch.from_numpy(contract.split_indices[name])
        for name in ("train", "val", "test")
    )
    return tensor_hash(tensors)


def preprocess(
    csv_path: Path,
    output_dir: Path,
    *,
    validate: bool = True,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate any older PASS marker before touching shared/path artifacts.
    # An interrupted rebuild must never look trainable to the benchmark.
    atomic_write_json(
        {
            "dataset": "IMDB",
            "mode": "invariant",
            "graph_schema": schema,
            "status": "BUILDING",
        },
        output_dir / "metadata.json",
    )
    contract = build_contract(csv_path, schema=schema)
    validate_contract(contract)
    save_shared(contract, output_dir, csv_path)
    physical_variants = base_variant_edges(contract.movies, contract.maps)

    physical_hashes: dict[str, str] = {}
    semantic_hashes: dict[str, str] = {}
    compiled_edges: dict[str, set[tuple[int, int]]] = {}
    context_audits: dict[str, dict[str, int | str]] = {}
    physical_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # First prove that every physical graph independently licenses the same
    # semantic graph.  This pass is edge-only and therefore cheap.
    for variant in INVARIANT_VARIANTS:
        physical = physical_variants[variant]
        raw_edge_index, raw_edge_type, _raw_adj, _raw_lookup = graph_tensors(
            physical, contract.node_types
        )
        semantic, audit = compile_invariant_edges(
            physical, variant, contract.node_types
        )
        edge_index, edge_type, _adjacency, _lookup = graph_tensors(
            semantic, contract.node_types
        )
        physical_hashes[variant] = tensor_hash((raw_edge_index, raw_edge_type))
        semantic_hashes[variant] = tensor_hash((edge_index, edge_type))
        compiled_edges[variant] = semantic
        context_audits[variant] = audit
        physical_tensors[variant] = (raw_edge_index, raw_edge_type)

    if len(set(physical_hashes.values())) != len(INVARIANT_VARIANTS):
        raise RuntimeError(
            "IMDb invariant inputs are not four distinct physical graphs: "
            f"{physical_hashes}"
        )
    if len(set(semantic_hashes.values())) != 1:
        raise RuntimeError(
            "IMDb1-4 did not compile to one semantic graph. This usually "
            "means a physical variant is missing information (especially "
            f"IMDb4 Actor1 edges): {semantic_hashes}"
        )

    universal_edges = physical_variants["universal"]
    universal_edge_index, universal_edge_type, _adj, _lookup = graph_tensors(
        universal_edges, contract.node_types
    )
    universal_hash = tensor_hash((universal_edge_index, universal_edge_type))
    semantic_hash = next(iter(semantic_hashes.values()))
    semantic_edges = compiled_edges[INVARIANT_VARIANTS[0]]

    # The semantic closure must never LOSE information relative to the union graph.
    missing = universal_edges - semantic_edges
    if missing:
        raise RuntimeError(
            f"The compiled IMDb semantic closure is missing {len(missing)} edges "
            "present in the physical four-variant union; inspect the switching "
            "contract before training"
        )
    extra = sorted(semantic_edges - universal_edges)

    # On the year-free graph the closure is exactly the union.  On the year graph it is
    # a strict SUPERSET: movie_to_year and movie_to_link are present in every variant,
    # so link -> movie -> year is derivable identically everywhere and the compiler
    # projects link-year, which no physical variant carries.  That is the method adding
    # structure, not a defect -- docs/DATASET_STATISTICS.md §1 records the same 4177-edge
    # surplus for RGCN.  It is recorded here rather than hidden.
    link_year_pairs = {(3, 4), (4, 3)}
    unexpected = [
        edge
        for edge in extra
        if (
            int(contract.node_types[edge[0]]),
            int(contract.node_types[edge[1]]),
        )
        not in link_year_pairs
    ]
    if unexpected:
        raise RuntimeError(
            f"The compiled IMDb semantic closure adds {len(unexpected)} edges that are "
            "not the expected projected imdb_link-year relation; inspect the switching "
            f"contract before training (first: {unexpected[:5]})"
        )
    if extra and contract.schema != "year":
        raise RuntimeError(
            "Only the year schema may produce a semantic closure larger than the union"
        )

    summary: dict = {
        "dataset": "IMDB",
        "mode": "invariant",
        "compiler": "imdb_movie_link_conditional_semantic_closure",
        "source_variants": list(INVARIANT_VARIANTS),
        "path_length": PATH_LENGTH,
        "path_semantics": "all_rooted_simple_paths_up_to_L_after_semantic_projection",
        "path_sampling": "none",
        "artifact_path_order": "decreasing_length_then_root",
        "runtime_path_grouping": "root_stable",
        "shared_contract_sha256": shared_contract_hash(contract),
        "physical_graph_sha256": physical_hashes,
        "semantic_graph_sha256": semantic_hashes,
        "graph_schema": contract.schema,
        "universal_graph_sha256": universal_hash,
        "semantic_equals_universal": not extra,
        "semantic_contains_universal": True,
        "num_universal_undirected_edges": len(universal_edges),
        "num_semantic_undirected_edges": len(semantic_edges),
        # Projected imdb_link-year edges: present in the semantic program, in no
        # physical variant and therefore not in the union graph.
        "semantic_only_undirected_edges": len(extra),
        "semantic_only_relation": "imdb_link-year" if extra else None,
        "variants": {},
    }

    path_hashes: dict[str, str] = {}
    for variant in INVARIANT_VARIANTS:
        artifact_name = f"invariant_{variant}"
        print(
            f"\n=== IMDB invariant PAIN-NC | source={variant} | "
            f"L={PATH_LENGTH} ===",
            flush=True,
        )
        semantic = compiled_edges[variant]
        edge_index, edge_type, adjacency, edge_lookup = graph_tensors(
            semantic, contract.node_types
        )
        paths, counts = enumerate_pain_paths(adjacency, edge_lookup)
        if validate:
            validate_paths(paths, adjacency)
        path_hash = tensor_hash(
            (edge_index, edge_type, *(paths[name] for name in PATH_FIELDS))
        )
        path_hashes[variant] = path_hash
        raw_edge_index, raw_edge_type = physical_tensors[variant]
        details = {
            "dataset": "IMDB",
            "variant": artifact_name,
            "source_variant": variant,
            "mapping_mode": "conditional_semantic_path_compile",
            "compiler": summary["compiler"],
            "path_length": PATH_LENGTH,
            "path_semantics": summary["path_semantics"],
            "path_sampling": "none",
            "artifact_path_order": summary["artifact_path_order"],
            "num_nodes": len(contract.node_types),
            "num_physical_undirected_edges": len(physical_variants[variant]),
            "num_semantic_undirected_edges": len(semantic),
            # Compatibility key consumed by the shared IMDb benchmark.  The
            # model-facing graph is the semantic graph, not the retained raw
            # physical provenance graph.
            "num_undirected_edges": len(semantic),
            "num_directed_edges": int(edge_index.shape[1]),
            "paths_exact_length_0_1_2_3": list(counts),
            "num_paths": sum(counts),
            "edge_type_names": list(edge_type_names(contract.schema)),
            "physical_graph_sha256": physical_hashes[variant],
            "semantic_graph_sha256": semantic_hashes[variant],
            "selected_path_program_sha256": path_hash,
            "shared_contract_sha256": summary["shared_contract_sha256"],
            "context_audit": context_audits[variant],
        }
        payload = {
            # Raw tensors are retained only as provenance. PAIN consumes the
            # canonical semantic tensors and their derived paths below.
            "physical_edge_index": raw_edge_index,
            "physical_edge_type": raw_edge_type,
            "edge_index": edge_index,
            "edge_type": edge_type,
            **paths,
            "meta": details,
        }
        atomic_torch_save(
            payload, output_dir / f"{artifact_name}_L{PATH_LENGTH}.pt"
        )
        summary["variants"][artifact_name] = details
        print(
            f"physical_edges={len(physical_variants[variant]):,} "
            f"semantic_edges={len(semantic):,} paths={sum(counts):,} "
            f"path_hash={path_hash[:12]}",
            flush=True,
        )
        del paths, payload, edge_index, edge_type, adjacency, edge_lookup
        gc.collect()

    if len(set(path_hashes.values())) != 1:
        raise RuntimeError(
            "IMDb1-4 semantic graphs matched but PAIN path programs did not: "
            f"{path_hashes}"
        )
    summary["selected_path_program_sha256"] = path_hashes
    summary["status"] = "PASS"
    atomic_write_json(summary, output_dir / "metadata.json")
    print(
        "\nIMDb invariant preprocessing PASS: physical hashes differ; "
        "semantic graph and exact PAIN path hashes match across v1-v4.",
        flush=True,
    )
    print(f"Saved invariant artifacts to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=paths.imdb_csv(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.preprocessed("IMDB_invariant"),
    )
    parser.add_argument(
        "--schema",
        choices=SCHEMAS,
        default=DEFAULT_SCHEMA,
        help=(
            "'year' (default) compiles the IMDb year graph; 'legacy' "
            "compiles the original year-free graph. Must match the schema used for "
            "the ordinary variants."
        ),
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate every materialized semantic path (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        args.csv, args.output_dir, validate=args.validate, schema=args.schema
    )
