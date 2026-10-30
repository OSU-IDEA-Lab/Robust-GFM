#!/usr/bin/env python3
"""
Create four leakage-free WordNet/WN18RR graph variants from official train.txt.

Variants
--------
no_changes/
    Original training triples only.
all_inverse_edges/
    Original training triples plus an explicit inverse for every base triple.
transitive_edges/
    Original training triples plus shortcut relations for forward-only relation
    compositions. No inverse edges are materialized in this variant.
universal_edges/
    Cumulative superset containing original training triples, every base inverse
    edge, every selected forward-only shortcut from transitive_edges, and every
    additionally selected shortcut whose composition may use inverse traversals.

Fast reuse of old output
------------------------
Use --reuse-old-output-dir to reuse the legacy inverse-enabled composition
catalog for the new universal variant. Old augmented edges are NEVER reused.
Every universal candidate is rescored and rematerialized using official
train.txt only. The final universal graph always inherits the selected
forward-only shortcut relations from transitive_edges, even when the reusable
legacy universal catalog contains only inverse-enabled compositions.

The old inverse-enabled catalog cannot define the new forward-only transitive
search space: an exhaustive legacy catalog can still contain zero compositions
without inverse hops. Therefore, transitive candidates are generated cheaply as
the Cartesian product of the base relation vocabulary at the requested hop
length (11^3 = 1,331 candidates for standard WN18RR and three-hop search), then
scored on train.txt. This avoids rerunning the expensive historical discovery
stage while preserving the intended forward-only search space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

CACHE_VERSION = 1
TRANSITIVE_PREFIX = "__transitive__"


def parse_dict_file(path: Path):
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad dictionary line {line_idx} in {path}: {line!r}")
            raw_id, name = parts
            idx = int(raw_id)
            if name in name_to_id or idx in id_to_name:
                raise ValueError(f"Duplicate dictionary entry in {path}: {line!r}")
            name_to_id[name] = idx
            id_to_name[idx] = name
    if not id_to_name:
        raise ValueError(f"Dictionary is empty: {path}")
    return name_to_id, id_to_name


def parse_triples(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
):
    triples: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad triple line {line_idx} in {path}: {line!r}")
            h, r, t = parts
            try:
                triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
            except KeyError as exc:
                raise KeyError(
                    f"Unknown entity/relation {exc.args[0]!r} at line {line_idx} in {path}"
                ) from exc
    return triples


def load_official_dataset(data_dir: Path):
    entity_to_id, id_to_entity = parse_dict_file(data_dir / "entities.dict")
    relation_to_id, id_to_relation = parse_dict_file(data_dir / "relations.dict")
    expected_ids = list(range(len(id_to_relation)))
    if sorted(id_to_relation) != expected_ids:
        raise ValueError(
            "Base relation IDs must be contiguous from 0 so their original order can be preserved."
        )
    train = parse_triples(data_dir / "train.txt", entity_to_id, relation_to_id)
    valid = parse_triples(data_dir / "valid.txt", entity_to_id, relation_to_id)
    test = parse_triples(data_dir / "test.txt", entity_to_id, relation_to_id)
    return {
        "entity_to_id": entity_to_id,
        "id_to_entity": id_to_entity,
        "relation_to_id": relation_to_id,
        "id_to_relation": id_to_relation,
        "train": train,
        "valid": valid,
        "test": test,
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
    }


def binarize_matrix(matrix: sp.spmatrix) -> sp.csr_matrix:
    out = matrix.tocsr(copy=True)
    out.sum_duplicates()
    if out.nnz:
        out.data[:] = 1
        out.eliminate_zeros()
    return out


def build_base_relation_matrices(
    num_nodes: int,
    triples: Sequence[tuple[int, int, int]],
    id_to_relation: dict[int, str],
):
    by_relation: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, t in triples:
        by_relation[r].append((h, t))

    matrices: dict[str, sp.csr_matrix] = {}
    for relation_id in sorted(id_to_relation):
        relation_name = id_to_relation[relation_id]
        pairs = by_relation.get(relation_id, [])
        if not pairs:
            matrices[relation_name] = sp.csr_matrix(
                (num_nodes, num_nodes), dtype=np.int8
            )
            continue
        rows = np.fromiter((h for h, _ in pairs), dtype=np.int32, count=len(pairs))
        cols = np.fromiter((t for _, t in pairs), dtype=np.int32, count=len(pairs))
        values = np.ones(len(pairs), dtype=np.int8)
        matrices[relation_name] = binarize_matrix(
            sp.coo_matrix(
                (values, (rows, cols)),
                shape=(num_nodes, num_nodes),
                dtype=np.int8,
            )
        )
    return matrices


def add_inverse_relation_matrices(base_matrices: dict[str, sp.csr_matrix]):
    matrices = dict(base_matrices)
    for relation_name, matrix in base_matrices.items():
        matrices[f"{relation_name}__inv"] = matrix.transpose().tocsr()
    return matrices


def normalize_composition(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [piece.strip() for piece in value.split(" ; ")]
    elif isinstance(value, (list, tuple)):
        parts = [str(piece).strip() for piece in value]
    else:
        return None
    parts = [piece for piece in parts if piece and piece != "<identity>"]
    if not parts:
        return None
    if any(piece.startswith(TRANSITIVE_PREFIX) for piece in parts):
        return None
    return tuple(parts)


def candidate_key(composition: Sequence[str]) -> str:
    return " ; ".join(composition)


def read_candidates_from_csv(csv_path: Path):
    candidates: dict[tuple[str, ...], dict] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "composition" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a composition column: {csv_path}")
        for row in reader:
            composition = normalize_composition(row.get("composition"))
            if composition is None:
                continue
            source_support = None
            for field in ("total_paths", "train_total_paths", "source_total_paths"):
                raw = row.get(field)
                if raw not in (None, ""):
                    try:
                        source_support = float(raw)
                    except ValueError:
                        source_support = None
                    break
            candidates.setdefault(
                composition,
                {
                    "composition": composition,
                    "source_files": [],
                    "source_support": source_support,
                },
            )["source_files"].append(str(csv_path))
    return candidates, {
        "type": "recovery_csv",
        "path": str(csv_path),
        "candidate_count": len(candidates),
        "catalog_is_exhaustive": None,
    }


def candidate_json_paths(old_output_dir: Path) -> list[Path]:
    if old_output_dir.is_file():
        return [old_output_dir]
    preferred = [
        old_output_dir / "universal_edges" / "metapaths.json",
        old_output_dir / "transitive_edges" / "metapaths.json",
        old_output_dir / "all_inverse_edges" / "metapaths.json",
        old_output_dir / "metapaths.json",
    ]
    paths = [path for path in preferred if path.is_file()]
    if paths:
        return paths
    # Avoid no_changes/metapaths.json because old no_changes files may contain
    # fragments rather than complete relation compositions.
    return sorted(
        path
        for path in old_output_dir.rglob("metapaths.json")
        if path.parent.name != "no_changes"
    )


def extract_candidates_from_metapaths_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates: dict[tuple[str, ...], dict] = {}

    mapping_rows = payload.get("composition_to_transitive_relation") or []
    if isinstance(mapping_rows, list) and mapping_rows:
        for row in mapping_rows:
            if not isinstance(row, dict):
                continue
            composition = normalize_composition(row.get("composition"))
            if composition is None:
                continue
            support = row.get("source_total_paths")
            if support is None:
                support = row.get("train_total_paths")
            candidates[composition] = {
                "composition": composition,
                "source_files": [str(path)],
                "source_support": support,
            }
    else:
        chains = payload.get("chains") or []
        if isinstance(chains, list):
            for chain in chains:
                composition = normalize_composition(chain)
                if composition is None:
                    continue
                candidates[composition] = {
                    "composition": composition,
                    "source_files": [str(path)],
                    "source_support": None,
                }

    selection = payload.get("selection") or {}
    top_k = selection.get("top_k")
    exhaustive = top_k == 0
    return candidates, {
        "type": "old_metapaths_json",
        "path": str(path),
        "variant": payload.get("variant"),
        "old_splits_used": payload.get("splits_used"),
        "old_top_k": top_k,
        "catalog_is_exhaustive": exhaustive,
        "candidate_count": len(candidates),
    }


def merge_candidate_maps(target: dict, source: dict):
    for composition, record in source.items():
        if composition not in target:
            target[composition] = {
                "composition": composition,
                "source_files": list(record.get("source_files", [])),
                "source_support": record.get("source_support"),
            }
        else:
            target[composition]["source_files"].extend(record.get("source_files", []))
            if target[composition].get("source_support") is None:
                target[composition]["source_support"] = record.get("source_support")


def load_candidate_catalog(
    recovery_csv: Path | None,
    old_output_dir: Path | None,
    allow_truncated_old_candidates: bool,
):
    candidates: dict[tuple[str, ...], dict] = {}
    sources: list[dict] = []

    if recovery_csv is not None:
        csv_candidates, metadata = read_candidates_from_csv(recovery_csv)
        merge_candidate_maps(candidates, csv_candidates)
        sources.append(metadata)

    if old_output_dir is not None:
        paths = candidate_json_paths(old_output_dir)
        if not paths:
            raise FileNotFoundError(
                f"No reusable metapaths.json found under {old_output_dir}"
            )
        for path in paths:
            json_candidates, metadata = extract_candidates_from_metapaths_json(path)
            if metadata["old_top_k"] not in (None, 0) and not allow_truncated_old_candidates:
                raise ValueError(
                    "The old candidate catalog is truncated (old top_k="
                    f"{metadata['old_top_k']}) in {path}. Reusing it can make the new "
                    "candidate universe depend on held-out data. Supply an exhaustive old "
                    "catalog (top_k=0), also supply --recovery-csv, or explicitly add "
                    "--allow-truncated-old-candidates to accept that limitation."
                )
            merge_candidate_maps(candidates, json_candidates)
            sources.append(metadata)

    if not candidates:
        raise ValueError(
            "No candidate compositions were loaded. Supply --reuse-old-output-dir "
            "and/or --recovery-csv."
        )

    for record in candidates.values():
        record["source_files"] = sorted(set(record["source_files"]))

    return candidates, sources


def generate_forward_candidate_catalog(
    base_relation_names: Sequence[str],
    hop_length: int,
):
    """Generate the complete forward-only relation-composition search space.

    For WN18RR with 11 base relations and hop_length=3, this creates only
    11^3 = 1,331 candidates, which is substantially cheaper than rerunning the
    historical graph/path discovery process. Each candidate is still rescored
    against official train.txt before ranking or materialization.
    """
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    ordered_relations = tuple(base_relation_names)
    candidates: dict[tuple[str, ...], dict] = {}
    source_label = (
        f"generated:forward_cartesian_product:length={hop_length}:"
        f"relations={len(ordered_relations)}"
    )
    for composition in product(ordered_relations, repeat=hop_length):
        candidates[composition] = {
            "composition": composition,
            "source_files": [source_label],
            "source_support": None,
        }
    metadata = {
        "type": "generated_forward_cartesian_product",
        "path": None,
        "hop_length": hop_length,
        "num_base_relations": len(ordered_relations),
        "candidate_count": len(candidates),
        "catalog_is_exhaustive": True,
    }
    return candidates, metadata


def file_sha256(path: Path, digest: hashlib._hashlib.HASH):
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)


def training_graph_fingerprint(data_dir: Path, binary_intermediate: bool) -> str:
    digest = hashlib.sha256()
    for filename in ("entities.dict", "relations.dict", "train.txt"):
        digest.update(filename.encode("utf-8"))
        file_sha256(data_dir / filename, digest)
    digest.update(f"binary_intermediate={binary_intermediate}".encode("utf-8"))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_support_cache(path: Path, fingerprint: str):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("version") != CACHE_VERSION:
        return {}
    if payload.get("training_graph_fingerprint") != fingerprint:
        return {}
    return {
        key: int(value)
        for key, value in (payload.get("support_by_composition") or {}).items()
    }


def save_support_cache(path: Path, fingerprint: str, support_by_key: dict[str, int]):
    atomic_write_json(
        path,
        {
            "version": CACHE_VERSION,
            "training_graph_fingerprint": fingerprint,
            "support_by_composition": support_by_key,
        },
    )


def compose_prefix(
    relation_matrices: dict[str, sp.csr_matrix],
    prefix: Sequence[str],
    binary_intermediate: bool,
) -> sp.csr_matrix | None:
    if not prefix:
        return None
    matrix = relation_matrices[prefix[0]].tocsr()
    if binary_intermediate:
        matrix = binarize_matrix(matrix)
    for relation_name in prefix[1:]:
        matrix = (matrix @ relation_matrices[relation_name]).tocsr()
        if binary_intermediate:
            matrix = binarize_matrix(matrix)
    return matrix


def composition_matrix_from_prefix(
    relation_matrices: dict[str, sp.csr_matrix],
    composition: Sequence[str],
    prefix_matrix: sp.csr_matrix | None,
    binary_intermediate: bool,
) -> sp.csr_matrix:
    if len(composition) == 1:
        matrix = relation_matrices[composition[0]].tocsr()
    else:
        if prefix_matrix is None:
            raise AssertionError("Missing prefix matrix for multi-hop composition")
        matrix = (prefix_matrix @ relation_matrices[composition[-1]]).tocsr()
    if binary_intermediate:
        matrix = binarize_matrix(matrix)
    return matrix


def validate_candidate_relations(
    candidates: dict[tuple[str, ...], dict],
    relation_matrices: dict[str, sp.csr_matrix],
):
    valid: dict[tuple[str, ...], dict] = {}
    skipped: list[dict] = []
    available = set(relation_matrices)
    for composition, record in candidates.items():
        missing = [relation for relation in composition if relation not in available]
        if missing:
            skipped.append(
                {
                    "composition": list(composition),
                    "missing_relations": missing,
                    "source_files": record.get("source_files", []),
                }
            )
        else:
            valid[composition] = record
    if not valid:
        raise ValueError("Every candidate composition references unknown relations.")
    return valid, skipped


def keep_inverse_enabled_candidates(candidates: dict[tuple[str, ...], dict]):
    """Keep only compositions that genuinely use at least one inverse traversal."""
    kept = {
        composition: record
        for composition, record in candidates.items()
        if any(relation.endswith("__inv") for relation in composition)
    }
    filtered = [
        {
            "composition": list(composition),
            "reason": "forward_only_composition_is_handled_by_transitive_selection",
        }
        for composition in sorted(set(candidates) - set(kept))
    ]
    return kept, filtered


def score_candidates(
    candidates: dict[tuple[str, ...], dict],
    relation_matrices: dict[str, sp.csr_matrix],
    binary_intermediate: bool,
    cache_path: Path,
    fingerprint: str,
    cache_every: int,
):
    support_by_key = load_support_cache(cache_path, fingerprint)
    cached_count = 0
    to_score: list[tuple[str, ...]] = []
    for composition in candidates:
        key = candidate_key(composition)
        if key in support_by_key:
            cached_count += 1
        else:
            to_score.append(composition)

    print(
        f"Candidate support cache: {cached_count} reused, {len(to_score)} to score "
        f"({cache_path})"
    )

    groups: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for composition in to_score:
        groups[composition[:-1]].append(composition)

    scored_since_flush = 0
    completed = cached_count
    total = len(candidates)
    for prefix in sorted(groups):
        prefix_matrix = compose_prefix(
            relation_matrices, prefix, binary_intermediate=binary_intermediate
        )
        for composition in sorted(groups[prefix]):
            matrix = composition_matrix_from_prefix(
                relation_matrices,
                composition,
                prefix_matrix,
                binary_intermediate=binary_intermediate,
            )
            support_by_key[candidate_key(composition)] = int(matrix.nnz)
            completed += 1
            scored_since_flush += 1
            if completed % 25 == 0 or completed == total:
                print(f"Scored {completed}/{total} candidate compositions on train.txt")
            if scored_since_flush >= cache_every:
                save_support_cache(cache_path, fingerprint, support_by_key)
                scored_since_flush = 0
        del prefix_matrix

    save_support_cache(cache_path, fingerprint, support_by_key)

    scored_records = []
    for composition, record in candidates.items():
        scored_records.append(
            {
                **record,
                "composition": composition,
                "train_support": support_by_key[candidate_key(composition)],
            }
        )
    return scored_records, cached_count


def select_top_k(
    scored_records: Sequence[dict],
    top_k: int,
    include_zero_support: bool,
    require_exact_top_k: bool,
    label: str,
):
    eligible = [
        record
        for record in scored_records
        if include_zero_support or int(record["train_support"]) > 0
    ]
    eligible.sort(
        key=lambda record: (
            -int(record["train_support"]),
            tuple(record["composition"]),
        )
    )
    if top_k == 0:
        selected = eligible
        exact = True
    else:
        selected = eligible[:top_k]
        exact = len(selected) == top_k
    if require_exact_top_k and not exact:
        raise ValueError(
            f"{label}: requested exactly {top_k} candidates but only "
            f"{len(eligible)} eligible candidates are available."
        )
    if not selected:
        raise ValueError(f"{label}: no eligible training-supported compositions.")
    return selected, {
        "requested_top_k": top_k,
        "eligible_candidate_count": len(eligible),
        "selected_candidate_count": len(selected),
        "exact_top_k_satisfied": exact,
        "include_zero_support": include_zero_support,
    }


def make_transitive_relation_name(composition: Sequence[str]) -> str:
    return TRANSITIVE_PREFIX + "|||".join(composition)


def materialize_selected_edges(
    selected_records: Sequence[dict],
    relation_matrices: dict[str, sp.csr_matrix],
    binary_intermediate: bool,
    label: str,
):
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for record in selected_records:
        composition = tuple(record["composition"])
        groups[composition[:-1]].append(record)

    edges_by_relation: dict[str, list[tuple[int, int]]] = {}
    mapping_rows: list[dict] = []
    completed = 0
    total = len(selected_records)
    for prefix in sorted(groups):
        prefix_matrix = compose_prefix(
            relation_matrices, prefix, binary_intermediate=binary_intermediate
        )
        for record in sorted(groups[prefix], key=lambda item: tuple(item["composition"])):
            composition = tuple(record["composition"])
            matrix = composition_matrix_from_prefix(
                relation_matrices,
                composition,
                prefix_matrix,
                binary_intermediate=binary_intermediate,
            ).tocoo()
            pairs = sorted(zip(matrix.row.tolist(), matrix.col.tolist()))
            source_nodes = set(matrix.row.tolist())
            target_nodes = set(matrix.col.tolist())
            endpoint_nodes = source_nodes | target_nodes
            relation_name = make_transitive_relation_name(composition)
            edges_by_relation[relation_name] = pairs
            mapping_rows.append(
                {
                    "composition": list(composition),
                    "transitive_relation": relation_name,
                    "train_total_paths": int(record["train_support"]),
                    "num_added_transitive_edges": len(pairs),
                    "num_source_nodes": len(source_nodes),
                    "num_target_nodes": len(target_nodes),
                    "num_unique_endpoint_nodes": len(endpoint_nodes),
                    "endpoint_node_coverage_fraction": (
                        len(endpoint_nodes) / matrix.shape[0] if matrix.shape[0] else 0.0
                    ),
                    "uses_inverse_relation": any(
                        relation.endswith("__inv") for relation in composition
                    ),
                    "source_support": record.get("source_support"),
                    "source_files": record.get("source_files", []),
                }
            )
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"Materialized {completed}/{total} {label} shortcut relations")
        del prefix_matrix
    return edges_by_relation, mapping_rows


def write_entities_dict(path: Path, id_to_entity: dict[int, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entity_id in sorted(id_to_entity):
            f.write(f"{entity_id}\t{id_to_entity[entity_id]}\n")


def write_relations_dict(path: Path, relation_names: Sequence[str]):
    if len(set(relation_names)) != len(relation_names):
        raise ValueError(f"Duplicate relation names while writing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for relation_id, relation_name in enumerate(relation_names):
            f.write(f"{relation_id}\t{relation_name}\n")
    return {name: idx for idx, name in enumerate(relation_names)}


def write_data_txt(path: Path, edges: Iterable[tuple[int, str, int]], id_to_entity):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, str, int]] = set()
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for h, relation_name, t in edges:
            edge = (int(h), relation_name, int(t))
            if edge in seen:
                continue
            seen.add(edge)
            f.write(
                f"{id_to_entity[edge[0]]}\t{edge[1]}\t{id_to_entity[edge[2]]}\n"
            )
            count += 1
    return count


def copy_official_files(data_dir: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        "entities.dict",
        "relations.dict",
        "train.txt",
        "valid.txt",
        "test.txt",
    ):
        shutil.copy2(data_dir / filename, destination / filename)


def base_named_edges(triples, id_to_relation):
    return [(h, id_to_relation[r], t) for h, r, t in triples]


def all_inverse_named_edges(base_edges: Sequence[tuple[int, str, int]]):
    return [(t, f"{relation_name}__inv", h) for h, relation_name, t in base_edges]


def shortcut_named_edges(edges_by_relation: dict[str, list[tuple[int, int]]]):
    rows: list[tuple[int, str, int]] = []
    for relation_name in sorted(edges_by_relation):
        rows.extend((h, relation_name, t) for h, t in edges_by_relation[relation_name])
    return rows


def validate_no_leakage(
    variants: dict[str, Sequence[tuple[int, str, int]]],
    base_train_edges: Sequence[tuple[int, str, int]],
    heldout_edges: Sequence[tuple[int, str, int]],
):
    base_set = set(base_train_edges)
    heldout_set = set(heldout_edges)
    heldout_inverse_set = {
        (t, f"{relation_name}__inv", h) for h, relation_name, t in heldout_edges
    }
    for variant_name, edges in variants.items():
        edge_set = set(edges)
        if len(edge_set) != len(edges):
            raise AssertionError(f"{variant_name} contains duplicate edges")
        missing_base = base_set - edge_set
        if missing_base:
            raise AssertionError(
                f"{variant_name} is missing {len(missing_base)} official train edges"
            )
        overlap = edge_set & heldout_set
        if overlap:
            raise AssertionError(
                f"{variant_name} contains {len(overlap)} official validation/test edges"
            )
        inverse_overlap = edge_set & heldout_inverse_set
        if inverse_overlap:
            raise AssertionError(
                f"{variant_name} contains {len(inverse_overlap)} inverse edges derived "
                "from held-out validation/test triples"
            )


def selection_payload(selected: Sequence[dict], metadata: dict):
    return {
        **metadata,
        "method": "top_k_by_train_support",
        "selected_compositions": [list(record["composition"]) for record in selected],
    }


def merge_selected_records(*named_groups: tuple[str, Sequence[dict]]):
    """Merge selected composition records and preserve every selection origin."""
    merged: dict[tuple[str, ...], dict] = {}
    for origin, records in named_groups:
        for record in records:
            composition = tuple(record["composition"])
            if composition not in merged:
                merged[composition] = {
                    **record,
                    "composition": composition,
                    "selection_origins": [origin],
                    "source_files": list(record.get("source_files", [])),
                }
                continue
            current = merged[composition]
            if origin not in current["selection_origins"]:
                current["selection_origins"].append(origin)
            current["source_files"] = sorted(
                set(current.get("source_files", []))
                | set(record.get("source_files", []))
            )
            # Both searches are evaluated on the same official training graph.
            # Keep the largest support defensively if cached metadata differs.
            current["train_support"] = max(
                int(current.get("train_support", 0)),
                int(record.get("train_support", 0)),
            )
    return [merged[key] for key in sorted(merged)]


def merge_mapping_rows(*named_groups: tuple[str, Sequence[dict]]):
    """Merge shortcut metadata by canonical composition."""
    merged: dict[tuple[str, ...], dict] = {}
    for origin, rows in named_groups:
        for row in rows:
            composition = tuple(row["composition"])
            if composition not in merged:
                merged[composition] = {
                    **row,
                    "selection_origins": [origin],
                }
                continue
            current = merged[composition]
            if current["transitive_relation"] != row["transitive_relation"]:
                raise AssertionError(
                    f"Conflicting shortcut names for composition {composition}"
                )
            if origin not in current["selection_origins"]:
                current["selection_origins"].append(origin)
            current["source_files"] = sorted(
                set(current.get("source_files", []))
                | set(row.get("source_files", []))
            )
    return [merged[key] for key in sorted(merged)]


def cumulative_universal_selection_payload(
    transitive_selected: Sequence[dict],
    transitive_metadata: dict,
    universal_selected: Sequence[dict],
    universal_metadata: dict,
    final_selected: Sequence[dict],
):
    transitive_keys = {tuple(record["composition"]) for record in transitive_selected}
    universal_keys = {tuple(record["composition"]) for record in universal_selected}
    overlap = transitive_keys & universal_keys
    return {
        "method": "cumulative_union_of_transitive_and_inverse_enabled_selections",
        "selected_candidate_count": len(final_selected),
        "inherited_transitive_candidate_count": len(transitive_keys),
        "universal_catalog_selected_candidate_count": len(universal_keys),
        "selection_overlap_candidate_count": len(overlap),
        "universal_specific_candidate_count": len(universal_keys - transitive_keys),
        "transitive_selection": selection_payload(
            transitive_selected, transitive_metadata
        ),
        "universal_catalog_selection": selection_payload(
            universal_selected, universal_metadata
        ),
        "selected_compositions": [
            list(record["composition"]) for record in final_selected
        ],
    }


def validate_cumulative_variant_overlap(
    variants: dict[str, Sequence[tuple[int, str, int]]],
):
    """Assert the intended graph lattice and return overlap statistics."""
    edge_sets = {name: set(edges) for name, edges in variants.items()}
    required_subsets = [
        ("no_changes", "all_inverse_edges"),
        ("no_changes", "transitive_edges"),
        ("all_inverse_edges", "universal_edges"),
        ("transitive_edges", "universal_edges"),
    ]
    for child, parent in required_subsets:
        missing = edge_sets[child] - edge_sets[parent]
        if missing:
            raise AssertionError(
                f"{parent} is missing {len(missing)} edges inherited from {child}"
            )

    def shortcut_edges(name: str):
        return {edge for edge in edge_sets[name] if edge[1].startswith(TRANSITIVE_PREFIX)}

    transitive_shortcuts = shortcut_edges("transitive_edges")
    universal_shortcuts = shortcut_edges("universal_edges")
    missing_shortcuts = transitive_shortcuts - universal_shortcuts
    if missing_shortcuts:
        raise AssertionError(
            "universal_edges is missing "
            f"{len(missing_shortcuts)} forward shortcut edges from transitive_edges"
        )
    return {
        "transitive_shortcut_edges": len(transitive_shortcuts),
        "universal_shortcut_edges": len(universal_shortcuts),
        "shared_shortcut_edges": len(transitive_shortcuts & universal_shortcuts),
        "transitive_shortcut_overlap_fraction": (
            len(transitive_shortcuts & universal_shortcuts)
            / len(transitive_shortcuts)
            if transitive_shortcuts
            else 1.0
        ),
        "transitive_graph_is_subset_of_universal": True,
        "inverse_graph_is_subset_of_universal": True,
    }


def write_coverage_rankings(path: Path, mappings_by_catalog: dict[str, Sequence[dict]]):
    """Write deterministic rankings for later reduced MAGNN preprocessing."""
    payload = {
        "ranking_definition": {
            "primary": "num_unique_endpoint_nodes descending",
            "secondary": "num_added_transitive_edges descending",
            "tertiary": "composition lexicographic",
            "note": (
                "Unique endpoint-node coverage counts the union of heads and tails "
                "connected by a materialized shortcut. It does not count intermediate "
                "witness nodes."
            ),
        },
        "catalogs": {},
    }
    for name, rows in mappings_by_catalog.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -int(row.get("num_unique_endpoint_nodes", 0)),
                -int(row.get("num_added_transitive_edges", 0)),
                tuple(row.get("composition", [])),
            ),
        )
        payload["catalogs"][name] = [
            {"rank": rank, **row} for rank, row in enumerate(ranked, start=1)
        ]
    atomic_write_json(path, payload)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-free unchanged, inverse, transitive, and universal "
            "WordNet training graphs."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--recovery-csv", type=Path, default=None)
    parser.add_argument(
        "--reuse-old-output-dir",
        type=Path,
        default=None,
        help=(
            "Reuse complete relation compositions from old metapaths.json files. "
            "Old data.txt/edge files are never reused."
        ),
    )
    parser.add_argument(
        "--allow-truncated-old-candidates",
        action="store_true",
        help=(
            "Allow reuse of an old metapath catalog whose old top_k was nonzero. "
            "This is faster but the candidate universe may reflect held-out data."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k for transitive and universal; 0 keeps all eligible candidates.",
    )
    parser.add_argument("--transitive-top-k", type=int, default=None)
    parser.add_argument(
        "--universal-top-k",
        type=int,
        default=None,
        help=(
            "Top-k inverse-enabled compositions added on top of the inherited "
            "transitive selection. The final universal catalog can therefore be larger."
        ),
    )
    parser.add_argument(
        "--transitive-hop-length",
        type=int,
        default=3,
        help=(
            "Length of the exhaustive forward-only composition search. "
            "Default: 3, yielding 11^3=1331 candidates on WN18RR."
        ),
    )
    parser.add_argument("--require-exact-top-k", action="store_true")
    parser.add_argument("--include-zero-support", action="store_true")
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=None,
        help="Default: OUTDIR/train_candidate_support_cache.json",
    )
    parser.add_argument(
        "--cache-every",
        type=int,
        default=25,
        help="Persist newly computed support values after this many candidates.",
    )
    parser.add_argument(
        "--binary-intermediate",
        dest="binary_intermediate",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-binary-intermediate",
        dest="binary_intermediate",
        action="store_false",
    )
    args = parser.parse_args()

    for name, value in (
        ("--top-k", args.top_k),
        ("--transitive-top-k", args.transitive_top_k),
        ("--universal-top-k", args.universal_top_k),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.cache_every <= 0:
        raise ValueError("--cache-every must be positive")
    if args.transitive_hop_length <= 0:
        raise ValueError("--transitive-hop-length must be positive")
    if args.recovery_csv is None and args.reuse_old_output_dir is None:
        raise ValueError(
            "Supply --reuse-old-output-dir and/or --recovery-csv as the candidate source."
        )

    data_dir = args.data_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    support_cache = (
        args.support_cache.resolve()
        if args.support_cache is not None
        else outdir / "train_candidate_support_cache.json"
    )

    transitive_top_k = (
        args.transitive_top_k
        if args.transitive_top_k is not None
        else args.top_k
    )
    universal_top_k = (
        args.universal_top_k
        if args.universal_top_k is not None
        else args.top_k
    )

    print("Loading official WordNet splits...")
    dataset = load_official_dataset(data_dir)
    base_relation_names = [
        dataset["id_to_relation"][idx]
        for idx in sorted(dataset["id_to_relation"])
    ]
    base_edges = base_named_edges(dataset["train"], dataset["id_to_relation"])
    valid_edges = base_named_edges(dataset["valid"], dataset["id_to_relation"])
    test_edges = base_named_edges(dataset["test"], dataset["id_to_relation"])
    print(
        f"  train={len(base_edges)}, valid={len(valid_edges)}, test={len(test_edges)}, "
        f"entities={dataset['num_entities']}, base_relations={len(base_relation_names)}"
    )

    print("Loading reusable universal candidate compositions...")
    universal_candidates, universal_candidate_sources = load_candidate_catalog(
        recovery_csv=args.recovery_csv.resolve() if args.recovery_csv else None,
        old_output_dir=(
            args.reuse_old_output_dir.resolve()
            if args.reuse_old_output_dir
            else None
        ),
        allow_truncated_old_candidates=args.allow_truncated_old_candidates,
    )
    universal_candidates_loaded_count = len(universal_candidates)
    universal_candidates, filtered_forward_universal_candidates = (
        keep_inverse_enabled_candidates(universal_candidates)
    )
    if not universal_candidates:
        raise ValueError(
            "The reusable universal catalog contains no inverse-enabled compositions."
        )
    print(
        f"  {universal_candidates_loaded_count} unique legacy candidates loaded; "
        f"kept {len(universal_candidates)} inverse-enabled candidates and filtered "
        f"{len(filtered_forward_universal_candidates)} forward-only candidates"
    )

    print(
        "Generating exhaustive forward-only transitive candidate compositions..."
    )
    transitive_candidates, transitive_candidate_source = (
        generate_forward_candidate_catalog(
            base_relation_names, hop_length=args.transitive_hop_length
        )
    )
    print(
        f"  {len(transitive_candidates)} forward-only candidates generated "
        f"({len(base_relation_names)}^{args.transitive_hop_length})"
    )

    base_matrices = build_base_relation_matrices(
        dataset["num_entities"], dataset["train"], dataset["id_to_relation"]
    )
    universal_matrices = add_inverse_relation_matrices(base_matrices)

    transitive_candidates, skipped_transitive_candidates = validate_candidate_relations(
        transitive_candidates, base_matrices
    )
    universal_candidates, skipped_universal_candidates = validate_candidate_relations(
        universal_candidates, universal_matrices
    )
    if skipped_transitive_candidates:
        print(
            "  Skipped "
            f"{len(skipped_transitive_candidates)} transitive candidates with "
            "unknown relation names"
        )
    if skipped_universal_candidates:
        print(
            "  Skipped "
            f"{len(skipped_universal_candidates)} universal candidates with "
            "unknown relation names"
        )

    fingerprint = training_graph_fingerprint(data_dir, args.binary_intermediate)

    # Score the generated forward-only space independently. The old universal
    # catalog may contain no forward-only candidates at all.
    transitive_scored, transitive_cached_count = score_candidates(
        transitive_candidates,
        base_matrices,
        binary_intermediate=args.binary_intermediate,
        cache_path=support_cache,
        fingerprint=fingerprint,
        cache_every=args.cache_every,
    )
    universal_scored, universal_cached_count = score_candidates(
        universal_candidates,
        universal_matrices,
        binary_intermediate=args.binary_intermediate,
        cache_path=support_cache,
        fingerprint=fingerprint,
        cache_every=args.cache_every,
    )

    transitive_selected, transitive_selection = select_top_k(
        transitive_scored,
        top_k=transitive_top_k,
        include_zero_support=args.include_zero_support,
        require_exact_top_k=args.require_exact_top_k,
        label="transitive",
    )
    universal_selected, universal_selection = select_top_k(
        universal_scored,
        top_k=universal_top_k,
        include_zero_support=args.include_zero_support,
        require_exact_top_k=args.require_exact_top_k,
        label="universal",
    )

    candidate_sources_by_variant = {
        "no_changes": [transitive_candidate_source],
        "all_inverse_edges": [transitive_candidate_source] + universal_candidate_sources,
        "transitive_edges": [transitive_candidate_source],
        "universal_edges": [transitive_candidate_source] + universal_candidate_sources,
    }

    universal_final_selected = merge_selected_records(
        ("inherited_transitive", transitive_selected),
        ("universal_catalog", universal_selected),
    )
    transitive_selected_keys = {
        tuple(record["composition"]) for record in transitive_selected
    }
    universal_additional_selected = [
        record
        for record in universal_selected
        if tuple(record["composition"]) not in transitive_selected_keys
    ]

    print(
        "Selected compositions: "
        f"transitive={len(transitive_selected)}, "
        f"universal_catalog={len(universal_selected)}, "
        f"universal_final={len(universal_final_selected)}"
    )

    transitive_shortcuts, transitive_mapping = materialize_selected_edges(
        transitive_selected,
        base_matrices,
        binary_intermediate=args.binary_intermediate,
        label="transitive",
    )
    universal_specific_shortcuts, universal_specific_mapping = materialize_selected_edges(
        universal_additional_selected,
        universal_matrices,
        binary_intermediate=args.binary_intermediate,
        label="universal-specific",
    )
    universal_shortcuts = {
        **transitive_shortcuts,
        **universal_specific_shortcuts,
    }
    universal_mapping = merge_mapping_rows(
        ("inherited_transitive", transitive_mapping),
        ("universal_catalog", universal_specific_mapping),
    )

    inverse_relation_names = [f"{name}__inv" for name in base_relation_names]
    inverse_edges = all_inverse_named_edges(base_edges)
    transitive_shortcut_edges = shortcut_named_edges(transitive_shortcuts)
    universal_shortcut_edges = shortcut_named_edges(universal_shortcuts)

    variants = {
        "no_changes": list(base_edges),
        "all_inverse_edges": list(base_edges) + inverse_edges,
        "transitive_edges": list(base_edges) + transitive_shortcut_edges,
        "universal_edges": (
            list(base_edges)
            + inverse_edges
            + universal_shortcut_edges
        ),
    }
    validate_no_leakage(
        variants,
        base_train_edges=base_edges,
        heldout_edges=valid_edges + test_edges,
    )
    overlap_validation = validate_cumulative_variant_overlap(variants)

    transitive_relation_names = sorted(transitive_shortcuts)
    universal_relation_names = sorted(universal_shortcuts)
    variant_relation_names = {
        "no_changes": base_relation_names,
        "all_inverse_edges": base_relation_names + inverse_relation_names,
        "transitive_edges": base_relation_names + transitive_relation_names,
        "universal_edges": (
            base_relation_names + inverse_relation_names + universal_relation_names
        ),
    }
    shared_relation_names = (
        base_relation_names
        + inverse_relation_names
        + sorted(set(transitive_relation_names) | set(universal_relation_names))
    )
    write_relations_dict(outdir / "shared_relations.dict", shared_relation_names)

    copy_official_files(data_dir, outdir / "original_splits")

    selection_by_variant = {
        "no_changes": {
            "note": "No shortcut composition selection is applied to unchanged.",
        },
        "all_inverse_edges": {
            "note": "Every base training relation is materialized in reverse.",
        },
        "transitive_edges": selection_payload(
            transitive_selected, transitive_selection
        ),
        "universal_edges": cumulative_universal_selection_payload(
            transitive_selected,
            transitive_selection,
            universal_selected,
            universal_selection,
            universal_final_selected,
        ),
    }
    mapping_by_variant = {
        "transitive_edges": transitive_mapping,
        "universal_edges": universal_mapping,
    }

    variant_edge_counts = {}
    for variant_name, edges in variants.items():
        variant_dir = outdir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_entities_dict(variant_dir / "entities.dict", dataset["id_to_entity"])
        relation_to_id = write_relations_dict(
            variant_dir / "relations.dict", variant_relation_names[variant_name]
        )
        edge_count = write_data_txt(
            variant_dir / "data.txt", edges, dataset["id_to_entity"]
        )
        variant_edge_counts[variant_name] = edge_count

        if variant_name == "no_changes":
            chains = [list(record["composition"]) for record in transitive_selected]
            direct_chains = []
            mapping_rows = []
        elif variant_name == "all_inverse_edges":
            chains = [
                list(record["composition"]) for record in universal_final_selected
            ]
            direct_chains = []
            mapping_rows = []
        elif variant_name == "transitive_edges":
            chains = [list(record["composition"]) for record in transitive_selected]
            direct_chains = [
                [make_transitive_relation_name(record["composition"])]
                for record in transitive_selected
            ]
            mapping_rows = mapping_by_variant[variant_name]
        else:
            chains = [
                list(record["composition"]) for record in universal_final_selected
            ]
            direct_chains = [
                [make_transitive_relation_name(record["composition"])]
                for record in universal_final_selected
            ]
            mapping_rows = mapping_by_variant[variant_name]

        chain_relation_ids = [
            [relation_to_id[relation] for relation in chain]
            for chain in chains
            if all(relation in relation_to_id for relation in chain)
        ]
        direct_relation_ids = [
            [relation_to_id[chain[0]]]
            for chain in direct_chains
            if chain[0] in relation_to_id
        ]
        atomic_write_json(
            variant_dir / "metapaths.json",
            {
                "variant": variant_name,
                "splits_used": ["train"],
                "selection": selection_by_variant[variant_name],
                "chains": chains,
                "chain_relation_ids": chain_relation_ids,
                "direct_transitive_chains": direct_chains,
                "direct_transitive_relation_ids": direct_relation_ids,
                "composition_to_transitive_relation": mapping_rows,
                "candidate_sources": candidate_sources_by_variant[variant_name],
                "note": {
                    "no_changes": (
                        "Original training edges only. Forward-only selected chains are "
                        "listed for comparison with transitive_edges."
                    ),
                    "all_inverse_edges": (
                        "Original training edges plus inverses for every base relation. "
                        "The cumulative universal chain catalog (forward-only plus "
                        "inverse-enabled compositions) is listed as explicit multi-hop paths."
                    ),
                    "transitive_edges": (
                        "Original training edges plus forward-only composed shortcut edges. "
                        "No inverse edges are materialized."
                    ),
                    "universal_edges": (
                        "Cumulative graph containing original training edges, all base "
                        "inverse edges, every forward-only shortcut from transitive_edges, "
                        "and additional inverse-enabled shortcut relations."
                    ),
                }[variant_name],
            },
        )

        inverse_relation_name_set = set(inverse_relation_names)
        num_inverse_edges = sum(
            1 for _, relation, _ in set(edges)
            if relation in inverse_relation_name_set
        )
        num_shortcut_edges = sum(
            1 for _, relation, _ in set(edges) if relation.startswith(TRANSITIVE_PREFIX)
        )
        atomic_write_json(
            variant_dir / "augmentation_summary.json",
            {
                "variant": variant_name,
                "augmentation_source": "official train.txt only",
                "official_train_edges": len(base_edges),
                "official_validation_queries": len(valid_edges),
                "official_test_queries": len(test_edges),
                "num_edges": edge_count,
                "num_active_relation_types": len(variant_relation_names[variant_name]),
                "num_added_inverse_edges": num_inverse_edges,
                "num_added_transitive_edges": num_shortcut_edges,
                "num_inherited_forward_transitive_edges": (
                    len(transitive_shortcut_edges)
                    if variant_name == "universal_edges"
                    else 0
                ),
                "num_universal_specific_transitive_edges": (
                    len(shortcut_named_edges(universal_specific_shortcuts))
                    if variant_name == "universal_edges"
                    else 0
                ),
                "selection": selection_by_variant[variant_name],
            },
        )
        print(f"Wrote {variant_name}: {edge_count} training edges")

    write_coverage_rankings(
        outdir / "metapath_coverage_rankings.json",
        {
            "transitive_forward_only": transitive_mapping,
            "universal_cumulative": universal_mapping,
            "universal_specific": universal_specific_mapping,
        },
    )

    atomic_write_json(
        outdir / "dataset_manifest.json",
        {
            "augmentation_source": "official train.txt only",
            "official_split_counts": {
                "train": len(base_edges),
                "valid": len(valid_edges),
                "test": len(test_edges),
            },
            "candidate_sources": {
                "transitive": [transitive_candidate_source],
                "universal": universal_candidate_sources,
            },
            "candidate_counts": {
                "transitive_generated": len(transitive_candidates),
                "universal_loaded_before_inverse_filter": (
                    universal_candidates_loaded_count
                ),
                "universal_inverse_enabled_kept": len(universal_candidates),
                "universal_forward_only_filtered": len(
                    filtered_forward_universal_candidates
                ),
                "transitive_skipped_unknown_relations": len(
                    skipped_transitive_candidates
                ),
                "universal_skipped_unknown_relations": len(
                    skipped_universal_candidates
                ),
            },
            "support_cache": {
                "path": str(support_cache),
                "training_graph_fingerprint": fingerprint,
                "transitive_support_values_reused": transitive_cached_count,
                "universal_support_values_reused": universal_cached_count,
            },
            "selection": {
                "transitive": transitive_selection,
                "universal": universal_selection,
            },
            "variant_training_edge_counts": variant_edge_counts,
            "cumulative_variant_overlap": overlap_validation,
            "universal_definition": (
                "no_changes + all base inverse edges + all selected forward-only "
                "transitive shortcuts + all additional inverse-enabled shortcuts"
            ),
            "metapath_coverage_rankings": str(
                outdir / "metapath_coverage_rankings.json"
            ),
            "shared_relation_vocabulary": str(outdir / "shared_relations.dict"),
            "old_augmented_edges_reused": False,
            "leakage_checks_passed": True,
        },
    )

    if (
        skipped_transitive_candidates
        or skipped_universal_candidates
        or filtered_forward_universal_candidates
    ):
        atomic_write_json(
            outdir / "skipped_candidate_compositions.json",
            {
                "transitive": skipped_transitive_candidates,
                "universal_unknown_relations": skipped_universal_candidates,
                "universal_forward_only_filtered": (
                    filtered_forward_universal_candidates
                ),
            },
        )

    print("Copied untouched official splits to original_splits/")
    print("Leakage checks passed.")
    print("Done.")


if __name__ == "__main__":
    main()