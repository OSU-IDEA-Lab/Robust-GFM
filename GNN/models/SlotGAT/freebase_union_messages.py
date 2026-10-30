#!/usr/bin/env python3
"""Canonical direct-message schedule for union-equivalent Freebase SlotGAT.

The recoverability-conditional implementation originally represented every
registered Exact-2 shortcut by a sequence of relation-wise reductions and then
emitted one record per reachable destination.  That operator is invariant, but
it is not the same operator as SlotGAT on the physical Exact-2 graph: a direct
union graph has one attention record for every shortcut endpoint pair.

This module prepares and loads a destination-CSR schedule that reproduces the
historical ``run_freebase_nc.py`` graph builder by default.  That builder keeps
one DGL edge per ordered node pair, lets the last physical relation win for a
parallel physical pair, adds a generated reverse only when the reverse pair is
absent, and finally guarantees one self edge per node.  An optional typed
multiedge mode preserves every typed row plus every reverse instead.

The resulting direct schedule is shared by all conditional raw contexts.  The
raw context remains variant-specific and must pass its own runtime audit; only
the propagation message program is canonical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from freebase_block_conditional import (
    ChannelRecord,
    FreebaseBlockContextPlan,
    RelationInfo,
    parse_info,
    sha256_file,
    sha256_json,
    write_json,
)


FORMAT_VERSION = 2
OPERATOR_NAME = "legacy_slotgat_exact2_union_message_csr_v2"


def _sha256_array_bytes(array: np.ndarray, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    view = np.asarray(array).view(np.uint8).reshape(-1)
    for start in range(0, int(view.size), int(chunk_bytes)):
        digest.update(memoryview(view[start : start + int(chunk_bytes)]))
    return digest.hexdigest()


def _channel_lookup(channels: Sequence[ChannelRecord]) -> dict[tuple[str, int], int]:
    result = {
        (str(record.semantic_relation), int(record.direction)): int(record.channel)
        for record in channels
    }
    if len(result) != len(channels):
        raise ValueError("Freebase channel catalog contains duplicate semantic keys")
    return result


def _iter_link_rows(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Malformed link.dat line {line_number} in {path}: {raw!r}")
            yield int(parts[0]), int(parts[1]), int(parts[2])


@dataclass(frozen=True)
class UnionMessageSchedule:
    root: Path
    total_nodes: int
    num_channels: int
    indptr: np.ndarray
    source: np.ndarray
    channel: np.ndarray
    message_count: int
    forward_edge_count: int
    physical_message_count: int
    reverse_message_count: int
    self_message_count: int
    construction_policy: str
    self_channel: int
    message_program_hash: str
    exact2_raw_link_hash: str
    channel_catalog_hash: str
    operator: str
    source_dir: str
    canonical_order: str

    @classmethod
    def load(cls, root: Path) -> "UnionMessageSchedule":
        root = Path(root).expanduser().resolve()
        manifest_path = root / "union_message_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing union message manifest: {manifest_path}. Run "
                "prepare_Freebase_union_message_equivalent.py first."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported union message format in {manifest_path}; regenerate the "
                "schedule with prepare_Freebase_union_message_equivalent.py --overwrite"
            )
        if str(payload.get("operator", "")) != OPERATOR_NAME:
            raise ValueError(
                f"Unexpected union message operator in {manifest_path}; regenerate the schedule"
            )
        indptr = np.load(root / "indptr.npy", mmap_mode="r")
        source = np.load(root / "source.npy", mmap_mode="r")
        channel = np.load(root / "channel.npy", mmap_mode="r")
        total_nodes = int(payload["total_nodes"])
        message_count = int(payload["message_count"])
        num_channels = int(payload["num_channels"])
        if indptr.dtype != np.int64 or indptr.shape != (total_nodes + 1,):
            raise ValueError("Invalid union-message indptr array")
        if int(indptr[0]) != 0 or int(indptr[-1]) != message_count:
            raise ValueError("Union-message indptr does not cover all messages")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("Union-message indptr is not monotone")
        if source.shape != (message_count,) or channel.shape != (message_count,):
            raise ValueError("Union-message arrays have inconsistent lengths")
        component_count = (
            int(payload.get("physical_message_count", payload["forward_edge_count"]))
            + int(payload.get("reverse_message_count", payload["forward_edge_count"]))
            + int(payload.get("self_message_count", total_nodes))
        )
        if component_count != message_count:
            raise ValueError(
                "Union-message manifest component counts do not sum to message_count"
            )
        if source.dtype not in (np.int32, np.int64):
            raise ValueError(f"Unsupported source dtype: {source.dtype}")
        if channel.dtype not in (np.uint16, np.int32, np.int64):
            raise ValueError(f"Unsupported channel dtype: {channel.dtype}")
        return cls(
            root=root,
            total_nodes=total_nodes,
            num_channels=num_channels,
            indptr=indptr,
            source=source,
            channel=channel,
            message_count=message_count,
            forward_edge_count=int(payload["forward_edge_count"]),
            physical_message_count=int(payload.get("physical_message_count", payload["forward_edge_count"])),
            reverse_message_count=int(payload.get("reverse_message_count", payload["forward_edge_count"])),
            self_message_count=int(payload.get("self_message_count", total_nodes)),
            construction_policy=str(payload.get("construction_policy", "typed_multiedge_plus_all_reverses")),
            self_channel=int(payload["self_channel"]),
            message_program_hash=str(payload["message_program_hash"]),
            exact2_raw_link_hash=str(payload["exact2_raw_link_hash"]),
            channel_catalog_hash=str(payload["channel_catalog_hash"]),
            operator=str(payload["operator"]),
            source_dir=str(payload["source_dir"]),
            canonical_order=str(payload["canonical_order"]),
        )

    def audit_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "operator": self.operator,
            "source_dir": self.source_dir,
            "total_nodes": self.total_nodes,
            "num_channels": self.num_channels,
            "message_count": self.message_count,
            "forward_edge_count": self.forward_edge_count,
            "physical_message_count": self.physical_message_count,
            "reverse_message_count": self.reverse_message_count,
            "self_message_count": self.self_message_count,
            "construction_policy": self.construction_policy,
            "self_channel": self.self_channel,
            "message_program_hash": self.message_program_hash,
            "exact2_raw_link_hash": self.exact2_raw_link_hash,
            "channel_catalog_hash": self.channel_catalog_hash,
            "canonical_order": self.canonical_order,
            "storage": {
                "indptr": "int64 destination CSR offsets",
                "source": str(self.source.dtype),
                "channel": str(self.channel.dtype),
                "destination": "implicit from indptr",
            },
        }

    def validate_against_plan(self, plan: FreebaseBlockContextPlan) -> None:
        if int(plan.layout.total) != self.total_nodes:
            raise ValueError("Union message node count differs from context plan")
        if int(plan.num_channels) != self.num_channels:
            raise ValueError("Union message channel count differs from context plan")
        catalog_hash = sha256_json([record.to_json() for record in plan.channels])
        if catalog_hash != self.channel_catalog_hash:
            raise ValueError("Union message channel catalog differs from context plan")
        if self.exact2_raw_link_hash == "":
            raise ValueError("Union message schedule has no Exact-2 provenance hash")
        plan.verify_runtime_context()

    def destination_blocks(self, max_edges: int) -> list[tuple[int, int]]:
        """Partition destination rows without splitting an incoming neighborhood."""
        max_edges = int(max_edges)
        if max_edges <= 0:
            return [(0, self.total_nodes)]
        blocks: list[tuple[int, int]] = []
        start = 0
        while start < self.total_nodes:
            target = int(self.indptr[start]) + max_edges
            stop = int(np.searchsorted(self.indptr, target, side="right") - 1)
            stop = max(start + 1, min(stop, self.total_nodes))
            # A single destination can exceed the requested edge budget.  It is
            # still kept intact because splitting it would change the softmax.
            blocks.append((start, stop))
            start = stop
        return blocks




def _legacy_candidate_priority(
    channels: np.ndarray,
    *,
    direction_by_channel: np.ndarray,
    rank_by_channel: np.ndarray,
    self_channel: int,
    relation_count: int,
) -> np.ndarray:
    """Return priorities that reproduce the legacy ordered-pair overwrite rules.

    Larger values win.  Physical rows outrank generated reverses and self.
    Among parallel physical rows the last relation in ``dl.links['data']``
    wins.  Among generated reverse candidates the first relation wins because
    the legacy loop inserts the reverse only while the ordered pair is absent.
    """
    channels = np.asarray(channels, dtype=np.int64)
    directions = direction_by_channel[channels]
    ranks = rank_by_channel[channels].astype(np.int64, copy=False)
    relation_count = max(int(relation_count), 1)
    priority = np.zeros(channels.shape, dtype=np.int64)
    physical = directions > 0
    reverse = directions < 0
    priority[physical] = 2 * relation_count + ranks[physical]
    priority[reverse] = relation_count + (relation_count - 1 - ranks[reverse])
    priority[channels == int(self_channel)] = 0
    return priority


def _select_legacy_ordered_pairs(
    source: np.ndarray,
    channel: np.ndarray,
    *,
    direction_by_channel: np.ndarray,
    rank_by_channel: np.ndarray,
    self_channel: int,
    relation_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Collapse one destination's candidates to one record per source.

    The implementation is vectorized over all sources in the destination row;
    this is important for Exact-2, whose candidate schedule has tens of millions
    of records.
    """
    source = np.asarray(source, dtype=np.int32)
    channel = np.asarray(channel, dtype=np.uint16)
    if source.shape != channel.shape:
        raise ValueError("source/channel candidate arrays have different shapes")
    if source.size == 0:
        return source.copy(), channel.copy(), {
            "physical": 0, "reverse": 0, "self": 0,
            "parallel_physical_excess": 0,
            "generated_reverse_suppressed": 0,
            "duplicate_same_channel": 0,
        }

    priority = _legacy_candidate_priority(
        channel,
        direction_by_channel=direction_by_channel,
        rank_by_channel=rank_by_channel,
        self_channel=self_channel,
        relation_count=relation_count,
    )
    # Sort by source first and priority second.  Selecting the last row of each
    # source group implements the overwrite/insert-if-absent policy above.
    order = np.lexsort((priority, source))
    source_sorted = source[order]
    channel_sorted = channel[order]
    directions_sorted = direction_by_channel[channel_sorted]
    group_end = np.empty(source_sorted.size, dtype=bool)
    group_end[:-1] = source_sorted[1:] != source_sorted[:-1]
    group_end[-1] = True
    chosen_source = source_sorted[group_end]
    chosen_channel = channel_sorted[group_end]
    chosen_direction = direction_by_channel[chosen_channel]

    physical_candidates = directions_sorted > 0
    reverse_candidates = directions_sorted < 0
    physical_group = np.maximum.reduceat(
        physical_candidates.astype(np.int8),
        np.r_[0, np.flatnonzero(source_sorted[1:] != source_sorted[:-1]) + 1],
    ).astype(bool)
    reverse_group_counts = np.add.reduceat(
        reverse_candidates.astype(np.int64),
        np.r_[0, np.flatnonzero(source_sorted[1:] != source_sorted[:-1]) + 1],
    )
    physical_group_counts = np.add.reduceat(
        physical_candidates.astype(np.int64),
        np.r_[0, np.flatnonzero(source_sorted[1:] != source_sorted[:-1]) + 1],
    )

    # Duplicate typed candidates are diagnostic only.  Lexicographically sort
    # (source, channel) and count adjacent repetitions.
    typed_order = np.lexsort((channel, source))
    typed_source = source[typed_order]
    typed_channel = channel[typed_order]
    duplicate_same_channel = int(np.count_nonzero(
        (typed_source[1:] == typed_source[:-1])
        & (typed_channel[1:] == typed_channel[:-1])
    )) if typed_source.size > 1 else 0

    stats = {
        "physical": int(np.count_nonzero(chosen_direction > 0)),
        "reverse": int(np.count_nonzero(chosen_direction < 0)),
        "self": int(np.count_nonzero(chosen_channel == int(self_channel))),
        "parallel_physical_excess": int(np.maximum(physical_group_counts - 1, 0).sum()),
        "generated_reverse_suppressed": int(
            reverse_group_counts[physical_group].sum()
            + np.maximum(reverse_group_counts[~physical_group] - 1, 0).sum()
        ),
        "duplicate_same_channel": duplicate_same_channel,
    }
    return chosen_source, chosen_channel, stats

def _validate_plan_catalog(plan: FreebaseBlockContextPlan, relations: Mapping[int, RelationInfo]) -> None:
    lookup = _channel_lookup(plan.channels)
    missing: list[str] = []
    for relation in relations.values():
        for direction in (1, -1):
            if (relation.name, direction) not in lookup:
                missing.append(f"{relation.name}:{direction}")
    if missing:
        raise ValueError(
            "Exact-2 link schema has relations absent from the conditional channel catalog: "
            + ", ".join(missing[:20])
        )


def _resolve_relation_schema(
    *,
    exact2_dir: Path,
    plan: FreebaseBlockContextPlan,
    relation_schema_info: Path | None = None,
) -> tuple[dict[int, RelationInfo], Path, list[dict[str, str]]]:
    """Resolve the authoritative relation-id schema for ``link.dat``.

    The aligned SlotGAT preprocessor writes ``info.dat`` as JSON and only stores
    endpoint type-pair labels such as ``BOOK-PEOPLE``.  Those labels are not
    sufficient to recover the original Freebase relation name when several
    relations share the same endpoint types.  The conditional context plan,
    however, records the original raw ``source_dir`` used during preprocessing.
    Its legacy ``info.dat`` contains the exact relation-id/name/type schema and
    corresponds to the byte-identical ``link.dat`` copied into the aligned
    directory.

    An explicit override is tried first, followed by the aligned directory and
    then the context plan's recorded raw source directory.  Candidates that do
    not contain the legacy relation schema are skipped with an audit trail.
    """
    candidates: list[tuple[str, Path]] = []
    if relation_schema_info is not None:
        candidates.append(
            ("explicit_override", Path(relation_schema_info).expanduser().resolve())
        )
    candidates.append(("exact2_directory", exact2_dir / "info.dat"))
    if str(plan.source_dir).strip():
        candidates.append(
            (
                "context_plan_source_dir",
                Path(plan.source_dir).expanduser().resolve() / "info.dat",
            )
        )

    attempts: list[dict[str, str]] = []
    seen: set[str] = set()
    for origin, candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.is_file():
            attempts.append(
                {"origin": origin, "path": key, "status": "missing"}
            )
            continue
        try:
            _node_names, relations = parse_info(candidate)
            _validate_plan_catalog(plan, relations)
        except Exception as error:
            attempts.append(
                {
                    "origin": origin,
                    "path": key,
                    "status": "rejected",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        attempts.append(
            {
                "origin": origin,
                "path": key,
                "status": "selected",
            }
        )
        return relations, candidate, attempts

    rendered = "; ".join(
        f"{row['origin']}={row['path']} ({row['status']}"
        + (f": {row['reason']}" if row.get("reason") else "")
        + ")"
        for row in attempts
    )
    raise ValueError(
        "Could not resolve the authoritative Freebase relation-id schema. "
        "The aligned JSON info.dat contains endpoint type labels, not unique "
        "relation names. Keep the raw directory recorded in context_plan.json, "
        "or pass --relation-schema-info PATH to its legacy info.dat. Attempts: "
        + rendered
    )


def prepare_union_message_schedule(
    *,
    exact2_dir: Path,
    context_plan_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    sort_within_destination: bool = True,
    legacy_pair_collapse: bool = True,
    progress_every_rows: int = 5_000_000,
    progress_every_destinations: int = 10_000,
    relation_schema_info: Path | None = None,
    anchor_context_kind: str | None = "physical_shortcut",
) -> dict[str, Any]:
    """Build a destination-CSR message schedule from one variant's ``link.dat``.

    ``anchor_context_kind`` defaults to ``"physical_shortcut"``, which is the
    Exact-2 anchor required for the SHARED union schedule used by the invariant
    arm. Pass ``None`` to build a schedule for any single variant -- that is how
    the matched per-variant control ("originals") and the universal arm are
    built: same builder, same ordered-pair policy, so the only difference
    between arms is which edges the schedule contains.

    The safety check that actually matters is retained in both cases: the
    ``link.dat`` digest must equal the anchoring plan's ``raw_link_hash``.
    """
    exact2_dir = Path(exact2_dir).expanduser().resolve()
    context_plan_dir = Path(context_plan_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    link_path = exact2_dir / "link.dat"
    if not link_path.is_file():
        raise FileNotFoundError(f"Exact-2 directory is missing link.dat: {exact2_dir}")
    plan = FreebaseBlockContextPlan.load(context_plan_dir)
    if anchor_context_kind is not None and plan.context_kind != anchor_context_kind:
        raise ValueError(
            f"The union schedule must be anchored to a {anchor_context_kind!r} plan; "
            f"found context_kind={plan.context_kind!r}"
        )
    exact2_raw_link_hash = sha256_file(link_path)
    if exact2_raw_link_hash != plan.raw_link_hash:
        raise ValueError(
            "The Exact-2 link.dat used for the union message schedule differs from the "
            "physical context plan. Re-run conditional preprocessing first."
        )
    relations, relation_schema_path, relation_schema_attempts = _resolve_relation_schema(
        exact2_dir=exact2_dir,
        plan=plan,
        relation_schema_info=relation_schema_info,
    )
    lookup = _channel_lookup(plan.channels)
    self_channel = int(plan.self_channel)
    total_nodes = int(plan.layout.total)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is nonempty; pass --overwrite: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: destination indegrees for the forward record, its reverse,
    # and one self record at every node.
    indegree = np.ones(total_nodes, dtype=np.int64)
    forward_edge_count = 0
    ordered_relations = [relation for _rid, relation in sorted(relations.items())]
    relation_counts: dict[str, int] = {relation.name: 0 for relation in ordered_relations}
    for source, destination, relation_id in _iter_link_rows(link_path):
        if not (0 <= source < total_nodes and 0 <= destination < total_nodes):
            raise ValueError("link.dat contains a node outside node.dat")
        relation = relations.get(relation_id)
        if relation is None:
            raise KeyError(f"Unknown relation id {relation_id} in {link_path}")
        indegree[destination] += 1
        indegree[source] += 1
        relation_counts[relation.name] += 1
        forward_edge_count += 1
        if progress_every_rows > 0 and forward_edge_count % int(progress_every_rows) == 0:
            print(
                f"  [union-message count] scanned {forward_edge_count:,} link rows",
                flush=True,
            )

    indptr = np.empty(total_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(indegree, out=indptr[1:])
    message_count = int(indptr[-1])
    candidate_message_count = int(message_count)
    np.save(output_dir / "indptr.npy", indptr)
    source_array = np.lib.format.open_memmap(
        output_dir / "source.npy", mode="w+", dtype=np.int32, shape=(message_count,)
    )
    channel_array = np.lib.format.open_memmap(
        output_dir / "channel.npy", mode="w+", dtype=np.uint16, shape=(message_count,)
    )
    cursor = indptr[:-1].copy()

    # Self records are inserted first and later canonical sorting places them by
    # (channel, source) together with all other incoming records.
    nodes = np.arange(total_nodes, dtype=np.int64)
    positions = cursor.copy()
    source_array[positions] = nodes.astype(np.int32, copy=False)
    channel_array[positions] = np.uint16(self_channel)
    cursor += 1

    # Second pass: place every direct relation edge and its reverse into the
    # destination CSR.  This exactly restores endpoint multiplicity instead of
    # emitting one path aggregate per reachable destination.
    filled_edge_count = 0
    for source, destination, relation_id in _iter_link_rows(link_path):
        relation = relations[relation_id]
        forward_channel = lookup[(relation.name, 1)]
        reverse_channel = lookup[(relation.name, -1)]
        position = int(cursor[destination])
        source_array[position] = np.int32(source)
        channel_array[position] = np.uint16(forward_channel)
        cursor[destination] += 1
        position = int(cursor[source])
        source_array[position] = np.int32(destination)
        channel_array[position] = np.uint16(reverse_channel)
        cursor[source] += 1
        filled_edge_count += 1
        if progress_every_rows > 0 and filled_edge_count % int(progress_every_rows) == 0:
            print(
                f"  [union-message fill] wrote candidates for {filled_edge_count:,} link rows",
                flush=True,
            )

    if not np.array_equal(cursor, indptr[1:]):
        raise AssertionError("Union message CSR fill did not match the counted indegrees")

    # The historical SlotGAT Freebase builder constructs one homogeneous DGL
    # edge per ordered node pair. ``edge2type[(u, v)] = relation`` means the
    # last physical relation wins for parallel physical rows; a generated
    # reverse is added only when no physical reverse pair exists. Reproduce
    # that exact message-set policy by default before canonical ordering.
    direction_by_channel = np.zeros(int(plan.num_channels), dtype=np.int8)
    rank_by_channel = np.full(int(plan.num_channels), np.iinfo(np.int32).max, dtype=np.int32)
    relation_rank = {relation.name: rank for rank, relation in enumerate(ordered_relations)}
    # The channel catalog is the UNION over variants (it names Exact-2's composite
    # relations even when the schedule is being built for `unchanged`, whose link.dat has
    # none of them; a direct lookup would raise KeyError). A channel whose semantic
    # relation is absent from THIS variant's relations can never be filled by the pass
    # above, so its rank is irrelevant; give it the lowest priority and count it.
    absent_rank = len(ordered_relations)
    absent_channel_relations: list[str] = []
    for record in plan.channels:
        channel_id = int(record.channel)
        direction_by_channel[channel_id] = int(record.direction)
        if record.kind == "self":
            rank_by_channel[channel_id] = -1
        else:
            name = str(record.semantic_relation)
            if name in relation_rank:
                rank_by_channel[channel_id] = int(relation_rank[name])
            else:
                rank_by_channel[channel_id] = absent_rank
                absent_channel_relations.append(name)
    if absent_channel_relations:
        print(
            f"  [union-message] {len(absent_channel_relations)} catalog channels name relations "
            f"absent from this variant's link.dat (e.g. {absent_channel_relations[0]}); "
            f"they carry no messages here",
            flush=True,
        )

    duplicate_same_channel_count = 0
    parallel_physical_excess = 0
    generated_reverse_suppressed = 0
    physical_message_count = 0
    reverse_message_count = 0
    self_message_count = 0

    if legacy_pair_collapse:
        collapsed_indptr = np.zeros(total_nodes + 1, dtype=np.int64)
        write_cursor = 0
        relation_count = len(relations)
        for destination in range(total_nodes):
            start_offset, stop_offset = int(indptr[destination]), int(indptr[destination + 1])
            selected_source_array, selected_channel_array, stats = _select_legacy_ordered_pairs(
                np.asarray(source_array[start_offset:stop_offset]),
                np.asarray(channel_array[start_offset:stop_offset]),
                direction_by_channel=direction_by_channel,
                rank_by_channel=rank_by_channel,
                self_channel=self_channel,
                relation_count=relation_count,
            )
            physical_message_count += stats["physical"]
            reverse_message_count += stats["reverse"]
            self_message_count += stats["self"]
            parallel_physical_excess += stats["parallel_physical_excess"]
            generated_reverse_suppressed += stats["generated_reverse_suppressed"]
            duplicate_same_channel_count += stats["duplicate_same_channel"]

            if sort_within_destination and selected_source_array.size:
                canonical = np.lexsort((selected_source_array, selected_channel_array))
                selected_source_array = selected_source_array[canonical]
                selected_channel_array = selected_channel_array[canonical]
            count = int(selected_source_array.size)
            source_array[write_cursor : write_cursor + count] = selected_source_array
            channel_array[write_cursor : write_cursor + count] = selected_channel_array
            write_cursor += count
            collapsed_indptr[destination + 1] = write_cursor
            if (
                progress_every_destinations > 0
                and (destination + 1) % int(progress_every_destinations) == 0
            ):
                print(
                    f"  [union-message collapse] destinations={destination + 1:,}/{total_nodes:,} "
                    f"retained={write_cursor:,}",
                    flush=True,
                )

        message_count = int(write_cursor)
        source_array.flush()
        channel_array.flush()
        final_source = np.lib.format.open_memmap(
            output_dir / "source.final.npy", mode="w+", dtype=np.int32, shape=(message_count,)
        )
        final_channel = np.lib.format.open_memmap(
            output_dir / "channel.final.npy", mode="w+", dtype=np.uint16, shape=(message_count,)
        )
        final_source[:] = source_array[:message_count]
        final_channel[:] = channel_array[:message_count]
        final_source.flush()
        final_channel.flush()
        del source_array, channel_array, final_source, final_channel
        (output_dir / "source.npy").unlink()
        (output_dir / "channel.npy").unlink()
        (output_dir / "source.final.npy").rename(output_dir / "source.npy")
        (output_dir / "channel.final.npy").rename(output_dir / "channel.npy")
        indptr = collapsed_indptr
        np.save(output_dir / "indptr.npy", indptr)
        source_array = np.load(output_dir / "source.npy", mmap_mode="r")
        channel_array = np.load(output_dir / "channel.npy", mmap_mode="r")
        construction_policy = "legacy_slotgat_one_message_per_ordered_pair"
    else:
        duplicate_count = 0
        if sort_within_destination:
            for destination in range(total_nodes):
                start_offset, stop_offset = int(indptr[destination]), int(indptr[destination + 1])
                if stop_offset - start_offset <= 1:
                    continue
                source_segment = np.asarray(source_array[start_offset:stop_offset])
                channel_segment = np.asarray(channel_array[start_offset:stop_offset])
                order = np.lexsort((source_segment, channel_segment))
                sorted_source = source_segment[order].copy()
                sorted_channel = channel_segment[order].copy()
                duplicate_count += int(
                    np.count_nonzero(
                        (sorted_source[1:] == sorted_source[:-1])
                        & (sorted_channel[1:] == sorted_channel[:-1])
                    )
                )
                source_array[start_offset:stop_offset] = sorted_source
                channel_array[start_offset:stop_offset] = sorted_channel
        source_array.flush()
        channel_array.flush()
        if duplicate_count:
            raise ValueError(
                f"Exact-2 link.dat produced {duplicate_count} duplicate (source, channel, destination) "
                "records. Deduplicate the source dataset first."
            )
        physical_message_count = int(forward_edge_count)
        reverse_message_count = int(forward_edge_count)
        self_message_count = int(total_nodes)
        duplicate_same_channel_count = int(duplicate_count)
        construction_policy = "typed_multiedge_plus_all_reverses"

    channel_catalog_hash = sha256_json([record.to_json() for record in plan.channels])
    message_program_hash = sha256_json(
        {
            "operator": OPERATOR_NAME,
            "total_nodes": total_nodes,
            "num_channels": int(plan.num_channels),
            "indptr_sha256": _sha256_array_bytes(indptr),
            "source_sha256": _sha256_array_bytes(source_array),
            "channel_sha256": _sha256_array_bytes(channel_array),
            "canonical_order": "destination_then_channel_then_source",
            "construction_policy": construction_policy,
        }
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "operator": OPERATOR_NAME,
        "source_dir": str(exact2_dir),
        "context_plan_dir": str(context_plan_dir),
        "relation_schema_info": str(relation_schema_path),
        "relation_schema_resolution": relation_schema_attempts,
        "total_nodes": total_nodes,
        "num_channels": int(plan.num_channels),
        "message_count": message_count,
        "candidate_message_count_before_pair_collapse": candidate_message_count,
        "forward_edge_count": int(forward_edge_count),
        "physical_message_count": int(physical_message_count),
        "reverse_message_count": int(reverse_message_count),
        "self_message_count": int(self_message_count),
        "reverse_edge_count": int(reverse_message_count),
        "self_edge_count": int(self_message_count),
        "construction_policy": construction_policy,
        "parallel_physical_record_excess": int(parallel_physical_excess),
        "generated_reverse_candidates_suppressed": int(generated_reverse_suppressed),
        "self_channel": self_channel,
        "canonical_order": "destination_then_channel_then_source",
        "message_program_hash": message_program_hash,
        "exact2_raw_link_hash": exact2_raw_link_hash,
        "channel_catalog_hash": channel_catalog_hash,
        "relation_forward_counts": relation_counts,
        "native_relation_order": [
            {"relation_id": int(relation.rid), "name": relation.name}
            for relation in ordered_relations
        ],
        "native_reference_builder": (
            "edge2type physical overwrite; reverse only if absent; remove/add self"
        ),
        "duplicate_message_count": int(duplicate_same_channel_count),
        "factorized_path_records_used": False,
        "direct_endpoint_records_used": True,
        "raw_context_graph_materialized_as_model_input": False,
        "shared_propagation_schedule": True,
        "semantic_statement": (
            "Distinct raw context plans activate the same direct Exact-2 SlotGAT ordered-pair message schedule."
        ),
    }
    write_json(output_dir / "union_message_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare direct Exact-2 union messages for conditional Freebase SlotGAT"
    )
    parser.add_argument("--exact2-dir", type=Path, required=True)
    parser.add_argument("--context-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--relation-schema-info",
        type=Path,
        default=None,
        help=(
            "Optional path to the legacy raw info.dat containing relation IDs, names, "
            "and endpoint types. By default the builder falls back to source_dir stored "
            "in the Exact-2 context plan when the aligned JSON info.dat is insufficient."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--any-context-kind",
        action="store_true",
        help=(
            "Build a schedule for ANY single variant instead of requiring the "
            "physical_shortcut (Exact-2) anchor. Use this for the matched "
            "per-variant control and the universal arm; the link.dat digest is "
            "still checked against the anchoring plan's raw_link_hash."
        ),
    )
    parser.add_argument("--preserve-link-order", action="store_true")
    parser.add_argument("--progress-every-rows", type=int, default=5_000_000)
    parser.add_argument("--progress-every-destinations", type=int, default=10_000)
    parser.add_argument(
        "--typed-multiedge-union",
        action="store_true",
        help=(
            "Preserve one message per typed link.dat row plus every reverse. The default "
            "reproduces run_freebase_nc.py's one-message-per-ordered-pair behavior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = prepare_union_message_schedule(
        exact2_dir=args.exact2_dir,
        context_plan_dir=args.context_plan_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        sort_within_destination=not args.preserve_link_order,
        legacy_pair_collapse=not args.typed_multiedge_union,
        progress_every_rows=args.progress_every_rows,
        progress_every_destinations=args.progress_every_destinations,
        relation_schema_info=args.relation_schema_info,
        anchor_context_kind=None if args.any_context_kind else "physical_shortcut",
    )
    print("Prepared union-equivalent Freebase SlotGAT messages.")
    print(f"  messages: {payload['message_count']:,}")
    print(f"  raw rows: {payload['forward_edge_count']:,}")
    print(f"  candidates before pair collapse: {payload['candidate_message_count_before_pair_collapse']:,}")
    print(f"  physical: {payload['physical_message_count']:,}")
    print(f"  reverse:  {payload['reverse_message_count']:,}")
    print(f"  policy:   {payload['construction_policy']}")
    print(f"  hash:     {payload['message_program_hash'][:16]}")
    print(f"  output:   {args.output_dir}")


if __name__ == "__main__":
    main()
