"""Raw-context conditional structural bias for genuine invariant SlotGAT.

Unlike the earlier wrapper, this module does *not* materialize one shared DGL
propagation graph.  Every IMDb variant is represented by a distinct
``ConditionalContextPlan`` containing its raw typed edges and the concrete
context witnesses found in that graph.  A genuine conditional SlotGAT layer
receives that plan and maps direct ``Movie--Role`` contexts and two-hop
``Movie--Link--Role`` contexts to the same semantic propagation events at
forward time.

Node types follow ``preprocess_IMDB.py``:
    0 = Movie, 1 = Director, 2 = Actor, 3 = Link, 4 = Year.

The Year type is OPTIONAL: layouts built from a four-count node table (the
older year-less ``IMDB_var1..5`` builds) simply carry zero year nodes, and every year
rule below then contributes nothing. That keeps the older data reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MOVIE = 0
DIRECTOR = 1
ACTOR = 2
LINK = 3
YEAR = 4

# Attribute roles that hang off the Movie/Link pair. Each is projected onto BOTH
# endpoints by the semantic closure, which is what makes the variants indistinguishable.
ROLE_TYPES = (DIRECTOR, ACTOR, YEAR)

CONTEXT_MOVIE_LINK = 0
CONTEXT_DIRECT = 1
CONTEXT_VIA_LINK = 2
CONTEXT_SELF = 3
CONTEXT_KIND_NAMES = {
    CONTEXT_MOVIE_LINK: "movie_link",
    CONTEXT_DIRECT: "direct",
    CONTEXT_VIA_LINK: "via_link",
    CONTEXT_SELF: "self",
}

# Fixed semantic propagation channels.  Physical relation IDs and witness shape
# are deliberately absent from this catalog; the raw context selects a channel
# only inside the conditional layer.
CHANNEL_NAMES: tuple[str, ...] = (
    "movie_to_link",
    "link_to_movie",
    "movie_to_director",
    "director_to_movie",
    "link_to_director",
    "director_to_link",
    "movie_to_actor",
    "actor_to_movie",
    "link_to_actor",
    "actor_to_link",
    "movie_to_year",
    "year_to_movie",
    "link_to_year",
    "year_to_link",
    "self",
)
CHANNEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# Ordered node-type pair -> semantic channel, used by the physical control arm.
PHYSICAL_CHANNEL_BY_TYPE_PAIR: dict[tuple[int, int], str] = {
    (MOVIE, LINK): "movie_to_link",
    (LINK, MOVIE): "link_to_movie",
    (MOVIE, DIRECTOR): "movie_to_director",
    (DIRECTOR, MOVIE): "director_to_movie",
    (LINK, DIRECTOR): "link_to_director",
    (DIRECTOR, LINK): "director_to_link",
    (MOVIE, ACTOR): "movie_to_actor",
    (ACTOR, MOVIE): "actor_to_movie",
    (LINK, ACTOR): "link_to_actor",
    (ACTOR, LINK): "actor_to_link",
    (MOVIE, YEAR): "movie_to_year",
    (YEAR, MOVIE): "year_to_movie",
    (LINK, YEAR): "link_to_year",
    (YEAR, LINK): "year_to_link",
}

# Semantic channel pair emitted for each role, as (movie-side, link-side) prefixes.
ROLE_CHANNELS: dict[int, tuple[str, str, str, str]] = {
    DIRECTOR: ("movie_to_director", "director_to_movie", "link_to_director", "director_to_link"),
    ACTOR: ("movie_to_actor", "actor_to_movie", "link_to_actor", "actor_to_link"),
    YEAR: ("movie_to_year", "year_to_movie", "link_to_year", "year_to_link"),
}


@dataclass(frozen=True)
class NodeTypeLayout:
    """Contiguous global-ID ranges for IMDb's node types.

    Four counts (movie, director, actor, link) or five (plus year). The four-count
    form is the older year-less build and is still accepted so those runs reproduce;
    it simply reports zero year nodes.
    """

    counts: tuple[int, ...]
    offsets: tuple[int, ...]
    total: int

    @classmethod
    def from_counts(cls, counts: Sequence[int] | Mapping[int, int]) -> "NodeTypeLayout":
        if isinstance(counts, Mapping):
            n = 5 if 4 in counts else 4
            values = tuple(int(counts[i]) for i in range(n))
        else:
            values = tuple(int(value) for value in counts)
        if len(values) not in (4, 5):
            raise ValueError(f"Expected four or five IMDb node-type counts, got {values}")
        if any(value <= 0 for value in values):
            raise ValueError(f"All node-type counts must be positive, got {values}")
        offsets = tuple(int(sum(values[:i])) for i in range(len(values)))
        return cls(values, offsets, int(sum(values)))

    @property
    def num_years(self) -> int:
        return self.counts[YEAR] if len(self.counts) > YEAR else 0

    def type_of(self, node_id: int) -> int:
        node_id = int(node_id)
        if not 0 <= node_id < self.total:
            raise IndexError(f"Node id {node_id} is outside [0, {self.total})")
        for node_type in range(len(self.offsets) - 1, 0, -1):
            if node_id >= self.offsets[node_type]:
                return node_type
        return MOVIE

    def ids_for_type(self, node_type: int) -> range:
        node_type = int(node_type)
        if node_type >= len(self.counts):
            return range(0)
        start = self.offsets[node_type]
        return range(start, start + self.counts[node_type])

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": list(self.counts),
            "offsets": list(self.offsets),
            "total": self.total,
        }


@dataclass(frozen=True, order=True)
class RawTypedEdge:
    src: int
    dst: int
    relation: int


@dataclass(frozen=True, order=True)
class SemanticMessage:
    channel: int
    src: int
    dst: int


@dataclass(frozen=True, order=True)
class ConditionalWitness:
    """One variant-specific context occurrence.

    ``context_nodes`` is intentionally retained.  Direct and Link-mediated
    variants therefore have different model-facing witness records even when
    they induce the same semantic source/destination messages.
    """

    family: int
    context_kind: int
    movie: int
    peer: int
    link: int
    context_nodes: tuple[int, int, int]
    context_relations: tuple[int, int]

    def integer_row(self) -> tuple[int, ...]:
        return (
            self.family,
            self.context_kind,
            self.movie,
            self.peer,
            self.link,
            *self.context_nodes,
            *self.context_relations,
        )


@dataclass
class ConditionalContextPlan:
    """Distinct raw graph plus conditional occurrences used by the model.

    The plan never stores a DGL semantic graph.  ``emit_semantic_messages``
    executes the context-to-propagation rules by branching on each witness.
    """

    layout: NodeTypeLayout
    raw_edges: tuple[RawTypedEdge, ...]
    witnesses: tuple[ConditionalWitness, ...]
    raw_graph_hash: str
    context_plan_hash: str
    semantic_propagation_hash: str
    context_audit: dict[str, Any] = field(default_factory=dict)
    # "semantic_closure": conditional rules map every context to the shared
    # semantic channel set (the invariant arm).
    # "physical": messages follow this variant's own edges only (the matched,
    # NON-invariant control -- same model class, same parameter count, same
    # channel vocabulary, so the message set is the single manipulated variable).
    mode: str = "semantic_closure"

    @property
    def num_edge_types(self) -> int:
        return len(CHANNEL_NAMES)

    @property
    def message_program_hash(self) -> str:
        """Compatibility name used by older aggregators."""
        return self.semantic_propagation_hash

    @property
    def messages(self) -> tuple[SemanticMessage, ...]:
        """Compatibility property; messages are generated from raw witnesses."""
        return self.emit_semantic_messages(verify=True)

    @property
    def src(self) -> np.ndarray:
        return np.fromiter((item.src for item in self.messages), dtype=np.int64)

    @property
    def dst(self) -> np.ndarray:
        return np.fromiter((item.dst for item in self.messages), dtype=np.int64)

    @property
    def channel_ids(self) -> np.ndarray:
        return np.fromiter((item.channel for item in self.messages), dtype=np.int64)

    def raw_edge_set(self) -> set[tuple[int, int]]:
        return {(edge.src, edge.dst) for edge in self.raw_edges}

    def _has_undirected(self, left: int, right: int, edge_set: set[tuple[int, int]]) -> bool:
        return (left, right) in edge_set or (right, left) in edge_set

    def validate_witnesses(self) -> None:
        edge_set = self.raw_edge_set()
        for witness in self.witnesses:
            if witness.context_kind == CONTEXT_SELF:
                if witness.movie != witness.peer:
                    raise AssertionError("Malformed self witness")
                continue
            if witness.context_kind == CONTEXT_MOVIE_LINK:
                if not self._has_undirected(witness.movie, witness.link, edge_set):
                    raise AssertionError(
                        f"Movie--Link witness is absent from raw graph: {witness}"
                    )
                continue
            if witness.context_kind == CONTEXT_DIRECT:
                if not self._has_undirected(witness.movie, witness.peer, edge_set):
                    raise AssertionError(
                        f"Direct conditional witness is absent from raw graph: {witness}"
                    )
                continue
            if witness.context_kind == CONTEXT_VIA_LINK:
                if not self._has_undirected(witness.movie, witness.link, edge_set):
                    raise AssertionError(
                        f"First edge of two-hop witness is absent: {witness}"
                    )
                if not self._has_undirected(witness.link, witness.peer, edge_set):
                    raise AssertionError(
                        f"Second edge of two-hop witness is absent: {witness}"
                    )
                continue
            raise AssertionError(f"Unknown context kind: {witness.context_kind}")

    def _emit_physical_messages(self) -> tuple[SemanticMessage, ...]:
        """Messages that follow this variant's own edges, with no closure.

        The channel vocabulary is unchanged, so a variant simply leaves the
        channels it does not physically contain empty. Model shape and
        parameter count are therefore identical to the invariant arm.
        """
        messages: set[SemanticMessage] = set()
        for edge in self.raw_edges:
            pair = (self.layout.type_of(edge.src), self.layout.type_of(edge.dst))
            channel = PHYSICAL_CHANNEL_BY_TYPE_PAIR.get(pair)
            if channel is None:
                raise AssertionError(
                    f"Raw edge {edge} has unsupported node-type pair {pair}"
                )
            messages.add(SemanticMessage(CHANNEL_TO_ID[channel], edge.src, edge.dst))
        for node_id in range(self.layout.total):
            messages.add(SemanticMessage(CHANNEL_TO_ID["self"], node_id, node_id))
        return tuple(sorted(messages, key=lambda item: (item.dst, item.channel, item.src)))

    def emit_semantic_messages(self, *, verify: bool = False) -> tuple[SemanticMessage, ...]:
        """Evaluate conditional rules against this raw context plan.

        This is intentionally a runtime rule evaluator rather than a shared
        graph constructor.  The output is sorted only after every distinct raw
        witness has been interpreted.
        """

        if verify:
            self.validate_witnesses()
        if self.mode == "physical":
            return self._emit_physical_messages()
        if self.mode != "semantic_closure":
            raise ValueError(f"Unknown context-plan mode: {self.mode!r}")
        messages: set[SemanticMessage] = set()

        def add(channel_name: str, src: int, dst: int) -> None:
            messages.add(
                SemanticMessage(CHANNEL_TO_ID[channel_name], int(src), int(dst))
            )

        for witness in self.witnesses:
            if witness.context_kind == CONTEXT_SELF:
                add("self", witness.movie, witness.movie)
                continue
            if witness.context_kind == CONTEXT_MOVIE_LINK:
                add("movie_to_link", witness.movie, witness.link)
                add("link_to_movie", witness.link, witness.movie)
                continue

            # The direct and two-hop branches have different raw contexts but
            # deliberately induce the same propagation closure. Every role is
            # projected onto BOTH endpoints -- including the year, whose movie-side
            # edge is present in every variant, so link -> movie -> year is derivable
            # identically everywhere.
            channels = ROLE_CHANNELS.get(witness.family)
            if channels is None:
                raise AssertionError(f"Unexpected witness family: {witness.family}")
            movie_fwd, movie_bwd, link_fwd, link_bwd = channels
            add(movie_fwd, witness.movie, witness.peer)
            add(movie_bwd, witness.peer, witness.movie)
            add(link_fwd, witness.link, witness.peer)
            add(link_bwd, witness.peer, witness.link)

        return tuple(sorted(messages, key=lambda item: (item.dst, item.channel, item.src)))

    def emit_semantic_numpy(self, *, verify: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        records = self.emit_semantic_messages(verify=verify)
        src = np.fromiter((record.src for record in records), dtype=np.int64)
        dst = np.fromiter((record.dst for record in records), dtype=np.int64)
        channels = np.fromiter((record.channel for record in records), dtype=np.int64)
        return src, dst, channels

    def channel_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CHANNEL_NAMES}
        for message in self.emit_semantic_messages():
            counts[CHANNEL_NAMES[message.channel]] += 1
        return counts

    def witness_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CONTEXT_KIND_NAMES.values()}
        for witness in self.witnesses:
            counts[CONTEXT_KIND_NAMES[witness.context_kind]] += 1
        return counts

    def to_manifest(self) -> dict[str, Any]:
        return {
            "format_version": "slotgat_raw_conditional_context_v2",
            "mode": self.mode,
            "layout": self.layout.as_dict(),
            "channels": list(CHANNEL_NAMES),
            "context_kinds": CONTEXT_KIND_NAMES,
            "raw_edge_count": len(self.raw_edges),
            "conditional_witness_count": len(self.witnesses),
            "semantic_message_count": len(self.emit_semantic_messages()),
            "channel_counts": self.channel_counts(),
            "witness_counts": self.witness_counts(),
            "raw_graph_hash": self.raw_graph_hash,
            "context_plan_hash": self.context_plan_hash,
            "semantic_propagation_hash": self.semantic_propagation_hash,
            "materialized_shared_dgl_graph": False,
            "context_audit": self.context_audit,
        }

    def save_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n")


# Backward-compatible type name; the object is now a raw conditional plan, not a
# canonical message graph.
ConditionalMessageProgram = ConditionalContextPlan


def _sha256_int_rows(rows: Iterable[Sequence[int]], *, width: int) -> str:
    rows_list = list(rows)
    arr = np.asarray(rows_list, dtype="<i8")
    if arr.size == 0:
        arr = np.empty((0, width), dtype="<i8")
    arr = arr.reshape((-1, width))
    header = np.asarray([arr.shape[0], arr.shape[1]], dtype="<i8").tobytes()
    return hashlib.sha256(header + arr.tobytes(order="C")).hexdigest()


def raw_edges_from_data_loader(dl: Any) -> tuple[NodeTypeLayout, tuple[RawTypedEdge, ...]]:
    layout = NodeTypeLayout.from_counts(dl.nodes["count"])
    edges: list[RawTypedEdge] = []
    for relation_key, matrix in sorted(dl.links["data"].items(), key=lambda item: int(item[0])):
        coo = matrix.tocoo(copy=False)
        relation = int(relation_key)
        edges.extend(
            RawTypedEdge(int(src), int(dst), relation)
            for src, dst in zip(coo.row.tolist(), coo.col.tolist())
        )
    return layout, tuple(sorted(set(edges)))


def _undirected_pairs_between(
    raw_edges: Iterable[RawTypedEdge],
    layout: NodeTypeLayout,
    left_type: int,
    right_type: int,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for edge in raw_edges:
        src_type = layout.type_of(edge.src)
        dst_type = layout.type_of(edge.dst)
        if src_type == left_type and dst_type == right_type:
            pairs.add((edge.src, edge.dst))
        elif src_type == right_type and dst_type == left_type:
            pairs.add((edge.dst, edge.src))
    return pairs


def _build_pair_relation_index(
    raw_edges: Iterable[RawTypedEdge],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Undirected node pair -> sorted relation ids, built in one pass.

    The previous ``_relations_for_pair`` rescanned every raw edge for each pair,
    which is O(pairs x edges): ~21k pairs x 42k edges = ~9e8 comparisons per
    variant, minutes of wall clock. This index makes compilation linear.
    """
    index: dict[tuple[int, int], set[int]] = {}
    for edge in raw_edges:
        key = (edge.src, edge.dst) if edge.src <= edge.dst else (edge.dst, edge.src)
        index.setdefault(key, set()).add(edge.relation)
    return {key: tuple(sorted(values)) for key, values in index.items()}


def _relations_for_pair(
    pair_relations: Mapping[tuple[int, int], tuple[int, ...]], left: int, right: int
) -> tuple[int, ...]:
    key = (left, right) if left <= right else (right, left)
    return pair_relations.get(key, ())


def _validate_movie_link_bijection(
    movie_link_pairs: set[tuple[int, int]],
    layout: NodeTypeLayout,
) -> tuple[dict[int, int], dict[int, int]]:
    movie_to_link: dict[int, int] = {}
    link_to_movie: dict[int, int] = {}
    for movie, link in sorted(movie_link_pairs):
        previous_link = movie_to_link.setdefault(movie, link)
        if previous_link != link:
            raise ValueError(f"Movie {movie} maps to multiple Link nodes")
        previous_movie = link_to_movie.setdefault(link, movie)
        if previous_movie != movie:
            raise ValueError(f"Link {link} maps to multiple Movie nodes")
    missing_movies = sorted(set(layout.ids_for_type(MOVIE)) - set(movie_to_link))
    missing_links = sorted(set(layout.ids_for_type(LINK)) - set(link_to_movie))
    if missing_movies or missing_links:
        raise ValueError(
            "Movie--Link context is not bijective: "
            f"missing_movies={len(missing_movies)}, missing_links={len(missing_links)}"
        )
    return movie_to_link, link_to_movie


def _build_role_witnesses(
    *,
    raw_edges: tuple[RawTypedEdge, ...],
    layout: NodeTypeLayout,
    role_type: int,
    movie_to_link: Mapping[int, int],
    link_to_movie: Mapping[int, int],
    pair_relations: Mapping[tuple[int, int], tuple[int, ...]],
) -> tuple[list[ConditionalWitness], dict[str, Any]]:
    direct = _undirected_pairs_between(raw_edges, layout, MOVIE, role_type)
    link_role = _undirected_pairs_between(raw_edges, layout, LINK, role_type)
    via_link = {(link_to_movie[link], role) for link, role in link_role}
    semantic = direct | via_link

    witnesses: list[ConditionalWitness] = []
    direct_selected = 0
    via_selected = 0
    both_available = 0
    for movie, role in sorted(semantic):
        link = movie_to_link[movie]
        if (movie, role) in direct:
            # Prefer the one-edge occurrence when both are physically present;
            # set semantics still emits one semantic relationship.
            relations = _relations_for_pair(pair_relations, movie, role)
            relation = relations[0] if relations else -1
            witnesses.append(
                ConditionalWitness(
                    family=role_type,
                    context_kind=CONTEXT_DIRECT,
                    movie=movie,
                    peer=role,
                    link=link,
                    context_nodes=(movie, role, -1),
                    context_relations=(relation, -1),
                )
            )
            direct_selected += 1
            if (movie, role) in via_link:
                both_available += 1
        else:
            movie_link_rel = _relations_for_pair(pair_relations, movie, link)
            link_role_rel = _relations_for_pair(pair_relations, link, role)
            witnesses.append(
                ConditionalWitness(
                    family=role_type,
                    context_kind=CONTEXT_VIA_LINK,
                    movie=movie,
                    peer=role,
                    link=link,
                    context_nodes=(movie, link, role),
                    context_relations=(
                        movie_link_rel[0] if movie_link_rel else -1,
                        link_role_rel[0] if link_role_rel else -1,
                    ),
                )
            )
            via_selected += 1

    audit = {
        "direct_pair_count": len(direct),
        "via_link_pair_count": len(via_link),
        "semantic_pair_count": len(semantic),
        "selected_direct_witness_count": direct_selected,
        "selected_via_link_witness_count": via_selected,
        "both_contexts_available_count": both_available,
        "direct_via_symmetric_difference_count": len(direct ^ via_link),
        "direct_via_symmetric_difference_examples": [
            list(item) for item in sorted(direct ^ via_link)[:20]
        ],
    }
    return witnesses, audit


def compile_conditional_context_plan(
    layout: NodeTypeLayout,
    raw_edges: Iterable[RawTypedEdge],
    *,
    strict: bool = True,
) -> ConditionalContextPlan:
    """Build a variant-specific conditional context plan.

    This function may run as preprocessing, but it only indexes raw occurrences.
    It does not create a union graph or a model-facing semantic adjacency.
    """

    raw_tuple = tuple(sorted(set(raw_edges)))
    if not raw_tuple:
        raise ValueError("Cannot compile an empty raw graph")
    raw_hash = _sha256_int_rows(
        ((edge.relation, edge.src, edge.dst) for edge in raw_tuple), width=3
    )

    pair_relations = _build_pair_relation_index(raw_tuple)
    movie_link_pairs = _undirected_pairs_between(raw_tuple, layout, MOVIE, LINK)
    movie_to_link, link_to_movie = _validate_movie_link_bijection(
        movie_link_pairs, layout
    )

    witnesses: list[ConditionalWitness] = []
    for movie, link in sorted(movie_link_pairs):
        relations = _relations_for_pair(pair_relations, movie, link)
        witnesses.append(
            ConditionalWitness(
                family=LINK,
                context_kind=CONTEXT_MOVIE_LINK,
                movie=movie,
                peer=link,
                link=link,
                context_nodes=(movie, link, -1),
                context_relations=(relations[0] if relations else -1, -1),
            )
        )

    role_audits: dict[int, dict[str, Any]] = {}
    for role_type in ROLE_TYPES:
        # The year block is absent from the older year-less builds; skip it there
        # rather than failing, so those runs still reproduce.
        if role_type == YEAR and layout.num_years == 0:
            continue
        role_witnesses, role_audits[role_type] = _build_role_witnesses(
            raw_edges=raw_tuple,
            layout=layout,
            role_type=role_type,
            movie_to_link=movie_to_link,
            link_to_movie=link_to_movie,
            pair_relations=pair_relations,
        )
        if strict and not role_witnesses:
            raise ValueError(f"No Movie--{role_type} associations were recovered")
        witnesses.extend(role_witnesses)
    director_audit = role_audits.get(DIRECTOR, {})
    actor_audit = role_audits.get(ACTOR, {})
    year_audit = role_audits.get(YEAR, {})

    for node_id in range(layout.total):
        witnesses.append(
            ConditionalWitness(
                family=layout.type_of(node_id),
                context_kind=CONTEXT_SELF,
                movie=node_id,
                peer=node_id,
                link=-1,
                context_nodes=(node_id, -1, -1),
                context_relations=(-1, -1),
            )
        )

    ordered_witnesses = tuple(
        sorted(
            witnesses,
            key=lambda item: (
                item.context_kind,
                item.family,
                item.movie,
                item.peer,
                item.link,
                item.context_nodes,
                item.context_relations,
            ),
        )
    )
    context_plan_hash = _sha256_int_rows(
        (item.integer_row() for item in ordered_witnesses), width=10
    )

    provisional = ConditionalContextPlan(
        layout=layout,
        raw_edges=raw_tuple,
        witnesses=ordered_witnesses,
        raw_graph_hash=raw_hash,
        context_plan_hash=context_plan_hash,
        semantic_propagation_hash="",
        context_audit={},
    )
    provisional.validate_witnesses()
    semantic_messages = provisional.emit_semantic_messages()
    semantic_hash = _sha256_int_rows(
        ((item.channel, item.src, item.dst) for item in semantic_messages), width=3
    )
    provisional.semantic_propagation_hash = semantic_hash
    provisional.context_audit = {
        "movie_link_pair_count": len(movie_link_pairs),
        "movie_link_is_bijection": True,
        "director": director_audit,
        "actor": actor_audit,
        "year": year_audit,
        "raw_context_retained": True,
        "conditional_rules_evaluated_in_layer": True,
        "materialized_shared_dgl_graph": False,
        "semantic_order": "(dst, channel, src)",
    }
    return provisional


def compile_physical_context_plan(
    layout: NodeTypeLayout,
    raw_edges: Iterable[RawTypedEdge],
    *,
    strict: bool = True,
) -> ConditionalContextPlan:
    """Matched NON-invariant control: propagate on this variant's own edges.

    Identical to :func:`compile_conditional_context_plan` in every respect --
    same witnesses, same channel vocabulary, same model class and parameter
    count -- except that ``emit_semantic_messages`` returns the variant's
    physical edges instead of the semantic closure. This is the arm the
    invariant arm should be compared against, because the message set is then
    the only manipulated variable.
    """
    plan = compile_conditional_context_plan(layout, raw_edges, strict=strict)
    plan.mode = "physical"
    messages = plan.emit_semantic_messages()
    plan.semantic_propagation_hash = _sha256_int_rows(
        ((item.channel, item.src, item.dst) for item in messages), width=3
    )
    plan.context_audit = dict(plan.context_audit)
    plan.context_audit.update(
        {
            "mode": "physical",
            "physical_message_count": len(messages),
            "semantic_closure_applied": False,
        }
    )
    return plan


def compile_conditional_message_program(
    layout: NodeTypeLayout,
    raw_edges: Iterable[RawTypedEdge],
    *,
    strict: bool = True,
) -> ConditionalContextPlan:
    """Backward-compatible entry point returning a raw context plan."""
    return compile_conditional_context_plan(layout, raw_edges, strict=strict)


def compile_from_data_loader(dl: Any, *, strict: bool = True) -> ConditionalContextPlan:
    layout, raw_edges = raw_edges_from_data_loader(dl)
    return compile_conditional_context_plan(layout, raw_edges, strict=strict)


def compile_physical_from_data_loader(dl: Any, *, strict: bool = True) -> ConditionalContextPlan:
    layout, raw_edges = raw_edges_from_data_loader(dl)
    return compile_physical_context_plan(layout, raw_edges, strict=strict)


def assert_context_plans_invariant(
    plans: Mapping[str, ConditionalContextPlan],
    *,
    require_distinct_raw_graphs: bool = True,
    require_distinct_context_plans: bool = True,
) -> None:
    if not plans:
        raise ValueError("No context plans were provided")
    semantic_hashes = {plan.semantic_propagation_hash for plan in plans.values()}
    if len(semantic_hashes) != 1:
        detail = {name: plan.semantic_propagation_hash for name, plan in plans.items()}
        raise AssertionError(f"Conditional propagation semantics differ: {detail}")
    if require_distinct_raw_graphs:
        raw_hashes = {plan.raw_graph_hash for plan in plans.values()}
        if len(raw_hashes) != len(plans):
            raise AssertionError(
                "Raw graph hashes are not all distinct; the experiment would not "
                "demonstrate separate transformed inputs"
            )
    if require_distinct_context_plans:
        plan_hashes = {plan.context_plan_hash for plan in plans.values()}
        if len(plan_hashes) != len(plans):
            raise AssertionError(
                "Conditional witness plans are not all distinct; direct and two-hop "
                "contexts may have been canonicalized too early"
            )


def assert_programs_invariant(
    programs: Mapping[str, ConditionalContextPlan],
    *,
    require_distinct_raw_graphs: bool = True,
) -> None:
    assert_context_plans_invariant(
        programs,
        require_distinct_raw_graphs=require_distinct_raw_graphs,
        require_distinct_context_plans=require_distinct_raw_graphs,
    )


def build_slotgat_graph(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "build_slotgat_graph was removed from the genuine conditional path. "
        "Use GenuineConditionalSlotGAT with a distinct ConditionalContextPlan."
    )
