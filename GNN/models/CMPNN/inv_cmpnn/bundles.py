"""Load the RGCN structural bundles as CMPNN link-prediction inputs.

The compiled semantic message program is a property of (dataset, transformation), not of
the model, so CMPNN reads the same per-variant bundles the invariant RGCN was trained on
(see ``models/CMPNN/README.md``):

* IMDb / DBLP: ``inv_rgcn.data.ConditionalGraphData`` (``graph.pt`` per variant)
* WordNet:     the ``wordnet_rgcn_conditional_universal_parity_v2`` dict

Both are turned into one :class:`LPBundle`. Two CMPNN-specific transformations happen
here and are hashed so the gauntlet can check them across variants:

1. **Forward-half relation selection.** The programs carry both directions of every
   relation (``movie_to_director`` and ``director_to_movie``; ``base::r`` and
   ``inverse::r__inv``). CMPNN adds inverses itself (``graph.undirected(add_inverse=True)``),
   so feeding both halves would encode every fact twice under two labels. Only the
   forward member of each pair is kept; the model's own ``r + R`` inverse covers the rest.
   This reproduces the original CMPNN runners' convention (forward-only triplets) for
   IMDb/DBLP physical graphs, and is applied to every semantic program. The WordNet
   *physical* graph is the exception: its named inverse / shortcut relations ARE the
   variant, so it is passed raw with its own relation vocabulary (stock behaviour).
2. **Alias map for easy-edge masking.** On IMDb the Link node is a bijective copy of the
   Movie node, and the program carries ``link_to_director(link(m), d)`` for training
   pairs. When CMPNN masks the query edge ``(m, d)`` during training it must mask
   ``(link(m), d)`` too, or a training-only 2-hop path to the label remains (RGCN guide
   failure mode #10). ``alias[n]`` is the copy of node ``n`` (itself where none).
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_RGCN_DIR = _HERE.parent.parent.parent / "common"
if str(_RGCN_DIR) not in sys.path:
    sys.path.insert(0, str(_RGCN_DIR))  # inv_rgcn (bundle dataclass), rgcn_resource_metrics

from inv_rgcn.data import ConditionalGraphData, load_graph_data  # noqa: E402


def tensor_sha256(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def canonical_edge_order(src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """Permutation sorting edges by (rel, dst, src). One fixed order for every variant."""
    key = (rel.to(torch.int64) * (int(dst.max()) + 1 if dst.numel() else 1) + dst.to(torch.int64))
    key = key * (int(src.max()) + 1 if src.numel() else 1) + src.to(torch.int64)
    return torch.argsort(key, stable=True)


@dataclass
class LPBundle:
    dataset: str            # IMDB | DBLP | WORDNET
    task: str               # movie_year | movie_director | paper_conference | link_prediction
    variant: str
    num_nodes: int
    # relation vocabularies (index = CMPNN relation id), one per graph; identical for
    # IMDb/DBLP, different for WordNet (raw physical vocabulary vs forward-half semantic)
    physical_relation_names: List[str]
    semantic_relation_names: List[str]
    # edges, each [E, 3] int64 (src, dst, rel), canonical order, duplicate-free
    physical_edges: torch.Tensor
    semantic_edges: torch.Tensor
    # supervision: [N, 2] (head, tail) for the single-relation tasks, [N, 3] (h, r, t)
    # for WordNet; negatives are tails [N, K] (IMDb/DBLP) or None (WordNet: ranked
    # against all entities, filtered)
    train_pos: torch.Tensor
    val_pos: torch.Tensor
    test_pos: torch.Tensor
    train_neg: Optional[torch.Tensor]
    val_neg: Optional[torch.Tensor]
    test_neg: Optional[torch.Tensor]
    # relation id queried by the task, per graph (None: WordNet, per-triple in *_pos[:, 1])
    physical_target_relation: Optional[int]
    semantic_target_relation: Optional[int]
    node_type: Optional[torch.Tensor]  # [num_nodes] for IMDb/DBLP; None for WordNet
    alias: torch.Tensor                # [num_nodes] copy-of map (identity where none)
    candidate_pool: Optional[torch.Tensor]  # DBLP: all conference node ids (full ranking)
    # relation NAMES whose (head, *) edges are masked with the query during training
    label_derived_relations: List[str]
    # provenance
    source_physical_hash: str
    source_semantic_hash: str
    source_supervision_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- derived hashes (what the gauntlet compares across variants) ----
    def physical_hash(self) -> str:
        return tensor_sha256(self.physical_edges)

    def semantic_hash(self) -> str:
        return tensor_sha256(self.semantic_edges)

    def supervision_hash(self) -> str:
        parts = [self.train_pos, self.val_pos, self.test_pos]
        for t in (self.train_neg, self.val_neg, self.test_neg):
            if t is not None:
                parts.append(t)
        if self.candidate_pool is not None:
            parts.append(self.candidate_pool)
        return tensor_sha256(*parts) + f":target={self.semantic_target_relation}"

    def relation_hash(self, graph: str = "semantic") -> str:
        return hashlib.sha256("\n".join(self.relation_names(graph)).encode()).hexdigest()

    def relation_names(self, graph: str) -> List[str]:
        return self.semantic_relation_names if graph == "semantic" else self.physical_relation_names

    def target_relation(self, graph: str) -> Optional[int]:
        return self.semantic_target_relation if graph == "semantic" else self.physical_target_relation

    def num_relation(self, graph: str) -> int:
        return len(self.relation_names(graph))

    def positives(self, split: str, graph: str) -> torch.Tensor:
        """[N, 3] (h, r, t) triples in the relation ids of ``graph``."""
        pos = {"train": self.train_pos, "val": self.val_pos, "test": self.test_pos}[split]
        if pos.shape[1] == 3:
            if graph == "semantic" or self.dataset != "WORDNET":
                return pos
            # WordNet physical vocabulary: base relation i is named without the prefix
            m = self.metadata["physical_task_relation_ids"]
            r = torch.tensor(m, dtype=torch.int64)[torch.tensor(self.metadata["task_relation_index_of_semantic_id"], dtype=torch.int64)[pos[:, 1]]]
            return torch.stack([pos[:, 0], r, pos[:, 2]], 1)
        r = torch.full((pos.shape[0],), int(self.target_relation(graph)), dtype=torch.int64)
        return torch.stack([pos[:, 0], r, pos[:, 1]], 1)

    def alias_hash(self) -> str:
        return tensor_sha256(self.alias) + ":" + ",".join(self.label_derived_relations)

    def label_derived_ids(self, graph: str) -> List[int]:
        names = self.relation_names(graph)
        return [names.index(n) for n in self.label_derived_relations if n in names]

    def edges(self, graph: str) -> torch.Tensor:
        if graph == "semantic":
            return self.semantic_edges
        if graph == "physical":
            return self.physical_edges
        raise ValueError(f"graph must be 'semantic' or 'physical', got {graph!r}")

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "task": self.task, "variant": self.variant,
            "num_nodes": self.num_nodes, "num_relation_physical": self.num_relation("physical"),
            "num_relation_semantic": self.num_relation("semantic"),
            "num_physical_edges": int(self.physical_edges.shape[0]),
            "num_semantic_edges": int(self.semantic_edges.shape[0]),
            "train_pos": int(self.train_pos.shape[0]), "val_pos": int(self.val_pos.shape[0]),
            "test_pos": int(self.test_pos.shape[0]),
            "neg_k": None if self.train_neg is None else int(self.train_neg.shape[1]),
            "target_relation": self.semantic_target_relation,
            "physical_hash": self.physical_hash(), "semantic_hash": self.semantic_hash(),
            "supervision_hash": self.supervision_hash(),
            "relation_hash_physical": self.relation_hash("physical"),
            "relation_hash_semantic": self.relation_hash("semantic"),
            "alias_hash": self.alias_hash(),
            "source_physical_hash": self.source_physical_hash,
            "source_semantic_hash": self.source_semantic_hash,
        }


# --------------------------------------------------------------------------------------
# forward-half selection
# --------------------------------------------------------------------------------------

def _forward_half_pairs(names: Sequence[str]) -> Tuple[List[int], Dict[int, int]]:
    """Return (forward relation ids in vocabulary order, reverse_id -> forward_id).

    IMDb/DBLP names are ``a_to_b`` / ``b_to_a``: the first of each unordered pair in
    vocabulary order is the forward member. WordNet names are ``base::r``,
    ``inverse::r__inv``, ``shortcut::...``: ``inverse::`` entries are the reverse of the
    ``base::`` entry with the same core name; everything else is forward.
    """
    forward: List[int] = []
    reverse_of: Dict[int, int] = {}
    seen: Dict[str, int] = {}
    for rid, name in enumerate(names):
        if "::" in name:
            kind, core = name.split("::", 1)
            if kind == "inverse":
                base = core[:-len("__inv")] if core.endswith("__inv") else core
                fwd = seen.get(f"base::{base}")
                if fwd is None:
                    raise ValueError(f"inverse relation {name!r} has no base:: partner")
                reverse_of[rid] = fwd
            else:
                seen[name] = rid
                forward.append(rid)
        elif "_to_" in name:
            a, b = name.split("_to_", 1)
            partner = f"{b}_to_{a}"
            if partner in seen and partner != name:
                reverse_of[rid] = seen[partner]
            else:
                seen[name] = rid
                forward.append(rid)
        else:
            seen[name] = rid
            forward.append(rid)
    return forward, reverse_of


def _select_forward(edges: torch.Tensor, names: Sequence[str]) -> Tuple[torch.Tensor, List[str], Dict[int, int]]:
    """Keep forward-half edges, remap relation ids densely, sort canonically, dedupe.

    Reverse-half edges are *verified* to be the mirror of their forward partner (same
    (dst, src) set) and then dropped — if a reverse relation carried a pair whose forward
    partner is absent, dropping it would lose information, so that is an error.
    """
    forward, reverse_of = _forward_half_pairs(names)
    new_id = {old: i for i, old in enumerate(forward)}
    src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
    keep = torch.zeros(len(names), dtype=torch.bool)
    keep[forward] = True
    fwd_edges = edges[keep[rel]]
    fwd_edges = torch.stack(
        [fwd_edges[:, 0], fwd_edges[:, 1],
         torch.tensor([new_id[int(r)] for r in fwd_edges[:, 2].tolist()], dtype=torch.int64)], dim=1
    ) if fwd_edges.numel() else fwd_edges.reshape(0, 3)
    # mirror check
    fwd_set = {(int(s), int(d), int(r)) for s, d, r in fwd_edges.tolist()}
    for rev, fwd in reverse_of.items():
        m = rel == rev
        for s, d in zip(src[m].tolist(), dst[m].tolist()):
            if (d, s, new_id[fwd]) not in fwd_set:
                raise ValueError(
                    f"reverse relation {names[rev]!r} carries ({s}->{d}) whose forward partner "
                    f"{names[fwd]!r} ({d}->{s}) is absent; cannot drop the reverse half"
                )
    order = canonical_edge_order(fwd_edges[:, 0], fwd_edges[:, 1], fwd_edges[:, 2])
    fwd_edges = fwd_edges[order]
    # duplicate-free (set semantics) — required so PNA degrees are variant-independent
    if fwd_edges.shape[0] > 1:
        same = (fwd_edges[1:] == fwd_edges[:-1]).all(dim=1)
        if bool(same.any()):
            raise ValueError(f"{int(same.sum())} duplicate (src,dst,rel) edges in the program")
    return fwd_edges.contiguous(), [names[i] for i in forward], new_id


# --------------------------------------------------------------------------------------
# IMDb / DBLP
# --------------------------------------------------------------------------------------

TARGET_RELATION = {
    ("IMDB", "movie_year"): "movie_to_year",
    ("IMDB", "movie_director"): "movie_to_director",
    ("IMDB", "movie_link"): "movie_to_link",
    ("DBLP", "paper_conference"): "paper_to_conference",
}

# node types whose members are bijective copies of another type (copy -> original)
ALIAS_TYPES = {"IMDB": ("link", "movie")}

# Relations whose edges INCIDENT TO THE QUERY HEAD exist only because of the head's own
# supervised (training) pair, i.e. they are a relabelled copy of the label. The runner
# masks them together with the query edge (``InvariantCMPNN.remove_easy_edges``), so a
# training query sees the same neighbourhood a held-out query will. Found by the leak
# audit (``validate_cmpnn_invariance.py`` step 7):
#   DBLP: the compiler emits ``paper_to_area`` for TRAINING papers only (area = the
#         conference's area, derived through the train-only paper_to_conference edge);
#         with ``conference_to_area`` present, paper -> area -> conference reaches the
#         answer for 100 % of training queries and 0 % of held-out ones.
#   IMDb Movie-Director: ``link_to_director(link(m), d)`` -- handled by the alias map,
#         which masks (alias(h), t) directly.
LABEL_DERIVED_RELATIONS = {
    ("DBLP", "paper_conference"): ["paper_to_area"],
}


def _alias_map(data: ConditionalGraphData) -> torch.Tensor:
    alias = torch.arange(data.num_nodes, dtype=torch.int64)
    dataset = str(data.metadata.get("dataset", "")).upper()
    if dataset in ALIAS_TYPES:
        copy_t, orig_t = ALIAS_TYPES[dataset]
        offsets = data.metadata.get("offsets") or {}
        counts = data.metadata.get("num_nodes_by_type") or {}
        if copy_t in offsets and orig_t in offsets and counts.get(copy_t) == counts.get(orig_t):
            n = int(counts[copy_t])
            co, oo = int(offsets[copy_t]), int(offsets[orig_t])
            # structural package numbers links by movie index -> link i is the copy of movie i
            alias[co:co + n] = torch.arange(oo, oo + n)
            alias[oo:oo + n] = torch.arange(co, co + n)
    return alias


def _load_conditional(path: Path) -> LPBundle:
    data = load_graph_data(path)
    dataset = str(data.metadata.get("dataset", "")).upper()
    task = str(data.metadata.get("task"))
    target_name = TARGET_RELATION[(dataset, task)]
    phys = torch.stack([data.physical_edge_index[0], data.physical_edge_index[1], data.physical_edge_type], 1)
    sem = torch.stack([data.semantic_edge_index[0], data.semantic_edge_index[1], data.semantic_edge_type], 1)
    if list(data.physical_relation_names) != list(data.semantic_relation_names):
        raise ValueError("physical and semantic relation vocabularies differ; loader assumes one vocabulary")
    phys_f, names, new_id = _select_forward(phys, data.physical_relation_names)
    sem_f, names2, _ = _select_forward(sem, data.semantic_relation_names)
    assert names == names2
    target = names.index(target_name)
    td = data.task_data
    as_t = lambda k: torch.as_tensor(np.asarray(td[k]), dtype=torch.int64) if k in td else None
    train_pos, val_pos, test_pos = as_t("train_pos"), as_t("val_pos"), as_t("test_pos")
    train_neg, val_neg, test_neg = as_t("train_neg"), as_t("val_neg"), as_t("test_neg")
    candidate_pool = None
    if dataset == "DBLP":
        # DBLP negatives are stored flat [N*K, 2]; the evaluation protocol ranks against every
        # conference (full_target_type), so expose the pool and drop the flat negatives.
        conf_type = data.metadata.get("node_type_ids", {}).get("conference") if isinstance(data.metadata.get("node_type_ids"), dict) else None
        if conf_type is None:
            # infer: the type of every training tail
            conf_type = int(data.node_type[train_pos[0, 1]])
        candidate_pool = torch.nonzero(data.node_type == conf_type).flatten()
        train_neg = val_neg = test_neg = None
    return LPBundle(
        dataset=dataset, task=task, variant=data.variant, num_nodes=data.num_nodes,
        physical_relation_names=names, semantic_relation_names=names,
        physical_edges=phys_f, semantic_edges=sem_f,
        train_pos=train_pos, val_pos=val_pos, test_pos=test_pos,
        train_neg=train_neg, val_neg=val_neg, test_neg=test_neg,
        physical_target_relation=target, semantic_target_relation=target,
        node_type=data.node_type.clone(), alias=_alias_map(data),
        candidate_pool=candidate_pool,
        label_derived_relations=list(LABEL_DERIVED_RELATIONS.get((dataset, task), [])),
        source_physical_hash=data.physical_hash, source_semantic_hash=data.semantic_hash,
        source_supervision_hash=data.supervision_hash,
        metadata={k: v for k, v in data.metadata.items() if k in (
            "dataset", "task", "variant_placement", "offsets", "num_nodes_by_type", "skip_neg_k",
            "movie_year_neg_policy", "mapping_scope", "shared_split_file", "feature_table_hash")},
    )


# --------------------------------------------------------------------------------------
# WordNet
# --------------------------------------------------------------------------------------

def _load_wordnet(path: Path) -> LPBundle:
    d = torch.load(path if path.is_file() else path / "graph.pt", map_location="cpu", weights_only=False)
    if d.get("format_version") != "wordnet_rgcn_conditional_universal_parity_v2":
        raise ValueError(f"unexpected WordNet store format {d.get('format_version')!r}")
    names = list(d["semantic_relation_names"])
    phys_names = list(d["physical_relation_names"])
    # physical edges are typed by the raw relation vocabulary (task relations + physical
    # inverses/shortcuts as the variant names them); map them onto the semantic names by
    # position (the store aligns the two lists index-wise).
    if len(phys_names) != len(names):
        raise ValueError("WordNet store: physical/semantic relation lists differ in length")
    phys = torch.stack([d["physical_edge_index"][0], d["physical_edge_index"][1], d["physical_edge_type"]], 1)
    sem = torch.stack([d["semantic_edge_index"][0], d["semantic_edge_index"][1], d["semantic_edge_type"]], 1)
    # semantic: forward half (base:: + dependency::), CMPNN adds the inverses
    sem_f, fnames, _ = _select_forward(sem, names)
    # physical: raw, exactly the relations the variant materialises (named inverses and
    # shortcuts included), densely renumbered in vocabulary order
    used = torch.unique(phys[:, 2]).tolist()
    pnames = [phys_names[i] for i in used]
    pmap = {old: i for i, old in enumerate(used)}
    phys_r = torch.stack([phys[:, 0], phys[:, 1], torch.tensor([pmap[int(r)] for r in phys[:, 2].tolist()], dtype=torch.int64)], 1)
    phys_r = phys_r[canonical_edge_order(phys_r[:, 0], phys_r[:, 1], phys_r[:, 2])].contiguous()
    if phys_r.shape[0] > 1 and bool((phys_r[1:] == phys_r[:-1]).all(dim=1).any()):
        raise ValueError("duplicate (src,dst,rel) edges in the WordNet physical graph")
    # supervision triples are (h, r, t) over the 11 task relations; stored ids index
    # task_relation_names. Semantic ids: base::<name>; physical ids: <name>.
    task_names = list(d["task_relation_names"])
    sem_ids = [fnames.index(f"base::{n}") for n in task_names]
    phys_ids = [pnames.index(n) for n in task_names]
    def remap(t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.int64)
        r = torch.tensor(sem_ids, dtype=torch.int64)[t[:, 1]]
        return torch.stack([t[:, 0], r, t[:, 2]], 1)
    task_index_of_sem = [-1] * len(fnames)
    for i, sid in enumerate(sem_ids):
        task_index_of_sem[sid] = i
    return LPBundle(
        dataset="WORDNET", task="link_prediction", variant=str(d["variant"]), num_nodes=int(d["num_entities"]),
        physical_relation_names=pnames, semantic_relation_names=fnames,
        physical_edges=phys_r, semantic_edges=sem_f,
        train_pos=remap(d["task_train_pos"]), val_pos=remap(d["val_pos"]), test_pos=remap(d["test_pos"]),
        train_neg=None, val_neg=None, test_neg=None,
        physical_target_relation=None, semantic_target_relation=None, node_type=None,
        alias=torch.arange(int(d["num_entities"]), dtype=torch.int64), candidate_pool=None,
        label_derived_relations=[],
        source_physical_hash=str(d["physical_hash"]), source_semantic_hash=str(d["semantic_hash"]),
        source_supervision_hash=str(d["supervision_hash"]),
        metadata={"budget": d.get("budget"), "mode": d.get("mode"), "num_task_relations": len(task_names),
                  "semantic_task_relation_ids": sem_ids, "physical_task_relation_ids": phys_ids,
                  "task_relation_index_of_semantic_id": task_index_of_sem},
    )


def load_bundle(path: str | Path) -> LPBundle:
    p = Path(path)
    f = p / "graph.pt" if p.is_dir() else p
    raw = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(raw, ConditionalGraphData):
        return _load_conditional(p)
    if isinstance(raw, dict) and "format_version" in raw:
        return _load_wordnet(p)
    raise TypeError(f"unrecognised bundle at {p}")
