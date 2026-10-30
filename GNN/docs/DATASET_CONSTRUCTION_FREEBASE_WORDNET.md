# How the Freebase and WordNet variant families were built — transformations, selection criteria, numbers

> Describes how the Freebase and WordNet variant families are constructed, using the two provenance
> folders (`data/dataset_variant_3hops_filtered_PROVENANCE/`, `data/wordnet_3hops_augmented_full_provenance/`),
> the dataset folders, and the mining scripts in `dataset_construction/`. Every number below is read
> from a file in the data download (path given); where the READMEs or overview files in the provenance
> folders disagree with the dataset files, this doc says so. IMDb and DBLP are summarised in §4 for
> completeness.

---

## 0. Summary

Both knowledge-graph families are built by the same **materialisation switching**: a relational
composition `r₁;r₂[;r₃]` — a typed path — is re-encoded as a single named *shortcut* relation
between its endpoints, added to the base graph (never replacing it). The base variant carries
the information only as paths; the shortcut variants carry the same information additionally as
direct edges, so the variants are information-equivalent and non-isomorphic. Which
compositions are materialised is decided by an explicit selection criterion: on **WordNet**,
a **block-bijection** test (the composed relation must define a reversible correspondence
between groups of structurally equivalent sources and targets) for the inverse-bearing
compositions, and non-zero training support for the forward-only ones; on **Freebase**, a
**type-signature** test (the composed relation's start/end node-type pair must not coincide
with any base relation's) so that the shortcut occupies a schema position no base relation
already has. The universal graph is the union of the variants' edge sets. Supervision
(labels / test triples) is identical in every variant and is never used to build shortcuts.

---

## 1. Freebase (node classification on BOOK)

### 1a. Base data (`unchanged`)

HGB-style heterogeneous graph (`01_original_data/` = the `unchanged/` variant):

| | value | source |
|---|---|---|
| node types | 8: BOOK 40,402 · FILM 19,427 · MUSIC 82,351 · SPORTS 1,025 · PEOPLE 17,641 · LOCATION 9,368 · ORGANIZATION 2,731 · BUSINESS 7,153 = **180,098** nodes | `unchanged/node.dat` |
| base relations | **36** directed, typed (`BOOK-and-BOOK`, `BOOK-to-FILM`, `BOOK-on-SPORTS`, `BOOK-on-LOCATION`, `BOOK-about-ORGANIZATION`, `FILM-and-FILM`, `MUSIC-in-BOOK`, … `BUSINESS-and-BUSINESS`); ids 0–35 | `unchanged/info.dat` |
| base edges | **1,057,688** (directed rows of `link.dat`; models add reverse edges themselves) | `all_variants_overview.json` |
| node features | none (`Attribute Dimension: 0`) — models use learned per-type embeddings | `info.dat` |
| labels | **7,954** BOOK nodes, **8 classes** (`label.dat` 2,386 + `label.dat.test` 5,568, merged into one `label.dat` per variant) | `01_original_data/label.dat*` |
| split | **4,772 / 1,591 / 1,591** (60 / 20 / 20), stratified by class, seed shared with the other models (`make_split`: nested `train_test_split` over the *sorted* labelled BOOK ids), stored as `Freebase_unchanged/train_val_test_idx.npz` and used by every model (RGCN, MAGNN, SlotGAT, SeHGNN, PathNet) | `common/freebase_block_common.py` (`make_split`), `Freebase_unchanged/train_val_test_idx.npz` |

### 1b. The transformation: typed-path compositions materialised as shortcut relations

Step 1 (`code/typed_book_paths_up_to_k_new_transitive_fast.py`; not included in `dataset_construction/`, its output summary is in the `dataset_variant_3hops_filtered_PROVENANCE/` download) enumerates every
typed relation chain `v₀ –r₁→ v₁ –r₂→ … –r_k→ v_k` with **k ∈ {2, 3}** whose node-type path
**passes through BOOK** (`--must-include-node-type BOOK`), using the 36 forward base relations
only (no inverses), pruning chains with adjacent repeated relations (36 pruned) and chains not
touching BOOK (238 pruned). Each chain is composed as a product of sparse binary adjacency
matrices, **binarised after every hop**, giving the endpoint relation `T = r₁ ∘ r₂ [∘ r₃]`;
a composition with an empty `T` is dropped (none were). Result: **182 candidate compositions
(39 two-hop, 143 three-hop)**, each with statistics — `total_paths`, `unique_endpoint_pairs`,
`max_paths_for_any_pair`, `one_to_one_pairs`, `one_to_many_start_nodes`,
`many_to_one_end_nodes` — in `02_transitive_fast_3hops/typed_book_paths_up_to_k_new_transitive_summary.csv`.
The overview records `reversibility_level: not_checked` and `filter_existing_exact:
exact_same_binarized_edge_set_same_type_signature` (a composition equal to an existing base
relation would be dropped; 0 were).

### 1c. The selection criterion: type-signature filter (+ subset filter)

Step 2b (`dataset_construction/freebase/typed_book_paths_up_to_k_new_transitive_fast_streaming_inverse_signature_filtered.py`)
re-enumerates from the original data and keeps a composition only if

1. **no base relation has the same start/end node-type signature** (the *signature filter*:
   every composed `BOOK→FILM` is dropped because `BOOK-to-FILM` exists; every composed
   `PEOPLE→BOOK` because `PEOPLE-to-BOOK` exists; …), and
2. its endpoint edge set is not a subset of a base relation with the same signature (the
   *subset filter*; never binding on its own — composed sets are millions of pairs, base
   relations ~30 k).

Overview: 420 typed prefixes seen → 154 signature-skipped, 238 missing BOOK, 36 duplicates →
**28 kept: 6 two-hop, 22 three-hop** (`03_transitive_3hops_filtered/…_subset_filtered_summary.csv`).
So **154 of the 182 candidates were rejected, all by the signature filter** (33 of the 39
two-hop ones). The block-bijection statistic was **not** applied on Freebase; the kept
compositions have witness multiplicity (`max_paths_for_any_pair` from 21 to 2,798;
`one_to_one_pairs / unique_endpoint_pairs` from .934 to .9999) and the rejected ones look
alike on those statistics (median paths-per-pair 1.64 vs 1.35 kept).

What the signature filter buys: the shortcut occupies a **new schema
position**, so every model — including those whose structural bias is keyed on node-type
pairs rather than relation names (SeHGNN's type-path channels, SlotGAT's per-type slots) — sees
the shortcut as a distinct relation and not as extra edges of an existing base relation. Without
it, a composed `BOOK→FILM` would be pooled with the base `BOOK-to-FILM` by any type-collapsing
bias, i.e. two relations that carry different information would be identified by the
architecture.

**The 6 two-hop compositions kept** (edge counts = unique endpoint pairs; the materialised
relation is binary):

| new relation (id) | composition | node-type path | edges |
|---|---|---|---|
| `BOOK-about-ORGANIZATION-for-BUSINESS` (36) | `BOOK-about-ORGANIZATION ; ORGANIZATION-for-BUSINESS` | BOOK → ORGANIZATION → BUSINESS | 177,260 |
| `BOOK-about-ORGANIZATION-to-MUSIC` (37) | `BOOK-about-ORGANIZATION ; ORGANIZATION-to-MUSIC` | BOOK → ORGANIZATION → MUSIC | 43,930,565 |
| `BUSINESS-about-BOOK-about-ORGANIZATION` (38) | `BUSINESS-about-BOOK ; BOOK-about-ORGANIZATION` | BUSINESS → BOOK → ORGANIZATION | 1,944,239 |
| `MUSIC-in-BOOK-about-ORGANIZATION` (39) | `MUSIC-in-BOOK ; BOOK-about-ORGANIZATION` | MUSIC → BOOK → ORGANIZATION | 186,795 |
| `ORGANIZATION-for-BUSINESS-about-BOOK` (40) | `ORGANIZATION-for-BUSINESS ; BUSINESS-about-BOOK` | ORGANIZATION → BUSINESS → BOOK | 64,076 |
| `ORGANIZATION-to-MUSIC-in-BOOK` (41) | `ORGANIZATION-to-MUSIC ; MUSIC-in-BOOK` | ORGANIZATION → MUSIC → BOOK | 8,368 |

All six involve ORGANIZATION, and that is the signature filter at work: the only (start type,
end type) pairs reachable in two hops through BOOK that **no base relation occupies** are
`BOOK→MUSIC`, `BOOK→BUSINESS`, `MUSIC→ORGANIZATION`, `BUSINESS→ORGANIZATION` and
`ORGANIZATION→BOOK` (there is no base `BOOK-…-MUSIC`, `BOOK-…-BUSINESS`, `MUSIC-…-ORGANIZATION`,
`BUSINESS-…-ORGANIZATION` or `ORGANIZATION-…-BOOK`); every other two-hop path lands on a type
pair such as `BOOK→FILM` or `PEOPLE→BOOK` that a base relation already names, and is rejected. The **22 three-hop** compositions are
their extensions by one `-and-` or one further hop (`BOOK-about-ORGANIZATION-to-MUSIC-and-MUSIC`
234,153,417 edges is the largest; `ORGANIZATION-and-ORGANIZATION-to-MUSIC-in-BOOK` 21,213 the
smallest); the full list with edge counts is `exact_3/new_transitive_relations.jsonl`.

### 1d. The variants

Step 3 (`dataset_construction/freebase/create_legacy_fb_transitive_variants_streaming.py`) writes, for each variant, the
unchanged `node.dat`, the merged `label.dat` (7,954 rows), an `info.dat` with the base relations
followed by the new ones, and a `link.dat` that streams the original 1,057,688 rows followed by
every materialised shortcut edge (weight 1.0). The shortcut relation is **added**; the base
edges it summarises are kept. Numbers (from the rebuild's per-relation counts and the registries;
see the caveat below):

| variant | new relations | new edges | total relations | total directed edges | role |
|---|---|---|---|---|---|
| `unchanged` | 0 | 0 | 36 | 1,057,688 | base variant |
| `exact_2` | 6 (ids 36–41) | 46,311,303 | 42 | 47,368,991 | 2-hop compositions materialised |
| `exact_3` | 22 (ids 36–57) | 445,909,448 | 58 | 446,967,136 | 3-hop compositions materialised |
| `union_exact_2_3` | 28 | 492,220,751 | 64 | 493,278,439 | **universal graph** `⋃ᵢ Freebaseᵢ` — the `{model}*` comparison point |

The variant family used by every model is **{unchanged, exact_2, exact_3}** with
universal **`union_exact_2_3`**.

**Provenance caveats.**
(i) `exact_3/link.dat` (8.48 GB) and `union_exact_2_3` are built by
`dataset_construction/freebase/freebase_rebuild_exact3_and_union.py` (driver
`rebuild_exact3_and_union.sh`) from the `range_2_3` store (the Step-2a materialisation of all 182
unfiltered candidates, 218 relations / 1.96 B edges — a build intermediate, not a variant and not
used by any model) by keeping the 36 base + the 22 registered relations; the acceptance test
compares per-relation row counts against the registry. (ii) `exact_3/variant_overview.json` and the
top-level `all_variants_overview.json` describe an earlier, unfiltered build (143 relations / 1.74 B
edges; `exact_2` as 39 relations / 217 M edges) and do **not** match the dataset files; the registry
(`new_transitive_relations.jsonl`, 22 rows) and the rebuild's `actual_counts_*.json` are
authoritative. (iii) `range_2_3` is the Step-2a (182-row) enumeration, not "exact_2 ∪ exact_3".

### 1e. How each model consumes the family

* The **semantic catalog** (which compositions are semantic relations) is derived from the
  registries (`dataset_construction/freebase/registries/{exact_2,exact_3}.jsonl`, 6 / 22
  lines) — the same 28 compositions. Every model's invariant program uses catalog `exact_3` = these 28 + the 36 base
  relations.
* Per variant the invariant compilers reconstruct each semantic relation from **whatever the
  variant offers**: the base chain (composed, interior witnesses collapsed — set semantics) on
  `unchanged`; the chain *and* the shortcut on `exact_k`, where the shortcut is verified against
  the composed chain (`freebase_block_common.verify_full_physical_pairs`
  `matches_composed_semantic_blocks`). Because the shortcut is derived from the chain, every
  variant compiles to the same program (RGCN: `7bd4b8b5ecc9`-style family hashes; MAGNN v2 / SeHGNN
  v2 / SlotGAT family plans likewise).
* Path-based models cap the instances per (BOOK, metapath) / walks per (target, channel) at 500,
  keyed on the semantic identity.
* Originals train stock models on each variant's physical graph; augmentation trains one model
  jointly across the three; universal trains on `union_exact_2_3`.

---

## 2. WordNet (WN18RR link prediction)

### 2a. Base data (`no_changes`)

| | value | source |
|---|---|---|
| entities | **40,943** | `00_source_data/entities.dict` |
| base relations | **11** (`_hypernym`, `_derivationally_related_form`, `_member_meronym`, `_synset_domain_topic_of`, `_has_part`, `_instance_hypernym`, `_also_see`, `_verb_group`, `_member_of_domain_region`, `_member_of_domain_usage`, `_similar_to`) | `relations.dict` |
| official split | train **86,835** · valid **3,034** · test **3,134** triples (total 93,003) | `00_source_data/`, `wordnet_splits.npz` |
| `no_changes` | `train.txt` byte-identical; the message graph of every variant is built from **train only** | `no_changes/augmentation_summary.json` (`augmentation_source: official train.txt only`) |
| supervision | the official train / valid / test triples, identical for every variant; filtered ranking over all 40,943 entities (`docs/results/METRICS.md`) | `original_splits/` |

### 2b. The transformations: inverse materialisation and 3-hop transitive shortcuts

Two switchings, applied to the training graph only:

1. **Inverse materialisation** (`all_inverse_edges`): for every train triple `(h, r, t)` add
   `(t, r__inv, h)` as a *named* relation — 11 new relation types, 86,835 new edges. (A model
   that virtualises inverses, e.g. via transposes, sees the same information on `no_changes`.)
2. **Transitive shortcuts** (`transitive_edges`): a 3-hop relational composition
   `r₁ ; r₂ ; r₃` over the train graph is materialised as one new relation
   `__transitive__r₁|||r₂|||r₃` between its endpoints (sparse adjacency product, binarised
   after each hop; an edge is emitted only where at least one concrete chain exists).

### 2c. Candidate discovery and the selection criteria

Two catalogs feed the final build; they were selected by **different criteria**:

* **Inverse-bearing 3-hop compositions → the block-bijection catalog** (Step 1,
  `dataset_construction/wn18rr/wn18rr_experiment_exact_hops_transitive.py`, on train+valid+test).
  Enumerates all `22³ = 10,648` sequences over the 11 base + 11 inverse relations, keeps only
  sequences that contain at least one `__inv`, prunes adjacent `r ; r__inv`, adjacent
  duplicates and all-same-relation sequences, composes each survivor as a binary edge set, and
  **keeps a composition only if `block_bijection_holds = 1`**. The test
  (`summarize_relation_edges`): group sources by their neighbour set (`s ~ s′` iff
  `R(s) = R(s′)`; these are the *row blocks*) and targets by their preimage set (*col blocks*);
  the composition is block-bijective iff the number of row blocks equals the number of col
  blocks, every row block's neighbours fall in exactly one col block, and every col block's
  preimages fall in exactly one row block — i.e. the relation is a **reversible correspondence
  between groups of structurally equivalent sources and targets**. The graded form
  `block_bijection_ratio = min(forward, backward block-functionality ratio) × (min/max block
  count)` is 1.0 for every survivor. Result: **1,639 compositions**
  (`01_candidate_discovery/transitive_3hops_inverse/transitive_summary.csv`), all with ratio
  1.0, small (median 6 paths, max 11,352; median 2 row blocks). Non-bijective candidates were
  discarded (a hard `continue`, script line 455) and can be regenerated in about a minute.
* **Forward-only 3-hop compositions → the train-support catalog** (Step 3,
  `dataset_construction/wn18rr/augment_wordnet_topk_transitive_selective_inverses(3).py`). All
  `11³ = 1,331` forward sequences of the base relations are composed on **train only**; the
  **526 with non-zero train support** are selected (`selection.method:
  top_k_by_train_support`, `requested_top_k: 0` = all eligible, `include_zero_support: false`).
  **No block-bijection filter is applied to this catalog.**

An intermediate build (Step 2, `02_first_augmented_build/old_…`) materialised all
1,639 bijective compositions on train+valid+test (316,487 edges, 1,661 relation types); it is
not used as a variant because held-out triples influence it, and only its composition catalog
(`metapaths.json`) is reused by Step 3.

### 2d. The variants (final, leakage-free build)

| variant | training edges | relation types | contents | source |
|---|---|---|---|---|
| `no_changes` | 86,835 | 11 | `train.txt` only | `no_changes/augmentation_summary.json` |
| `all_inverse_edges` | 173,670 | 22 | + a named inverse of every train triple (86,835) | `all_inverse_edges/…` |
| `transitive_edges` | 706,128 | 537 | + **526 forward-only 3-hop shortcut relations, 619,293 shortcut edges**; no inverse edges | `transitive_edges/…` |
| `universal_edges` | 893,629 | 2,087 | `no_changes` + all 11 base inverses (86,835) + the 526 forward shortcuts (619,293, inherited) + **1,539 inverse-enabled block-bijective shortcuts with non-zero train support (100,666 universal-specific edges; 719,959 shortcut edges in total)** | `universal_edges/…`, `dataset_manifest.json` (`universal_definition`) |

Leakage checks passed (`leakage_checks_passed: true`): no shortcut edge reproduces a valid or
test triple; `old_augmented_edges_reused: false`. A shared relation vocabulary of 2,087 ids
(`shared_relations.dict`) is used by every variant so relation ids agree across them.

The variant family used by every model is **{no_changes, all_inverse_edges, transitive_edges}**
with universal **`universal_edges`**. Note that `universal_edges` is *more*
than the union of the three variants: it also carries the 1,539 inverse-bearing shortcuts that
no single variant has (100,666 edges); it is defined by the manifest (`universal_definition`),
not as "the union of the variants".

### 2e. How each model consumes the family

* The semantic catalog is the reduced / full dependency list (`models/RGCN/wordnet_structural_catalogs/`); each
  dependency `q₁;q₂;q₃` names its `shortcut_relation` where one exists.
* Per variant the invariant RGCN compiler reconstructs each dependency's endpoint set by
  composing the three-hop physical context (inverse hops via virtual transposes on
  `no_changes`, physical inverses on `all_inverse_edges`), collapses interior witnesses, applies
  a deterministic per-source cap (`rotate_and_cap`, keyed on the dependency id), and — where the
  variant materialises the shortcut — **verifies `shortcut == composed chain`** (raising on any
  mismatch). Every variant therefore compiles to the same program (τ = 1 on all 18 pairs).
* Supervision hashes are identical across variants; negatives and batch order are fixed by the
  shared split file (`wordnet_splits.npz`).

---

### 2f. What the criteria actually selected

Re-running the WordNet discovery with the bijection filter disabled gives 2,822 non-empty
inverse-bearing candidates: the 1,639 survivors are small and structurally degenerate (median 6
endpoint pairs and 2 row blocks; the largest survivors, 11,352 pairs, consist of a single
block — a complete bipartite relation), while the 1,183 rejected ones are the large, informative
compositions (median 186 pairs, up to 185,888) with `block_bijection_ratio` spread over
.26–.88. Applying the same test to Freebase's 39 two-hop candidates: **none** is
block-bijective (ratio ≤ .17), kept or rejected, and witness multiplicity is negligible (mean
paths per pair ≤ 1.03; ≤ 1.1 % of pairs have more than one chain). So, as built, the WordNet
`universal_edges` catalog is the strict block-bijective one, the WordNet `transitive_edges`
catalog and the whole Freebase family are selected by support / schema-position criteria and
are not block-bijective — each family is selected by its own criterion, not by one rule for both.

## 3. The two criteria side by side

| | Freebase | WordNet |
|---|---|---|
| switching | typed-path composition (k = 2, 3) through BOOK → shortcut relation, added | (i) forward triple → named inverse, added; (ii) 3-hop relational composition → shortcut relation, added |
| candidates | 182 (39 / 143) | forward-only 1,331; inverse-bearing 10,648 → pruned |
| **selection criterion** | **type-signature**: no base relation with the same (start type, end type); plus subset test | **block bijection** for inverse-bearing compositions (1,639 survive); **non-zero train support** for forward-only (526 survive) |
| what the criterion guarantees | the shortcut is a new schema position — distinguishable from every base relation even by type-keyed structural biases | the composed relation is a reversible correspondence between blocks of structurally equivalent sources and targets — the shortcut determines its chain occurrences up to structural equivalence (the condition under which skip nodes and set semantics lose nothing) |
| what it does **not** guarantee | block bijection (kept compositions have multiplicity up to 2,798 paths per pair) | schema-position novelty (a composed `_hypernym;…` may share the signature of a base relation — WN18RR has a single node type, so every relation shares it) |
| variants | unchanged / exact_2 (6 rel, 46.3 M edges) / exact_3 (22 rel, 445.9 M) ; universal = union (28 rel, 493.3 M) | no_changes / all_inverse_edges / transitive_edges (526 rel, 619 k edges) ; universal_edges (2,087 rel types, 893,629 edges, more than the union) |
| supervision | 7,954 labelled BOOK, 8 classes, 4,772 / 1,591 / 1,591 stratified | official 86,835 / 3,034 / 3,134 triples; message graphs from train only |
| rejected candidates available? | yes — 154 rows with statistics on disk | regenerable in ~1 min (disable the filter) |

---

## 4. IMDb and DBLP in the same vocabulary (pointer only)

These are **relocation** switchings (information moved, not duplicated), built directly from
the raw tables; full detail in `dataset_construction/` and the `PLACEMENT` tables in
`models/RGCN/preprocess_IMDB_rgcn_structural.py`; statistics in `docs/DATASET_STATISTICS.md`.

| | IMDb (year graph) | DBLP |
|---|---|---|
| switching | an attribute relation `movie–X` (X ∈ {director, actor}) is re-attached to the movie's Link node (`link–X`); Link is a bijective copy of Movie; year is fixed on the movie (NC, MD) or is the target (MY) | the Area label is attached to the Paper (v1), to its Conference (v2) or to its Authors (v3) |
| variants | v1 both on movie · v2 both on link · v3 director on movie / actor on link · v4 the reverse; universal = both endpoints | v1 / v2 / v3; universal = all three at once |
| selection / cleaning | `title_year` present for 4,177 / 4,180 movies; Movie–Link frozen (leaky by construction) | a paper's area = its authors' label; **866 / 14,328 papers whose authors disagree are dropped** so the relocation is exact per block; Area edges emitted for training-support papers only |
| what makes the switching invertible | Link ↔ Movie is one-to-one | per-paper **block certificate** (training-only) — without it `P→C→Area` offers ≈ 3.96 areas per paper |
| supervision | canonical nested split seed 1566911444 (NC: 2,926 / 418 / 836; LP: shared positives, K = 19 negatives) | canonical MAGNN partition, 9,448 / 1,348 / 2,666 pairs, K = 19 = all other conferences |

---

## 5. Relevant files

* Freebase: `data/dataset_variant_3hops_filtered_PROVENANCE/{README_PROVENANCE.md, code/, 02_transitive_fast_3hops/, 03_transitive_3hops_filtered/}`;
  `data/dataset_variant_3hops_filtered/{unchanged, exact_2, exact_3, union_exact_2_3}/{info.dat, new_transitive_relations.jsonl}`;
  `dataset_construction/freebase/` (Step 2b / Step 3 scripts, rebuild, `registries/`).
* WordNet: `data/wordnet_3hops_augmented_full_provenance/{README.md, scripts/, 01_candidate_discovery/}`;
  `data/wordnet_3hops_augmented_full/{dataset_manifest.json, */augmentation_summary.json, shared_relations.dict, original_splits/, wordnet_splits.npz}`.
