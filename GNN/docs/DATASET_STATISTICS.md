# Dataset statistics

> Computed from the bundles, manifests and raw files the experiments consume. Layout: nodes per type, edges per relation per variant, splits; §5 adds the model-specific structural statistics (semantic programs, metapaths / instances, channels, message schedules). `IMDb*`, `DBLP*` etc. = the union (universal) graph.

## 0. At a glance

| dataset | task | node types / nodes | variants (universal) | edges per variant (directed) | supervision |
|---|---|---|---|---|---|
| IMDb (year graph) | NC, 3 genres | 5 / 15,789 | v1–v4 (IMDb*) | 50,142 each (universal 83,570; semantic program 91,924) | 2,926 / 418 / 836 movies |
| IMDb | LP Movie–Year | 5 / 15,789 | v1–v4 (IMDb*) | 47,638 each (universal = program 81,066) | 2,925 / 418 / 834 pairs × 19 negatives |
| IMDb | LP Movie–Director | 5 / 15,789 | v1, v3 (IMDb*) | 47,634 each (universal 72,702; program 86,908) | 2,926 / 418 / 836 pairs × 19 negatives |
| DBLP | LP Paper–Conference | 5 / 26,076 | v1–v3 (DBLP*) | 287,236 / 268,490 / 275,288 (universal = program 294,334) | 9,448 / 1,348 / 2,666 pairs, 20-way ranking |
| WN18RR | LP | 1 / 40,943 | no_changes, all_inverse_edges, transitive_edges (universal_edges) | 86,835 / 173,670 / 706,128 (universal = program 893,629) | 86,835 / 3,034 / 3,134 triples, filtered full-entity ranking |
| Freebase | NC, 8 genres | 8 / 180,098 | unchanged, exact_2, exact_3 (union_exact_2_3) | 1.06 M / 47.4 M / 447.0 M (universal 493.3 M) | 4,772 / 1,591 / 1,591 labelled books |

(Per-relation counts and splits are detailed in the sections below.)

## 0b. Which artifacts each model reads

Every arm of every model reads preprocessed bundles under `data/preprocessed/`. The RGCN structural
bundles (`RGCN_structural_v10_year_canonfeat` for IMDb, `RGCN_structural_v8_dblp` for DBLP,
`RGCN_structural/WORDNET/universal_parity_full` for WN18RR, `freebase_rgcn_block_conditional` for
Freebase) hold the compiled semantic programs; MAGNN, SeHGNN, SlotGAT and PathNet compile their own
inputs from them (or from the same raw variants with the same catalog), and CMPNN reads them directly.
PAIN builds its path artifacts from the raw IMDb csv and the raw Freebase variants with its own compiler.
The exact bundle each arm reads, and the script that builds it, are listed in `models/<MODEL>/README.md`
and `scripts/<MODEL>/preprocess_*.sh`; §5 below gives the per-model structural statistics of those bundles.

## 1. IMDb (year graph)

Five node types; the Link node is a bijective copy of the movie; the year is fixed on the movie except in Movie–Year where it is the held-out target. Variants relocate {director, actor} between the movie and its Link (NC, Movie–Year: v1 both on the movie, v2 both on the Link, v3 director on the movie / actor on the Link, v4 the reverse; Movie–Director: only the actor relocates, v1 / v3). Movie features: 3,008-d bag of words over plot keywords; director / actor / link / year features = mean of their movies' rows (canonical table, identical in every variant). Held-out target edges are absent from the graph (training pairs only).

### 1.1 Node classification (movie genre)

| node type | count |
|---|---|
| movie | 4,180 |
| director | 2,081 |
| actor | 5,257 |
| link | 4,180 |
| year | 91 |
| **total** | **15,789** |

Physical edges per relation (undirected pairs; the bundles store both directions) and the compiled semantic program:

| relation (pair count) | v1 | v2 | v3 | v4 | universal | semantic program (every variant) |
|---|---|---|---|---|---|---|
| movie–director | 4,180 | – | 4,180 | – | 4,180 | 4,180 |
| movie–actor | 12,534 | – | – | 12,534 | 12,534 | 12,534 |
| movie–link | 4,180 | 4,180 | 4,180 | 4,180 | 4,180 | 4,180 |
| movie–year | 4,177 | 4,177 | 4,177 | 4,177 | 4,177 | 4,177 |
| link–director | – | 4,180 | – | 4,180 | 4,180 | 4,180 |
| link–actor | – | 12,534 | 12,534 | – | 12,534 | 12,534 |
| link–year | – | – | – | – | – | 4,177 |
| **directed edges, total** | **50,142** | **50,142** | **50,142** | **50,142** | **83,570** | **91,924** |

Supervision: 2,926 / 418 / 836 labelled target nodes (train / val / test), 3 classes. Split = nested stratified `train_test_split(random_state=1566911444)` on the 4,180 movies.
Semantic hash `942317043f15`; physical hashes differ per variant. Universal graph = the `universal` column.

### 1.2 Link prediction Movie–Year

| node type | count |
|---|---|
| movie | 4,180 |
| director | 2,081 |
| actor | 5,257 |
| link | 4,180 |
| year | 91 |
| **total** | **15,789** |

Physical edges per relation (undirected pairs; the bundles store both directions) and the compiled semantic program:

| relation (pair count) | v1 | v2 | v3 | v4 | universal | semantic program (every variant) |
|---|---|---|---|---|---|---|
| movie–director | 4,180 | – | 4,180 | – | 4,180 | 4,180 |
| movie–actor | 12,534 | – | – | 12,534 | 12,534 | 12,534 |
| movie–link | 4,180 | 4,180 | 4,180 | 4,180 | 4,180 | 4,180 |
| movie–year | 2,925 | 2,925 | 2,925 | 2,925 | 2,925 | 2,925 |
| link–director | – | 4,180 | – | 4,180 | 4,180 | 4,180 |
| link–actor | – | 12,534 | 12,534 | – | 12,534 | 12,534 |
| **directed edges, total** | **47,638** | **47,638** | **47,638** | **47,638** | **81,066** | **81,066** |

Supervision (identical in every variant and every model): train 2,925 pairs × 19 negatives; val 418 pairs × 19 negatives; test 834 pairs × 19 negatives. Same movie partition as Movie–Director (2,926 / 418 / 836 movies; the 3 movies without a year are dropped). Negatives: 19 years drawn from the empirical year marginal (frequency-matched); candidate pool = 91 years.
Semantic hash `d74dc76282a9`; physical hashes differ per variant. Universal graph = the `universal` column.

### 1.3 Link prediction Movie–Director

| node type | count |
|---|---|
| movie | 4,180 |
| director | 2,081 |
| actor | 5,257 |
| link | 4,180 |
| year | 91 |
| **total** | **15,789** |

Physical edges per relation (undirected pairs; the bundles store both directions) and the compiled semantic program:

| relation (pair count) | v1 | v3 | universal | semantic program (every variant) |
|---|---|---|---|---|
| movie–director | 2,926 | 2,926 | 2,926 | 2,926 |
| movie–actor | 12,534 | – | 12,534 | 12,534 |
| movie–link | 4,180 | 4,180 | 4,180 | 4,180 |
| movie–year | 4,177 | 4,177 | 4,177 | 4,177 |
| link–actor | – | 12,534 | 12,534 | 12,534 |
| link–director | – | – | – | 2,926 |
| link–year | – | – | – | 4,177 |
| **directed edges, total** | **47,634** | **47,634** | **72,702** | **86,908** |

Supervision (identical in every variant and every model): train 2,926 pairs × 19 negatives; val 418 pairs × 19 negatives; test 836 pairs × 19 negatives. Canonical movie partition (`IMDB_md_shared_splits.npz`); negatives: 19 uniformly sampled directors.
Semantic hash `63298526d97b`; physical hashes differ per variant. Universal graph = the `universal` column.

## 2. DBLP (Paper–Conference link prediction)

Five node types. The Area of a paper is derived from its authors' labels; variants attach the Area to the Paper (v1), the Conference (v2) or the Author (v3). Area edges exist for **training-support papers only** in every variant (a held-out paper's area would narrow its conference to ~5 of 20), and Paper–Conference edges are the training pairs only. Featureless (learned embeddings / identity inputs).

### 2.1 Paper–Conference

| node type | count |
|---|---|
| author | 3,948 |
| paper | 13,462 |
| term | 8,642 |
| conference | 20 |
| area | 4 |
| **total** | **26,076** |

Physical edges per relation (undirected pairs; the bundles store both directions) and the compiled semantic program:

| relation (pair count) | v1 | v2 | v3 | universal | semantic program (every variant) |
|---|---|---|---|---|---|
| author–paper | 17,568 | 17,568 | 17,568 | 17,568 | 17,568 |
| paper–term | 107,154 | 107,154 | 107,154 | 107,154 | 107,154 |
| paper–conference | 9,448 | 9,448 | 9,448 | 9,448 | 9,448 |
| paper–area | 9,448 | – | – | 9,448 | 9,448 |
| conference–area | – | 75 | – | 75 | 75 |
| author–area | – | – | 3,474 | 3,474 | 3,474 |
| **directed edges, total** | **287,236** | **268,490** | **275,288** | **294,334** | **294,334** |

Supervision (identical in every variant and every model): train 9,448 pairs × 19 negatives; val 1,348 pairs × 19 negatives; test 2,666 pairs × 19 negatives. Split = `DBLP_shared_splits_CANONICAL` (paper-disjoint 70/10/20, seed 1566911444, reproducible from `dataset_construction/dblp/paper-venue_shared_splits.py`); the 19 negatives are every other conference, i.e. full 20-way ranking.
Semantic hash `308cec36b62b`; physical hashes differ per variant. Universal graph = the `universal` column.

## 3. WN18RR (link prediction)

40,943 entities (one node type), 11 base relations; official splits 86,835 / 3,034 / 3,134 triples (the same supervision for every variant). Variants add relations to the message graph without touching the supervision:

| variant | relations | edges (`data.txt`) | composition |
|---|---|---|---|
| no_changes | 11 | 86,835 | the 11 base relations (= official training triples) |
| all_inverse_edges | 22 | 173,670 | 11 base + 11 named physical inverses |
| transitive_edges | 537 | 706,128 | 11 base + 526 mined 2/3-hop shortcut relations (compositions of base relations) |
| universal_edges | 2,087 | 893,629 | 11 base + 11 inverses + 2,065 shortcuts (union; the universal graph) |

Compiled semantic program (RGCN `inv_rgcn` structural package, full catalog, intersection mode) — identical for the four variants:

* 2,087 semantic relations = base 11, dependency 2,065, inverse 11; 893,629 semantic edges = base 86,835, dependency 719,959, inverse 86,835 (exact parity with the universal graph's 893,629 edges).
* CMPNN's invariant arm consumes the 11 `base::` relations only (86,835 edges; inverses are CMPNN's own `add_inverse`, the `dependency::` shortcuts are context — a 6-layer path model reaches the 3-hop chain itself, and the shortcut edges mined from the training triples would leak the masked query).
* Evaluation: filtered full-entity ranking over the 40,943 entities for Hits/MRR; 50 fixed seeded negatives per query (`candidate_seed 1566911444`) for the invariance statistic.

## 4. Freebase (BOOK genre node classification)

| node type | count |
|---|---|
| 0 BOOK | 40,402 |
| 1 FILM | 19,427 |
| 2 MUSIC | 82,351 |
| 3 SPORTS | 1,025 |
| 4 PEOPLE | 17,641 |
| 5 LOCATION | 9,368 |
| 6 ORGANIZATION | 2,731 |
| 7 BUSINESS | 7,153 |
| **total** | **180,098** |

Labelled books: 7,954 of 40,402 (7 classes present of the 8 in the schema); split 4,772 / 1,591 / 1,591 (RGCN partition, seed 1566911444, shared by every model). Featureless. Classes: 0 poem/poem_character.

Relations and edges per variant (counted from `link.dat`); shortcut relations are named by the base-relation composition they materialise:

| variant | relations | edges | new relations vs. `unchanged` |
|---|---|---|---|
| unchanged | 36 | 1,057,688 | 0:  |
| exact_2 | 42 | 47,368,991 | 6: BOOK-about-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-to-MUSIC, BUSINESS-about-BOOK-about-ORGANIZATION, MUSIC-in-BOOK-about-ORGANIZATION, ORGANIZATION-for-BUSINESS-about-BOOK, ORGANIZATION-to-MUSIC-in-BOOK |
| exact_3 | 58 | 446,967,136 | 22: BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, BOOK-about-ORGANIZATION-for-BUSINESS-about-MUSIC, BOOK-about-ORGANIZATION-for-BUSINESS-and-BUSINESS, … (22) |
| union_exact_2_3 | 64 | 493,278,439 | 28: BOOK-about-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-to-MUSIC, BUSINESS-about-BOOK-about-ORGANIZATION, MUSIC-in-BOOK-about-ORGANIZATION, … (28) |

Per-relation edge counts (rows = every relation of the union graph; a dash = not materialised in that variant):

| id | relation | unchanged | exact_2 | exact_3 | union_exact_2_3 |
|---|---|---|---|---|---|
| 4 | BOOK-about-ORGANIZATION | 21,900 | 21,900 | 21,900 | 21,900 |
| 0 | BOOK-and-BOOK | 202,674 | 202,674 | 202,674 | 202,674 |
| 3 | BOOK-on-LOCATION | 26,921 | 26,921 | 26,921 | 26,921 |
| 2 | BOOK-on-SPORTS | 6,615 | 6,615 | 6,615 | 6,615 |
| 1 | BOOK-to-FILM | 38,299 | 38,299 | 38,299 | 38,299 |
| 30 | BUSINESS-about-BOOK | 18,625 | 18,625 | 18,625 | 18,625 |
| 31 | BUSINESS-about-FILM | 8,397 | 8,397 | 8,397 | 8,397 |
| 32 | BUSINESS-about-MUSIC | 24,764 | 24,764 | 24,764 | 24,764 |
| 33 | BUSINESS-about-SPORTS | 610 | 610 | 610 | 610 |
| 35 | BUSINESS-and-BUSINESS | 4,448 | 4,448 | 4,448 | 4,448 |
| 34 | BUSINESS-on-LOCATION | 6,647 | 6,647 | 6,647 | 6,647 |
| 5 | FILM-and-FILM | 87,838 | 87,838 | 87,838 | 87,838 |
| 23 | LOCATION-and-LOCATION | 47,817 | 47,817 | 47,817 | 47,817 |
| 22 | LOCATION-in-FILM | 21,299 | 21,299 | 21,299 | 21,299 |
| 8 | MUSIC-and-MUSIC | 283,670 | 283,670 | 283,670 | 283,670 |
| 9 | MUSIC-for-SPORTS | 8,975 | 8,975 | 8,975 | 8,975 |
| 6 | MUSIC-in-BOOK | 31,486 | 31,486 | 31,486 | 31,486 |
| 7 | MUSIC-in-FILM | 11,291 | 11,291 | 11,291 | 11,291 |
| 10 | MUSIC-on-LOCATION | 42,915 | 42,915 | 42,915 | 42,915 |
| 28 | ORGANIZATION-and-ORGANIZATION | 1,101 | 1,101 | 1,101 | 1,101 |
| 29 | ORGANIZATION-for-BUSINESS | 1,073 | 1,073 | 1,073 | 1,073 |
| 24 | ORGANIZATION-in-FILM | 13,128 | 13,128 | 13,128 | 13,128 |
| 27 | ORGANIZATION-on-LOCATION | 2,696 | 2,696 | 2,696 | 2,696 |
| 25 | ORGANIZATION-to-MUSIC | 10,702 | 10,702 | 10,702 | 10,702 |
| 26 | ORGANIZATION-to-SPORTS | 559 | 559 | 559 | 559 |
| 18 | PEOPLE-and-PEOPLE | 22,813 | 22,813 | 22,813 | 22,813 |
| 21 | PEOPLE-in-BUSINESS | 5,378 | 5,378 | 5,378 | 5,378 |
| 20 | PEOPLE-in-ORGANIZATION | 2,215 | 2,215 | 2,215 | 2,215 |
| 19 | PEOPLE-on-LOCATION | 15,134 | 15,134 | 15,134 | 15,134 |
| 14 | PEOPLE-to-BOOK | 35,587 | 35,587 | 35,587 | 35,587 |
| 15 | PEOPLE-to-FILM | 17,604 | 17,604 | 17,604 | 17,604 |
| 16 | PEOPLE-to-MUSIC | 10,948 | 10,948 | 10,948 | 10,948 |
| 17 | PEOPLE-to-SPORTS | 14,850 | 14,850 | 14,850 | 14,850 |
| 12 | SPORTS-and-SPORTS | 1,290 | 1,290 | 1,290 | 1,290 |
| 11 | SPORTS-in-FILM | 6,763 | 6,763 | 6,763 | 6,763 |
| 13 | SPORTS-on-LOCATION | 656 | 656 | 656 | 656 |
| 42 | BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS | – | – | 121,333 | 121,333 |
| 43 | BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC | – | – | 1,138,594 | 1,138,594 |
| 36 | BOOK-about-ORGANIZATION-for-BUSINESS | – | 177,260 | – | 177,260 |
| 44 | BOOK-about-ORGANIZATION-for-BUSINESS-about-MUSIC | – | – | 87,324,926 | 87,324,926 |
| 45 | BOOK-about-ORGANIZATION-for-BUSINESS-and-BUSINESS | – | – | 10,357,574 | 10,357,574 |
| 37 | BOOK-about-ORGANIZATION-to-MUSIC | – | 43,930,565 | – | 43,930,565 |
| 46 | BOOK-about-ORGANIZATION-to-MUSIC-and-MUSIC | – | – | 234,153,417 | 234,153,417 |
| 47 | BOOK-and-BOOK-about-ORGANIZATION-for-BUSINESS | – | – | 500,231 | 500,231 |
| 48 | BOOK-and-BOOK-about-ORGANIZATION-to-MUSIC | – | – | 104,975,615 | 104,975,615 |
| 38 | BUSINESS-about-BOOK-about-ORGANIZATION | – | 1,944,239 | – | 1,944,239 |
| 49 | BUSINESS-about-BOOK-about-ORGANIZATION-and-ORGANIZATION | – | – | 775,784 | 775,784 |
| 50 | BUSINESS-about-BOOK-and-BOOK-about-ORGANIZATION | – | – | 1,972,151 | 1,972,151 |
| 51 | BUSINESS-about-MUSIC-in-BOOK-about-ORGANIZATION | – | – | 52,292 | 52,292 |
| 52 | BUSINESS-and-BUSINESS-about-BOOK-about-ORGANIZATION | – | – | 468,521 | 468,521 |
| 53 | MUSIC-and-MUSIC-in-BOOK-about-ORGANIZATION | – | – | 1,116,816 | 1,116,816 |
| 39 | MUSIC-in-BOOK-about-ORGANIZATION | – | 186,795 | – | 186,795 |
| 54 | MUSIC-in-BOOK-about-ORGANIZATION-and-ORGANIZATION | – | – | 350,368 | 350,368 |
| 55 | MUSIC-in-BOOK-about-ORGANIZATION-for-BUSINESS | – | – | 891,278 | 891,278 |
| 56 | MUSIC-in-BOOK-and-BOOK-about-ORGANIZATION | – | – | 793,019 | 793,019 |
| 57 | ORGANIZATION-and-ORGANIZATION-for-BUSINESS-about-BOOK | – | – | 53,433 | 53,433 |
| 58 | ORGANIZATION-and-ORGANIZATION-to-MUSIC-in-BOOK | – | – | 21,213 | 21,213 |
| 40 | ORGANIZATION-for-BUSINESS-about-BOOK | – | 64,076 | – | 64,076 |
| 59 | ORGANIZATION-for-BUSINESS-about-BOOK-and-BOOK | – | – | 307,884 | 307,884 |
| 60 | ORGANIZATION-for-BUSINESS-about-MUSIC-in-BOOK | – | – | 47,274 | 47,274 |
| 61 | ORGANIZATION-for-BUSINESS-and-BUSINESS-about-BOOK | – | – | 334,395 | 334,395 |
| 62 | ORGANIZATION-to-MUSIC-and-MUSIC-in-BOOK | – | – | 23,220 | 23,220 |
| 41 | ORGANIZATION-to-MUSIC-in-BOOK | – | 8,368 | – | 8,368 |
| 63 | ORGANIZATION-to-MUSIC-in-BOOK-and-BOOK | – | – | 130,110 | 130,110 |

## 5. Model-specific structural statistics

### 5.1 RGCN — compiled semantic programs (also the input of CMPNN)

| dataset / task | physical relations (directed) | semantic relations (directed) | semantic edges (directed, every variant) | CMPNN forward half |
|---|---|---|---|---|
| IMDb NC | 8 / 8 / 8 / 8 / 12 (v1 / v2 / v3 / v4 / universal) | 14 | 91,924 | 7 relations, 45,962 edges |
| IMDb Movie–Year | 8 / 8 / 8 / 8 / 12 (v1 / v2 / v3 / v4 / universal) | 12 | 81,066 | 6 relations, 40,533 edges |
| IMDb Movie–Director | 8 / 8 / 10 (v1 / v3 / universal) | 14 | 86,908 | 7 relations, 43,454 edges |
| DBLP Paper–Conference | 8 / 8 / 8 / 12 (v1 / v2 / v3 / universal) | 12 | 294,334 | 6 relations, 147,167 edges |
| WN18RR | 11 / 22 / 537 / 2,087 per variant | 2,087 | 893,629 | 11 base relations, 86,835 edges (context-only program) |
| Freebase (block-conditional, `all_graph` scope) | 36 / 42 / 58 / 64 per variant | 64 semantic relations = 128 directed channels | 986,569,594 (sampled at training time with deterministic fan-out 4 + 4) | – |

### 5.2 MAGNN — metapaths and instances

Instances are enumerated with set semantics and capped at 500 per (source node, metapath) by an RNG keyed on the semantic identity; "capped cells" = (source, metapath) pairs whose instance set exceeded the cap. The runner samples 100 instances per (node, metapath) per batch.

**IMDb NC** (`imdb_magnn_v2/nc`; sources = the 4,180 movies):

| arm | variant | metapaths | instance rows | capped cells / non-empty cells | program hash |
|---|---|---|---|---|---|
| stock | v1 | 4: MDM MAM MLM MYM | 741,997 | 0 / 16,717 | `61523a07bf` |
| stock | v2 | 4: MLM MYM MLDLM MLALM | 741,997 | 0 / 16,717 | `41c8bcf4d1` |
| stock | v3 | 4: MDM MLM MYM MLALM | 741,997 | 0 / 16,717 | `602003518d` |
| stock | v4 | 4: MAM MLM MYM MLDLM | 741,997 | 0 / 16,717 | `f9a5c9f6f0` |
| stock | universal | 6: MDM MAM MLM MYM MLDLM MLALM | 847,875 | 0 / 25,077 | `5c31f4cd01` |
| invariant | v1 | 4: MDM MAM MLM MYM | 741,997 | 0 / 16,717 | `61523a07bf` |
| invariant | v2 | 4: MDM MAM MLM MYM | 741,997 | 0 / 16,717 | `61523a07bf` |
| invariant | v3 | 4: MDM MAM MLM MYM | 741,997 | 0 / 16,717 | `61523a07bf` |
| invariant | v4 | 4: MDM MAM MLM MYM | 741,997 | 0 / 16,717 | `61523a07bf` |
| invariant_union | v1 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 1,479,814 | 0 / 29,254 | `d1c4019be5` |
| invariant_union | v2 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 1,479,814 | 0 / 29,254 | `d1c4019be5` |
| invariant_union | v3 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 1,479,814 | 0 / 29,254 | `d1c4019be5` |
| invariant_union | v4 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 1,479,814 | 0 / 29,254 | `d1c4019be5` |

**IMDb movie_year** (`imdb_magnn_v2/movie_year`):

| arm | variant | mode0 metapaths (head side) | mode1 metapaths (tail side) | instance rows | capped cells / non-empty cells | program hash |
|---|---|---|---|---|---|---|
| stock | v1 | 4: MDM MAM MLM MYM | 1: YMY | 425,274 | 0 / 15,553 | `4fae33a1cf` |
| stock | v2 | 4: MLM MYM MLDLM MLALM | 1: YMY | 425,274 | 0 / 15,553 | `df51d9902c` |
| stock | v3 | 4: MDM MLM MYM MLALM | 1: YMY | 425,274 | 0 / 15,553 | `245cc1e157` |
| stock | v4 | 4: MAM MLM MYM MLDLM | 1: YMY | 425,274 | 0 / 15,553 | `d15c5c8b0e` |
| stock | universal | 6: MDM MAM MLM MYM MLDLM MLALM | 1: YMY | 531,152 | 0 / 23,913 | `d571cfbafb` |
| invariant | v1 | 4: MDM MAM MLM MYM | 1: YMY | 425,274 | 0 / 15,553 | `4fae33a1cf` |
| invariant | v2 | 4: MDM MAM MLM MYM | 1: YMY | 425,274 | 0 / 15,553 | `4fae33a1cf` |
| invariant | v3 | 4: MDM MAM MLM MYM | 1: YMY | 425,274 | 0 / 15,553 | `4fae33a1cf` |
| invariant | v4 | 4: MDM MAM MLM MYM | 1: YMY | 425,274 | 0 / 15,553 | `4fae33a1cf` |
| invariant_union | v1 | 6: MDM MAM MLM MYM MLDLM MLALM | 1: YMY | 531,152 | 0 / 23,913 | `d571cfbafb` |
| invariant_union | v2 | 6: MDM MAM MLM MYM MLDLM MLALM | 1: YMY | 531,152 | 0 / 23,913 | `d571cfbafb` |
| invariant_union | v3 | 6: MDM MAM MLM MYM MLDLM MLALM | 1: YMY | 531,152 | 0 / 23,913 | `d571cfbafb` |
| invariant_union | v4 | 6: MDM MAM MLM MYM MLDLM MLALM | 1: YMY | 531,152 | 0 / 23,913 | `d571cfbafb` |

**IMDb movie_director** (`imdb_magnn_v2/movie_director`):

| arm | variant | mode0 metapaths (head side) | mode1 metapaths (tail side) | instance rows | capped cells / non-empty cells | program hash |
|---|---|---|---|---|---|---|
| stock | v1 | 4: MDM MAM MLM MYM | 1: DMD | 736,921 | 0 / 17,139 | `a1951d1e8b` |
| stock | v3 | 4: MDM MLM MYM MLALM | 1: DMD | 736,921 | 0 / 17,139 | `e94eeef13c` |
| stock | universal | 5: MDM MAM MLM MYM MLALM | 1: DMD | 826,297 | 0 / 21,319 | `cc76438de3` |
| invariant | v1 | 4: MDM MAM MLM MYM | 2: DMD DLD | 739,847 | 0 / 18,815 | `8f8f700ba7` |
| invariant | v3 | 4: MDM MAM MLM MYM | 2: DMD DLD | 739,847 | 0 / 18,815 | `8f8f700ba7` |
| invariant_union | v1 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 2: DMD DLD | 1,469,662 | 0 / 30,098 | `8698995733` |
| invariant_union | v3 | 7: MDM MAM MLM MYM MLDLM MLALM MLYLM | 2: DMD DLD | 1,469,662 | 0 / 30,098 | `8698995733` |

**DBLP Paper–Conference** (`dblp_magnn_v2`; sources = 13,462 papers / 20 conferences):

| arm | variant | mode0 metapaths (head side) | mode1 metapaths (tail side) | instance rows | capped cells / non-empty cells | program hash |
|---|---|---|---|---|---|---|
| stock | v1 | 4: PAP PTP PVP PArP | 2: VPV VPAPV | 15,671,874 | 28,813 / 45,860 | `2e336fd554` |
| stock | v2 | 4: PAP PTP PVP PVArVP | 2: VPV VPAPV | 15,671,874 | 28,813 / 45,860 | `1997791c92` |
| stock | v3 | 4: PAP PTP PVP PAArAP | 2: VPV VPAPV | 17,474,874 | 32,419 / 49,466 | `62b1efe0cc` |
| stock | universal | 6: PAP PTP PVP PArP PAArAP PVArVP | 2: VPV VPAPV | 26,922,874 | 51,315 / 68,362 | `3c53019696` |
| invariant | v1 | 4: PAP PTP PVP PArP | 2: VPV VPAPV | 15,671,874 | 28,813 / 45,860 | `2e336fd554` |
| invariant | v2 | 4: PAP PTP PVP PArP | 2: VPV VPAPV | 15,671,874 | 28,813 / 45,860 | `2e336fd554` |
| invariant | v3 | 4: PAP PTP PVP PArP | 2: VPV VPAPV | 15,671,874 | 28,813 / 45,860 | `2e336fd554` |
| invariant_union | v1 | 6: PAP PTP PVP PArP PAArAP PVArVP | 2: VPV VPAPV | 26,922,874 | 51,315 / 68,362 | `3c53019696` |
| invariant_union | v2 | 6: PAP PTP PVP PArP PAArAP PVArVP | 2: VPV VPAPV | 26,922,874 | 51,315 / 68,362 | `3c53019696` |
| invariant_union | v3 | 6: PAP PTP PVP PArP PAArAP PVArVP | 2: VPV VPAPV | 26,922,874 | 51,315 / 68,362 | `3c53019696` |

**Freebase** (`freebase_magnn_v2`; sources = the 7,954 labelled books):

| arm | variant | metapaths | instance rows | capped cells / non-empty cells | program hash |
|---|---|---|---|---|---|
| stock | unchanged | 8: BOOK-about-ORGANIZATION BOOK-and-BOOK BOOK-on-LOCATION BOOK-on-SPORTS BOOK-to-FILM BUSINESS-about-BOOK MUSIC-in-BOOK PEOPLE-to-BOOK | 3,199,952 | 5,063 / 16,474 | `80fa923662` |
| stock | exact_2 | 12 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 5,015,042 | 8,518 / 20,593 | `7449c3802e` |
| stock | exact_3 | 22 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 13,502,663 | 25,332 / 37,612 | `64cb7f1565` |
| stock | union_exact_2_3 | 26 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 15,317,753 | 28,787 / 41,731 | `9fa90f3c62` |
| invariant | unchanged | 26 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 15,317,753 | 28,787 / 41,731 | `9fa90f3c62` |
| invariant | exact_2 | 26 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 15,317,753 | 28,787 / 41,731 | `9fa90f3c62` |
| invariant | exact_3 | 26 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 15,317,753 | 28,787 / 41,731 | `9fa90f3c62` |
| invariant_union | unchanged | 44 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 27,490,662 | 52,705 / 66,988 | `d694b5239c` |
| invariant_union | exact_2 | 44 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 27,490,662 | 52,705 / 66,988 | `d694b5239c` |
| invariant_union | exact_3 | 44 (BOOK-about-ORGANIZATION, BOOK-about-ORGANIZATION-and-ORGANIZATION-for-BUSINESS, BOOK-about-ORGANIZATION-and-ORGANIZATION-to-MUSIC, …; full list in metapaths.json) | 27,490,662 | 52,705 / 66,988 | `d694b5239c` |

### 5.3 SeHGNN — pre-computed metapath channels

Flavours used: IMDb `full`, Freebase `full_rel` (relation-aware). One channel per type path (IMDb: K ≤ 4, plus label-propagation channels along the same paths, the node's own label removed; Freebase: K ≤ 2, channels are sparse selectors over learned embeddings, capped at 500 seeded walks per (target, channel)). `physical` = channels of the variant's own graph (originals / universal arms); `semantic` = channels on the compiled semantic program (invariant arm; one program hash). On IMDb the semantic channels are enumerated over the canonical schema with the Link type skipped (one channel per τ-equivalence class).

| dataset | flavour | graph | variant | channels | label channels | program hash |
|---|---|---|---|---|---|---|
| IMDB | full | physical | universal | 101 | 31 | `908b4a627e` |
| IMDB | full | physical | v1 | 41 | 21 | `3ec5f6be06` |
| IMDB | full | physical | v2 | 27 | 9 | `c58c7bfdb2` |
| IMDB | full | physical | v3 | 32 | 14 | `c5bb2a24e1` |
| IMDB | full | physical | v4 | 32 | 14 | `806801256b` |
| IMDB | full | semantic | v1 | 25 | 13 | `8e48cd229f` |
| IMDB | full | semantic | v2 | 25 | 13 | `8e48cd229f` |
| IMDB | full | semantic | v3 | 25 | 13 | `8e48cd229f` |
| IMDB | full | semantic | v4 | 25 | 13 | `8e48cd229f` |
| FREEBASE | full_rel | physical | exact_2 | 159 | 0 | `491e562861` |
| FREEBASE | full_rel | physical | exact_3 | 443 | 0 | `27b950f487` |
| FREEBASE | full_rel | physical | unchanged | 91 | 0 | `b5ee053b69` |
| FREEBASE | full_rel | physical | union_exact_2_3 | 597 | 0 | `3681e462e6` |
| FREEBASE | full_rel_uov511 (invariant) | semantic, limited to the union of the variants' physical biases | exact_2 / exact_3 / unchanged | 511 of 597 | 0 | `5ddf2735ff` (subset keys `ecdfe37fda`) |

`full_rel_uov511` = the 597-channel `full_rel` program minus the 86 channels no variant's own bias contains (91 ∪ 159 ∪ 443 = 511; by hop count: 0-hop 1, 1-hop 9, 2-hop 85, 3-hop 66, 4-hop 196, 6-hop 154); the same 511 keys on every variant, while the universal arm is the stock 597-channel `full_rel` on the union graph.

### 5.4 SlotGAT — message schedules

**IMDb NC** (raw-context conditional operator; from the run records):

| variant | raw context edges | semantic messages (propagation) | message edge types | conditional witnesses | message-program hash |
|---|---|---|---|---|---|
| v1 | 50,142 | 107,713 | 15 | 40,860 | `d58612aaf5` |
| v2 | 50,142 | 107,713 | 15 | 40,860 | `d58612aaf5` |
| v3 | 50,142 | 107,713 | 15 | 40,860 | `d58612aaf5` |
| v4 | 50,142 | 107,713 | 15 | 40,860 | `d58612aaf5` |

**Freebase** (recoverability-conditional operator, family {unchanged, exact_2, exact_3}; `physical` = the variant's own one-message-per-ordered-pair schedule (originals / universal arms), `union` = the family's union schedule every invariant run consumes):

| schedule | variant | forward edges | messages | channels | base messages | conditional path records | catalog hash |
|---|---|---|---|---|---|---|---|
| physical | unchanged | 1,057,688 | 1,623,206 | 129 | 2,295,474 | 420,063 | `1d2524d725` |
| physical | exact_2 | 47,368,991 | 94,234,382 | 129 | 2,295,474 | 420,063 | `1d2524d725` |
| physical | exact_3 | 446,967,136 | 750,258,416 | 129 | 2,295,474 | 420,063 | `1d2524d725` |
| physical | union_exact_2_3 | 493,278,439 | 750,545,656 | 129 | 2,295,474 | 420,063 | `1d2524d725` |
| union (invariant, all variants) | union_exact_2_3 | 493,278,439 | 750,545,656 | 129 | 2,295,474 | 420,063 | `1d2524d725` |

**Link prediction (SlotGAT LP):** the deterministic parity SlotGAT consumes the RGCN structural bundles of §5.1 as plain edge lists (physical edges for originals / universal / augmentation, the compiled semantic program for the invariant arm; one shared edge-type vocabulary per family, synthetic reverse edges only where a physical reverse is missing, one self loop per node). Message counts are therefore the edge counts of §5.1 (plus reverses and self loops); no separate schedule is compiled.

