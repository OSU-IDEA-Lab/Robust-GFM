# RGCN: original / universal / augmentation / invariant

RGCN code for every RGCN experiment: IMDb NC, IMDb LP Movie–Year and Movie–Director,
DBLP LP Paper–Conference, WN18RR LP and Freebase NC. The joint-training (augmentation)
runners for DBLP and WN18RR live in `augmentation/`.
Run everything through `scripts/RGCN/*.sh`. They are plain bash; on a Slurm cluster, wrap
them in `sbatch`/`srun`.

## How the invariant RGCN works

(See also `docs/INVARIANT_MECHANISMS.md` §2 and `docs/THEORY.md` §2.1.)
Each dataset has a **conditional compiler** that never modifies the input graph. For each
physical variant, it turns every occurrence of a *context pattern* into a propagation edge
typed by its **semantic relation**. The result is a *semantic message program*: a set of
`(src, semantic_relation, dst)` messages that is identical across information-equivalent
variants. This is checked by hashing: the physical hashes differ, but the semantic hash is
the same. Nodes that only license a message (skip nodes) contribute no features.

The RGCN then propagates **only** on those edges (`common/inv_rgcn/model.py: ConditionalRGCNLayer`):

* basis-decomposed relation weights are indexed by the semantic relation *name*;
* reductions are `segment_reduce` sums in a fixed destination order.

With the same program, supervision, initialisation and reduction order, training is
bit-identical across variants (spread 0, Kendall-τ = 1). Set semantics on `(src, rel, dst)`
is the message-level quotient, so no extra redundancy step is needed (Lemma 1 holds with
`b_τ` = the context realisation of the same semantic relation).

| dataset | compiler | program vs. union graph |
|---|---|---|
| IMDb (year graph) | `preprocess_IMDB_rgcn_structural.py` (`compile_semantic`, `link_to_{director,actor,year}` projections) | NC / Movie–Director: strict superset (`link–year`, `link–director`); Movie–Year: equal |
| DBLP | `preprocess_DBLP_rgcn_structural.py` (block-local Paper–Area pair filter, training-support papers only) | equal (294,334 edges) |
| WN18RR | `preprocess_WORDNET_rgcn_structural.py` + `wordnet_structural_catalogs/full_selection.json` (virtual transposes where inverses are absent, verified shortcuts where present; intersection mode) | equal edge set (893,629 edges, 2,087 relations); objective `task_only` |
| Freebase | `preprocess_Freebase_rgcn_block_conditional.py` + `catalogs/exact_3` (block-local composition over CSR channels; sampled 2-layer RGCN with deterministic splitmix64 fan-out 4+4, `freebase_conditional_rgcn.py`) | the union graph's 64 relations, rebuilt from the base relations per variant; +12,716 walk-composed `BOOK-and-BOOK` pairs (0.0013 %) |

Arms:

| arm | how |
|---|---|
| `original` | the same runner on each variant's own physical graph (`--graph physical` on IMDb / DBLP / WN18RR; physical-only catalogs on Freebase) |
| `universal` | the same runner on the union graph (`universal` / `universal_edges` / `union_exact_2_3`) |
| `augmentation` | one model trained jointly on the variants' physical graphs (union relation vocabulary) |
| `invariant` | the compiled semantic program (`--graph semantic`; `exact_3` catalog on Freebase) |

## Software

Python 3.11, PyTorch 2.4.0+cu124, DGL 2.4.0 (`DGLBACKEND=pytorch`), NumPy, SciPy,
scikit-learn and pandas (`envs/gnn.txt`). The WN18RR augmentation arm also needs
**`torch_geometric`**, because `augmentation/model_RGCN_lp_wordnet.py` uses
`torch_geometric.utils.scatter`.
Shared modules come from `common/`: `inv_rgcn/`, `rgcn_resource_metrics.py` and
`imdb_canonical_features.py`. All expected results below were produced on an NVIDIA
V100-SXM3-32GB.

## Data inputs

All paths are under `data/` at the repository root (`models/RGCN/data -> ../../data`).

### Built by `dataset_construction/` (variant level, not built here)

| path | what | size |
|---|---|---|
| `raw/IMDB/movie_metadata.csv` | Kaggle IMDb 5000 (raw download) | 1.5 MB |
| `raw/DBLP/{author_label,paper_author,paper_conf,paper_term}.txt` | DBLP 4-area (raw download; only these 4 files are read) | 0.1 / 0.5 / 0.2 / 1.2 MB |
| `preprocessed/IMDB_shared_splits/IMDB_md_shared_splits.npz` | Movie–Director shared split (only the positives are read) | 68 KB |
| `preprocessed/IMDB_shared_splits/IMDB_my_shared_splits.npz` | Movie–Year shared split | 70 KB |
| `preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz` | DBLP P–C canonical split (seed 1566911444) | 4.8 MB |
| `wordnet_3hops_augmented_full/wordnet_splits.npz` | WN18RR: every variant's training triples (`train_pos_<variant>`) + official val/test, 2,087-relation vocabulary | 24.7 MB |
| `wordnet_3hops_augmented_full/` (rest) | variant folders (built by `dataset_construction/`); the RGCN scripts read only `wordnet_splits.npz` from this folder | ~150 MB |
| `dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/{info,node,link,label}.dat` | Freebase variants (HGB format) | link.dat 20 MB / 0.89 GB / 8.5 GB / 9.3 GB; node.dat 5.1 MB, label.dat 0.3 MB each |

The IMDb NC split is drawn inside `preprocess_IMDB_rgcn_structural.py` (seed 1566911444).
The Freebase split is drawn inside `common/freebase_block_common.make_split` (seed 1566911444).

### Built here (in order)

| step | command | output (under `data/preprocessed/`) | read by |
|---|---|---|---|
| 1 | `bash scripts/RGCN/preprocess_imdb.sh` (CPU, minutes) | `RGCN_structural_v10_year_canonfeat/IMDB/{nc,movie_year,movie_director}/<variant>/{graph.pt,audit.json}` (~2.4 GB; 13 bundles of ~195 MB) | **shared**: RGCN, CMPNN (MY/MD), MAGNN, SeHGNN, SlotGAT (LP) |
| 2 | `bash scripts/RGCN/preprocess_dblp.sh` (CPU, ~2 min) | `RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}` (~115 MB) | **shared**: RGCN, CMPNN, MAGNN (DBLP), SlotGAT (LP) |
| | (same script) | `DBLP_rgcn_{v1,v2,v3}_canonical/{graph_data.pt,meta.pt}` (~6.6 MB each) | RGCN augmentation arm only |
| 3 | `bash scripts/RGCN/preprocess_wn18rr.sh` (CPU, ~4 min, ~24 GB RAM) | `RGCN_structural/WORDNET/universal_parity_full/full/intersection/{no_changes,all_inverse_edges,transitive_edges,universal_edges}` (~415 MB) | **shared**: RGCN, CMPNN, SlotGAT (LP) |
| 4 | `bash scripts/RGCN/preprocess_freebase.sh` (CPU, ~48 GB RAM, hours) | `freebase_rgcn_physical/<V>/physical_<V>/union/all_graph` (V ∈ unchanged, exact_2, exact_3, union_exact_2_3; ≈ 2 M / 95 M / 894 M / 987 M edges) and `freebase_rgcn_block_conditional/<V>/exact_3/union/all_graph` (V ∈ unchanged, exact_2, exact_3; 987 M edges each, hard-link deduplicated) | RGCN |

Other models hard-code these bundle names, so keep the `--out-root`s used in the scripts.

Expected hashes for checking a rebuild (printed by the builders and stored in `audit.json` /
`preprocess_summary.json`):

* IMDb semantic: nc `942317043f15`, movie_director `63298526d97b`, movie_year `d74dc76282a9`;
  feature table `66c99116b7530a47`
* DBLP semantic: `308cec36b62b`
* WN18RR semantic: `64d2c91fa9d7…`

`audit.json` stores the split path as it was passed on the command line, so its
`shared_split_file` / `shared_npz` string depends on where you run from. The graph hashes do
not depend on it.

## Experiments

Run `ARM=<arm> bash scripts/RGCN/<script>`. Defaults:

* `SEEDS="1566911444 20241017 20251017"`
* `DEVICE=cuda:0`
* results go to `models/RGCN/results/<dataset>_<task>/<arm>/` (override with `RESULTS=`)

Expected results are mean ± population std over the 3 seeds, as in `docs/results/*.tex`.
For `original` and `augmentation`, the per-variant values are listed in variant order.
Time is training seconds per run on a V100.

### IMDb NC (micro-F1): `run_imdb_nc.sh`

| arm | output dir | expected | time / GPU |
|---|---|---|---|
| original | `results/imdb_nc/original` | .6515±.0171 / .6033±.0030 / .6491±.0157 / .6180±.0089 | ~45 s per run, <1 GB |
| universal | `results/imdb_nc/universal` | .6535±.0089 | ~45 s |
| augmentation | `results/imdb_nc/augmentation` | .6136 / .5981 / .6077 / .5993 (mean .6047±.0159) | ~15 s per seed |
| invariant | `results/imdb_nc/invariant` | .6531±.0107 on every variant (τ = 1) | ~45 s |

### IMDb LP Movie–Year (MRR, 19 frequency-matched negatives): `run_imdb_movie_year.sh`

| arm | output dir | expected | time / GPU |
|---|---|---|---|
| original | `results/imdb_movie_year/original` | .2005±.0102 / .2020±.0065 / .1973±.0090 / .2012±.0026 | ~240 s, 1.2 GB |
| universal | `results/imdb_movie_year/universal` | .2003±.0082 | ~220 s |
| augmentation | `results/imdb_movie_year/augmentation` | .2004 / .1945 / .1983 / .1993 (mean .1981±.0022) | ~95 s per seed |
| invariant | `results/imdb_movie_year/invariant` | .2003±.0082 on every variant (= universal) | ~220 s, 1.4 GB |

### IMDb LP Movie–Director (MRR; variants v1, v3): `run_imdb_movie_director.sh`

| arm | output dir | expected | time |
|---|---|---|---|
| original | `results/imdb_movie_director/original` | .1845±.0498 / .2063±.0579 | ~220 s |
| universal | `results/imdb_movie_director/universal` | .1766±.0272 | ~210 s |
| augmentation | `results/imdb_movie_director/augmentation` | .2194 / .2077 (mean .2135±.0264) | ~115 s per seed |
| invariant | `results/imdb_movie_director/invariant` | .2001±.0047 on v1 and v3 | ~220 s |

### DBLP LP Paper–Conference (MRR over all 20 conferences): `run_dblp_lp.sh`

| arm | output dir | expected | time / GPU |
|---|---|---|---|
| original | `results/dblp_lp/original` | .3254±.0177 / .3289±.0172 / .3767±.0161 | ~50 s, 2.8 GB |
| universal | `results/dblp_lp/universal` | .4024±.0606 | ~55 s |
| augmentation | `results/dblp_lp/augmentation` | .3632 / .3161 / .3163 (mean .3319±.0257) | ~250 s per seed |
| invariant | `results/dblp_lp/invariant` | .4024±.0606 on every variant (= universal) | ~55 s |

### WN18RR LP (filtered MRR): `run_wn18rr_lp.sh`

| arm | output dir | expected | time / GPU |
|---|---|---|---|
| original | `results/wn18rr_lp/original` | .3819±.0076 / .3977±.0029 / .3663±.0017 | 0.5–1 h per run, 1–2 GB |
| universal | `results/wn18rr_lp/universal` | .3761±.0025 | ~40 min, 2.5 GB |
| augmentation | `results/wn18rr_lp/augmentation` | .3591 / .3752 / .3278 (mean .3540±.0125) | ~5.5 h per seed, 11 GB |
| invariant | `results/wn18rr_lp/invariant` | .3802±.0019 on every variant | ~42 min, 2.5 GB |

### Freebase NC (micro-F1): `run_freebase_nc.sh`

| arm | output dir | expected | time |
|---|---|---|---|
| original | `results/freebase_nc/original` | .6824±.0053 / .6857±.0095 / .6885±.0038 | 8–17 min per run |
| universal | `results/freebase_nc/universal` | .6864±.0031 | ~20 min |
| augmentation | `results/freebase_nc/augmentation` | .6906 / .6897 / .6916 (mean .6906±.0042) | ~12 min per seed |
| invariant | `results/freebase_nc/invariant` | .6895±.0047 on every variant | ~25 min per run |

## Other files

- `catalogs/`: the Freebase block-conditional catalogs the scripts use.
  `exact_3/union/all_graph.json` is the invariant arm's catalog.
  `physical_{unchanged,exact_2,exact_3,union_exact_2_3}/union/all_graph.json` are used for the
  original / universal / augmentation stores; `build_physical_only_catalog.py` regenerates
  them (idempotently) inside `scripts/RGCN/preprocess_freebase.sh`.
- `derive_Freebase_rgcn_catalogs_from_MAGNN.py` + `catalogs/CATALOG_DERIVATION_SUMMARY.json`:
  documents how the `exact_3` catalog was derived from the canonical metapath manifests and
  shortcut registries of a Freebase MAGNN package. That package is not included, so the
  scripts do not run this file. The summary is the script's output; it also lists catalogs
  that no experiment here uses (not included).
- `wordnet_structural_catalogs/`: `full_selection.json` is the WN18RR catalog used.
  `reduced_selection.json` is required by the existence check in `run_wordnet_rgcn_structural_full.sh`.
- `validate_Freebase_rgcn_block_invariance.py`, `validate_WORDNET_rgcn_structural.py`:
  pre-training invariance audits run by the preprocessing scripts (`VALIDATE=1` for WN18RR).
- `aggregate_WORDNET_rgcn_structural.py`, `aggregate_Freebase_rgcn_block_conditional.py`:
  per-arm aggregation (the Freebase augmentation runner imports the latter).
- `dedupe_freebase_channels.py`: hard-links identical Freebase channel files (saves disk only).
