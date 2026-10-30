# CMPNN (NBFNet / C-MPNN): link prediction, four arms

CMPNN is evaluated on link prediction only: IMDb Movie–Year (MY), IMDb Movie–Director (MD),
DBLP Paper–Conference and WN18RR. There is no node classification and no Freebase experiment.

## How the invariant version works

`cmpnn/{layer,model}.py` is the torchdrug C-MPNN/NBFNet model. A query `(h, r, ?)` puts the
relation embedding on `h`, runs 6 layers of query-conditioned Bellman–Ford
(`GeneralizedRelationalConv`, DistMult messages, PNA aggregation on IMDb/DBLP and SUM on WN18RR),
and scores every candidate tail with an MLP. `forward()` adds inverse edges itself
(`graph.undirected(add_inverse=True)`), and training removes the query's own edges
(`remove_easy_edges`).

A switch between variants (for example Area on Paper vs. Area on Venue, a materialised `r⁻¹`,
or a 2-hop path vs. a named shortcut) changes the set of relational paths rooted at `h`. That
changes the PNA inputs, so the stock model is not τ-invariant. The fix is Mechanism A
(`docs/THEORY.md` §2.5, `docs/INVARIANT_MECHANISMS.md` §4):

* **No compiler of its own.** CMPNN reads the per-variant **compiled semantic program** that
  the RGCN package builds (the RGCN structural bundles). These programs are hash-identical
  across variants. `inv_cmpnn/bundles.py` keeps the *forward half* of every relation pair,
  because CMPNN adds the inverses itself. It also builds the Movie↔Link alias map (IMDb).
  Because the labelled edge multiset is identical, the query-rooted path set is identical for
  every `h`, so every node at every layer gets identical PNA inputs and the scores are identical.
* **Masking derived copies.** `inv_cmpnn/model.py: InvariantCMPNN.remove_easy_edges` masks
  the query edge together with its alias copies (IMDb `link(m)→d`) and the head's label-derived
  edges (DBLP `paper_to_area`). This avoids a train-only shortcut to the label.
* **Bitwise determinism.** `inv_cmpnn/conv.py: DeterministicRelationalConv` sorts the edges
  by `(dst, rel, src)` and reduces with `torch.segment_reduce`, with no atomics. WN18RR uses a
  chunked mode (`--chunk-edges`). On a V100, the invariant arm gives identical checkpoints on
  every variant (τ = 1).
* **WN18RR invariant arm = context-only program.** `run_cmpnn_lp_wordnet_context.py`
  keeps only the 11 `base::` relations of the semantic program. The `inverse::` relations are
  CMPNN's own inverses. The `dependency::` 3-hop shortcuts are context: the chain is already
  inside the 6-layer path bias, and the shortcut edges mined from training triples leak the
  masked query. This program is byte-identical to the `no_changes` physical graph, so the
  invariant arm reproduces the `no_changes` original bitwise.

Arms (`run_cmpnn_lp.py` unless noted):

| arm | how |
|---|---|
| `original` | `--graph physical`, one run per variant |
| `universal` | `--graph physical --variant universal` (WN18RR: `universal_edges`) |
| `augmentation` | `run_cmpnn_lp_augmentation.py`: one model over the union relation vocabulary, trained jointly on all variants' physical graphs in super-epochs (budget ÷ #variants), selected on mean val MRR |
| `invariant` | `--graph semantic`, one run per variant (WN18RR: `run_cmpnn_lp_wordnet_context.py --graph semantic`) |

All arms share the same supervision, negatives and seeds, taken from the bundle.

## Required inputs under `data/`

CMPNN reads **only** the RGCN structural bundles (`graph.pt` per variant). It reads no
other file and no node features.

| relative path under `data/` | size | built by (RGCN package) |
|---|---|---|
| `preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/movie_year/{v1,v2,v3,v4,universal}/graph.pt` | 196.5 MB each (universal 197.8 MB) | `scripts/RGCN/preprocess_imdb.sh` (`preprocess_IMDB_rgcn_structural.py --tasks nc,movie_director,movie_year`, written to the `_v10_year_canonfeat` root) |
| `preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/movie_director/{v1,v3,universal}/graph.pt` | 196.7 MB each (universal 197.7 MB) | same builder |
| `preprocessed/RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}/graph.pt` | 28.2–29.2 MB | `scripts/RGCN/preprocess_dblp.sh` (`preprocess_DBLP_rgcn_structural.py`, canonical shared split, written to the `_v8_dblp` root) |
| `preprocessed/RGCN_structural/WORDNET/universal_parity_full/full/intersection/{no_changes,all_inverse_edges,transitive_edges,universal_edges}/graph.pt` | 70.2 / 77.8 / 124.6 / 141.1 MB | `scripts/RGCN/preprocess_wn18rr.sh` (`run_wordnet_rgcn_structural_full.sh` → `preprocess_WORDNET_rgcn_structural.py`, budget `full`, mode `intersection`, `PREPROCESSED_ROOT=…/universal_parity_full`) |

Nothing here is downloaded directly. Build the bundles in this order:

1. Run `dataset_construction/` (or download tier A) to get the raw/variant graphs and shared splits.
2. Run the RGCN preprocessing scripts. For the exact commands and their raw inputs, see
   `models/RGCN/README.md`.

The runner hard-codes the bundle **directory names** above (`run_cmpnn_lp.py: BUNDLE_ROOTS`).
If you build with the RGCN builders' own defaults (`preprocessed/RGCN_structural/...`), pass the
corresponding `--out-root` or move the bundles to these names. `--data-root` defaults to
`data/preprocessed`; the scripts pass `$INV_GNN_DATA/preprocessed`.

Expected hashes (recorded in every `metrics.json`):

* IMDb MD v1 semantic, forward half: `dc2d516ca000…`
* bundle semantic hash: MY `d74dc76282a9…`, MD `63298526d97b…`
* WN18RR base-relation program: `73bf23174f47…` on all four variants

Verification step (not an experiment): the data-level gauntlet checks the bundles before training
(hashes, duplicate edges, ≤2-hop leak audit; CPU only, no training):
`ARM=gauntlet bash scripts/CMPNN/run_<experiment>.sh`.

## Environment

See `envs/cmpnn.txt`. Tested with Python 3.10, PyTorch 2.0.1+cu118, torchdrug 0.2.1
and torch_scatter 2.1.2. torchdrug JIT-compiles a CUDA extension, so `nvcc` must match:

* tested with CUDA 11.7 on V100
* `CMPNN_MODULE_SETUP=1` sources `models/CMPNN/scripts/setup.sh`, which loads the CUDA module
  and sets a per-architecture `TORCH_EXTENSIONS_DIR` (needs an environment-module system)
* `ENV_ACTIVATE="..."` is evaluated first, if set

Run everything from the repository root:
`ARM=<arm> [SEEDS="..."] bash scripts/CMPNN/run_<experiment>.sh`.
Runs are skip-if-done (`metrics.json`) and resumable (`--resume`), and they write
`results/<experiment>/<arm>/<DATASET>/<task>/<variant>/seed_<s>/{metrics.json,scores.npz,checkpoint.pt,trace.json}`
under `models/CMPNN/`. After the runs, the arm is aggregated with
`common/aggregate_rgcn_structural.py`, which reports per-variant means, pairwise Kendall τ
and checkpoint-hash equality.

## Expected results (test MRR, mean ± population std over seeds 1566911444 / 20241017 / 20251017; V100)

The full tables are in `docs/results/{IMDB_lp,DBLP_lp,WN18RR_lp}.tex`. The variant mapping is
IMDb1–4 = v1–v4, DBLP1–3 = v1–v3, and WN18RR1–3 = no_changes / all_inverse_edges / transitive_edges.

### IMDb Movie–Year (`scripts/CMPNN/run_imdb_movie_year.sh`)

A run takes about 10 min and peaks at about 4.3 GiB.

| arm | command | result dir (`models/CMPNN/`) | expected MRR |
|---|---|---|---|
| original | `ARM=original bash scripts/CMPNN/run_imdb_movie_year.sh` | `results/IMDB_movie_year/original` | v1 .1887±.0115 / v2 .1928±.0058 / v3 .1939±.0044 / v4 .1918±.0052 |
| universal | `ARM=universal …` | `results/IMDB_movie_year/universal` | .1931±.0031 |
| augmentation | `ARM=augmentation …` | `results/IMDB_movie_year/augmentation` | .1919 / .1991 / .1965 / .1928 (mean .1951±.0012) |
| invariant | `ARM=invariant …` | `results/IMDB_movie_year/invariant` | .1931±.0031 on every variant (τ = 1; ≡ universal bitwise) |

### IMDb Movie–Director (`scripts/CMPNN/run_imdb_movie_director.sh`)

A run takes about 15 min and peaks at about 4.5 GiB.

| arm | command | result dir | expected MRR |
|---|---|---|---|
| original | `ARM=original bash scripts/CMPNN/run_imdb_movie_director.sh` | `results/IMDB_movie_director/original` | v1 .4152±.0109 / v3 .3430±.0072 |
| universal | `ARM=universal …` | `…/universal` | .4171±.0061 |
| augmentation | `ARM=augmentation …` | `…/augmentation` | v1 .4140±.0093 / v3 .3478±.0041 (mean .3809±.0059) |
| invariant | `ARM=invariant …` | `…/invariant` | .4228±.0066 on v1 and v3 (τ = 1) |

### DBLP Paper–Conference (`scripts/CMPNN/run_dblp_lp.sh`)

A run takes about 4.8 h (full 20-conference ranking, about 14 min per
epoch) and peaks at about 11 GiB. Run one arm or seed per job.

| arm | command | result dir | expected MRR |
|---|---|---|---|
| original | `ARM=original bash scripts/CMPNN/run_dblp_lp.sh` | `results/DBLP_paper_conference/original` | v1 .6195±.0026 / v2 .6132±.0039 / v3 .6150±.0053 |
| universal | `ARM=universal …` | `…/universal` | .6194±.0012 |
| augmentation | `ARM=augmentation …` | `…/augmentation` | .6178 / .6166 / .6161 (mean .6168±.0055) |
| invariant | `ARM=invariant …` | `…/invariant` | .6194±.0012 on every variant (τ = 1; ≡ universal bitwise) |

### WN18RR (`scripts/CMPNN/run_wn18rr_lp.sh`), filtered full-entity MRR

Runner flags: `EXTRA="--batch-size 64 --chunk-edges 200000"` (the script default).

Time and memory per run on a V100:

* original: 2–14 h
* universal: about 16.5 h
* augmentation: about 6 h (joint)
* invariant (context program): about 3 h
* memory: about 17 GiB

| arm | command | result dir | expected MRR |
|---|---|---|---|
| original | `ARM=original bash scripts/CMPNN/run_wn18rr_lp.sh` | `results/WN18RR_link_prediction/original` | .4238±.0095 / .3075±.0559 / .2927±.0306 |
| universal | `ARM=universal …` | `…/universal` | .3524±.0036 |
| augmentation | `ARM=augmentation …` | `…/augmentation` | .3625 / .3136 / .2748 (mean .3169±.0143) |
| **invariant** (context-only, 11 base relations) | `ARM=invariant …` | `…/invariant` | .4238±.0095 on every variant (τ = 1; = no_changes original bitwise) |

Hits@1/3, precision/recall/F1 and τ tables are in the same `.tex` files.

## Files

| file | role |
|---|---|
| `run_cmpnn_lp.py` | runner for the original / universal / invariant arms (dataset settings in `DEFAULTS`) |
| `run_cmpnn_lp_augmentation.py` | augmentation arm |
| `run_cmpnn_lp_wordnet_context.py` | WN18RR invariant arm (wraps `run_cmpnn_lp.py`, base-relation program) |
| `validate_cmpnn_invariance.py` | data-level gauntlet |
| `inv_cmpnn/{bundles,conv,model}.py` | bundle loader, deterministic conv, `InvariantCMPNN` |
| `cmpnn/{layer,model}.py` | stock torchdrug C-MPNN |
| `scripts/setup.sh` | optional CUDA-module / extension-cache setup (V100 / sm 7.0 only; add other GPUs to its `case`) |
| `tests/` | CPU unit tests: `PYTHONPATH=. python -m unittest tests.test_inv_cmpnn`; `PYTHONPATH=. python tests/test_remove_easy_edges_one_hop_false.py` |
| `../../common/inv_rgcn/`, `../../common/rgcn_resource_metrics.py` | bundle dataclass, metrics and determinism utils shared with RGCN |

The scripts set `OPENBLAS/OMP/MKL_NUM_THREADS=4` by default.
