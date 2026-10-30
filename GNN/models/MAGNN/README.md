# MAGNN — four arms on IMDb (NC, LP Movie–Year, LP Movie–Director), DBLP (LP Paper–Conference) and Freebase (NC)

Stock MAGNN (`model/MAGNN_nc_mb.py` for NC, `model/MAGNN_lp.py` for LP; RotatE0 instance encoder,
intra-/inter-metapath attention) run under one contract per task. What changes between the arms is
**only the metapath-instance tables the model is fed**; those are compiled per variant by three
compilers that share `magnn_compile_common.py`. MAGNN is not run on WN18RR.

## How the invariant version works

MAGNN's structural bias is the multiset of metapath instances rooted at a node. A switching between
information-equivalent variants changes which label sequence realises a semantic relation (IMDb
`M–D` vs `M–L–D`, DBLP `P–Ar` vs `P–V–Ar` vs `P–A–Ar`, Freebase a direct shortcut vs its 2-/3-hop
composition), so the stock instance multisets differ across variants. The invariant arm uses
**skip metapaths**, compiled rather than hand-written:

* A semantic relation's endpoint set on a variant is its *skip-compiled* realisation (the direct edge
  where the variant materialises it, else the binarised composition along its context, interior
  witnesses collapsed). For IMDb / DBLP this is read from the RGCN structural bundles (their compiled
  semantic program is hash-identical across variants); for Freebase it is computed here from the raw
  variants and the shortcut registries of the family {unchanged, exact_2, exact_3}.
* The metapath for relation `r` is the symmetric `C –r→ X –r⁻¹→ C`; its instances `(c, x, c′)` are
  stored **exactly once** (set semantics, canonical order), so skip nodes never enter the instance.
* A deterministic cap of 500 instances per (source, metapath) is drawn by an RNG seeded from a hash of
  `(seed, metapath name, source id)` — a function of the semantic instance set, never of the physical
  layout. The runners sample 100 instances per (node, metapath) per batch with the stock
  `count^¾` rule and a per-(epoch, batch) seeded generator.

Identical instance tables ⇒ identical batches ⇒ identical first-step gradients, per-epoch state hashes
and checkpoints on every variant (τ = 1, spread 0 across variants for every seed).

There are two invariant **flavours** (both under `ARM=invariant`, selected by `FLAVOUR`):

| `FLAVOUR` | runner arm / bundle dir | metapath set | row in `docs/results` |
|---|---|---|---|
| `union` (default) | `invariant_union` | intersection set + the long forms whose interior nodes are real on every variant (IMDb `MLDLM MLALM MLYLM`; DBLP `PAArAP PVArVP`; Freebase 18 long forms over base relations) | "invariant + union metapaths" |
| `intersection` | `invariant` | one skip metapath per semantic relation incident to the target (IMDb `MDM MAM MLM MYM`; DBLP `PAP PTP PVP PArP`, tail side `VPV VPAPV`; Freebase 26 `BOOK–r→Y–r⁻¹→BOOK`) | "invariant (intersection metapaths)" |

Recoverability identities (same program hash ⇒ bitwise the same model): `stock/universal` ≡
`invariant_union` on DBLP (`3c5301969657`), Freebase (`9fa90f3c62f7` — there the universal graph's
stock set equals the *intersection* program) and IMDb Movie–Year (`d571cfbafb`); `stock/v1` ≡
`invariant` on IMDb NC (`61523a07bf`) and DBLP (`2e336fd55421`). See `docs/THEORY.md` §2.2 / §5 and
`docs/INVARIANT_MECHANISMS.md` §3.

LP training masks (every arm): for a training query (h, t) the query edge is removed from the instance
sets of the positive **and every negative candidate**, plus its alias copy (IMDb `link(m)`) and the
head's label-derived metapaths (DBLP `paper_to_area`, IMDb MD `link_to_director`) — CMPNN's
`remove_easy_edges`, so a training query looks like a held-out one.

Other arms: `original` = stock metapaths of each variant's own physical graph (bundle `stock/<v>`);
`universal` = stock metapaths on the union graph (`stock/universal`, Freebase `stock/union_exact_2_3`;
no skip paths); `augmentation` = one model over the union metapath vocabulary of the per-variant stock
bundles (absent metapaths = empty slots), trained on every variant in seeded shuffled super-epochs
(budget = 100 variant-epochs ÷ #variants, patience ÷ #variants), selected on the mean validation metric.

## Files

| file | role |
|---|---|
| `preprocess_imdb_magnn_v2.py` | IMDb compiler (nc / movie_year / movie_director) from `RGCN_structural_v10_year_canonfeat` |
| `preprocess_dblp_magnn_v2.py` | DBLP compiler from `RGCN_structural_v8_dblp` (imports `relation_matrices` from the IMDb compiler) |
| `preprocess_freebase_magnn_v2.py` | Freebase compiler from the raw HGB-format variants + registries |
| `magnn_compile_common.py` | set-semantics instances, seeded cap, exact hierarchical long-path sampler, hashes |
| `run_freebase_magnn_nc.py` | NC runner (`--dataset FREEBASE|IMDB`), arms `invariant`/`invariant_union`/`originals`/`universal` |
| `run_freebase_magnn_nc_augmentation.py` | NC augmentation runner |
| `run_magnn_lp.py` | LP runner (`--dataset IMDB|DBLP`), same four runner arms |
| `run_magnn_lp_augmentation.py` | LP augmentation runner |
| `model/{__init__,MAGNN_nc,MAGNN_nc_mb,MAGNN_lp,base_MAGNN}.py` | the MAGNN modules of the original MAGNN repository (unchanged) |
| `../../common/{inv_rgcn/, rgcn_resource_metrics.py, freebase_block_common.py, aggregate_rgcn_structural.py}` | shared helpers (hashes / determinism / ranking metrics / GPU accounting; `inv_rgcn.data.load_graph_data` reads the RGCN bundles; the canonical Freebase split; the per-arm aggregator) |

**Also consumed by SeHGNN.** `models/SeHGNN` imports `preprocess_imdb_magnn_v2` (`TYPES`,
`metapath_specs`, `relation_matrices`), `preprocess_freebase_magnn_v2` (`build_catalog`, `Variant`,
`endpoint_pairs`, …) and `magnn_compile_common` from this directory (its `sys.path` points at
`../MAGNN`) for its type, relation-matrix and Freebase catalog helpers when it compiles its
channels (it reads no MAGNN-built data). Keep these files here and unchanged.

## Required inputs under `data/`

Downloaded, or produced by `dataset_construction/` / the RGCN preprocessing (not by this model):

| path | what | produced by | size |
|---|---|---|---|
| `data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/{nc,movie_year}/{v1,v2,v3,v4,universal}/graph.pt`, `.../movie_director/{v1,v3,universal}/graph.pt` | RGCN structural bundles (physical graph, compiled semantic program, canonical pooled-BoW features, labels, shared splits, K=19 negatives) | `scripts/RGCN/preprocess_imdb.sh` (see `models/RGCN/README.md`) | ~197 MB each, 13 files |
| `data/preprocessed/RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}/graph.pt` | RGCN DBLP bundles (Area on Paper / Conference / Author; universal = all) | `scripts/RGCN/preprocess_dblp.sh` (see `models/RGCN/README.md`) | ~28–29 MB each, 4 files |
| `data/dataset_variant_3hops_filtered/unchanged/{info,node,link,label}.dat` | Freebase base variant (HGB format) | `dataset_construction/` | link.dat 20 MB |
| `data/dataset_variant_3hops_filtered/exact_2/{info,node,link,label}.dat`, `new_transitive_relations.jsonl` | 2-hop shortcut variant + its shortcut registry | `dataset_construction/` | link.dat 886 MB |
| `data/dataset_variant_3hops_filtered/exact_3/{info,node,link,label}.dat`, `new_transitive_relations.jsonl` | 3-hop shortcut variant + registry | `dataset_construction/` | link.dat 8.5 GB |
| `data/dataset_variant_3hops_filtered/union_exact_2_3/{info,node,link,label}.dat` | Freebase universal (union) graph | `dataset_construction/` | link.dat 9.3 GB |

Built here (in this order; CPU only):

| path | command |
|---|---|
| `data/preprocessed/imdb_magnn_v2/<task>/{invariant,invariant_union,stock}/<variant>/` (34 bundles, ~0.2 GB each — the canonical feature table is stored per bundle) | `scripts/MAGNN/preprocess_imdb.sh` |
| `data/preprocessed/dblp_magnn_v2/{invariant,invariant_union,stock}/<variant>/` (10 bundles) | `scripts/MAGNN/preprocess_dblp.sh` |
| `data/preprocessed/freebase_magnn_v2/{invariant,invariant_union,stock}/<variant>/` (10 bundles, ≤ ~0.3 GB each; large-memory node for exact_3 / union) | `scripts/MAGNN/preprocess_freebase.sh` |

Every bundle's `summary.json` carries `program_hash` (instance tables), `physical_hash`,
`supervision_hash` (and `feature_hash` / `task_data_hash`). The run scripts' data-level gauntlet checks
that the invariant bundles share one program hash and that supervision is identical everywhere.
Expected program hashes (for checking a rebuild; invariant / union flavour): IMDb NC invariant `61523a07bf…` ×4, union `d1c4019be5…` ×4; MY `4fae33a1cf…` /
`d571cfbafb…`; MD `8f8f700ba7…` / `8698995733…`; DBLP `2e336fd55421…` / `3c5301969657…`; Freebase
`9fa90f3c62f7…` / `d694b5239c5e…`.

## Running

Environment: Python 3.11, PyTorch 2.4.0+cu124, DGL 2.4.0 (`envs/`). Optional `ENV_ACTIVATE=<path to
activate script>`. Data root: `INV_GNN_DATA` (default `<repo>/data`). Results:
`models/MAGNN/results/<dataset>_<task>/<ARM>/<runner arm>/<DATASET>/<task>/<variant>/seed_<s>/`
(`metrics.json`, `scores.npz`, `trace.json`, `checkpoint.pt`) and `.../aggregate/` (per-arm summary +
pairwise invariance CSV from `common/aggregate_rgcn_structural.py`); augmentation writes
`results/<dataset>_<task>/augmentation/<DATASET>/<task>/<variant>/seed_<s>/`. Seeds default to
`1566911444 20241017 20251017`; every number below is mean ± population std over them.
The run scripts pass `--output-root results/<dataset>_<task>/<ARM>` and the runners append their own arm
name (`originals`, `universal`, `invariant`, `invariant_union`), hence the `<ARM>/<runner arm>/` nesting.
All scripts are skip-if-done and pass `--resume` (per-epoch resume) where the runner supports it.

```bash
ARM=original                         scripts/MAGNN/run_imdb_nc.sh
ARM=universal                        scripts/MAGNN/run_imdb_nc.sh
ARM=augmentation                     scripts/MAGNN/run_imdb_nc.sh
ARM=invariant FLAVOUR=union          scripts/MAGNN/run_imdb_nc.sh
ARM=invariant FLAVOUR=intersection   scripts/MAGNN/run_imdb_nc.sh
```
(identically `run_imdb_movie_year.sh`, `run_imdb_movie_director.sh`, `run_dblp_lp.sh`, `run_freebase_nc.sh`.)

Other environment variables: `SEEDS`, `RESULTS` (results root), `RUN_TIMEOUT` (optional per-run wall
limit, e.g. `4h`; runs resume per epoch when restarted), `EXTRA` (extra runner flags), `GAUNTLET=0`
(skip the data-level hash check). Determinism environment set by the scripts: `DGLBACKEND=pytorch`,
`*_NUM_THREADS=4`, `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Resources: tested on one NVIDIA V100 32 GB with 4 CPU cores and 48 GB RAM. Times below are training
time per run; augmentation is one joint run per seed.

## Expected results (mean ± std over seeds 1566911444, 20241017, 20251017; V100)

### IMDb NC (v1–v4) — micro-F1 (`docs/results/IMDB_nc.tex`)

| arm | command | result dir | expected (micro-F1) | time / GPU |
|---|---|---|---|---|
| original | `ARM=original scripts/MAGNN/run_imdb_nc.sh` | `results/imdb_nc/original/originals/` | v1 .6248±.0073 · v2 .6272±.0065 · v3 .6192±.0015 · v4 .6168±.0176 | ~3 min/run, <0.4 GB |
| universal | `ARM=universal …` | `results/imdb_nc/universal/universal/` | .6320±.0060 | ~4 min |
| augmentation | `ARM=augmentation …` | `results/imdb_nc/augmentation/` | v1 .6248±.0143 · v2 .6100±.0076 · v3 .6248±.0174 · v4 .6172±.0093 (mean .6192±.0118) | ~4 min/seed |
| invariant, union | `ARM=invariant FLAVOUR=union …` | `results/imdb_nc/invariant/invariant_union/` | .6308±.0117 on every variant (τ = 1) | ~4–5 min |
| invariant, intersection | `ARM=invariant FLAVOUR=intersection …` | `results/imdb_nc/invariant/invariant/` | .6248±.0073 on every variant (τ = 1; = original v1) | ~2 min |

### IMDb LP Movie–Year (v1–v4) — MRR (`docs/results/IMDB_lp.tex`)

| arm | command | result dir | expected (MRR) | time / GPU |
|---|---|---|---|---|
| original | `ARM=original scripts/MAGNN/run_imdb_movie_year.sh` | `results/imdb_movie_year/original/originals/` | v1 .1856±.0030 · v2 .1949±.0031 · v3 .1940±.0035 · v4 .1996±.0069 | ~11 min/run, ~2.6 GB |
| universal | `ARM=universal …` | `results/imdb_movie_year/universal/universal/` | .1954±.0070 | ~24 min, 3.1 GB |
| augmentation | `ARM=augmentation …` | `results/imdb_movie_year/augmentation/` | v1 .1933±.0027 · v2 .1997±.0063 · v3 .1973±.0036 · v4 .1969±.0033 (mean .1968±.0032) | ~17 min/seed (25 super-epochs, patience 3) |
| invariant, union | `ARM=invariant FLAVOUR=union …` | `results/imdb_movie_year/invariant/invariant_union/` | .1954±.0070 on every variant (= universal) | ~24 min |
| invariant, intersection | `ARM=invariant FLAVOUR=intersection …` | `results/imdb_movie_year/invariant/invariant/` | .1856±.0030 on every variant | ~10 min |

### IMDb LP Movie–Director (v1, v3) — MRR (`docs/results/IMDB_lp.tex`)

| arm | command | result dir | expected (MRR) | time / GPU |
|---|---|---|---|---|
| original | `ARM=original scripts/MAGNN/run_imdb_movie_director.sh` | `results/imdb_movie_director/original/originals/` | v1 .2846±.0055 · v3 .2852±.0037 | ~15 min/run, ~2.7 GB |
| universal | `ARM=universal …` | `results/imdb_movie_director/universal/universal/` | .2922±.0055 | ~23 min |
| augmentation | `ARM=augmentation …` | `results/imdb_movie_director/augmentation/` | v1 .2850±.0043 · v3 .2847±.0042 (mean .2849±.0042) | ~14 min/seed (50 super-epochs, patience 5) |
| invariant, union | `ARM=invariant FLAVOUR=union …` | `results/imdb_movie_director/invariant/invariant_union/` | .2845±.0045 on both variants | ~17 min, 4.6 GB |
| invariant, intersection | `ARM=invariant FLAVOUR=intersection …` | `results/imdb_movie_director/invariant/invariant/` | .2811±.0120 on both variants | ~11 min |

### DBLP LP Paper–Conference (v1–v3) — MRR over all 20 conferences (`docs/results/DBLP_lp.tex`)

| arm | command | result dir | expected (MRR) | time / GPU |
|---|---|---|---|---|
| original | `ARM=original scripts/MAGNN/run_dblp_lp.sh` | `results/dblp_lp/original/originals/` | v1 .3181±.0048 · v2 .3138±.0071 · v3 .4237±.0929 | ~1.5 h/run (≤ 3.3 h), ~6.8 GB |
| universal | `ARM=universal …` | `results/dblp_lp/universal/universal/` | .4014±.0631 | ~1.8 h |
| augmentation | `ARM=augmentation …` | `results/dblp_lp/augmentation/` | v1 .2868±.0265 · v2 .2868±.0265 · v3 .2869±.0264 (mean .2869±.0264) | ~1.6 h/seed (33 super-epochs, patience 3) |
| invariant, union | `ARM=invariant FLAVOUR=union …` | `results/dblp_lp/invariant/invariant_union/` | .4014±.0631 on every variant (= universal) | ~1.9 h |
| invariant, intersection | `ARM=invariant FLAVOUR=intersection …` | `results/dblp_lp/invariant/invariant/` | .3181±.0048 on every variant (= original v1) | ~1.6 h, 5.2 GB |

### Freebase NC (unchanged / exact_2 / exact_3 = Freebase1–3) — micro-F1 (`docs/results/Freebase_nc.tex`)

| arm | command | result dir | expected (micro-F1) | time / GPU |
|---|---|---|---|---|
| original | `ARM=original scripts/MAGNN/run_freebase_nc.sh` | `results/freebase_nc/original/originals/` | .6135±.0103 · .6059±.0037 · .6036±.0008 | ~21 min/run, <0.6 GB |
| universal | `ARM=universal …` | `results/freebase_nc/universal/universal/` | .6021±.0062 | ~29 min |
| augmentation | `ARM=augmentation …` | `results/freebase_nc/augmentation/` | .6107±.0028 · .6103±.0019 · .6099±.0031 (mean .6103±.0026) | ~45 min/seed, no resume |
| invariant, union | `ARM=invariant FLAVOUR=union …` | `results/freebase_nc/invariant/invariant_union/` | .6107±.0093 on every variant | ~1.5 h, 0.8 GB |
| invariant, intersection | `ARM=invariant FLAVOUR=intersection …` | `results/freebase_nc/invariant/invariant/` | .6021±.0062 on every variant (= universal) | ~29 min |

Note: `run_freebase_magnn_nc_augmentation.py` writes `contract.features = "sparse identity per type"` /
`feature_hash = "identity"` into `metrics.json` regardless of dataset. On IMDb it actually loads the bundle's
canonical pooled-BoW tables (`features_identity` uses `features_*.npy` when present), so IMDb NC augmentation
uses the same canonical features as every other arm; only the recorded metadata is misleading.
