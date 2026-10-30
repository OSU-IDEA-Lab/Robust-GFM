# SeHGNN — node classification (IMDb year graph, Freebase)

SeHGNN pre-computes, **before training**, one feature channel per metapath rooted at the target type
(the nested mean `rownorm(A_01)·rownorm(A_12)⋯X_k`; on Freebase a sparse selector over learned per-type
embeddings), projects every channel, fuses them with a transformer over the channel axis and classifies with
an MLP. The model is the original SeHGNN in a mask-aware copy (`inv_sehgnn/model.py`; with an
all-one mask it is the stock forward — only the augmentation arm uses masks). Only the *inputs* differ
between the arms. All arms use SeHGNN's native **full-K** channel set ("full-K, relation-aware" in
`docs/results`).

## How the invariant version works

See also `docs/INVARIANT_MECHANISMS.md` §6 and `docs/THEORY.md` §2.4.

* The **invariant** arm computes the channels on the variant's **compiled semantic relations** instead of its
  physical edges (`preprocess_sehgnn_channels.py --graph semantic`). Every channel is a product of
  relation-step matrices along the variant's realisation of the same semantic path, so every channel tensor —
  and therefore training — is bit-identical across variants (τ = 1, spread 0; checked by
  `validate_sehgnn_invariance.py` and by equal checkpoint hashes).
* IMDb (channel set `full`, K = 4): the semantic program is read from the RGCN structural bundles; **skip types**
  (Link) are context only — never a step, never a feature source — so the invariant channel set is the stock
  type-path bias over {M, D, A, Y}: 25 channels + label channels (label channels never contain the node's own
  label). The other arms use stock SeHGNN's type-path enumeration on their physical graph (on IMDb every typed
  pair carries exactly one relation, so type paths = relation paths).
* Freebase (channel set `full_rel`, K = 2): the semantic program is the family catalog (base relations ∪
  registered shortcuts of unchanged / exact_2 / exact_3), realised per variant by MAGNN v2's `endpoint_pairs`
  (direct edges if materialised, else the binarised composition). Channels are **relation-aware** (every
  relation its own step) and capped at 500 seeded walks per (target, channel). The semantic program has 597
  channels; the invariant arm trains on the **511** of them that occur in at least one variant's own stock bias
  (`run_sehgnn_nc_subset.py --subset union_of_variants`, result dir `invariant_full_rel_uov511`).

## Requirements

Python 3.11, PyTorch 2.4.0 (+cu124), DGL 2.4.0 (only for `tests/` and `utils/`), scikit-learn, scipy, pandas,
and **`torch_sparse`** (PyG extra; the Freebase channel selectors use it — it is the bitwise-deterministic backend,
`torch.sparse.mm` is not). Shared modules: `<repo>/common/{inv_rgcn/, rgcn_resource_metrics.py,
freebase_block_common.py, aggregate_rgcn_structural.py}`.

**SeHGNN still depends on `models/MAGNN` (code only, no MAGNN data):** `inv_sehgnn/channels.py`,
`preprocess_sehgnn_channels.py` and `tests/test_inv_sehgnn.py` import `magnn_compile_common.py`
(`binarize`, `pair_hash`, `h64`, `sha`), and the preprocessing imports `preprocess_imdb_magnn_v2.py`
(`TYPES`, `relation_matrices` — the IMDb relation matrices) and `preprocess_freebase_magnn_v2.py`
(`build_catalog`, `Variant`, `endpoint_pairs`, `relation_types`, … — the Freebase semantic catalog and its
per-variant realisation). These imports are module-level and used by the `full` / `full_rel` paths.

## Data under `data/`

| path | kind | produced by |
|---|---|---|
| `data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc/{v1,v2,v3,v4,universal}/graph.pt` (≈ 196 MB each) | IMDb year graph: physical + semantic relations, canonical pooled-BoW features, labels, split | **built** by `scripts/RGCN/preprocess_imdb.sh` (see `models/RGCN/README.md`) — or downloaded |
| `data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/{info.dat,node.dat,link.dat,label.dat}` + `new_transitive_relations.jsonl` (exact_2, exact_3, union) | Freebase variants (link.dat 20 MB / 0.89 GB / 8.5 GB / 9.3 GB) | **built** by `dataset_construction/` (Freebase) — or downloaded |
| `data/preprocessed/sehgnn_v2/IMDB/full/semantic/{v1,v2,v3,v4}/`, `data/preprocessed/sehgnn_v2/IMDB/full/physical/{v1,v2,v3,v4,universal}/` (≈ 0.24–0.28 GB each, 9 bundles) | SeHGNN channel bundles (`channels/*.npz`, `manifest.json`, features, labels, split) | **built here**: `scripts/SeHGNN/preprocess_imdb.sh` |
| `data/preprocessed/sehgnn_v2/FREEBASE/full_rel/semantic/{unchanged,exact_2,exact_3}/`, `data/preprocessed/sehgnn_v2/FREEBASE/full_rel/physical/{unchanged,exact_2,exact_3,union_exact_2_3}/` (7 bundles) | SeHGNN channel bundles | **built here**: `scripts/SeHGNN/preprocess_freebase.sh` |

Freebase split = RGCN's `make_split` (seed 1566911444), recomputed inside the preprocessing; no MAGNN-built data
is read.

## Preprocessing (in order)

1. IMDb RGCN bundles (`scripts/RGCN/preprocess_imdb.sh`) and Freebase variants (`dataset_construction/`).
2. `bash scripts/SeHGNN/preprocess_imdb.sh` (CPU, minutes). Ends with the data-level gauntlet.
3. `bash scripts/SeHGNN/preprocess_freebase.sh` (CPU, ~10–40 min and up to 128 GB RAM per variant; split with
   `VARIANTS=` / `GRAPHS=`). Ends with the data-level gauntlet.

Expected channel-program hashes, for checking a rebuild (first 12 hex of `manifest.json:program_hash`; identical across variants for every
semantic bundle): IMDb semantic `full` `8e48cd229f7e` (25 ch); IMDb physical `full` v1–v4 / universal
`3ec5f6be06a6` `c58c7bfdb22b` `c5bb2a24e1ec` `806801256bd0` / `908b4a627e8c`. Freebase semantic `full_rel`
`3681e462e6e4` (597) = physical `union_exact_2_3`; physical `full_rel` unchanged / exact_2 / exact_3
`b5ee053b6983` (91) `491e5628610c` (159) `27b950f487a3` (443) — the semantic channels whose keys occur in one
of these three bundles form the invariant arm's 511-channel set. Supervision hashes: IMDb `c331ab896883`, Freebase `d9e6b4c1cc66`.

## Runs

Run from the repository root; `ARM` is required, seeds default to `1566911444 20241017 20251017`. Results land in
`models/SeHGNN/results/<DATASET>_nc/<armdir>_<channels>/<DATASET>/nc/<variant>/seed_<s>/` where `<armdir>` is the
runner's internal arm name (`original` → `originals`). Expected results are **test micro-F1, mean ± population std over seeds 1566911444, 20241017, 20251017**
(`docs/results/IMDB_nc.tex`, `docs/results/Freebase_nc.tex`; one NVIDIA V100 32 GB, 4 CPU cores, 48–64 GB RAM). The invariant arm is
identical on every variant, so one number is given.

### IMDb NC — `scripts/SeHGNN/run_imdb_nc.sh` (channel set `full`, K = 4)

| arm | command | result dir | expected micro-F1 (IMDb1 / 2 / 3 / 4) | GPU / time per run |
|---|---|---|---|---|
| original | `ARM=original bash scripts/SeHGNN/run_imdb_nc.sh` | `IMDB_nc/originals_full` | .6336±.0039 / .6388±.0157 / .6356±.0187 / .6455±.0189 | 7 GB, ~1.5 min |
| universal | `ARM=universal bash scripts/SeHGNN/run_imdb_nc.sh` | `IMDB_nc/universal_full` | .6431±.0039 (union graph) | 16 GB, ~3.5 min |
| augmentation | `ARM=augmentation bash scripts/SeHGNN/run_imdb_nc.sh` | `IMDB_nc/augmentation_full` | .6256±.0134 / .6308±.0113 / .6256±.0147 / .6284±.0123 (mean .6276) | 13 GB, ~3.5 min / seed (joint) |
| **invariant** | `ARM=invariant bash scripts/SeHGNN/run_imdb_nc.sh` | `IMDB_nc/invariant_full` | **.6535±.0064** (25 ch, τ = 1) | 4.5 GB, ~1 min |

### Freebase NC — `scripts/SeHGNN/run_freebase_nc.sh` (relation-aware full-K, K = 2, `--batch-size 1024`)

| arm | command | result dir | expected micro-F1 (Freebase1 / 2 / 3) | GPU / time per run |
|---|---|---|---|---|
| original | `ARM=original bash scripts/SeHGNN/run_freebase_nc.sh` | `FREEBASE_nc/originals_full_rel` | .6681±.0000 / .6759±.0063 / .6671±.0103 | 11 GB, ~3 min |
| universal | `ARM=universal bash scripts/SeHGNN/run_freebase_nc.sh` | `FREEBASE_nc/universal_full_rel` | .6587±.0041 (597 ch, union graph) | 16 GB, ~6 min |
| augmentation | `ARM=augmentation bash scripts/SeHGNN/run_freebase_nc.sh` | `FREEBASE_nc/augmentation_full_rel` | .6537±.0147 / .6541±.0192 / .6537±.0099 (mean .6538) | 17 GB, ~5 min (joint) |
| **invariant** | `ARM=invariant bash scripts/SeHGNN/run_freebase_nc.sh` | `FREEBASE_nc/invariant_full_rel_uov511` | **.6637±.0080** (511 ch, τ = 1) | 13 GB, ~5 min |

Aggregates / Kendall τ: add `AGGREGATE=1` to any run command, or run
`python models/SeHGNN/aggregate_sehgnn_v2.py --cell IMDB_nc --results-root models/SeHGNN/results`
(likewise `FREEBASE_nc`). Unit tests: `cd models/SeHGNN && python -m unittest tests.test_inv_sehgnn`
(the compiler reproduces stock SeHGNN's DGL propagation key-for-key and value-for-value).

Other environment variables of the run scripts: `SEEDS`, `OUT` (results root, default
`models/SeHGNN/results`), `EXTRA` (extra runner flags), `ENV_ACTIVATE` (command that activates the
environment). Restarted runs resume per epoch (`--resume`).

## Hyper-parameters

Recorded in every run's `metrics.json:config`: IMDb embed/hidden 512, ≤ 200 epochs, patience 50, batch
10 000, per-type input projection; Freebase embed/hidden 256, ≤ 100 epochs, patience 30, batch 1024; Adam 5e-3 /
wd 1e-4, dropout .5, 2 projection + 4 task layers, selection and early stopping on validation micro-F1;
`torch_sparse` backend; env `DGLBACKEND=pytorch PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`, 4 BLAS threads.
