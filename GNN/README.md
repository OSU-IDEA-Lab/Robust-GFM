# Information-based Invariance for Graph Neural Networks

Code for *Information-based Invariance, Equivariance, and Expressivity for Graph Neural
Networks*.

The same information can be stored in graphs that are not isomorphic. For example, a movie's
director can be attached to the movie node or to a copy of it, and a two-hop path can also
appear as a single shortcut edge. A standard GNN gives different answers on such
*information-equivalent* variants. This repository makes seven existing architectures invariant
to these changes without altering the architectures themselves. Instead, each model runs on a
*compiled semantic program*: the variant's graph rewritten into a form that is identical for
every equivalent variant.

**Models:** RGCN, MAGNN, SlotGAT, SeHGNN, CMPNN (NBFNet / C-MPNN), PathNet, PAIN.
**Datasets:** IMDb, DBLP, WN18RR and Freebase, each in several equivalent variants.

## Experimental arms

Every experiment compares four ways of training the same model:

| arm | what is trained |
|---|---|
| `original` | the unmodified model, one run per variant on that variant's own graph |
| `universal` | the unmodified model, one run on the union graph of all variants |
| `augmentation` | the unmodified model, one run trained jointly on all variants' graphs and evaluated on each |
| `invariant` | our method: one run per variant on its compiled semantic program |

All four arms of an experiment share the architecture, loss, optimiser, training budget,
seeds, splits, features and evaluation; only the structural input differs. The invariant arm
gives **bit-identical** models and predictions across variants (Kendall τ = 1) on a given GPU.

`docs/INVARIANT_MECHANISMS.md` describes how the invariant input is built for each model.
`docs/THEORY.md` connects that construction to the theory.

## Repository layout

```
envs/                   Python requirements: gnn.txt (all models except CMPNN and PAIN), cmpnn.txt, pain.txt
data/                   empty; download the data here (see data/README.md)
dataset_construction/   builds the dataset variants and shared splits from the public raw data
common/                 modules shared by several models
models/<MODEL>/         model code; each has a README with details and expected results
scripts/<MODEL>/        entry points: preprocess_<dataset>.sh and run_<dataset>_<task>.sh
docs/                   theory, per-model mechanisms, dataset statistics, experiment settings
docs/results/           result tables (LaTeX)
```

Each `models/<MODEL>/` folder contains a symlink `data -> ../../data`. To keep the data
somewhere else, replace `data/` with a symlink or set `INV_GNN_DATA=/path/to/data`.

## Installation

```bash
# RGCN, MAGNN, SlotGAT, SeHGNN, PathNet (Python 3.11, CUDA 12.4)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r envs/gnn.txt
export DGLBACKEND=pytorch
```

CMPNN needs its own environment (Python 3.10, PyTorch 2.0.1, torchdrug 0.2.1). Its setup
steps are at the top of `envs/cmpnn.txt`. PAIN also needs its own environment (Python 3.11,
PyTorch 2.13.0+cu126; `envs/pain.txt`), because its deterministic segment sum relies on a CUDA
`cumsum` that PyTorch 2.4 cannot run deterministically.

The scripts are plain bash. Set `ENV_ACTIVATE=/path/to/env/bin/activate` to have a script
activate an environment itself, which is useful on a cluster:

```bash
sbatch --gres=gpu:1 --wrap "ARM=invariant bash scripts/RGCN/run_imdb_nc.sh"
```

Each script's header lists the resources it needs.

## Data

Download the archives linked in `data/README.md` and unpack them into `data/`.

* **Tier A (required):** raw IMDb data, the WN18RR and Freebase variant graphs, and the
  shared splits. DBLP must be downloaded separately because of its licence; see
  `data/README.md`.
* **Tier B (optional):** preprocessed model inputs, which let you skip the preprocessing step.

`dataset_construction/` rebuilds every tier-A file from the public sources, and its README
gives sha256 checksums.

## Preprocessing

Skip anything you downloaded as tier B. Run the RGCN steps first: the RGCN bundles hold the
compiled programs for IMDb, DBLP and WN18RR, which the other models also read, and PathNet
samples its Freebase walks from the RGCN Freebase stores.

```bash
# 1. shared compiled programs (needed by every model on IMDb / DBLP / WN18RR)
bash scripts/RGCN/preprocess_imdb.sh
bash scripts/RGCN/preprocess_dblp.sh
bash scripts/RGCN/preprocess_wn18rr.sh

# 2. model-specific inputs (CMPNN and PathNet on IMDb have none)
bash scripts/MAGNN/preprocess_imdb.sh
bash scripts/MAGNN/preprocess_dblp.sh
bash scripts/SeHGNN/preprocess_imdb.sh
bash scripts/SlotGAT/preprocess_imdb.sh
bash scripts/SlotGAT/preprocess_lp.sh       # checks that the RGCN bundles exist
bash scripts/PAIN/preprocess_imdb.sh         # PAIN environment; reads the raw IMDb csv

# 3. Freebase (CPU, large memory, tens of GB of disk)
bash scripts/RGCN/preprocess_freebase.sh
bash scripts/MAGNN/preprocess_freebase.sh
bash scripts/SeHGNN/preprocess_freebase.sh
bash scripts/SlotGAT/preprocess_freebase.sh
bash scripts/PathNet/preprocess_freebase.sh # needs the RGCN Freebase stores
bash scripts/PAIN/preprocess_freebase.sh    # PAIN environment; reads the raw Freebase variants
```

Every builder records hashes of the physical graph, the compiled program and the supervision.
Across the variants of a dataset, the physical hashes differ while the program and
supervision hashes are identical. The model READMEs list the expected values.

## Running experiments

```bash
ARM=original|universal|augmentation|invariant [SEEDS="1566911444 20241017 20251017"] \
  bash scripts/<MODEL>/run_<dataset>_<task>.sh
```

| dataset / task | metric | RGCN | MAGNN | SlotGAT | SeHGNN | CMPNN | PathNet | PAIN |
|---|---|---|---|---|---|---|---|---|
| IMDb node classification (v1–v4) | micro-F1 | `run_imdb_nc.sh` | `run_imdb_nc.sh` | `run_imdb_nc.sh` | `run_imdb_nc.sh` | — | `run_imdb_nc.sh` | `run_imdb_nc.sh` |
| IMDb link prediction, Movie–Year (v1–v4) | MRR | `run_imdb_movie_year.sh` | `run_imdb_movie_year.sh` | `run_imdb_movie_year.sh` | — | `run_imdb_movie_year.sh` | — | — |
| IMDb link prediction, Movie–Director (v1, v3) | MRR | `run_imdb_movie_director.sh` | `run_imdb_movie_director.sh` | `run_imdb_movie_director.sh` | — | `run_imdb_movie_director.sh` | — | — |
| DBLP link prediction, Paper–Conference (v1–v3) | MRR | `run_dblp_lp.sh` | `run_dblp_lp.sh` | `run_dblp_lp.sh` | — | `run_dblp_lp.sh` | — | — |
| WN18RR link prediction (3 variants) | filtered MRR | `run_wn18rr_lp.sh` | — | `run_wn18rr_lp.sh` | — | `run_wn18rr_lp.sh` | — | — |
| Freebase node classification (3 variants) | micro-F1 | `run_freebase_nc.sh` | `run_freebase_nc.sh` | `run_freebase_nc.sh` | `run_freebase_nc.sh` | — | `run_freebase_nc.sh` | `run_freebase_nc.sh` |

To run everything:

```bash
for M in RGCN MAGNN SlotGAT SeHGNN CMPNN PathNet PAIN; do
  for S in scripts/$M/run_*.sh; do
    for ARM in original universal augmentation invariant; do ARM=$ARM bash "$S"; done
  done
done
```

Model-specific options:

* **MAGNN:** `FLAVOUR=union` (default) or `FLAVOUR=intersection` selects the metapath set of
  the invariant arm.
* **SlotGAT:** the IMDb `original` arm runs both the upstream SlotGAT implementation and our
  deterministic port of it. `IMPL=upstream` or `IMPL=parity` runs only one of them. The other
  arms always use the deterministic port.
* **CMPNN:** `ARM=gauntlet` runs only the data-level invariance checks; it trains nothing.
* **CMPNN, PAIN:** run with their own environments (`ENV_ACTIVATE=...`).

Each model README lists, for every dataset and arm, the command, the output directory, the
expected result and the approximate run time.

### Outputs

Results are written under `models/<MODEL>/results/`, one directory per (variant, seed). Each
holds a `metrics.json` (or `result.json`) with:

* task metrics and the best epoch;
* training and inference time;
* parameter count and peak GPU memory;
* per-epoch state hashes.

With `AGGREGATE=1` (or `--aggregate`), a run also writes a cross-variant summary with the
invariance statistic: the Kendall τ-b between the per-node (node classification) or
per-query (link prediction) score vectors of each pair of variants.

## Reproducibility

* **Determinism.** The runs use `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, sorted edge insertion and segment
  (not atomic) reductions. The invariant arm is bit-identical across variants on a given
  GPU type. The results in `docs/results/` were obtained on NVIDIA V100-SXM3-32GB GPUs; other
  GPUs give close but not bit-identical numbers.
* **Upstream SlotGAT** uses non-deterministic DGL kernels, so reruns vary slightly.
* **Model selection.** Early stopping and checkpoint selection use validation micro-F1
  (node classification) or MRR (link prediction). The augmentation arm selects on the mean
  over variants.
* **Results** are mean ± population standard deviation over seeds 1566911444,
  20241017 and 20251017.

`docs/EXPERIMENT_SETTINGS.md` lists all hyper-parameters. `docs/DATASET_STATISTICS.md` and
`docs/DATASET_CONSTRUCTION_FREEBASE_WORDNET.md` describe the datasets.

## Citation

```bibtex
@inproceedings{TODO,
  title  = {Information-based Invariance, Equivariance, and Expressivity for Graph Neural Networks},
  author = {TODO},
  year   = {TODO}
}
```

## Acknowledgements

The model code builds on the original implementations of RGCN (DGL), MAGNN
(https://github.com/cynricfu/MAGNN), SlotGAT, SeHGNN, NBFNet / C-MPNN (torchdrug), PathNet
(https://github.com/zjunet/PathNet) and PAIN (https://github.com/ocatias/ExpressivePathGNNs; the
node-classification port https://github.com/waiyanzt/PAIN-nc). Please cite those works as well.
