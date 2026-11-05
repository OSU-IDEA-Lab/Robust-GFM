# MOTIF

[MOTIF](https://github.com/HxyScotthuang/MOTIF) (Huang et al., ICML 2025) is a knowledge-graph foundation model for
link prediction. This folder evaluates its released checkpoint `motif_3g` on the lab's IMDb, DBLP, WN18RR and
Freebase variants: zero-shot, and fine-tuned on each variant with the lab's three seeds.

Only our own code is tracked here. The MOTIF library, configs and checkpoints come from the upstream repository at
a pinned commit, with one small patch.

## Setup

```bash
# Upstream MOTIF at the pinned commit, plus our patch (motif/rspmm/__init__.py loads the native
# C++/CUDA extension only when it is called, so the Triton path works without a C++ toolchain)
git clone https://github.com/HxyScotthuang/MOTIF.git upstream
git -C upstream checkout 30b2ff3c64892a14f1715b3e929855fc1abd2c81
git -C upstream apply ../upstream.patch

# Environment (Python 3.10, CUDA 12.1)
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt -r requirements-eval.txt

mkdir -p logs   # Slurm writes job logs here
```

`slurm/env.sh` (sourced by every job) loads the cluster modules, activates `.venv` and sets the compiler and
determinism variables; edit its `module load` line for another cluster.

## Data

See [`rai-datasets/README.md`](rai-datasets/README.md): where to download each dataset and how to build the MOTIF
inputs. The builders are the lab's own, read from `../../GNN/`, so the variants are identical to those of the GNN
experiments.

## Experiments

Run from this folder. Each job writes to `results/` (git-ignored).

| experiment | command | details |
| --- | --- | --- |
| IMDb Movie–Year, zero-shot (v1–v4) | `sbatch --export=ALL,DATASET=imdb,TASK=movie_year,RUN_TAG=full slurm/zeroshot.sbatch` | [zero-shot](docs/ZERO_SHOT_EVALUATION.md) |
| IMDb Movie–Year, fine-tuned | `sbatch slurm/finetune_seeds.sbatch` | [IMDb fine-tuning](docs/IMDB_FINE_TUNING.md) |
| IMDb Movie–Director (v1, v3) | `sbatch --export=ALL,TASK=movie_director,DATA=rai-datasets/preprocessed/IMDB_movie_director slurm/finetune_seeds.sbatch` | [seed tables](docs/SEED_TABLES.md) §3 |
| DBLP Paper–Conference, zero-shot (v1–v3) | `sbatch --export=ALL,DATASET=dblp,TASK=paper_conference,RUN_TAG=full slurm/zeroshot.sbatch` | [zero-shot](docs/ZERO_SHOT_EVALUATION.md) |
| DBLP Paper–Conference, fine-tuned | `sbatch --export=ALL,DATASET=dblp slurm/finetune_seeds.sbatch` | [DBLP fine-tuning](docs/DBLP_FINE_TUNING.md) |
| WN18RR (v1–v3) | `sbatch --array=0-2 --export=ALL,MODE=eval slurm/wordnet.sbatch`<br>`sbatch --array=0-8 --export=ALL,MODE=finetune slurm/wordnet.sbatch` | [seed tables](docs/SEED_TABLES.md) §4 |
| IMDb node classification (v1–v4) | `sbatch --array=0-3 --export=ALL,DATASET=imdb,MODE=eval slurm/nc.sbatch`<br>`sbatch --array=0-11 --export=ALL,DATASET=imdb,MODE=finetune slurm/nc.sbatch` | [node classification](docs/NODE_CLASSIFICATION.md) |
| Freebase node classification (v1–v3) | as IMDb with `DATASET=freebase` and `--array=0-2` / `0-8`; v3 needs `SCRIPT=script/nc_motif_dense.py` | [node classification](docs/NODE_CLASSIFICATION.md) |

Seed-averaged tables: `script/report_seeds.py` ([seed tables](docs/SEED_TABLES.md)). Slurm usage, GPU types and
memory: [Slurm guide](docs/SLURM_GUIDE.md). The `#SBATCH` account and partition lines are for the OSU HPC cluster.

The released checkpoint was pretrained on FB15k-237, WN18RR and CoDEx-Medium, so WN18RR is **not** held out.

## Layout

```
script/          our entry points: prepare_*_lp.py (data), eval_lp.py (zero-shot), finetune_*_lp.py,
                 run_finetune_matrix.py, wordnet_lp.py, nc_motif.py (node classification), report_*.py
slurm/           job scripts; env.sh is the shared job environment
tests/           data-contract and metric tests: python -m unittest discover -s tests
docs/            protocol and command details for each experiment
rai-datasets/    data (git-ignored except README.md and SHA256SUMS)
upstream.patch   our one change to upstream MOTIF
upstream/        upstream MOTIF checkout (git-ignored; see Setup)
```

## Citation

```bibtex
@inproceedings{huang2025how,
  title     = {How Expressive are Knowledge Graph Foundation Models?},
  author    = {Xingyue Huang and Pablo Barcel{\'o} and Michael M. Bronstein and {\.{I}}smail {\.{I}}lkan Ceylan and Mikhail Galkin and Juan L. Reutter and Miguel Romero Orth},
  booktitle = {Proceedings of the Forty-second International Conference on Machine Learning},
  year      = {2025}
}
```
