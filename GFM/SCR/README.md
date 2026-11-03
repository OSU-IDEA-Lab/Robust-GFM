# SCR on the original DBLP and IMDb graphs

SCR (Semantic-Conditioned Reasoning) is the graph foundation model from
[arXiv:2410.12609](https://arxiv.org/abs/2410.12609). The released `scr_3g`
checkpoint was pretrained on FB15k-237, WN18RR, and CoDEx-M. IMDb and DBLP are
not in that set.

This folder evaluates that checkpoint on the **original** physical variants
only, in two settings:

- **Zero-shot.** Weights stay frozen.
- **Fine-tune.** One checkpoint is trained per variant, starting from `scr_3g`,
  and tested on that same variant.

Invariant (compiled) graphs are not exported or trained here.

| task | graphs | query |
| --- | --- | --- |
| DBLP paper–venue | `v1`, `v2`, `v3` | `(paper, published_in, venue)` over 20 venues |
| IMDb node classification | `v1`–`v4` | `(movie, has_genre, genre)` over 3 genres |
| IMDb movie–director | `v1`, `v3` | `(movie, movie_to_director, director)` |
| IMDb movie–year | `v1`–`v4` | `(movie, movie_has_year, year)` over the observed years |

On IMDb, `v1`–`v4` move the director and actor edges between the movie and its
link node. On DBLP, `v1`–`v3` move the research-area edges among paper, venue,
and author. Held-out target edges are not written into the message graph.

## Setup

```bash
bash setup_upstream.sh          # clones https://github.com/KyneWang/SCR at 5e46b54
pip install -r requirements.txt # install a CUDA torch build first
```

`upstream/` is the untouched SCR checkout (the `ultra` package, configs, and
`ckpts/scr_3g.pth`). If `upstream/resource/` is missing, download the SCR
resource bundle named in that repository's README. The runners read relation
text from it.

Preprocessed GNN bundles are not stored in this repository. Point the builders
at the MAGNN preprocessed directory that contains `DBLP_lp_pc_var*_train_pc`,
`IMDB_preprocessed_star`, `IMDB_rgcn_v*`, and `IMDB_rgcn_lp_md_v*`.

## Run

```bash
export MAGNN_ROOT=/path/to/MAGNN/data/preprocessed
export IMDB_CSV=/path/to/movie_metadata.csv
export IMDB_ML_SPLIT=/path/to/IMDB_ml_shared_splits.npz
bash scripts/run_original.sh
```

The script is resumable: a variant/seed is skipped when `metrics.json` already
exists. Run one dataset at a time. Two jobs on the same dataset both write
`kg-datasets/.../processed/` and will collide.

Outputs:

```
results/zs/scr_3g/<dataset>/seed<seed>/{scores.npz,metrics.json}
results/ft/scr_3g/<dataset>/seed<seed>/{scores.npz,metrics.json}
```

Node classification uses seeds `1024`, `42`, `1337`, five epochs, and selects
the checkpoint by validation accuracy. Link prediction uses seeds
`1566911444`, `20241017`, `20251017`. DBLP fine-tune is one epoch. IMDb link
fine-tune is five epochs. Both select by validation MRR.

Zero-shot does not train, so train time is 0 and there is no training GPU
peak. Fine-tune records train time, steps, parameter size, and peak GPU memory
in `metrics.json`.
