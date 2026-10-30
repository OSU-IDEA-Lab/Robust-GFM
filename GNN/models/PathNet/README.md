# PathNet — original / universal / augmentation / invariant (node classification)

PathNet (Sun et al., *Beyond Homophily: Structure-aware Path Aggregation Graph Neural Network*,
IJCAI 2022, <https://github.com/zjunet/PathNet>) samples maximal-entropy random walks (MERW)
from every target node and encodes each walked node by its distance to the target. An LSTM
reads each walk, the walks are aggregated, and the result is concatenated with the target's
own (ego) representation. Upstream has two variants: `PathNet` for heterophilous graphs and
`PathNet_homo` for homophilous ones. IMDb and Freebase are homophilous (neighbouring labelled
nodes tend to share a label), so both use `PathNet_homo` with upstream's training protocol:
a fresh walk set every epoch and 1000 epochs. Cells: **IMDb NC** and **Freebase NC**, all four arms.

## Code and licence

`pathnet/` is a PyTorch port of upstream PathNet (MIT licence, see `pathnet/LICENSE`). Only its
MERW sampler (`pathnet/walks.py`) is used here. `pathnet_nc.py` holds both models, written line
for line after upstream: `PathNetHomoNC` (`PathNet_homo`: ReLU after `fc0` and the
distance-specific layers, a `(1 + attention)` gate on each walk, mean over walks) and `PathNetNC`
(`PathNet`: softmax attention over walks). The remaining files are the experiment harness:

| file | role |
|---|---|
| `pathnet_nc.py` | `PathNetHomoNC` (used here) and `PathNetNC` |
| `fast_walks.py` | per-epoch MERW sampler with exact BFS distance buckets (IMDb) |
| `fast_walks_targets.py`, `freebase_pairs.py` | the same sampler for Freebase's hub-heavy graphs, with a cached exact 2-hop bitset |
| `walk_dist.py` | exact BFS distance-to-target buckets (upstream `gen_merw.cpp --dist bfs`) |
| `run_pathnet_imdb_nc.py` | IMDb NC runner |
| `build_freebase_walks.py`, `freebase_samples.py` | Freebase walk-set builder and store loader |
| `run_pathnet_freebase_nc.py` | Freebase NC runner |
| `summarize_arms.py`, `invariance_stat.py` | per-arm means, pairwise Kendall τ, identical-checkpoint check |

`freebase_samples.py` and `freebase_pairs.py` import `freebase_conditional_rgcn` /
`freebase_block_common` from `models/RGCN` and `common/`; the IMDb runner uses `inv_rgcn` from `common/`.

## How the invariant PathNet works

PathNet's structural object is its set of **walks** (40 walks × length 4 per target) with
their distance buckets. The invariant arm applies the τ-invariant mapping to exactly this
object and keeps architecture, loss and protocol unchanged: it runs **the same sampler on each
variant's compiled semantic program** (the program RGCN, MAGNN, SeHGNN and CMPNN use) instead
of on the variant's physical graph. The programs are hash-identical across variants. The
per-epoch walks are drawn with an RNG keyed on (seed, epoch), so every variant gets the same walks
in every epoch. Training is then bit-identical: Kendall τ = 1 with identical checkpoints.
The model code is the same in all four arms.

| dataset | walk graph per arm |
|---|---|
| IMDb | undirected 0/1 adjacency of the arm's edges from the RGCN year bundles `RGCN_structural_v10_year_canonfeat/IMDB/nc/<v>`: physical v1–v4 (original), universal physical (universal), the compiled semantic program (invariant; identical on v1–v4) |
| Freebase | RGCN's Freebase stores: `freebase_rgcn_physical/<V>` (original), `.../union_exact_2_3` (universal), `freebase_rgcn_block_conditional/<V>/exact_3` = the family program, 64 relations (invariant; pair hash `8ecdb0bdd522` on all three variants) |

Distances are exact BFS distances from the target, as in upstream's `gen_merw.cpp`
(`--dist bfs`).

Settings (upstream `PathNet_run.py` defaults, the same in every arm and on both datasets):
`PathNet_homo`, hidden 64, 40 MERW walks of length 4 resampled every epoch, Adam lr 5e-3,
dropout .7, weight decay 5e-4, 1000 epochs without early stopping, best epoch selected on
validation micro-F1. IMDb uses the canonical 3,008-d BoW features. Freebase has no node features,
so it uses a fixed seeded random vector per node plus a learned node-type embedding
(`--input rand_type`). Augmentation runs 1000 / (number of variants) super-epochs, one step per
variant each. The scripts pass these settings; the runners' CLI defaults differ. See also
`docs/EXPERIMENT_SETTINGS.md`.

## Data inputs (under `data/`; `models/PathNet/data -> ../../data`)

| path | built by |
|---|---|
| `preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc/{v1,v2,v3,v4,universal}/graph.pt` | `bash scripts/RGCN/preprocess_imdb.sh` (shared prerequisite) |
| `preprocessed/freebase_rgcn_physical/<V>/physical_<V>/union/all_graph` (4 graphs) and `preprocessed/freebase_rgcn_block_conditional/{unchanged,exact_2,exact_3}/exact_3/union/all_graph` | `bash scripts/RGCN/preprocess_freebase.sh` (shared prerequisite) |
| `preprocessed/pathnet_inv/freebase_walks/<graph>_<variant>_bfs/` (7 dirs, ~59 MB each) | `bash scripts/PathNet/preprocess_freebase.sh` (CPU; 0.5–9 min each, up to ~50 GB RAM) |
| `preprocessed/pathnet_inv/freebase_twohop_bits/<graph>_<variant>_<hash>.npz` (5 files, 171 MiB each) | built by the first Freebase run on each graph (up to ~30 min); also in tier B |

IMDb walks are sampled inside the runner, so IMDb has no PathNet preprocessing.

## Commands

```bash
bash scripts/PathNet/preprocess_freebase.sh                                # once (after the RGCN Freebase stores)
ARM=original|universal|augmentation|invariant bash scripts/PathNet/run_imdb_nc.sh
ARM=original|universal|augmentation|invariant bash scripts/PathNet/run_freebase_nc.sh
cd models/PathNet && python summarize_arms.py results/imdb_nc IMDB/nc        # per-arm means, pairwise τ, identical ckpts
cd models/PathNet && python summarize_arms.py results/freebase_nc FREEBASE/nc
```

Optional environment variables: `SEEDS` (default `1566911444 20241017 20251017`), `OUT`
(results root, relative to `models/PathNet`), `ENV_ACTIVATE` (path to an environment activate script).

Results: `models/PathNet/results/{imdb_nc,freebase_nc}/<arm>/<DATASET>/nc/<variant>/seed_<s>/`
(`metrics.json`, `scores.npz`, `history.json`). The original arm's folder is named
`originals` (the runners' internal name).

Resources: tested on one NVIDIA V100 32 GB. Each IMDb run takes about 6 minutes (peak GPU
memory 6.6 GiB). Each Freebase run takes 25–45 minutes once the 2-hop bitsets exist (peak GPU
memory 12.2 GiB, ~70 GB host RAM). Software: the main environment (`envs/gnn.txt`); PathNet
needs no DGL.

## Expected results (test micro-F1, mean ± std over seeds 1566911444, 20241017, 20251017; V100)

| cell | original | universal | augmentation | invariant |
|---|---|---|---|---|
| IMDb NC | v1 .6009 ± .0030, v2 .5993 ± .0043, v3 .5957 ± .0078, v4 .6005 ± .0049; mean .5991 ± .0057, τ = .8579 | .5929 ± .0118 | .6007 ± .0117 (τ = .9958) | .5997 ± .0078 on every variant (τ = 1, 18/18 identical checkpoints) |
| Freebase NC | TODO | TODO | TODO | TODO |

## Verifying invariance

On the invariant arm every variant of a seed records the same `walk_hash`,
`supervision_hash`, `init_hash` and `checkpoint_hash` in `metrics.json` (IMDb also
`feature_hash`), while the physical hashes differ. `init_hash` / `checkpoint_hash` depend on
the device (CPU vs GPU), so compare them within one device.
