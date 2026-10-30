# PAIN — original / universal / augmentation / invariant (node classification)

PAIN (PAth Isomorphism Network; Graziani et al., *The Expressive Power of Path-Based Graph
Neural Networks*, ICML 2024, <https://github.com/ocatias/ExpressivePathGNNs>) represents each
node by the set of simple paths rooted at it. Every rooted path is turned into a sequence of
(node feature, positional encoding, edge-type embedding, neighbour mark) and read by an LSTM.
The path encodings are summed at the root, and a residual MLP layer with jumping knowledge
follows. Here the graph readout is replaced by a node-classification head. Cells:
**IMDb NC** and **Freebase NC**, all four arms.

## Code

The model, the data loader and the preprocessors are adapted from the node-classification port
PAIN-nc (<https://github.com/waiyanzt/PAIN-nc>, commit `59eaa40`), which follows upstream's
`Models/path_gnn.py` / `path_conv.py`. Only the files these experiments need are included.

| file | role |
|---|---|
| `pain_nc/model.py` | `PainNodeClassifier`: path encoder (packed LSTM over each rooted path, deterministic segment sum at the root), residual PAIN layers, node head |
| `pain_nc/data.py`, `pain_nc/config.py`, `pain_nc/paths.py` | artifact loader, YAML configs, data locations (`INV_GNN_DATA`, `PAIN_PREPROCESSED_ROOT`) |
| `preprocessing/imdb_node_classification.py` | IMDb year graph, canonical features, all rooted simple paths with ≤ 3 edges per variant and for the union graph |
| `preprocessing/imdb_invariant.py`, `build_imdb_invariant_minimal.py` | IMDb invariant artifacts (see below) |
| `preprocessing/freebase_node_classification.py`, `preprocessing/freebase_invariant.py` | Freebase sampled paths per variant / union graph, and the invariant artifacts |
| `run_pain_nc.py` | four-arm runner for both datasets |
| `pain_memfix.py` | memory-only override of the path aggregator (see below) |
| `experiments/node_classification/` | `build_model`, `set_seed` and the artifact checks used by the runner |
| `configs/{imdb,freebase}_nc.yaml` | model and optimiser settings (upstream's ZINC configuration) |
| `summarize_arms.py`, `invariance_stat.py` | per-arm means, pairwise Kendall τ, identical-checkpoint check |

`pain_memfix.py` changes only where tensors live: the checkpointed path chunks are moved to the
GPU inside the checkpointed function, so the saved inputs stay in host memory and the IMDb path
programs fit on a 32 GB V100. Operations, dtypes, chunk boundaries and order are unchanged.

## How the invariant PAIN works

PAIN's structural object is the **set of rooted simple paths** with up to L = 3 edges. The
invariant arm applies the τ-invariant mapping to exactly this object and keeps architecture, loss
and protocol unchanged. Each variant's own graph is compiled into a semantic program, and the same
path enumeration runs on that program instead of on the physical graph. The programs, and
therefore the path programs, are byte-identical across variants. Training is then bit-identical:
Kendall τ = 1 with identical checkpoints.

| dataset | invariant program | check |
|---|---|---|
| IMDb | **semantic program**: the variants move the director and the actor between the movie and its Link node; each variant's graph is compiled so that these relations are carried by conditional edges, giving the same program for every variant. Built by `build_imdb_invariant_minimal.py`. | 4 distinct physical hashes → 1 semantic graph `9825cee1bec6` → 1 path program `f0c474c9df47`; 11,179,673 paths |
| Freebase | **family program**: the 36 base relations plus the 28 2- and 3-hop compositions, each rebuilt from the variant's own base relations. Paths are sampled (4 relations, fan-out 8, cap 256 per root) with a sampler keyed on semantic identity. Built by `preprocessing.freebase_invariant`. | 3 distinct physical hashes → 1 semantic graph `ec58fb18c88d` → 1 path program `4f27282d6b2f`; 42,038,832 paths |

The runner checks the invariant artifacts before training: status PASS, distinct physical
hashes, one semantic graph and one path program.

## Settings

Upstream's ZINC configuration with one change, the number of PAIN layers: hidden 128, 2-layer
shared LSTM, reversed paths, learned positional encoding, neighbour marks, sum path aggregation,
2-layer head, no dropout, Adam lr 1e-3, ReduceLROnPlateau on validation loss (factor .5,
min 1e-5), gradient clipping 5, full batch. The depth was selected on the original arm:
1 PAIN layer (5 layers do not learn on IMDb). The same settings are used in every arm.

| | IMDb | Freebase |
|---|---|---|
| input | canonical 3,008-d BoW + node-type embedding | learned node-id + node-type embeddings |
| paths | all rooted simple paths, L = 3 | sampled (4 relations / fan-out 8 / cap 256), L = 3 |
| budget | ≤ 300 epochs, patience 40, LR-plateau patience 20 | ≤ 100 epochs, patience 20, LR-plateau patience 10 |
| augmentation | 75 super-epochs (one step per variant), patience 10 | 33 super-epochs, patience 6 |

Selection and early stopping use validation micro-F1 (ties: lower validation loss). See also
`docs/EXPERIMENT_SETTINGS.md`.

## Environment

PAIN needs its own environment (`envs/pain.txt`: Python 3.11, PyTorch 2.13.0+cu126). Its
deterministic segment sum uses a CUDA `cumsum` that PyTorch 2.4 (the main environment) cannot
run in deterministic mode. Point the scripts at it with
`ENV_ACTIVATE=/path/to/.venv-pain/bin/activate`.

## Data inputs (under `data/`; `models/PAIN/data -> ../../data`)

| path | built by |
|---|---|
| `raw/IMDB/movie_metadata.csv` | tier A download |
| `dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/` | tier A download |
| `preprocessed/pain_nc/IMDB/` (4.6 GB), `preprocessed/pain_nc/IMDB_invariant_minimal/` (6.2 GB) | `bash scripts/PAIN/preprocess_imdb.sh` (CPU, ~20 min) |
| `preprocessed/pain_nc/Freebase/` (6.9 GB), `preprocessed/pain_nc/Freebase_invariant/` (5.5 GB) | `bash scripts/PAIN/preprocess_freebase.sh` (CPU, ~6 h, resumable) |

PAIN computes its splits itself. They equal the splits of the other models: IMDb 2,926 / 418 / 836
movies and Freebase 4,772 / 1,591 / 1,591 books. The IMDb features are byte-identical to
`common/imdb_canonical_features.py`.

## Commands

```bash
export ENV_ACTIVATE=/path/to/.venv-pain/bin/activate
bash scripts/PAIN/preprocess_imdb.sh
bash scripts/PAIN/preprocess_freebase.sh
ARM=original|universal|augmentation|invariant bash scripts/PAIN/run_imdb_nc.sh
ARM=original|universal|augmentation|invariant bash scripts/PAIN/run_freebase_nc.sh
cd models/PAIN && python summarize_arms.py results/imdb_nc IMDB/nc
cd models/PAIN && python summarize_arms.py results/freebase_nc FREEBASE/nc
```

Optional environment variables: `SEEDS` (default `1566911444 20241017 20251017`), `OUT`
(results root, relative to `models/PAIN`), `ENV_ACTIVATE`.

Results: `models/PAIN/results/{imdb_nc,freebase_nc}/<arm>/<DATASET>/nc/<variant>/seed_<s>/`
(`metrics.json`, `scores.npz`, `history.json`). The original arm's folder is named `originals`
(the runner's internal name). A run keeps a per-epoch state file under `<results root>/_resume/`
and continues from it when restarted; the file is removed when the run finishes.

Resources: tested on one NVIDIA V100 32 GB. IMDb runs take 15–50 minutes (about 6 GB of GPU
memory); Freebase runs take 1.5–3 hours (about 4 GB of GPU memory). Use 96 GB of host RAM.

## Expected results (test micro-F1, mean ± std over seeds 1566911444, 20241017, 20251017; V100)

| cell | original | universal | augmentation | invariant |
|---|---|---|---|---|
| IMDb NC | v1 .5670 ± .0054, v2 .5538 ± .0026, v3 .5594 ± .0039, v4 .5670 ± .0078; mean .5618 ± .0077, τ = .7484 | .5813 ± .0083 | .5589 ± .0102 (τ = .9033) | .5813 ± .0083 on every variant (τ = 1, 18/18 identical checkpoints) |
| Freebase NC | unchanged .6141 ± .0113, exact_2 .5653 ± .0139, exact_3 .5338 ± .0146; mean .5711 ± .0356, τ = .5616 | .5475 ± .0076 | .5461 ± .0216 (τ = .8681) | .5391 ± .0185 on every variant (τ = 1, 9/9 identical checkpoints) |

## Verifying invariance

On the invariant arm every variant of a seed records the same `artifact_meta.semantic_graph_sha256`,
`artifact_meta.selected_path_program_sha256`, `supervision_hash`, `init_hash` and
`checkpoint_hash` in `metrics.json`, while `artifact_meta.physical_graph_sha256` differs.
`init_hash` / `checkpoint_hash` depend on the device (CPU vs GPU), so compare them within one
device.
