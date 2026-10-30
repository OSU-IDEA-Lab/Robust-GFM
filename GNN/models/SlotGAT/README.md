# SlotGAT

SlotGAT (typed-slot graph attention; upstream: Zhou et al., "SlotGAT: Slot-based Message Passing for
Heterogeneous Graphs", ICML 2023) under the four-arm protocol:

| arm | meaning here |
|---|---|
| `original` | one model per variant on its own physical graph |
| `universal` | one model on the union graph of the variants (`SlotGAT*`) |
| `augmentation` | one shared model trained on every variant's physical graph |
| `invariant` | one model per variant on that variant's **compiled semantic program** (ours) |

The directory holds two code trees:

* **NC** (IMDb, Freebase): this directory (`run_IMDB_nc_*.py`, `run_Freebase_nc_*.py`, `model/`, ...).
* **LP** (IMDb Movie–Year, IMDb Movie–Director, DBLP Paper–Conference, WN18RR): `lp/`.

## How the invariant version works

**Structural bias.** SlotGAT aggregates 1-hop typed edges. Each node keeps one feature slot per node type,
a neighbour's message lands in the slot of its type, and slot attention combines the slots (upstream:
`l2byslot` readout, projected residuals, attention residual α = 0.05). A switching (for example, moving
Director from the Movie onto its Link copy, or replacing a 2-hop path with a named shortcut edge) changes
which slot receives which messages, so the stock model is not τ-invariant.

**Upstream model and parity port.** `model/gnn.py` and `model/conv.py` are the upstream DGL SlotGAT.
`model/parity_slotgat.py` (IMDb) and `model/genuine_union_message_slotgat.py --model-impl parity`
(Freebase) port upstream SlotGAT onto a **plan-driven, deterministic** operator. They use degree-bucket or
segment softmax with a fixed summation order instead of DGL's atomics. On IMDb the port matches upstream to
float precision with copied weights (`tests/test_parity_slotgat.py`); on the year graph the models have
3,846,298 (upstream) and 3,847,405 (parity) parameters.

How the arms map onto these implementations:

* `original` has two rows on IMDb NC, selected with `IMPL` in `scripts/SlotGAT/run_imdb_nc.sh`:
  * **upstream**: the stock DGL SlotGAT (`model/gnn.py`) on each variant's physical graph;
  * **matched control**: the parity port on each variant's physical graph.

  On Freebase NC and on the LP cells, `original` is the matched control (parity port, physical graph).
* `universal`, `augmentation` and `invariant` always use the parity port.

So the matched control, `universal`, `augmentation` and `invariant` share one architecture, and the
invariant arm changes only the *plan* the operator receives.

* **IMDb NC: raw-context conditional plan** (`conditional_bias.py`). Each variant keeps its own raw typed
  edges and the context witnesses found in it. At forward time, direct `Movie–Role` contexts and two-hop
  `Movie–Link–Role` contexts become the same semantic propagation events, typed by the semantic relation
  (15 edge types). Every variant has 50,142 raw context edges, which compile to 107,713 semantic messages
  with an identical program hash. No shared DGL graph is materialised.
* **Freebase NC: recoverability-conditional plan** (`freebase_block_conditional.py`,
  `freebase_union_messages.py`) for the family {unchanged, exact_2, exact_3}:
  * one 129-channel catalog: base relations plus the registered 2- and 3-hop compositions in
    `registries/exact_{2,3}.jsonl`;
  * one context plan per variant, in which physical shortcut edges are provenance validated against the
    registry;
  * the **union-message schedule**: one message per ordered `(dst, channel, src)` record, 750.5 M messages.
  Every variant's invariant run executes this same schedule, which is also the union graph's schedule, so
  invariant equals universal bitwise.
* **LP** (`lp/lp_parity_model.py`, `lp/lp_parity_hetero_model.py`): the upstream `slotGATConv` ported onto
  a destination-sorted edge list with `torch.segment_reduce`. Slots are the node types (IMDb, DBLP), or a
  single slot on WN18RR. The readout is concat-of-layers plus LayerNorm, the decoder is DistMult, and the
  loss is `neg_logsigmoid`. The inputs are the **RGCN structural bundles**: physical edges for
  original/universal/augmentation, and the compiled semantic program for invariant. IMDb and DBLP mask the
  batch's target edges (and their alias / label-derived copies) from the message graph.

See `docs/INVARIANT_MECHANISMS.md` §5 and §5a, and `docs/THEORY.md`.

## Required inputs under `data/` (`models/SlotGAT/data -> ../../data`; override with `INV_GNN_DATA`)

### Built by shared steps (not by this model)

| path | source | size |
|---|---|---|
| `data/raw/IMDB/movie_metadata.csv` | raw download (IMDb 5000 movie dataset) | 1.5 MB |
| `data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/` (`node.dat`, `link.dat`, `label.dat`, `info.dat`) | `dataset_construction/` (Freebase variants) | link.dat: 20 MB / 886 MB / 8.5 GB / 9.3 GB |
| `data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/{movie_year/{v1,v2,v3,v4,universal},movie_director/{v1,v3,universal}}/graph.pt` | RGCN preprocessing (`scripts/RGCN/`) | ~197 MB each, 8 files |
| `data/preprocessed/RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}/graph.pt` | RGCN preprocessing | ~29 MB each |
| `data/preprocessed/RGCN_structural/WORDNET/universal_parity_full/full/intersection/{no_changes,all_inverse_edges,transitive_edges,universal_edges}/graph.pt` | RGCN preprocessing | 70 / 78 / 125 / 141 MB |

The LP runners also import shared code from `common/`: `common/inv_rgcn/` (bundle loader) and
`common/wordnet_structural_eval.py` + `common/wordnet_structural_common.py` (WN18RR filtered ranking).

### Built here (in this order)

| step | command | writes | cost |
|---|---|---|---|
| 1 | `bash scripts/SlotGAT/preprocess_imdb.sh` (`python preprocess_IMDB.py --suffix _year`) | `data/IMDB_var{1..5}_year/` (var5 = union graph), 7 files each, ~115 MB each | CPU, ~1 min |
| 2 | `STEP=1 bash scripts/SlotGAT/preprocess_freebase.sh` | `data/preprocessed/freebase_slotgat_block_family/{unchanged,exact_2,exact_3,union_exact_2_3}/` + `preprocessing_audit.json` | CPU, 64 GB RAM, ~20 min, 221 MB |
| 3 | `STEP=2 VARIANT=<v> bash scripts/SlotGAT/preprocess_freebase.sh` for v in unchanged, exact_2, exact_3, union_exact_2_3 | `data/preprocessed/freebase_slotgat_physical_messages_family/{unchanged,exact_2,exact_3}/`, `data/preprocessed/freebase_slotgat_union_messages/union_exact_2_3/` (+ relative symlink `freebase_slotgat_physical_messages_family/union_exact_2_3`) | CPU, 96 GB RAM; exact_3 and union ~2.5 h and 4.4 GB each |
| — | `bash scripts/SlotGAT/preprocess_lp.sh` | nothing: it only checks that the RGCN bundles above exist | — |

Notes:

* The IMDb build is deterministic: a rebuild gives byte-identical files.
* Freebase splits are not a file. The runner takes `label.dat` of the `unchanged` source dir and makes a
  stratified 60/20/20 split with seed 1566911444 (`run_Freebase_nc_block_conditional.load_dataset`).
* The context plans record the absolute `source_dir` they were built from, so rebuild them (step 2) after
  moving `data/`, or pass `--reference-data-dir`.

## Running

Every script takes `ARM=original|universal|augmentation|invariant`, `SEEDS` (default
`"1566911444 20241017 20251017"`), optional `ENV_ACTIVATE=/path/to/venv/bin/activate` and `GPU`. It
`cd`s into `models/SlotGAT`, and results land under `models/SlotGAT/results/...`. The scripts set
`PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each run reseeds from its own seed, so running
arms or seeds separately gives the same numbers as running them together.

Expected results below are **mean ± population std over seeds 1566911444, 20241017, 20251017**, each run
on one NVIDIA V100 32 GB. The full tables are in `docs/results/IMDB_nc.tex`, `docs/results/Freebase_nc.tex`,
`docs/results/IMDB_lp.tex`, `docs/results/DBLP_lp.tex` and `docs/results/WN18RR_lp.tex`; settings are in
`docs/EXPERIMENT_SETTINGS.md`.

**Determinism caveat.** The upstream SlotGAT (`IMPL=upstream`) uses DGL kernels that are not
deterministic, so reruns of that row vary slightly. The parity port uses a fixed summation order.

### IMDb NC (test micro-F1) — `scripts/SlotGAT/run_imdb_nc.sh`

Shared flags: `--data-suffix _year --feature-mode pooled_bow --select-on val_micro_f1 --epochs 300 --patience 40`
(the augmentation runner uses its default selection criterion).

| arm | command | result dir | expected (IMDb1 / 2 / 3 / 4) | time (V100) |
|---|---|---|---|---|
| original, upstream SlotGAT | `ARM=original IMPL=upstream` (default `IMPL=both` runs both original rows) | `results/imdb_nc/original/upstream/original/` | .6535±.0065 / .6627±.0068 / .6559±.0057 / .6467±.0041 | ~8 s/run |
| original, matched control (parity, physical) | `ARM=original IMPL=parity` | `results/imdb_nc/original/parity/physical/` | .6507±.0039 / .6575±.0084 / .6571±.0006 / .6555±.0153 | ~40 s/run |
| universal (parity, union graph var5, physical) | `ARM=universal` | `results/imdb_nc/universal/physical/` | .6519±.0102 | ~40 s/run |
| augmentation (parity) | `ARM=augmentation` | `results/imdb_nc/augmentation/seed_*/` | .6435±.0122 / .6463±.0195 / .6455±.0187 / .6531±.0141 (mean .6471±.0159) | ~40 s/seed |
| invariant (parity, conditional) | `ARM=invariant` (`VALIDATE=1` adds the bitwise audit) | `results/imdb_nc/invariant/conditional/` | .6611±.0145 on every variant | ~55 s/run |

`IMPL` (`both|upstream|parity`) is read only for `ARM=original`. The runner's `--conditions`, `--ablate`
and `--model-impl` options accept further values (ablations) that the table above does not use. The whole
cell takes under 1 h on one GPU (4 CPUs, 48 GB RAM).

### Freebase NC (test micro-F1) — `scripts/SlotGAT/run_freebase_nc.sh`

Parity port, lr 0.005, stop on val loss, select on val micro-F1, 150 epochs, patience 25 (augmentation:
450 epochs, patience 75), 250k-edge chunks. Runs are resumable (`--resume`): rerun the same command after a
time limit and it continues from the last saved epoch. Features, labels and splits always come from the
`unchanged` source dir, so the universal arm passes
`--reference-data-dir data/dataset_variant_3hops_filtered/unchanged`.

| arm | command | result dir | expected (Freebase1 / 2 / 3) | time (V100) |
|---|---|---|---|---|
| original (matched control, physical) | `ARM=original` | `results/freebase_nc/original/recoverability_conditional/{unchanged,exact_2,exact_3}/` | .6602±.0100 / .6570±.0065 / .6478±.0119 | ~1.5 min / 32 min / 12.8 h per run |
| universal (physical union_exact_2_3) | `ARM=universal` | `results/freebase_nc/universal/recoverability_conditional/union_exact_2_3/` | .6396±.0061 | ~10 h per run |
| augmentation | `ARM=augmentation` | `results/freebase_nc/augmentation/seed_*/` | .6503±.0100 / .6489±.0057 / .5990±.0027 (mean .6327±.0048) | ~5.4 h per seed |
| invariant (exact_union schedule) | `ARM=invariant` | `results/freebase_nc/invariant/recoverability_conditional/<v>/` | .6396±.0061 on every variant | ~10 h per variant-run (~90 GPU-h total) |

After each seed, the script runs `scripts/gate_freebase_runs.py`, which checks the message mode, the
selection criterion and the epoch budget of every run (and, for the invariant arm, that all variants
reach the same test micro-F1). Peak GPU memory is ~10 GiB when training on the union schedule; 32 GB RAM,
4 CPUs and 12 h time slices (with `--resume`) are enough.

Do not pass `--aggregate` to `run_Freebase_nc_block_conditional.py`: it imports
`aggregate_Freebase_nc_block_conditional`, which is not included. Read the per-run `result.json` files instead.

### LP (test MRR) — `lp/`, RGCN bundles as input

| cell | script | result dir | original (per variant) | universal | augmentation (per variant) | invariant | time (V100) |
|---|---|---|---|---|---|---|---|
| IMDb Movie–Year | `run_imdb_movie_year.sh` | `results/lp_hetero/<arm>/IMDB/movie_year/<v>/seed_*/` | .2107±.0055 / .2114±.0016 / .2160±.0070 / .2165±.0099 | .2171±.0087 | .2206 / .2201 / .2196 / .2193 | .2171±.0087 (≡ universal) | ~30–65 s/run |
| IMDb Movie–Director | `run_imdb_movie_director.sh` | `results/lp_hetero/<arm>/IMDB/movie_director/<v>/seed_*/` | .2637±.0149 / .2431±.0066 (v1, v3) | .2610±.0066 | .2543±.0190 / .2526±.0167 | .2514±.0088 | ~30–65 s/run |
| DBLP Paper–Conference | `run_dblp_lp.sh` | `results/lp_hetero/<arm>/DBLP/paper_conference/<v>/seed_*/` | .5561±.0064 / .5418±.0079 / .5557±.0081 | .5565±.0101 | .5330±.0145 / .5332±.0154 / .5367±.0154 | .5565±.0101 (≡ universal) | ~2.5–7 min/run |
| WN18RR | `run_wn18rr_lp.sh` | `results/wn18rr_lp/<arm>/full/intersection/<v>/seed_*/` | .3658±.0011 / .3660±.0003 / .3652±.0009 | .3635±.0001 | .3638±.0013 / .3635±.0016 / .3624±.0018 | .3662±.0017 | ~20–37 min/run |

`<arm>` in the LP result dirs is the runner's own name: `originals`, `universal`, `augmentation` or
`invariant` (the scripts map `ARM=original` to `--arm originals`). Aggregate with
`python lp/aggregate_lp_arms.py --family hetero_nls|wordnet_nls`. Each run saves its full configuration in
`metrics.json`. Flags on top of the runner defaults:

* IMDb/DBLP: `--mask-batch-edges --loss neg_logsigmoid --eval-every 1 --patience 20`;
* WN18RR: `--loss neg_logsigmoid` (runner defaults: hidden 128, 2×4 heads, `concat_ln` readout,
  batch 2048 × 256 negatives, lr 1e-3, eval every 5, patience 8; ~17 s/epoch, 48 GB RAM).

## Outputs

* **NC runs** write `result.json` (test/val metrics, best epoch, timing, peak memory, hashes) and
  `history.json` per `<condition>/<variant>/seed_<s>/`, plus `preflight_audit.json` and `run_manifest.json`
  at the result root; checkpoints go to `checkpoint/{imdb_nc,freebase_nc}/<arm>/`. On IMDb, `--aggregate`
  (on by default) also writes `aggregate_per_variant.csv`, `aggregate_pairwise_invariance.csv` and
  `aggregate_summary.json`. `VALIDATE=1` on the IMDb invariant arm writes
  `results/imdb_nc/invariant/validation_model_audit.json`.
* **LP runs** write `metrics.json`, `history.json`, test scores (`scores.npz` / `test_scores.npz`) and
  `best_model.pt` per `seed_<s>/`, in the RGCN result layout.

## Verifying invariance

The invariant runs record the hashes needed to check invariance:

1. Physical graph hashes differ across variants.
2. Program and supervision hashes are equal across variants.
3. Parameter hashes at every evaluation and the final checkpoint hashes are equal across variants.
4. Hence the spread across variants is 0 and Kendall τ = 1.

On IMDb, `VALIDATE=1 ARM=invariant bash scripts/SlotGAT/run_imdb_nc.sh` runs
`validate_IMDB_slotgat_conditional.py --model-check --require-bitwise`, which checks that the raw graphs
and witness plans differ, that every variant compiles to the same propagation program (107,713 semantic
messages from 50,142 raw context edges), and that evaluation logits and first-step gradients are bitwise
equal. On Freebase, the invariant runs equal the universal run bitwise, and
`scripts/gate_freebase_runs.py --require-invariance` checks the per-seed agreement.

## Tests

`tests/` holds unit tests for the parity port (`test_parity_slotgat.py`, against upstream with copied
weights), the IMDb conditional plan, the Freebase family preprocessing and the union-message schedule:

```bash
cd models/SlotGAT && python -m pytest tests
```
