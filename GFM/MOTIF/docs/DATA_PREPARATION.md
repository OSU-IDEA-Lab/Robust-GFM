# IMDb and DBLP preprocessing for MOTIF

Run from the MOTIF repository root. Raw datasets and all generated data live in
the git-ignored `rai-datasets/`, separated into raw inputs and preprocessed outputs:

```text
MOTIF/
  rai-datasets/
    raw/
      IMDB/
        movie_metadata.csv
      DBLP/
        author_label.txt
        paper_author.txt
        paper_conf.txt
        paper_term.txt
    preprocessed/
      IMDB/                  # adapter-ready structural arrays + provenance
        manifest.json
        movie_year/
        movie_director/      # when this task is requested
      DBLP/
        manifest.json
        paper_conference/
      _build/                # intermediate canonical splits and graph.pt bundles
        IMDB/
        DBLP/
```

Place your own raw files in `raw/IMDB/` and `raw/DBLP/`. No raw data is downloaded
or copied by these commands. Additional DBLP files can stay in `raw/DBLP/`; only the
four listed files are required. Raw hashes are checked against the canonical
dataset checksums in `rai-datasets/SHA256SUMS`.

MOTIF's upstream benchmark loaders use the YAML `dataset.root`, whose default is
`~/MOTIF/kg-datasets/`, with dataset-specific `raw/` and `processed/` directories.
These custom datasets use `rai-datasets/` through the preprocessing entry points
below. They do not change upstream benchmark paths.

## Environment and commands

These commands run on CPU and import no MOTIF, PyG, Triton, or custom CUDA
extension. They need Python 3.10+, NumPy, PyTorch, pandas, SciPy and scikit-learn.
Use your activated MOTIF venv; its existing `torch==2.1.0+cu121` installation
satisfies the PyTorch requirement:

```bash
python -m pip install -r requirements-preprocess.txt
python script/prepare_imdb_lp.py
python script/prepare_dblp_lp.py
```

For a CPU-only preprocessing environment, install the CPU PyTorch wheel first:

```bash
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-preprocess.txt
```

Defaults omit the universal arm, as requested. IMDb exports Movie-Year v1-v4
and Movie-Director v1/v3; DBLP exports Paper-Conference v1-v3. Every variant also
contains its compiled view. Add `--include-universal` to either command to
include that arm. Movie-Year's compiled view equals its universal graph even
when the universal arm is omitted.

## Task-specific variant families

Use the actual canonical builders and `../../GNN/docs/DATASET_STATISTICS.md`
alongside the general `../docs/EVALUATION_GUIDE.md`
(sections B1-B4). The guide's combined IMDb overview does not mean every IMDb
task has four physical variants:

| Task | Physical variants | Context that moves | Test queries / candidate protocol |
| --- | --- | --- | --- |
| IMDb Movie-Year | v1, v2, v3, v4 | Director and Actor between Movie and Link | 834; true year + 19 fixed frequency-matched negatives |
| IMDb Movie-Director | v1, v3 | Actor on Movie (v1) or Link (v3) | 836; true director + 19 fixed uniform negatives |
| DBLP Paper-Conference | v1, v2, v3 | Area on Paper (v1), Conference (v2) or Author (v3) | 2,666; all 20 conferences |

Movie-Director keeps the supervised relation as Movie-to-Director, with only
training positives in the message graph. Link-to-Director copies are omitted
from the physical inputs. The year remains fixed Movie context. Do not invent
Movie-Director v2/v4 by moving its target: that would change the task contract.
Its two physical variants have one pairwise comparison, v1 vs v3.

DBLP keeps Author-Paper and Paper-Term context in every variant. Paper-Conference
target edges and the support used to construct relocated Area edges are
training-only; held-out target edges and held-out paper-specific Area context
are excluded. The canonical split verifier runs during preprocessing. Every
variant is exported with its physical and compiled view. Universal remains
optional and is omitted for this preliminary experiment.

## Preparing one task or adding Movie-Director later

To prepare only the four-variant IMDb task:

```bash
python script/prepare_imdb_lp.py --tasks movie_year
```

The default command, without `--tasks`, prepares both IMDb LP tasks. Selecting
`--task movie_year` at evaluation time only runs that task even if both were
prepared. To see which tasks the HPC export contains, inspect its root
`manifest.json` or list `rai-datasets/preprocessed/IMDB/`.

If Movie-Director was not prepared and the Movie-Year export already exists,
build a separate director-only export in fresh directories:

```bash
python script/prepare_imdb_lp.py --tasks movie_director \
  --work-dir rai-datasets/preprocessed/_build/IMDB_director \
  --output rai-datasets/preprocessed/IMDB_director
```

Pass `--data rai-datasets/preprocessed/IMDB_director` to `eval_lp.py` for this
export. Do not merge files manually into the existing IMDb export: its root and
task manifests record the file hashes. If canonical Movie-Director bundles
already exist, `--bundle-root /path/to/IMDB --tasks movie_director` with a fresh
`--output` can export them without rebuilding.

For DBLP, place the four raw files in `rai-datasets/raw/DBLP/`, then run:

```bash
python script/prepare_dblp_lp.py
```

This creates canonical intermediate bundles under
`rai-datasets/preprocessed/_build/DBLP/` and adapter-ready inputs under
`rai-datasets/preprocessed/DBLP/paper_conference/`.

Options `--csv` (IMDb) and `--raw-dir` (DBLP) select raw files elsewhere without
copying them. `--work-dir` selects a fresh build directory and `--output` selects
the final export. Existing build/export directories are rejected to prevent
mixing runs. After an interrupted build, choose a new work directory. To export
already completed canonical bundles, skip rebuilding with:

```bash
python script/prepare_imdb_lp.py --bundle-root /path/to/IMDB --output rai-datasets/preprocessed/IMDB_retry
python script/prepare_dblp_lp.py --bundle-root /path/to/DBLP/paper_conference --output rai-datasets/preprocessed/DBLP_retry
```

`--source-repo` optionally uses another INV-GNN source checkout; by default the
builders are read from `Robust-GFM/GNN/` (`../../GNN` from this folder).

The adapter defaults to `rai-datasets/preprocessed/IMDB/` or
`rai-datasets/preprocessed/DBLP/`. If you already have old-layout portable
exports, move the complete `motif_imdb_lp/` directory to `preprocessed/IMDB/`
and `motif_dblp_lp/` to `preprocessed/DBLP/`, provided the destinations do not
already exist. Existing exports remain valid after relocation; rebuilding is
not required. Alternatively, point evaluation at the old export with `--data`.
Old intermediate build directories can be supplied via `--bundle-root`.

## What is preserved

The original split builders, structural preprocessors, feature helpers and bundle
classes are used unchanged from `Robust-GFM/GNN/`. Their expected file hashes are
recorded in `tests/PROVENANCE.json` and checked by `tests/test_preprocessing.py`. The two
export entry points and graph/query checks are adapted from SCR's corresponding files.

Preserved contracts:

* IMDb: 15,789 nodes; Movie-Year train/val/test positives 2,925 / 418 / 834 and
  19 fixed frequency-matched negative years; Movie-Director 2,926 / 418 / 836
  and 19 fixed uniform negative directors. Repeated sampled years stay repeated.
* DBLP: 26,076 nodes; canonical paper-disjoint split, verified by the original
  verifier; post-filter positives 9,448 / 1,348 / 2,666. Rank over all 20
  conferences. Mapping scope and target edges are training-only.
* Shared node IDs, relation vocabularies (including empty relation slots), and
  supervision across variants; one common compiled program per task.
* Audits reject held-out target edges, their IMDb Link copies, and held-out
  paper-specific DBLP Area context.
* Source inverse pairs are verified, then only forward triples are exported.
  The MOTIF adapter appends internal inverses exactly once, with inverse
  relation ID `r + len(relation_names)`, before building MOTIF relation graphs.
* Edge arrays use `(head, tail, relation)` columns and unit-weight set semantics,
  matching SCR's exporter. Source bundle hashes retain the original weighted
  program identity separately from the effective exported edge hashes.

The source IMDb bundles retain canonical BoW features to preserve the SCR
construction and provenance. The portable MOTIF exports omit these features:
MOTIF is structure-only here. SCR's SVD projection, ontology features, semantic
neighbors and training/model code are not part of these entry points.

## Portable outputs and current scope

Each export has a root `manifest.json`, then a directory per task containing:

```text
manifest.json
queries.npz
v1.npz
v2.npz  # where defined for this task
...
```

Each variant NPZ contains `physical` and `compiled` forward edge arrays.
IMDb queries contain `node_type`, `train_pos`, `val_pos`, `test_pos` and their
19-negative tail matrices. DBLP queries contain the same positives and node types
plus `candidate_pool`, the shared 20-conference pool; negatives are its complement
for each positive. Manifests record code, source-bundle, supervision, raw-data and
effective edge hashes, counts, relation names, target relation and feature policy.

This prepares datasets only. The separate `script/eval_lp.py` adapter reads these
arrays, calls MOTIF's native Lift and model, and scores the fixed candidates.
See [zero-shot evaluation](ZERO_SHOT_EVALUATION.md) for commands and report formats.
The upstream `upstream/script/run.py --dataset IMDB` or `--dataset DBLP` entry points are
not added. The separate extension preflight is optional. The lazy initializer
in `motif/rspmm/__init__.py` (applied to `upstream/` by `upstream.patch`) lets the Triton inference path import its kernels
without compiling the separate native C++/CUDA extension. That native backend
still needs a compatible compiler if explicitly used.

Contract checks:

```bash
python -m unittest discover -s tests -p test_preprocessing.py -v
```
