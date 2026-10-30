# Release conventions (internal — how this folder is assembled)

Source of truth for *which* code/run is reported: the working repo
`/nfs/hpc/share/mousavij/gnn/new_code/INV-GNN` —
`paper_tables/SOURCES.md` (result root behind every table block),
`paper_tables/build_paper_tables.py` (registry), `docs/EXPERIMENT_SETTINGS_all_dgx2.md`
(contracts), per-model handoffs in `docs/`.

## Scope
Models: RGCN, MAGNN, SlotGAT, SeHGNN, CMPNN, PathNet, PAIN (minimal program only) (MAGNA later).
Cells: IMDb NC; IMDb LP Movie–Year (MY) and Movie–Director (MD); DBLP LP Paper–Conference;
WN18RR LP; Freebase NC. Only the blocks registered in SOURCES.md (reported rows + the
labelled ablations listed there). No robustness / message-budget / theory-minimal studies
in this pass.

## Arm names (use exactly these, everywhere: folder names, script flags, READMEs)
| arm | paper meaning | old names in the working repo |
|---|---|---|
| `original` | one model per variant on its own physical graph | originals, physical, baseline |
| `universal` | one model on the union graph of the variants (`{model}*`) | canonical, universal, union |
| `augmentation` | one model trained jointly on all variants' physical graphs | augmentation, aug |
| `invariant` | one model per variant on its compiled semantic program (ours) | invariant, structural, conditional, semantic |

## Layout
```
INV-GNN-release/
  README.md                  top-level: what's here, env, data, how to run every experiment
  data/                      EMPTY in the repo (README.md only); user downloads into it
  common/                    inv_gnn_paths.py, imdb_canonical_features.py
  dataset_construction/      raw -> variant graphs + shared splits (shared by all models)
  models/<MODEL>/            minimal closure of the model's code, original relative layout kept
     README.md               per-cell table: preprocessing cmd + one cmd per arm + result dir
     data -> ../../data      relative symlink (code uses data/... relative to model dir)
  scripts/<MODEL>/           run_<dataset>_<task>.sh  (ARM=original|universal|augmentation|invariant)
                             (plain bash, Slurm optional via sbatch wrappers)
  envs/                      requirements
  docs/                      theory, dataset construction, settings, this file, RELEASE_PROGRESS.md
```

## Rules
1. **Never modify anything in the working repo.** Copy only. Jobs are queued against it.
2. Copy the *exact* files that produced the reported roots (trace sbatch -> runner -> imports;
   read run dirs' saved args/config where present). Keep the model's internal relative layout
   so imports work unchanged. No `.bak*`, `__pycache__`, results, logs, checkpoints, data.
3. Allowed edits in the copies: (a) absolute paths (`/nfs/...`, `~/hpc-share/...`) -> relative
   `data/...` or `$INV_GNN_DATA` (default `<release>/data`); (b) env activation lines -> removed /
   `${ENV_ACTIVATE:-}`; (c) Slurm `#SBATCH` partition/account lines kept as comments/examples only;
   (d) output roots -> `results/<dataset>_<task>/<arm>/...`. No logic changes. Record every
   non-path edit in the model README "Changes vs. working repo" section.
4. Data layout under `data/` mirrors `<authors-cluster>/gnn/data` exactly (same relative
   names), so `data/preprocessed/RGCN_structural_v10_year_canonfeat/...` etc. keep working.
5. Every model README lists: required inputs under `data/` (exact relative paths), which ones are
   *downloaded* vs *built by a step in this repo*, the build commands in order, and for each
   (dataset, task, arm): command, seeds (1566911444 20241017 20251017), expected metric from
   the paper table (mean ± pop. std), GPU/time hint.
6. Verify: every copied entry point at least `python -m py_compile`s and `--help`s (inside
   `srun`, never heavy work on the login node) with `OPENBLAS_NUM_THREADS=1`.
