# GFM release: progress and provenance (internal; delete before publishing)

Same convention as `GNN/docs/RELEASE_*`: this file is the only place for provenance, dates and job ids.

## Layout decision (2026-10-29, user request)

Repo root `INV-GNN-GFM-release/` = `GNN/` (byte-identical `cp -a` of `new_code/INV-GNN-release/` on 2026-10-29,
verified with `diff -rq --no-dereference`) + `GFM/` (shared code + one folder per GFM; colleagues add folders).
`new_code/INV-GNN-release/` itself was not touched. Future GNN-release changes (e.g. MAGNA) must go into
`GNN/` here, or into both.

## Sources (working repo, copied, never edited)

| release file | source in `INV-GNN/src/baselines/ULTRA_pilot/` | changes |
| --- | --- | --- |
| `common/kg_format.py` | `pilot_common.py` | renamed; docstring |
| `common/build_kg_variants.py`, `build_wn18rr_variants.py`, `compile_back.py`, `compare_variants.py` | same names | import name; `--data-root` / `GFM_DATA` (default `GFM/data`); WN18RR source `../GNN/data/...`; docstrings |
| `ULTRA/ultra_eval.py`, `ref_check.py`, `ref_check_inference.yaml` | same names | `../common` on sys.path; yaml paths templated (`--output_root`, `--dataset_root`); `dataset_dir` in metrics.json = basename |
| `ULTRA/aggregate.py`, `make_figure.py` | `aggregate_pilot.py`, `make_pilot_figure.py` | `RESULTS` env; outputs `summary.*`, `figure.*` |
| `ULTRA/expected_results.md` | `results/pilot_summary.md` | title paragraph |
| `docs/EVALUATION_GUIDE.md` | `GFM_EVALUATION_GUIDE.md` | rewritten in release vocabulary |
| `ULTRA/README.md` | `README.md` + `EVALUATION_DETAILS.md` | merged, release style |
| new | — | `scripts/{_common,download_raw,build_data}.sh`, `ULTRA/setup_upstream.sh`, `ULTRA/requirements.txt`, `ULTRA/scripts/*.sh`, `TEMPLATE/*`, READMEs |

`upstream/` is not copied: `setup_upstream.sh` clones DeepGraphLearning/ULTRA at `427966ad` (checkpoints are tracked
in that repo). Raw data: villmow FB15k-237 and ULTRA's Hetionet dropbox URLs; sha256 equal to the pilot's raw files
(checked 2026-10-29).

## Verification (2026-10-29, test copy in ~/gfm_release_test, because /nfs/hpc/share was over quota)

- `download_raw.sh`: all six files downloaded, sha256 OK.
- `summarize.sh` on a copy of the pilot's comparison.json files: summary.md / summary.csv identical to
  `pilot_summary.*`.
- build job 21971125 (`build_data.sh`, all three datasets, 6 min 42 s): 33/33 manifests hash-identical to the pilot's
  data; every `<V>__compiled` byte-identical to the compiled reference (WN18RR `20df0974c9eb3282`).
- GPU job 21971126 (RTX 2080 Ti, cn-gpu4, 11 min 39 s): `LIMIT=64 run_zero_shot.sh WN18RR ultra_4g` ranks + candidate
  scores bitwise equal to the pilot on all 4 variants + repeat; `TEMPLATE/evaluate.py` with ULTRA's hooks
  (~/gfm_release_test/test_template_ultra.py) bitwise equal on no_changes + transitive_edges; `run_compiled.sh WN18RR`
  bitwise equal to pilot results_compiled on all 4; `run_ref_check.sh ultra_4g` test MRR 0.483783 (= pilot).
- Not run end to end: FB15k-237 / Hetionet full evaluations (same code path as WN18RR; data hash-identical).

## Before publishing

- delete this file;
- `GNN/docs/RELEASE_*` and `GNN/docs/progress/` likewise (see `GNN/docs/RELEASE_PROGRESS.md`).
