# SeHGNN release progress (agent notes; resume here)

## 2026-09-30 tracing
- Reported roots (SOURCES.md): IMDb `SeHGNN/results/SeHGNN_v2/IMDB_nc/*` ; Freebase `SeHGNN_v2_dgx2/FREEBASE_nc/*`, `SeHGNN_v2/FREEBASE_nc/*` (dgxh), invariant headline `SeHGNN_v2_subset{_dgx2}/FREEBASE_nc/invariant_full_rel_uov511`.
- Saved metrics.json per run has `config` + `bundle.dir` (no argv). Configs consistent per root:
  IMDb: embed/hidden 512, batch 10000, patience 50, projection=type, 200 epochs.
  Freebase full/restricted*: batch 10000; full_rel (all arms) + uov511: batch 1024; embed 256, 100 epochs, patience 30.
- Producing sbatch: IMDb `scripts/sbatch_SEHGNN_imdb_suite.sbatch` (runs Sep 12; invariant_full re-run Sep 14 with skip schema -> job 21331802; old run renamed invariant_full_noskip, bundle now `full/semantic-noskip`).
  Freebase dgx2 full/restricted: `scripts/sbatch_SEHGNN_freebase_dgx2.sbatch`; full_rel dgx2: `scripts/probe_freebase_full_rel.sbatch` (--batch-size 1024);
  dgxh: `scripts/sbatch_SEHGNN_freebase_dgxh.sbatch` (+ CELLS/EXTRA for full_rel); uov511: `probe_freebase_invariant_uov511.sbatch` (dgx2) / `sbatch_SEHGNN_freebase_uov511_dgxh.sbatch` (dgxh).
  Preprocess: `scripts/sbatch_SEHGNN_freebase_preprocess.sbatch` -> preprocess_sehgnn_channels.py.
- dgxh full_rel (21337375) = sbatch_SEHGNN_freebase_dgxh.sbatch with CELLS=4 full_rel cells, EXTRA=--batch-size 1024 (log headers confirm).
- External imports (NOT copied; resolved via sibling models/RGCN, models/MAGNN through the runners' unchanged sys.path lines):
  RGCN: inv_rgcn/{__init__,data,model,utils}.py, rgcn_resource_metrics.py, freebase_block_common.py, aggregate_rgcn_structural.py
  MAGNN: magnn_compile_common.py, preprocess_imdb_magnn_v2.py, preprocess_freebase_magnn_v2.py (code only; SeHGNN reads NO MAGNN-built data)
- Data inputs: IMDb data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc/{v1..v4,universal}/graph.pt (RGCN builder);
  Freebase data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/ (dataset_construction).
## 2026-09-30 copy done
- Copied 15 py files to models/SeHGNN (+ data symlink). Path edits: DEFAULT_ROOT in run_sehgnn_nc.py / run_sehgnn_nc_subset.py, 3 constants + `import os` in preprocess_sehgnn_channels.py. All py_compile OK.
- Next: scripts/SeHGNN/{preprocess_imdb,preprocess_freebase,run_imdb_nc,run_freebase_nc}.sh, README, srun --help check.
## 2026-09-30 (cont.)
- Switched to release convention: RGCN deps from <release>/common (byte-identical, verified with cmp); MAGNN code from models/MAGNN (only path edits by MAGNN agent).
- Scripts written: scripts/SeHGNN/{preprocess_imdb,preprocess_freebase,run_imdb_nc,run_freebase_nc}.sh; README.md written.
- srun -p share check: all py_compile OK, all 6 entry points --help OK, lazy preprocess imports resolve to common/ + models/MAGNN, torch_sparse 0.6.18 present, unit tests 3/3 OK.
- CPU smoke of run scripts against the working-repo bundles (INV_GNN_DATA=<authors-cluster>/gnn/data, output to node /tmp) in progress.
- NB: srun from /tmp scripts must be passed via `bash -s <` (scratchpad not visible on compute nodes); dgx2 queue is slow, use -p share for CPU checks.
- CPU smoke (share, job 21491605) PASS: run_imdb_nc.sh invariant/restricted (v1-v4, 2 epochs) + augmentation/restricted, run_freebase_nc.sh invariant/restricted (unchanged, exact_2) against working-repo bundles; program hashes eb0c22f6c36c / 02b8b4098f38 as expected. (exact_3 cut by `head` SIGPIPE in the smoke script, not a code issue.)
## STATUS: SeHGNN DONE 2026-09-30 (final report sent to coordinator)
