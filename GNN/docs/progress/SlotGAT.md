# SlotGAT release progress
- 2026-09-30: started. Reported roots from SOURCES.md lines 12-18 (IMDb NC), 42-45/59-62/76-82/101-104 (LP), 125-132 (Freebase NC).
## Trace (done)
- IMDb NC: sbatch_IMDB_year_nc_v2.sbatch (upstream original, var5 universal; job 21247232, 09-11) ; sbatch_IMDB_parity_ablation.sbatch task 0 `parity` (physical/conditional + _universal; 21331639, 09-14 20:xx) ; sbatch_IMDB_year_augmentation_parity.sbatch (21332762, 09-14 21:11). Runner run_IMDB_nc_conditional.py mtime 09-14 20:32 (AFTER v2 runs 09-11: v2 'original' produced by pre-parity version; no .bak of that state; original path uses model/gnn.py unchanged since 07-26).
- Data IMDB_var{1..5}_year built 09-01 19:21 by `preprocess_IMDB.py --suffix _year` (script mtime 19:21 = data mtime).
- Freebase: sbatch_FREEBASE_family_{physical,invariant,augmentation}_parity_dgx2.sbatch (LR 0.005) and _dgxh (default LR 0.05 in script but result.json say lr 0.005 -> submitted LR=0.005). Prep: sbatch_FREEBASE_family_prep.sbatch + sbatch_FREEBASE_family_schedules.sbatch (array 0-3). Runner mtime 09-14 21:34 < runs 09-15..09-19.
- LP: sbatch_arms_full_dgx2.sbatch (OUT_ROOT=.../arms_nls EXTRA='--loss neg_logsigmoid', array 21353337) and sbatch_hetero_arms_dgx2.sbatch (OUT_ROOT=.../arms_hetero_nls EXTRA='--mask-batch-edges --loss neg_logsigmoid --eval-every 1 --patience 20', 21353338) -- confirmed from metrics.json config. Deps on RGCN: inv_rgcn/data.py (+inv_rgcn pkg), wordnet_structural_eval.py (+wordnet_structural_common.py).
- DBLP policy ablations: sbatch_dblp_perturb_dgx2.sbatch + run_hetero_lp_arms_perturb.py (copy of hetero runner + --bundle-root); bundles from src/baselines/perturb (robustness, out of scope).
## Copy + edits (done 20:55)
- 40 files copied to models/SlotGAT (NC root, model/, utils/, scripts/, registries/, tests/, lp/). data symlink made.
- Path edits: utils/data.py, preprocess_IMDB.py, run_Freebase_nc_block_conditional.py (INV_GNN_DATA const), run_Freebase_nc_augmentation.py (native.INV_GNN_DATA), lp runners (DATA/BUNDLE_ROOT env default; RGCN_DIR -> <release>/common), lp/aggregate_lp_arms.py nls roots -> models/SlotGAT/results/{wn18rr_lp,lp_hetero}.
- common/: added wordnet_structural_eval.py + wordnet_structural_common.py (from RGCN, byte-identical); common/inv_rgcn already present (identical to working repo).
- NEXT: scripts/SlotGAT/*.sh, README, py_compile + --help under srun.
## Scripts + README (done ~21:15)
- scripts/SlotGAT: run_imdb_nc.sh (IMPL=upstream|parity|both|ablation_conditional), run_freebase_nc.sh, run_imdb_movie_year.sh, run_imdb_movie_director.sh, run_dblp_lp.sh, run_wn18rr_lp.sh, run_dblp_lp_policy_ablation.sh, preprocess_imdb.sh, preprocess_freebase.sh, preprocess_lp.sh (checker only).
- Removed aggregate_Freebase_nc_block_conditional.py from release copy (py3.11 SyntaxError, only used under opt-in --aggregate).
- Verify: all .py compile, all 15 entry points --help OK (srun job 21491348). Freebase/family unit tests pass (7). IMDb rebuild byte-compare: running (verify2).
- Verify2 (srun 21491512): IMDb rebuild from release copy byte-identical to data/IMDB_var{1..5}_year (35/35 files); load_data, LP bundle load (DBLP v1), WN eval import via common/, Freebase plan load (semantic hash 7bd4b8b5ecc9) all OK. __pycache__ removed. DONE.
