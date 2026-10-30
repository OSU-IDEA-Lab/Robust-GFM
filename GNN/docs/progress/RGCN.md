# RGCN release progress (agent notes)
- 2026-09-30 start. Conventions read. Roots from SOURCES.md lines 3-6,29-32,46-49,63-66,83-90,105-114.

## Trace findings (2026-09-30)
- IMDb NC/MY/MD (orig/univ/inv): sbatch_IMDB_year_suite.sbatch (TASK, DATA=v10, OUT=IMDB_year_<task>_canonfeat; RUN_TIMEOUT 20m nc / 45m LP) jobs 21247205 (nc), 21248828 (my), 21248827 (md) -> run_rgcn_structural.py (mtime Sep10 23:31 < runs). 
- IMDb aug: sbatch_IMDB_year_nc_augmentation.sbatch (21247212) -> run_rgcn_structural_nc_augmentation.py; sbatch_IMDB_year_lp_augmentation.sbatch (21248829 md, 21248830 my) -> run_rgcn_structural_lp_augmentation.py
- v10 bundle: preprocess_IMDB_rgcn_structural.py --tasks nc,movie_director,movie_year --shared-movie-director IMDB_md --shared-movie-year IMDB_my --out-root RGCN_structural_v10_year_canonfeat (nc built Sep10 23:18, md/my Sep11 09:49 after LP-feature patch; current file diff vs .bak_pre_lp_features = LP features only; HANDOFF: dist rebuild byte-identical md/my, nc identical data fields). Imports src/baselines/imdb_canonical_features.py.
- DBLP orig/univ/inv: sbatch_ONE_RUNNER_controls.sbatch task 0 (21127312_0) -> run_rgcn_structural.py (Sep 1 version = .bak_pre_progress_lines; diff vs current = NC metrics + progress prints only).
- DBLP bundle v8: preprocess_DBLP_rgcn_structural.py defaults + --out-root RGCN_structural_v8_dblp (built 18:09:40 Sep1 after file mtime 18:09:35).
- DBLP aug: sbatch_RGCN_LP_instr_resume.sbatch (cd rgcn_data_augmentation) run_DBLP_rgcn_augmentation.py --data-suffix _canonical ... -> needs DBLP_rgcn_v{1,2,3}_canonical from RGCN/preprocess_DBLP_rgcn.py --variant v1,v2,v3 --data-suffix _canonical (post-build edit = universal only).
- WN inv: sbatch_WORDNET_patience_suite.sbatch task1 -> run_wordnet_rgcn_structural_full.sh TRAIN_OBJECTIVE=task_only (runner = .bak_pre_graph_flag; current adds --graph default semantic only).
- WN orig/univ: sbatch_ONE_RUNNER_controls task 1 -> run_WORDNET_rgcn_structural.py --graph physical.
- WN bundle: run_wordnet_rgcn_structural_full.sh PREPROCESS=1 TRAIN=0 (sbatch_WORDNET_rgcn_structural_transitive.sbatch, Aug21) -> universal_parity_full/full/intersection; input wordnet_3hops_augmented_full/wordnet_splits.npz + wordnet_structural_catalogs/full_selection.json
- WN aug: sbatch_WORDNET_aug_dgx2_backup.sbatch (seed 1566911444 copied from 21115988_3) -> rgcn_data_augmentation/run_WORDNET_rgcn_augmentation.py (needs torch_geometric)
- FB inv: sbatch_FREEBASE_exact3_compile.sbatch + sbatch_FREEBASE_exact3_invariant.sbatch; ablation range_2_3: sbatch_FREEBASE_rgcn_block_preprocess_all / _compile_exact3 + sbatch_FREEBASE_selF1_invariant
- FB orig/univ: sbatch_FREEBASE_physical_baselines.sbatch (catalogs+stores) + sbatch_FREEBASE_selF1_baselines.sbatch
- FB aug: sbatch_FREEBASE_aug_instr.sbatch -> run_FREEBASE_rgcn_augmentation_sampled.py (current adds --resume, default off)
- freebase_conditional_rgcn.py edited Aug28 19:16 after baselines: shared-vocab, default None = unchanged.
- edit: preprocess_IMDB_rgcn_structural.py sys.path -> <release>/common for imdb_canonical_features
- per coordinator: inv_rgcn/ + rgcn_resource_metrics.py removed from models/RGCN (moved to scratchpad), imported from <release>/common via one added sys.path line in run_rgcn_structural{,_nc_augmentation,_lp_augmentation}.py, preprocess_DBLP_rgcn_structural.py, run_FREEBASE_rgcn_augmentation_sampled.py (preprocess_IMDB already points at common/).
- NOTE: common/ also holds freebase_block_common.py, wordnet_structural_{common,eval}.py (copied by another agent); RGCN keeps its own local copies (imported implicitly by ~12 files with no sys.path line). Flag to coordinator.
- 21:xx: scripts/RGCN written (_common.sh, _imdb.sh, run_imdb_{nc,movie_year,movie_director}.sh, run_dblp_lp.sh, run_wn18rr_lp.sh, run_freebase_nc.sh, preprocess_{imdb,dblp,wn18rr,freebase}.sh), bash -n OK. README written. py_compile all OK.
- pending: srun DBLP v8 rebuild-vs-reference hash check; srun --help on all entry points.
- DONE: srun --help OK on all 21 entry points (job 21491575); DBLP v8 rebuilt from release code (job 21491604) -> physical/semantic/supervision hashes + edge counts + context_audit identical to data/preprocessed/RGCN_structural_v8_dblp for v1,v2,v3,universal.
- Status: RGCN package complete (pending coordinator decision on duplicate freebase_block_common / wordnet_structural_* in common/).
