# CMPNN release progress
- 2026-09-30: started. Read conventions, SOURCES.md CMPNN lines, CMPNN_HANDOFF.md.
- Traced: IMDb MY/MD <- scripts/sbatch_CMPNN_lp_suite.sbatch (jobs 21256772-76, 09-11); DBLP <- scripts/sbatch_CMPNN_dblp_array.sbatch (21260314 + reruns 21265365/21330358);
  WN dgx2 orig/univ/aug/leaky-inv <- scripts/sbatch_CMPNN_wordnet_dgx2.sbatch (21339940, batch 64 chunk 200k); WN ctx invariant <- sbatch_CMPNN_wordnet_context_dgx2.sbatch (21355786,21355934-41);
  dgxh twins: sbatch_CMPNN_wordnet_dgxh.sbatch (21339944, chunk 400k default) + sbatch_CMPNN_wordnet_context_dgxh.sbatch (21369460-68).
- Runners: run_cmpnn_lp.py (orig/univ/inv), run_cmpnn_lp_augmentation.py (aug), run_cmpnn_lp_wordnet_context.py (WN inv). CMPNN_data_augmentation/ NOT used (no reported root).
- Closure: cmpnn/{__init__,layer,model}.py, inv_cmpnn/{__init__,bundles,conv,model}.py; outside: RGCN/inv_rgcn/{__init__,data,model,metrics,utils}.py + RGCN/rgcn_resource_metrics.py -> common/.
- Inputs: RGCN bundles v10_year_canonfeat (IMDb), v8_dblp (DBLP), RGCN_structural/WORDNET/universal_parity_full/full/intersection (WN).
- mtimes: model.py changed 09-15 (remove_one_hop=False branch only; True path identical to .bak) - IMDb/DBLP ran before; run_cmpnn_lp*.py 09-12 02:05 (resume fix; straight runs unchanged per handoff) - IMDb ran 09-11 before it.
- Env: sbatch PATH -> ~/hpc-share/gnn/conda_envs/torchdrug_env (= /nfs/hpc/share/mousavij/gnn/conda_envs/torchdrug_env), NOT INV-GNN/torchdrug_env.
- copied code, path edits, scripts/CMPNN/*.sh, envs/cmpnn.txt; srun share job 21491334: --help x4 OK, 10 unit tests + one_hop_false test OK, bundle load OK (md v1 hash dc2d516ca000 = run's). Next: README.
- 2026-09-30: README written; DONE. Open: env Python 3.10 (settings doc says 3.11); IMDb ran on pre-09-12 runner, DBLP mostly pre-09-12 runner + 09-11 model.py; dgxh WN ctx invariant not bitwise.
