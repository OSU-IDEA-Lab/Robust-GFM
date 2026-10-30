# MAGNN release progress
- 2026-09-30: started. SOURCES.md roots: MAGNN_imdb_v2/{nc,movie_year,movie_director}/{originals,universal,augmentation,invariant,invariant_union}, MAGNN_dblp_v2/*, MAGNN_freebase_v2_dgx2/* (dgx2), MAGNN_freebase_v2/* (dgxh).
- Traced producers:
  - IMDb NC: scripts/sbatch_MAGNN_imdb_nc_suite.sbatch (job 21259478, 09-11 21:25, dgx2) -> run_freebase_magnn_nc.py --dataset IMDB (+ run_freebase_magnn_nc_augmentation.py --dataset IMDB --group mode0 --variants v1,v2,v3,v4)
  - IMDb MY/MD: scripts/sbatch_MAGNN_imdb_lp_suite.sbatch (MY 21333243-245 started 09-14 23:03:42 > run_magnn_lp.py mtime 23:03:31; MD 21333442/447/452) -> run_magnn_lp.py / run_magnn_lp_augmentation.py
  - DBLP: scripts/sbatch_MAGNN_dblp_lp_suite.sbatch (21333438-451 + universal 21337218-220)
  - Freebase: scripts/sbatch_MAGNN_freebase_dgx2.sbatch (21334039/21334040) and _dgxh.sbatch (21259884) -> same runners.
  - Bundles: preprocess_{imdb,dblp,freebase}_magnn_v2.py (+ magnn_compile_common.py) from RGCN_structural_v10_year_canonfeat/IMDB, RGCN_structural_v8_dblp/DBLP, dataset_variant_3hops_filtered.
- Deps outside model dir: RGCN/inv_rgcn, rgcn_resource_metrics (already in common/, identical), RGCN/freebase_block_common.py (compiler split), RGCN/aggregate_rgcn_structural.py (aggregation) -> common/.
- Flags: run_freebase_magnn_nc.py mtime 09-15 15:42 (after all runs; adds INV_GNN_DETERMINISM_STRICT env switch only). preprocess_freebase_magnn_v2.py mtime 09-11 21:06 > FB invariant/stock bundle compile (20:11-20:40) -> verify by recompiling.
- IMDb NC augmentation metrics.json says features 'sparse identity' but code loads canonical BoW (label hard-coded) -- metadata only.
- Copied: models/MAGNN/{8 py + model/5 py}, common/{freebase_block_common.py, aggregate_rgcn_structural.py} (new, byte-identical to RGCN/); common/inv_rgcn + rgcn_resource_metrics already there (identical). data symlink made.
- Edits: sys.path ../RGCN -> ../../common; DEFAULT_ROOTs via INV_GNN_DATA; import os added to 3 compilers. README written. scripts/MAGNN: _common.sh, _run_lp.sh, run_{imdb_nc,imdb_movie_year,imdb_movie_director,dblp_lp,freebase_nc}.sh, preprocess_{imdb,dblp,freebase}.sh.
- FLAVOUR default = union (tables list union block first). py_compile OK. srun verification (help + recompile hash compare) running.
- 2026-09-30 VERIFIED (srun share job 21491472): all 7 entry points + aggregator --help OK; release compilers recompile 9 sample bundles
  (FB stock/invariant/invariant_union unchanged; IMDb nc inv v1, nc union v2, MD stock v3, MY stock universal; DBLP stock v2, union v3)
  with every summary hash + every per-metapath instance hash EQUAL to the bundles behind the reported runs -> FB compiler mtime concern resolved.
- Gauntlet helper tested on real bundles (MD, DBLP, FB): PASS. DONE.
