# Release assembly progress (resume here after a disconnect)

Started 2026-09-30. Goal: a self-contained public repo for the paper, built by COPYING from
`/nfs/hpc/share/mousavij/gnn/new_code/INV-GNN` (never editing it). Conventions:
`docs/RELEASE_CONVENTIONS.md`.

## Status
| part | status | notes |
|---|---|---|
| skeleton + conventions | done 2026-09-30 | |
| dataset_construction/ | done 2026-09-30 | all variants/splits rebuild byte-identical except IMDb md/ml (recovered recipe, new reconstruct script) |
| models/RGCN | done 2026-09-30 | 54 files; DBLP v8 rebuild hash-identical; inv_rgcn in common/; RGCN keeps local copies of freebase_block_common / wordnet_structural_* (also in common/, identical) — decide in final pass; FB dgxh exact_3 invariant root has 0 runs |
| models/MAGNN | done 2026-09-30 | docs/progress/MAGNN.md; uses common/{inv_rgcn,rgcn_resource_metrics,freebase_block_common,aggregate_rgcn_structural} |
| models/SlotGAT | done 2026-09-30, pruned 2026-10-01 | NC top-level + lp/; IMDb rebuild byte-identical 35/35; upstream IMDb rows predate 09-14 runner edit (old path unchanged);  see progress/SlotGAT.md |
| models/SeHGNN | done 2026-09-30 | headline IMDb=full, Freebase inv=full_rel_uov511; imports MAGNN code (not data); runner/compiler mtimes post-date some runs, no .bak — README lists expected program hashes |
| models/CMPNN | done 2026-09-30 | env = gnn/conda_envs/torchdrug_env (py3.10); runner mtime 09-12 after IMDb runs (resume fix only); bundles from RGCN; see progress/CMPNN.md |
| top-level README | done 2026-09-30 | |
| data/README + rsync list | done 2026-09-30 | data/download_lists/{tierA_required,tierB_imdb_dblp_wn18rr,tierB_freebase}.txt; raw/DBLP excluded (licence) |
| models/PathNet | done 2026-10-01 | reported roots PathNet_inv/results/{imdb_nc_tuned_bfs,freebase_nc_rt_bfs} (jobs 21491083, 21491091 + walks 21491084-90); release runner == working runner (CPU hashes); not in paper_tables .tex |
| models/PathNet | UPDATED 2026-10-08 | homo + per-epoch MERW protocol; roots PathNet_inv/results/{imdb_nc_homo_merw,freebase_nc_homo_merw}; README Freebase row = TODO until 21686898/99 finish |
| models/PAIN | done 2026-10-08 | MINIMAL only; IMDb inv = imdb_nc_L1_minimal (= universal bitwise), other IMDb arms imdb_nc_L1, Freebase all arms freebase_nc_L1; own env envs/pain.txt; runner pruned of closure/L2 options |
| MAGNA | later (user) | |

## Log
- 2026-09-30: skeleton created; per-model tracing delegated to parallel agents, each writes
  its findings into `docs/progress/<MODEL>.md` as it goes.
- 2026-09-30: copied reference docs into docs/ (THEORY, INVARIANT_MECHANISMS, THEORY_PATTERN_REDUNDANCY,
  DATASET_CONSTRUCTION_FREEBASE_WORDNET, DATASET_STATISTICS, EXPERIMENT_SETTINGS_*, CANONICAL_SPLITS,
  IMDB_YEAR_REDESIGN) and the paper tables (*.tex, METRICS, README, SOURCES) into docs/paper_results/.
  These still contain working-repo paths/dates — scrub pass TODO before publishing.
- Agents running (one per part): DATASETS, RGCN, MAGNN, SlotGAT, SeHGNN, CMPNN. Each writes docs/progress/<X>.md.
- 2026-09-30 final pass: script names normalised (run_<dataset>_<task>.sh), envs/gnn.txt written, RGCN duplicate
  helpers -> symlinks to common/, cluster paths scrubbed (except RGCN catalogs JSON source_variant_dir, left
  byte-identical), MAGNN Freebase hash sentence fixed in docs/INVARIANT_MECHANISMS.md, tier B pinned to what run
  scripts read. Remaining for the user: citation block, delete
  docs/RELEASE_* + docs/progress before publishing, PAIN/MAGNA/PathNet next.
- 2026-10-01: PathNet added (models/PathNet, scripts/PathNet, top README matrix, tierB_freebase list += 7 BFS walk dirs).
- 2026-10-01 PRUNE to reported rows only (user decisions): kept both MAGNN invariant flavours, SlotGAT matched-control
  original; removed all ablations (RGCN range_2_3; SeHGNN restricted/restricted_union/noskip/type-collapsed/597 closure;
  SlotGAT upstream-universal, parity_universal/conditional, DBLP policy; CMPNN 2076), all dgxh/H100 material, legacy
  dataset builders, history docs (IMDB_YEAR_REDESIGN, CANONICAL_SPLITS, THEORY_PATTERN_REDUNDANCY, EXPERIMENT_SETTINGS_dgx2_dgxh).
  Renamed docs: EXPERIMENT_SETTINGS.md, paper_results/{Freebase_nc,WN18RR_lp}.tex. Removed files are in the session
  scratchpad pruned/ (not in the release). Runner internals left byte-identical (unreachable flavour branches remain).
- 2026-10-03 STYLE pass (user: READMEs must read like a normal repo, not "the paper"/"reported"): all READMEs, docs,
  script headers and code comments restyled per scratchpad STYLE.md; `# release:` comments stripped; docs/paper_results
  -> docs/results (SOURCES.md removed); data/SHA256SUMS added; dated PROVENANCE_*.md, freebase_rebuild/ output and
  adopt_rgcn_freebase_splits.py removed from dataset_construction/. No code logic changed (help strings / default
  output dirs only). Compile + bash -n OK.
- 2026-10-08 (session 2) user: (1) update PathNet to the new reported protocol = PathNet_homo + fresh MERW walks per
  epoch + 1000 ep (lr 5e-3, dp .7, wd 5e-4), roots PathNet_inv/results/{imdb_nc_homo_merw, freebase_nc_homo_merw}
  (Freebase seeds 2-3 still running: 21686898/99); (2) add PAIN, MINIMAL program only (IMDb invariant =
  PAIN_inv/results/imdb_nc_L1_minimal, built by build_imdb_invariant_minimal.py, == universal bitwise; other IMDb arms
  imdb_nc_L1; Freebase all arms freebase_nc_L1 — Freebase closure == minimal, compiled by colleague's
  preprocessing.freebase_invariant); (3) tone check everything touched.
- 2026-10-08 (session 2) DONE: PathNet recopied (pathnet_nc/runners + fast_walks*, freebase_pairs; ablate diag removed;
  scripts/README/docs updated; tierB += 5 twohop bitsets matching walk-meta pair hashes). PAIN added: models/PAIN (PAIN-nc subset
  + run_pain_nc/pain_memfix/build_imdb_invariant_minimal), scripts/PAIN, envs/pain.txt, docs (THEORY 2.7, MECHANISMS 6b,
  SETTINGS rows, THEORY 6 holds-table + cat.3), top README, data README/lists. Release runner edits for PAIN: removed --program
  (IMDb inv always IMDB_invariant_minimal, FB inv always Freebase_invariant), removed --path-length (fixed 3), paths.py env
  INV_GNN_DATA. Equivalence smoke (release vs working, same node, 1-2 epochs, param hashes): job 21718250 (share), outputs
  /nfs/hpc/share/mousavij/gnn/new_code/release_smoke/ (outside both repos; delete after).
- 2026-10-08 user: removed all int8-overflow text and all minimal-vs-closure / "= union graph" commentary from PAIN README,
  build script docstring, runner docstring, scripts, THEORY (2.7, §6 rows), MECHANISMS 6b, SETTINGS, DATASET_STATISTICS.
  Kept file/dir names (build_imdb_invariant_minimal.py, IMDB_invariant_minimal/) and the artifact metadata string so tier-B data match.
- 2026-10-09 smoke 21718250 (ran on CPU: cn-gpu3 CUDA unknown error): release == working, identical per-epoch param hashes +
  ckpt for PathNet IMDb inv v2 (6042442448a7), PathNet FB inv exact_2 (9c1641d441db), PAIN IMDb inv v3 (bc7d56bd1290).
  PAIN FB timed out on CPU -> rerun 21727594 (share, --exclude=cn-gpu3, GPU-asserted).
- 2026-10-09 smoke 21727594 (cn-gpu4 RTX 2080 Ti): PAIN FB inv exact_3 release == working (ckpt cafa58ce229d). All four
  equivalence checks pass. release_smoke/ can be deleted.
