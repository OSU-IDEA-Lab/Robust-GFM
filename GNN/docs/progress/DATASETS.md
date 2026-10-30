# DATASETS release progress (agent notes)
- 2026-09-30 start. Conventions read. Scope: dataset_construction/{imdb,dblp,wn18rr,freebase}, data/README.md.

## Provenance findings so far
- WN18RR variants `wordnet_3hops_augmented_full/` = provenance Step 3 `augment_wordnet_topk_transitive_selective_inverses(3).py` (2026-07-25; data dir copied in 07-26 00:37).
- `wordnet_splits.npz` (data/ and data/wordnet_3hops_augmented_full/, byte-identical sha256 d139fabc...) has format_version `wordnet_lp_four_variants_v1`, uncompressed savez -> written by `rgcn_data_augmentation/preprocess_WORDNET_rgcn_augmentation.py` (same keys/format); the SlotGAT `preprocess_wordnet_lp_slotgat.py` writes the identical payload into data/ root. NOT `RGCN/prepare_WORDNET_rgcn_structural_splits.py` (different format string + train_pos_supervision key).
- Freebase variants: provenance Steps 1/2b/3 (May 4) + exact_3 rebuild + union (2026-08-22, `RGCN/sbatch_FREEBASE_rebuild_exact3_union.sbatch` + `freebase_rebuild_exact3_and_union.py`). Registries in RGCN/registries == variant new_transitive_relations.jsonl (sha-verified).
- Freebase split: `freebase_block_common.make_split` (seed 1566911444, 0.6/0.2) -> SlotGAT `adopt_rgcn_freebase_splits.py` wrote Freebase_{unchanged,exact_2}/train_val_test_idx.npz (2026-09-01).
- DBLP split: `MAGNN/paper-venue_shared_splits.py` -> DBLP_shared_splits_magnn == CANONICAL (sha 3686f59a...).
- IMDb md/ml npz contain keys seed/num_movies/num_directors that the in-repo CMPNN builders do NOT write -> builder version that wrote them is gone; positives to be checked by regeneration. my = CMPNN/build_IMDB_my_shared_splits.py (09-01 18:40 same minute as npz).

## 2026-09-30 copies + verification
- Copied: freebase/ (provenance code/ scripts, rebuild py+sh, registries/, freebase_rebuild/ record, adopt_rgcn_freebase_splits.py), wn18rr/ (3 provenance scripts + preprocess_WORDNET_rgcn_augmentation.py), dblp/ (paper-venue_shared_splits.py, verify_dblp_canonical_splits.py), imdb/ (build_IMDB_my_shared_splits.py; legacy/ md+ml CMPNN builders).
- Path edits only (INV_GNN_DATA); rebuild sh: ROOT/D/df paths, SLURM_SUBMIT_DIR, mail-user scrubbed. DBLP builder OUT -> DBLP_shared_splits_CANONICAL (+ DBLP_SPLITS_OUT env override).
- VERIFIED (srun 21491374): wordnet_splits.npz sha-identical from preprocess_WORDNET_rgcn_augmentation.py; DBLP canonical npz sha-identical (3686f59a); verify_dblp exit 0; IMDB_my sha-identical (44008c22); Freebase make_split == Freebase_unchanged npz == RGCN store split.
- IMDb md/ml: in-repo builders do NOT reproduce (different partition). Recovered recipe = nested train_test_split(0.3, then 2/3, seed 1566911444), sorted -> NEW file imdb/reconstruct_IMDB_md_ml_shared_splits.py: ml sha-identical, md all arrays identical (zip container bytes differ).
- Running: WN Step 3 rebuild compare (21491423), FB Steps 1/2b compare (21491424), hashes (bg).
- FB Step 3 (--max-hops 2, filtered summary) reproduces unchanged/ + exact_2/ BYTE-IDENTICALLY (info/node/label/link/registry/overview), 45 s.
- sha256 of all tier-A files computed (data/README.md table). Sizes measured (sizes.out).
- Written: dataset_construction/README.md (draft; §5 re-run log pending), data/README.md (tier A/B, provisional B).
- --help OK for all 14 argparse entry points (paper-venue_shared_splits.py has no CLI, runs at import).
- Resubmitted (short limits after Priority pending): WN Step 3 compare, FB Steps 1/2b compare, FB full Step 3 + rebuild_exact3_and_union.sh in node-local /tmp.
- WN Step 3 rerun (12 s): all data.txt/dicts/support cache/summaries byte-identical; 2 metapaths.json + coverage rankings identical modulo embedded paths; manifest differs in paths + cache-reuse counters.
- FB Step 1 + 2b: summaries byte-identical. FB full Step 3 (filtered summary, 16 min): exact_3/link.dat BYTE-IDENTICAL to live rebuilt (sha 514ee418). Fresh exact_3/variant_overview.json is correct (live is stale).
- rebuild sh bug in my edit: SLURM_SUBMIT_DIR is also set under srun -> ROOT wrong. Fixed (script dir first, then SLURM_SUBMIT_DIR, FB_REBUILD_ROOT override). Re-running union build on fresh data (fb_full2).
- Union build on fresh data: exit 0, union_exact_2_3 info/link/registry/overview BYTE-IDENTICAL to live; freebase_rebuild record identical except map_r23_to_exact3.tsv (fresh range_2_3 = 28 rels). => Freebase family fully reproducible from HGB raw with shipped code.
- DONE: dataset_construction/README.md §5 complete, data/README.md (tier B provisional).
