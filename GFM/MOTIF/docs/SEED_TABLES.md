# Seed-averaged tables (lab format)

`script/report_seeds.py` produces two tables in the lab format:

* **Performance:** one row per method and variant, with MRR, Hits@1, Hits@3, Precision and Recall.
* **Invariance:** one Kendall-τ row per variant pair.

Metric definitions and aggregation follow the lab's GNN metrics (`METRICS.md` beside the GNN result tables),
so the MOTIF rows can be compared directly with the lab's CMPNN results:

* MRR and Hits@k use the 20 fixed candidates per query, and ties get the averaged rank.
* Precision and Recall are binary metrics, pooled over all (query, candidate) pairs. A pair
  counts as predicted positive when σ(score) ≥ 0.5, i.e. MOTIF's logit ≥ 0. With 1 positive
  in 20 candidates, a model that predicts every pair positive shows Precision ≈ 0.05 and
  Recall = 1.
* Kendall-τ is the per-query τ_b between two variants' candidate scores, averaged over
  queries. It is computed within one seed.
* Each cell is mean ± population std (ddof = 0) over the training seeds
  1566911444, 20241017 and 20251017.
* **MOTIF (FT)** rows use the model trained and tested on the same variant, which is the
  lab's "original" arm. FT τ compares the independently trained v_i and v_j models.
* **MOTIF (ZS)** has no training seed. Its single deterministic run is shown with std 0.

## 1. Fine-tune with the three lab seeds

From the repository root on the submit node:

```bash
sbatch slurm/finetune_seeds.sbatch                              # IMDb movie_year
sbatch --export=ALL,DATASET=dblp slurm/finetune_seeds.sbatch    # DBLP paper_conference
```

This submits one array task per seed. Each task runs the full train/test matrix
(`script/run_finetune_matrix.py`) under the run tag `seeds`. The output goes to
`results/motif_finetune_matrix_<dataset>_lp_seeds_s<seed>/`, and Slurm logs go to
`logs/motif-ft-seeds-<job>_<task>.out`. You can resubmit after a failure: `--resume`
reuses completed cells.

## 2. Build the tables

```bash
python script/report_seeds.py --dataset imdb \
  --zeroshot results/motif_zeroshot_imdb_lp_full \
  --finetune results/motif_finetune_matrix_imdb_lp_seeds_s* \
  --baseline-tex $BASELINES/IMDB_lp.tex
```

This step runs on CPU in a few seconds. It writes `results/seed_summary_<dataset>_<task>/`:
`performance.tex`, `invariance.tex`, `report.md`, `performance.csv`, `invariance.csv` and
`summary.json`.

* `--baseline-tex` adds the CMPNN "original" rows from the lab's results file. `$BASELINES` is the folder
  holding the GNN result tables (`IMDB_lp.tex`, `IMDB_nc.tex`, `WN18RR_lp.tex`, ...); they are not part of this repo.
  `--baseline` selects a different model block, e.g. `RGCN`.
* Drop `--zeroshot` or `--finetune` to report only one of them.
* All matrices passed together must share code and settings, apart from the seed.

## 3. IMDb Movie-Director (v1, v3)

The task is exported from the lab's own bundles (the same ones CMPNN used):

```bash
python script/prepare_imdb_lp.py --tasks movie_director \
  --bundle-root rai-datasets/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB \
  --output rai-datasets/preprocessed/IMDB_movie_director
MD=DATASET=imdb,TASK=movie_director,DATA=rai-datasets/preprocessed/IMDB_movie_director
sbatch --export=ALL,$MD,RUN_TAG=md_full slurm/zeroshot.sbatch
sbatch -p dgx2 --constraint=v100 --export=ALL,$MD,RUN_TAG=md_seeds2 slurm/finetune_seeds.sbatch
python script/report_seeds.py --dataset imdb --task movie_director \
  --zeroshot results/motif_zeroshot_imdb_lp_md_full \
  --finetune results/motif_finetune_matrix_imdb_lp_md_seeds2_s* \
  --baseline-tex $BASELINES/IMDB_lp.tex
```

## 4. WN18RR (v1 no_changes, v2 all_inverse_edges, v3 transitive_edges)

`script/wordnet_lp.py` follows CMPNN's WordNet protocol (`../../GNN/models/CMPNN/run_cmpnn_lp.py`):

* MRR and Hits use filtered ranking over all 40,943 entities, for both head and tail queries,
  with ties averaged.
* Precision/Recall and τ use the positive plus 100 candidates drawn with each lab seed.
* Fine-tuning trains on the official base-relation triples and removes only the exact batch
  edges.

The released checkpoint was **pretrained on WN18RR**, so its "zero-shot" rows are not
zero-shot. v3 (537 relations) needs about 45 GB, so all WordNet runs use one H200:

```bash
sbatch -p dgxh --constraint=h200 --array=0-2 --export=ALL,MODE=eval,RUN_TAG=full slurm/wordnet.sbatch
sbatch -p dgxh --constraint=h200 --array=0-8 --export=ALL,MODE=finetune,RUN_TAG=seeds slurm/wordnet.sbatch
python script/report_seeds.py --dataset wordnet \
  --zeroshot results/motif_zeroshot_wordnet_lp_full \
  --finetune results/motif_finetuned_wordnet_lp_seeds_s* \
  --baseline-tex $BASELINES/WN18RR_lp.tex
```

## 5. Node classification (IMDb genre, Freebase BOOK)

`script/nc_motif.py` turns classification into link prediction to one node per class. Only
training nodes get a `node_to_class` edge, and validation/test nodes are checked to have none.
The labels, splits and variants are the lab's own (see the script docstring). There is no
CMPNN baseline, so the tables use the lab's NC baselines.

```bash
# IMDb: build the lab's NC bundles, export, run (V100)
python ../../GNN/models/RGCN/preprocess_IMDB_rgcn_structural.py --tasks nc \
  --variants v1,v2,v3,v4 --raw-csv rai-datasets/raw/IMDB/movie_metadata.csv \
  --out-root rai-datasets/preprocessed/_build/IMDB_nc
python script/nc_motif.py export --dataset imdb --bundle-root rai-datasets/preprocessed/_build/IMDB_nc/IMDB/nc
sbatch --array=0-3  --export=ALL,DATASET=imdb,MODE=eval,RUN_TAG=full slurm/nc.sbatch
sbatch --array=0-11 --export=ALL,DATASET=imdb,MODE=finetune,RUN_TAG=seeds slurm/nc.sbatch
python script/report_seeds.py --dataset nc_imdb --zeroshot results/motif_zeroshot_nc_imdb_full \
  --finetune results/motif_finetuned_nc_imdb_seeds_s* \
  --baseline-tex $BASELINES/IMDB_nc.tex --baseline RGCN SlotGAT

# Freebase: CPU export (parses the 7.9 GB exact_3 link.dat), then H200 runs
sbatch --export=ALL,DATASET=freebase slurm/nc_export.sbatch
```

## 6. Overview table (averages over all variants)

```bash
python script/report_overview.py   # reads every results/seed_summary_*/summary.json
```

It writes `results/seed_summary_overview/overview_{lp,nc}.tex` with one row per dataset and
method. The columns are the average metric over variants (MRR and Hits@1 for link prediction,
accuracy for node classification), the average |difference| between variants (computed from
the variant means, the same way for every method), and the average Kendall-τ over variant
pairs. Rerun it after adding any `seed_summary_*` folder.

## 7. Overleaf

`results/overleaf/motif_seed_tables/main.tex` is a standalone file that collects every
`seed_summary_*` table. Upload that single file to Overleaf.

Git tracks only `results/overleaf/` and the `results/{seed,zeroshot}_summary_*` tables.
Per-run outputs (checkpoints, scores, histories, logs) stay on the HPC.
