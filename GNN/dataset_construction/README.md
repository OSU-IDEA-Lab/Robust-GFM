# Dataset construction

These scripts build the dataset variants and the shared splits from the public raw data. If
you downloaded tier A (`data/README.md`), you do not need to run them; they are here so every
input can be regenerated and checked. Model-specific preprocessing is in
`scripts/<MODEL>/preprocess_*.sh` and starts from the outputs listed here.

All paths are relative to the data root (`$INV_GNN_DATA`, default `<repo>/data`).

* **Software:** Python 3 with `numpy`, `scipy`, `pandas` and `scikit-learn` (`envs/gnn.txt`).
* **Hardware:** everything runs on CPU. Set `OPENBLAS_NUM_THREADS=1`.
* **Seed:** every split uses 1566911444.

| dataset | variants (union) | built here | built by the model preprocessing |
|---|---|---|---|
| IMDb | v1–v4 (union) | link-prediction split archives | the variant graphs (director / actor attached to the movie or to its Link copy): `scripts/RGCN/preprocess_imdb.sh`, `scripts/SlotGAT/preprocess_imdb.sh` |
| DBLP | v1–v3 (union) | Paper–Conference split | the variant graphs (Area attached to Paper, Conference or Author): `scripts/RGCN/preprocess_dblp.sh` |
| WN18RR | `no_changes`, `all_inverse_edges`, `transitive_edges` (`universal_edges`) | variant graphs and `wordnet_splits.npz` | — |
| Freebase | `unchanged`, `exact_2`, `exact_3` (`union_exact_2_3`) | variant graphs | — |

---

## 1. IMDb (`imdb/`)

**Raw input:** `raw/IMDB/movie_metadata.csv`, the Kaggle "IMDB 5000 Movie Dataset"
(https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset; the same file MAGNN
uses). Its sha256 is `74e4a9af314eea255147a6d48a5678ad7d0be3cb8ba3fdb819ef6edddb36e2ca`.

Every model uses the same set of movies, built as follows:

1. drop duplicate `movie_imdb_link`;
2. drop rows without `actor_1_name` or `director_name`;
3. keep movies whose first matching genre is Action, Comedy or Drama.

This gives 4,180 movies, 2,081 directors, 5,257 actors and 91 years.

```bash
# Movie–Director and Movie–Link split archives
python dataset_construction/imdb/reconstruct_IMDB_md_ml_shared_splits.py
# Movie–Year split archive (inherits the movie partition; drops the 3 movies without a year)
python dataset_construction/imdb/build_IMDB_my_shared_splits.py \
    --csv "$INV_GNN_DATA/raw/IMDB/movie_metadata.csv" \
    --inherit-from "$INV_GNN_DATA/preprocessed/IMDB_shared_splits/IMDB_md_shared_splits.npz" \
    --out "$INV_GNN_DATA/preprocessed/IMDB_shared_splits/IMDB_my_shared_splits.npz"
```

| output (`preprocessed/IMDB_shared_splits/`) | content |
|---|---|
| `IMDB_md_shared_splits.npz` | (movie, director) positives, 2,926 / 418 / 836 |
| `IMDB_ml_shared_splits.npz` | (movie, Link) positives, 2,926 / 418 / 836 |
| `IMDB_my_shared_splits.npz` | (movie, year) positives, 2,925 / 418 / 834 |

The split arrays match the distributed files exactly. The sha256 of the `.npz` files can
differ, because the archives also store metadata such as the path of the archive they
inherit from.

The archives contain positives only. Each model's preprocessing draws 19 negatives per
positive deterministically: frequency-matched years for Movie–Year, uniform directors for
Movie–Director.

**Node-classification split.** This split has no file. The preprocessing scripts compute it
with two `train_test_split` calls (seed 1566911444), giving 2,926 / 418 / 836 movies.

**Node features.** These have no file either. `common/imdb_canonical_features.py` defines one
feature rule shared by all models:

* **movie:** raw-count bag of words over `plot_keywords` (`CountVectorizer(min_df=2)`,
  3,008 terms);
* **director, actor, link, year:** the mean of the bag-of-words rows of the node's movies;
* **link prediction:** for the held-out target type, only training pairs are counted.

---

## 2. DBLP (`dblp/`)

**Raw input:** `raw/DBLP/`, the DBLP four-area dataset (Ji et al., ECML-PKDD 2010), also
distributed as `DBLP_four_area` with HAN (https://github.com/Jhy1993/HAN) and MAGNN. Its
licence does not allow redistribution, so download it from the original source
(http://ews.uiuc.edu/~jinggao3/doc/BGCM.zip) or the HAN repository.
`data/SHA256SUMS` lists checksums for the four files that are read.

```bash
python dataset_construction/dblp/paper-venue_shared_splits.py      # writes the split (no CLI arguments)
python dataset_construction/dblp/verify_dblp_canonical_splits.py   # exit code 0 = split reproduced
```

Output: `preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz` (sha256
`3686f59a9bab465fe3bb6b526f1a8bfd0da27cba5ba93157639778d178b38794`). It is a paper-disjoint
70/10/20 split of the 14,328 (paper, conference) pairs, 10,029 / 1,433 / 2,866, with every
other conference as a negative. Set `DBLP_SPLITS_OUT` to write it somewhere else.

The model preprocessing then makes two changes, giving 9,448 / 1,348 / 2,666 pairs:

* it drops 866 papers whose authors disagree on the research area;
* it adds Area edges only for training papers, so a test paper's area cannot reveal its
  conference.

---

## 3. WN18RR (`wn18rr/`)

**Raw input:** the official WN18RR split (Dettmers et al., 2018) in the format of the RotatE
repository (https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding, `data/wn18rr`):
40,943 entities, 11 relations, and 86,835 / 3,034 / 3,134 triples. The tier-A folder
`wordnet_3hops_augmented_full_provenance/00_source_data/` contains it.

```bash
W=$INV_GNN_DATA/wordnet_3hops_augmented_full_provenance
# Step 1: candidate 3-hop compositions (block-bijective, inverse-bearing; ~1 min)
python dataset_construction/wn18rr/wn18rr_experiment_exact_hops_transitive.py \
  --dataset-format wn18rr --data-dir $W/00_source_data --exact-hops 3 \
  --add-inverse-relations --prune-immediate-inverse-cancel --prune-consecutive-duplicates \
  --prune-all-same-relation --outdir $W/01_candidate_discovery/transitive_3hops_inverse
# Step 2: candidate catalog (only its metapaths.json files are used by step 3)
python dataset_construction/wn18rr/augment_wordnet_topk_transitive_selective_inverses.py \
  --data-dir $W/00_source_data \
  --recovery-csv $W/01_candidate_discovery/transitive_3hops_inverse/transitive_summary.csv \
  --top-k 0 --outdir $W/02_first_augmented_build/old_wordnet_3hops_augmented_full
# Step 3: the variants, built from the training triples only
python "dataset_construction/wn18rr/augment_wordnet_topk_transitive_selective_inverses(3).py" \
  --data-dir $W/00_source_data \
  --reuse-old-output-dir $W/02_first_augmented_build/old_wordnet_3hops_augmented_full \
  --top-k 0 --transitive-hop-length 3 --outdir $INV_GNN_DATA/wordnet_3hops_augmented_full
# Step 4: split archive shared by all models
python dataset_construction/wn18rr/preprocess_WORDNET_rgcn_augmentation.py \
  --data-dir $INV_GNN_DATA/wordnet_3hops_augmented_full
cp $INV_GNN_DATA/wordnet_3hops_augmented_full/wordnet_splits.npz $INV_GNN_DATA/wordnet_splits.npz
```

The tier-A provenance folder already holds the outputs of steps 1–2, so a rebuild normally
runs only steps 3–4.

| output (`wordnet_3hops_augmented_full/`) | content |
|---|---|
| `no_changes/data.txt` | 86,835 edges, 11 relations (the official training set) |
| `all_inverse_edges/data.txt` | 173,670 edges, 22 relations (adds named inverses) |
| `transitive_edges/data.txt` | 706,128 edges, 537 relations (adds 526 forward-only 3-hop shortcut relations) |
| `universal_edges/data.txt` | 893,629 edges, 2,087 relations (union, including inverse-bearing shortcuts) |
| `shared_relations.dict`, `original_splits/` | shared relation vocabulary; the official split files |
| `wordnet_splits.npz` | training triples of each variant, plus the official validation (3,034) and test (3,134) triples; sha256 `d139fabc4797728d3b8418601390571561357113ed43e348b2438dd39032e869` |

---

## 4. Freebase (`freebase/`)

**Raw input:** the Freebase node-classification dataset of the HGB benchmark (Lv et al.,
KDD 2021, https://github.com/THUDM/HGB). It has:

* 180,098 nodes in 8 types, 36 relations and 1,057,688 directed edges, without node features;
* 7,954 labelled BOOK nodes.

The tier-A folder `dataset_variant_3hops_filtered_PROVENANCE/01_original_data/` contains it.

```bash
P=$INV_GNN_DATA/dataset_variant_3hops_filtered_PROVENANCE
D=$INV_GNN_DATA/dataset_variant_3hops_filtered
# Step 1: type-signature filter -> 28 compositions (6 two-hop, 22 three-hop); ~2 min
python dataset_construction/freebase/typed_book_paths_up_to_k_new_transitive_fast_streaming_inverse_signature_filtered.py \
  --data-dir $P/01_original_data --outdir $P/03_transitive_3hops_filtered \
  --min-hops 2 --max-hops 3 --must-include-node-type BOOK --prune-consecutive-duplicates
# Step 2: materialise unchanged / exact_2 / exact_3 (~16 min, ~20 GB; use an empty output dir)
python dataset_construction/freebase/create_legacy_fb_transitive_variants_streaming.py \
  --data-dir $P/01_original_data \
  --summary $P/03_transitive_3hops_filtered/typed_book_paths_up_to_k_new_transitive_subset_filtered_summary.csv \
  --outdir $D --min-hops 2 --max-hops 3
# Step 3: verify per-relation row counts and build the union graph union_exact_2_3 (~15 min)
bash dataset_construction/freebase/rebuild_exact3_and_union.sh
```

Step 2 also writes a `range_2_3/` folder. The step-3 script reads it; no experiment does.
You can delete it after step 3.

**Split.** `common/freebase_block_common.make_split(labels, node_types, 0, 0.6, 0.2,
1566911444)` produces the BOOK split, 4,772 / 1,591 / 1,591; the model preprocessing computes
it on the fly. The same split is distributed as `Freebase_unchanged/train_val_test_idx.npz`.

| output (`dataset_variant_3hops_filtered/`) | relations | directed edges |
|---|---|---|
| `unchanged/` | 36 | 1,057,688 |
| `exact_2/` | 42 (+6 two-hop shortcuts) | 47,368,991 |
| `exact_3/` | 58 (+22 three-hop shortcuts) | 446,967,136 |
| `union_exact_2_3/` | 64 (+28) | 493,278,439 |

Each variant contains `node.dat`, `link.dat`, `label.dat` (7,954 labelled books), `info.dat`
and `variant_overview.json`; the shortcut variants also contain
`new_transitive_relations.{jsonl,csv}`. `freebase/registries/{exact_2,exact_3}.jsonl` list the
shortcut relations and their expected row counts, which step 3 checks. The sha256 of every
`link.dat`, `node.dat`, `label.dat` and `info.dat` is in `data/SHA256SUMS`.

---

## Files

```
imdb/      reconstruct_IMDB_md_ml_shared_splits.py, build_IMDB_my_shared_splits.py
dblp/      paper-venue_shared_splits.py, verify_dblp_canonical_splits.py
wn18rr/    wn18rr_experiment_exact_hops_transitive.py, augment_wordnet_topk_transitive_selective_inverses.py,
           augment_wordnet_topk_transitive_selective_inverses(3).py, preprocess_WORDNET_rgcn_augmentation.py
freebase/  typed_book_paths_up_to_k_new_transitive_fast_streaming_inverse_signature_filtered.py,
           create_legacy_fb_transitive_variants_streaming.py, rebuild_exact3_and_union.sh,
           freebase_rebuild_exact3_and_union.py, registries/
```
