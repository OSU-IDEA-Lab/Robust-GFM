# Data

Everything in this folder except this README and `SHA256SUMS` is git-ignored. Download the inputs below, then
build the MOTIF inputs with the commands at the end.

## Layout

```
rai-datasets/
  SHA256SUMS                checksums of the raw files and tier-A files (tracked)
  raw/
    IMDB/movie_metadata.csv   downloaded
    DBLP/*.txt                downloaded
  data_tierA/               downloaded: the tier-A data of the GNN experiments
    wordnet_splits.npz                  WN18RR variants and the official split
    dataset_variant_3hops_filtered/     Freebase variants (unchanged, exact_2, exact_3)
    Freebase_unchanged/train_val_test_idx.npz
  preprocessed/             built by script/prepare_*.py and script/nc_motif.py export
    IMDB/  DBLP/  IMDB_movie_director/  NC/{imdb,freebase}/
    _build/                 intermediate split archives and RGCN bundles
```

## Downloads

The datasets and their variants are the same ones as in the GNN experiments (`../../../GNN/`);
`../../../GNN/dataset_construction/README.md` describes how each one is built.

* **IMDb:** `raw/IMDB/movie_metadata.csv`, the Kaggle "IMDB 5000 Movie Dataset"
  (https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset; the same file MAGNN uses).
* **DBLP:** `raw/DBLP/`, the DBLP four-area dataset (Ji et al., ECML-PKDD 2010), also distributed as
  `DBLP_four_area` with HAN (https://github.com/Jhy1993/HAN) and MAGNN. Its licence does not allow redistribution,
  so download it from the original source (http://ews.uiuc.edu/~jinggao3/doc/BGCM.zip) or the HAN repository.
* **WN18RR and Freebase:** the tier-A files listed under Layout, from the GNN experiments' tier-A data. To rebuild
  them from the public sources instead, follow `../../../GNN/dataset_construction/README.md` (sections 3 and 4).

Check the files against `SHA256SUMS`. The `raw/` entries are relative to this folder, the others to
`data_tierA/`:

```bash
cd rai-datasets
sha256sum -c --ignore-missing SHA256SUMS                                  # raw files
(cd data_tierA && sha256sum -c --ignore-missing ../SHA256SUMS)           # tier-A files
```

`script/prepare_imdb_lp.py` and `script/prepare_dblp_lp.py` also refuse raw files whose checksum differs.

## Building the MOTIF inputs

CPU only. Run from `GFM/MOTIF/`; each command refuses to overwrite an existing output.

```bash
python script/prepare_imdb_lp.py                  # -> preprocessed/IMDB/
python script/prepare_dblp_lp.py                  # -> preprocessed/DBLP/
# IMDb node classification: the lab's NC bundles, then the MOTIF export
python ../../GNN/models/RGCN/preprocess_IMDB_rgcn_structural.py --tasks nc \
  --variants v1,v2,v3,v4 --raw-csv rai-datasets/raw/IMDB/movie_metadata.csv \
  --out-root rai-datasets/preprocessed/_build/IMDB_nc
python script/nc_motif.py export --dataset imdb --bundle-root rai-datasets/preprocessed/_build/IMDB_nc/IMDB/nc
# Freebase node classification (CPU, parses the 7.9 GB exact_3 link.dat; slurm/nc_export.sbatch)
python script/nc_motif.py export --dataset freebase
```

WN18RR needs no preprocessing: `script/wordnet_lp.py` reads `data_tierA/wordnet_splits.npz` directly.
`../docs/DATA_PREPARATION.md` gives the details, including the Movie–Director export and the NC bundles.
