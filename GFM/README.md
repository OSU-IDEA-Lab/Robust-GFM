# Graph foundation models across information-equivalent representations

This folder tests whether pretrained graph foundation models (GFMs) give the same answers when the same facts are
written as different graphs: inverse edges materialized, facts reified through a node, multi-hop paths
materialized as shortcut edges. It also tests whether compiling every representation back to one canonical graph,
via the declared transformations, restores identical answers without retraining.

All GFMs share one data format, one set of representations and one comparison script. Each GFM has its own folder
with its evaluation code, scripts and README.

| folder | model | task | status |
| --- | --- | --- | --- |
| `ULTRA/` | ULTRA 3g / 4g / 50g (KG link prediction, relation-agnostic) | zero-shot LP on WN18RR, FB15k-237, Hetionet | complete; expected results in `ULTRA/README.md` |
| `TEMPLATE/` | skeleton for a new GFM | — | copy to start a new folder |

## Layout

```
common/                 GFM-agnostic code
  kg_format.py            shared on-disk format (base ids, splits, fixed negatives, variants, manifests)
  build_kg_variants.py    representations of a raw-triple KG (FB15k-237, Hetionet)
  build_wn18rr_variants.py  exports the WN18RR variants of the GNN experiments
  compile_back.py         compiled view: undoes declared transformations, verifying every removed edge
  compare_variants.py     per-variant metrics and per-pair Kendall tau / delta MRR / flip rates
scripts/
  download_raw.sh         raw FB15k-237 and Hetionet triples (sha256-checked)
  build_data.sh           every representation + its compiled view, under data/
data/                   generated; see data/README.md
docs/EVALUATION_GUIDE.md  how to evaluate a GFM: zero-shot, controls, the GNN datasets, fine-tuning
<GFM>/                  one folder per model (ULTRA/, ...)
```

## Data

```bash
bash scripts/download_raw.sh      # FB15k-237, Hetionet raw triples
bash scripts/build_data.sh        # ~15 min on CPU, ~64 GB RAM, ~0.6 GB disk
```

WN18RR is exported from the GNN data (`../GNN/data/wordnet_3hops_augmented_full/`, tier A in
`../GNN/data/README.md`), so the GFM and GNN experiments use exactly the same WN18RR variants. Set
`GFM_DATA=/path` to keep the data elsewhere.

Each dataset directory holds one `base/` (vocabularies, train / valid / test triples, filter set, 50 fixed filtered
negatives per query and direction) and one `variants/<V>/` per representation (`graph.tsv` plus its vocabularies and
a manifest with counts, hashes and the relations whose queries it makes unanswerable). Every variant keeps the base
entity and relation ids as a prefix and appends whatever it adds, so the same queries, filters and negatives apply to
every variant. `common/kg_format.py` documents the format. `build_data.sh` also writes `<V>__compiled` for every
variant; all of them are byte-identical to the compiled reference.

| dataset | representations (reference first) |
| --- | --- |
| WN18RR | `no_changes`, `all_inverse_edges`, `transitive_edges`, `universal_edges` |
| FB15k237 | `original`, `inverse`, `reified_cvt_strong`, `reified_cvt_light`, `shortcut2_c5k`, `shortcut2_c50k`, `shortcut2_c200k` |
| Hetionet | `original`, `inverse`, `reified_regulation`, `shortcut2_c50k` |

## Evaluator output contract

`common/compare_variants.py` works for any model whose evaluator writes, for each variant `V`, a directory
`<results>/<DATASET>/<CHECKPOINT>/<V>/` with:

| file | content |
| --- | --- |
| `scores.npz` | `query_index` int [Q] (rows of `base/test.tsv` evaluated), `rank_pess` int [Q, 2] and `rank_tie` float [Q, 2] (filtered ranks for tail and head prediction; ties against the answer / 1 + #better + #tied / 2), `cand_scores` float [Q, 2, 51] (scores of the true answer followed by the 50 fixed negatives of `base/negatives_test.npy`) |
| `metrics.json` | at least: checkpoint name and SHA-256, variant and base hashes from the manifests, query counts (evaluated / skipped as unanswerable), MRR and Hits@1/3/10 under both rank conventions, runtime, peak GPU memory, device |

Rules the evaluator must follow (see `docs/EVALUATION_GUIDE.md` §A for the reasons):

- rank **base entities only** (the first `len(base/entities.txt)` ids) and filter against the **base** truth set;
- skip queries whose relation is listed in the variant manifest's `unanswerable_query_relations`;
- run deterministically, and evaluate the reference twice (`<reference>__repeat`) to measure the noise floor;
- never train on validation or test triples.

Then:

```bash
python common/compare_variants.py --results <results>/<DATASET>/<CHECKPOINT> --reference <reference variant>
```

writes `comparison.{json,md}` in that directory.

## Adding a GFM

1. Copy `TEMPLATE/` to `<NAME>/` and follow its README.
2. Pin the upstream code and checkpoints with a setup script (`ULTRA/setup_upstream.sh` is an example); never edit
   the upstream checkout.
3. Reproduce one published number with the model's own code, then match it with your evaluator on the reference
   variant (`ULTRA/scripts/run_ref_check.sh`).
4. Write the evaluator so it follows the output contract above, and reuse `scripts/_common.sh` for the variant lists.
5. Document the model, its pretraining graphs (seen / held out / unknown for each dataset), resources and expected
   results in `<NAME>/README.md`, and add a row to the table at the top of this file.

A model that reads node features or text, or performs node classification, needs its own task format; the GNN
datasets (`../GNN/`) provide IMDb, DBLP and Freebase representations for that (`docs/EVALUATION_GUIDE.md` §B).
