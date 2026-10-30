# TEMPLATE: adding a graph foundation model

Copy this folder to `GFM/<NAME>/` (e.g. `GFM/GraphAny/`), then:

1. **`setup_upstream.sh`**: set `REPO`, `COMMIT` and the checkpoint files. The script clones the model's code at a
   fixed commit into `upstream/`; never edit that checkout (apply shims from your own files, as
   `ULTRA/ultra_eval.py` does).
2. **`requirements.txt`**: the exact versions you tested. Reuse the torch / PyG pins of `../ULTRA/requirements.txt`
   if you can, so one environment serves several GFMs.
3. **`evaluate.py`**: fill in the three functions marked `MODEL` (`load_model`, `prepare_graph`, `score`). The rest
   of the file (unanswerable queries, base-entity candidates, base-truth filtering, both rank conventions, the 51
   candidate scores, `scores.npz` and `metrics.json`) implements the output contract of `../README.md`; leave it
   unchanged so results are comparable across GFMs.
4. **Reference check**: reproduce one published number with the model's own evaluation script, then match it with
   `evaluate.py` on the reference variant before running anything else (`../ULTRA/scripts/run_ref_check.sh`).
5. **`scripts/run_zero_shot.sh <DATASET> <CHECKPOINT>`** runs every representation plus the repeat of the
   reference and calls `../common/compare_variants.py`. Start with `LIMIT=64`.
6. **This README**: replace it with the model's description, following `../ULTRA/README.md`:
   - what the model reads (structure only / features / text; does it add inverse edges itself?), which decides
     which transformations it can see (`../docs/EVALUATION_GUIDE.md` §A3);
   - each checkpoint's pretraining graphs and, for every dataset, whether it is seen, documented held out or unknown;
   - setup, resources (GPU memory, time per dataset), expected results and the reference check.
7. Add a row for the model to the table at the top of `../README.md`.

A model for node classification or one that reads node features or text needs a different task format; see
`../docs/EVALUATION_GUIDE.md` §B for the GNN datasets (IMDb, DBLP, Freebase) and how to pose their tasks.
