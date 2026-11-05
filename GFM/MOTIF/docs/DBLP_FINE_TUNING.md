# DBLP MOTIF fine-tuning on the HPC

Train three independent checkpoints from `upstream/ckpts/motif_3g.pth`, one on each
physical DBLP variant v1-v3. Evaluate each selected checkpoint on all three
test graphs: **three training runs and nine evaluations**. Reuse the existing
preprocessed DBLP export at `rai-datasets/preprocessed/DBLP/`.

Run from the MOTIF root in the same GPU environment as the successful IMDb run.
No new dependencies are required. The DBLP training path has not been executed
locally; run the short all-variant smoke on the HPC first.

## Training regimes

Each training run starts afresh from the released checkpoint and trains on one
variant only. It does not continue training the previous variant's model.

| Train / Test | v1 | v2 | v3 |
|---|---|---|---|
| v1 | C-a | C-b | C-b |
| v2 | C-b | C-a | C-b |
| v3 | C-b | C-b | C-a |

C-a is matched-variant evaluation; C-b is transfer to another representation.
One model trained on a mixture of all variants would be C-c augmentation, which
is outside this experiment. C-a/C-b share the same performance tables, with
explicit Train/Test columns.

## Protocol

* Paper-Conference uses the same **20 conference entities** for every metric.
  There is no additional node-classification model or larger candidate pool.
* Train on the 9,448 shared training papers and all 19 alternative conferences.
  Validation MRR on the 1,348 validation queries selects each checkpoint.
  Test evaluation uses all 2,666 shared test papers. Test labels are not used
  for training or checkpoint selection.
* Use the same initial weights, seed, keyed batch order, step budget, loss,
  optimizer settings and batch size across training variants. Defaults match
  the IMDb pilot: Adam, learning rate `5e-4`, batch size 4, 1,000 steps and
  validation every 100 steps. These are starting settings, not paper-tuned DBLP
  hyperparameters; 1,000 steps does not traverse all training papers at batch 4.
* Remove each batch's target Paper-Conference edges and their internal inverses
  before rebuilding Lift. Area context is supported by training papers: Paper-Area
  in v1, Conference-Area in v2, and Author-Area in v3. Withdraw the batch papers'
  support too. Remove an Area edge and its inverse if no other training paper
  still supports it; retain shared evidence from other papers. The trainer
  reconstructs and audits this support using the existing v1 export and shared
  author/paper links, so no new raw-data preprocessing is required.
* The trainer records direct-target and Area-context removals separately in
  `history.csv` and `training_summary.json`.

## 1. All-variant smoke

After syncing the updated scripts to the HPC:

```bash
cd /nfs/hpc/share/tinw/idealab/MOTIF
source .venv/bin/activate
mkdir -p logs
set -o pipefail

python -u script/run_finetune_matrix.py \
  --dataset dblp --task paper_conference \
  --steps 2 --validate-every 2 --val-limit 16 --eval-limit 64 \
  --run-tag smoke \
  2>&1 | tee logs/dblp_finetune_smoke.log
```

This checks the new training/removal path on all three variants and evaluates
64 test queries per cell. Its tables are explicitly labelled as limited-query
debugging results. It is not the full experiment.

## 2. Full pilot

After the smoke completes, start each model again from the released checkpoint:

```bash
python -u script/run_finetune_matrix.py \
  --dataset dblp --task paper_conference \
  --steps 1000 --validate-every 100 \
  --run-tag matrix \
  2>&1 | tee logs/dblp_finetune_matrix.log
```

Training, all nine evaluations and table generation run sequentially. To reuse
verified completed training runs and evaluation cells after a stop, rerun the
same command with `--resume`. Incomplete optimizer runs are not restored; an
incomplete training directory is reported rather than silently reused. A new
run tag starts a fresh experiment.

Outputs:

```text
results/motif_finetune_dblp_lp_matrix/paper_conference/train_vN/seed1024/
results/motif_finetuned_dblp_lp_matrix_train_vN_s1024/paper_conference/physical/vM/
results/motif_finetune_matrix_dblp_lp_matrix_s1024/tables/
```

## 3. Three presentation tables

The combined report generates exactly these three tables:

1. `conference_prediction.tex`: nine Train/Test rows with macro Precision,
   macro Recall and Macro-F1 over all 20 conferences. Accuracy and Micro-F1 are
   omitted because they equal Hits@1 in the ranking table.
2. `link_prediction.tex`: the same nine rows with MRR and Hits@1/3/10, also over
   those same 20 conferences. Hits@1 and Accuracy coincide when scores are untied;
   MRR/Hits use average-rank ties, while top-1 prediction uses the smallest global ID.
3. `invariance.tex`: C-a comparisons between the three independently trained
   diagonal models, plus C-b comparisons between test variants with each trained
   checkpoint fixed. Regime and Train columns distinguish them. Metrics are
   Agreement@1, Kendall tau-b and Delta MRR (second variant minus first).

For C-a pair comparisons, both weights and graphs vary. For C-b groups, the
checkpoint stays fixed and its matched-variant evaluation is the reference.
These are different questions, even when shown in one table.

The pilot uses training seed 1024. Error bars are query-bootstrap SE, not
across-training-seed SD. The guide's training-repeat and multi-seed noise-floor
study remain pending. Cost and edge-removal evidence appear as method notes.

To regenerate reports on the HPC without retraining/inference, while the score
arrays and checkpoints are available:

```bash
python script/report_finetune_matrix.py \
  --matrix results/motif_finetune_matrix_dblp_lp_matrix_s1024
```

## 4. Overleaf

After the full report exists:

```bash
python script/export_overleaf.py \
  --run results/motif_finetune_matrix_dblp_lp_matrix_s1024 \
  --output results/overleaf/motif_dblp_finetuned_s1024
```

Upload the ZIP and compile `main.tex` on Overleaf. Add IMDb/DBLP zero-shot and
IMDb fine-tuning result roots as additional `--run` arguments to combine them
in a new project. No model or score-array loading is needed to export saved tables.

