# IMDb MOTIF fine-tuning on the HPC

Train four checkpoints independently from `upstream/ckpts/motif_3g.pth`, one on each
physical Movie-Year variant v1-v4. Evaluate each selected checkpoint on all four
test graphs: **four training runs and 16 evaluations**. Reuse the existing IMDb
preprocessing and keep the zero-shot results for comparison.

Run from the MOTIF repository root, inside a GPU allocation, with the modules,
activated venv and exports of `slurm/env.sh`. No new dependencies are required.
The matrix runner and reporter have not been executed locally; runtime validation
is on the HPC.

## Protocol and what the tables mean

This follows section C of
`../docs/EVALUATION_GUIDE.md`:

* **C-a, matched variant:** train v1/test v1, train v2/test v2, and so on.
  These are the four diagonal cells, the per-variant adaptation baseline.
* **C-b, cross-representation transfer:** train v1/test v2, train v1/test v3,
  and the other off-diagonal cells. A row holds the trained checkpoint fixed.
* Each training run starts from the same released weights, with the same seed,
  optimizer-step budget, batch size, shared targets, fixed negatives and keyed
  batch order. Training one variant never initializes the next variant's model.
* Training uses the 2,925 shared Movie-Year positives and their 19 fixed negative
  years. Native MOTIF removes batch target edges and their inverses, then rebuilds
  Lift. Context relations are not extra training targets.
* Validation MRR on all 418 validation queries selects each checkpoint. Test
  labels are not used for training or checkpoint selection. Step-zero validation
  is recorded separately; validation ties retain the earlier trained checkpoint.
* Each evaluation uses all 834 shared test queries and the original 20 candidates:
  the correct year and the same 19 fixed incorrect years. All presentation metrics
  use this candidate protocol, following guide B2.

The pilot defaults are Adam, learning rate `5e-4`, batch size 4, gradient-norm
clipping at 1 and 1,000 optimizer steps **per variant**. All existing trainable
model parameters are updated; static positional encodings stay frozen as defined
by MOTIF. The loss is MOTIF's binary cross-entropy with detached adversarial
negative weights. These are starting settings, not established IMDb
hyperparameters or a reproduced paper configuration.

The experiment uses one training seed, 1024. Values reported as `estimate ± SE`
use query-bootstrap standard errors, not variation across training seeds. A later
study should use at least three training seeds and report across-seed mean/SD
separately, following C4/C6. Universal, compiled and augmented training arms are
outside this experiment.

## 1. Training smoke, already completed if the v1 smoke succeeded

The existing two-update v1 command checks the backward path and parameter updates:

```bash
mkdir -p logs
set -o pipefail

python -u script/finetune_imdb_lp.py \
  --task movie_year --train-variant v1 \
  --steps 2 --validate-every 2 --val-limit 16 --run-tag smoke \
  2>&1 | tee logs/imdb_movie_year_finetune_smoke.log
```

If that run succeeded, proceed to the matrix command; it need not be repeated.
The full run starts from the released checkpoint, not the smoke weights.
Training retains the lazy native-extension import used by the working Triton path.

## 2. Train all variants and evaluate the full matrix

Sync the new scripts to the HPC first. Then run:

```bash
mkdir -p logs
set -o pipefail

python -u script/run_finetune_matrix.py \
  --task movie_year --steps 1000 --validate-every 100 \
  --run-tag matrix \
  2>&1 | tee logs/imdb_movie_year_finetune_matrix.log
```

For each training variant, the runner trains from the released checkpoint,
selects its best validation checkpoint and evaluates it on v1-v4. After all
16 evaluations finish, it generates the matrix reports and LaTeX tables.
No preprocessing or zero-shot rerun is needed.

Optional controls:

* `--batch-size 4` sets training batch size; `--eval-batch-size 4` sets test
  batch size. Keep these fixed across the experiment.
* `--seed 2048 --run-tag matrix` starts a separate seed's matrix. This does not
  automatically aggregate multiple training seeds into one table.
* `--skip-report` finishes training/inference without bootstrapping tables;
  use the report command below later.
* `--val-limit` and `--eval-limit` are for debugging. Leave both at their default
  zero to use all validation and test queries in the full experiment.

To continue with **verified completed runs and evaluation cells** after a stop:

```bash
python -u script/run_finetune_matrix.py \
  --task movie_year --steps 1000 --validate-every 100 \
  --run-tag matrix --resume \
  2>&1 | tee logs/imdb_movie_year_finetune_matrix_resume.log
```

Use the same compute settings and source code. `--resume` skips completed work;
it does not restore optimizer state from an interrupted training run. An
incomplete training directory is reported as an error, rather than treated as
a completed checkpoint. A fresh tag starts an entirely new experiment.

## 3. Results and table generation

Training outputs, for each `vN`:

```text
results/motif_finetune_imdb_lp_matrix/movie_year/train_vN/seed1024/
  best.pth
  training_config.json
  training_summary.json
  history.csv
```

Evaluation outputs, for each training variant `vN` and test variant `vM`:

```text
results/motif_finetuned_imdb_lp_matrix_train_vN_s1024/movie_year/physical/vM/
  metrics.json
  scores.npz
```

Combined experiment and reports:

```text
results/motif_finetune_matrix_imdb_lp_matrix_s1024/
  experiment.json
  logs/
  tables/
    summary.json
    report.md
    performance.csv
    comparisons.csv
    transfer_performance.tex
    invariance.tex
```

The runner generates these automatically. To regenerate tables without running
training or inference, use this on the HPC while the score arrays and checkpoints
are still available:

```bash
python script/report_finetune_matrix.py \
  --matrix results/motif_finetune_matrix_imdb_lp_matrix_s1024
```

The presentation tables are:

* `transfer_performance.tex`: 16 rows, one per Fine-tuned on/Test graph combination,
  with MRR and Hits@1/3/10 columns, matching the zero-shot table. Four matched pairs are C-a; the other 12 are C-b.
  A row such as v1/v1 evaluates the v1-trained checkpoint on the v1 graph.
* `invariance.tex`: one table with Regime, Checkpoint trained on and Test graphs
  compared labels. C-a contains the six
  pairs of separately trained diagonal models; C-b contains all six test-variant
  pairs for each of the four fixed checkpoints, including the matched reference.
  Columns are Agreement@1, Kendall tau-b and Delta MRR (second variant minus
  first), all over the original 20 candidates. C-a changes both weights and
  graphs; each C-b group holds weights fixed.

The 16 performance evaluations produce outputs that can be compared in different
pairs. The invariance table has 30 comparisons: six C-a pairs and six C-b pairs
for each of four checkpoints. For example, C-b checkpoint v1, test graphs v1 vs
v2 compares the v1-trained checkpoint's outputs on those two graphs. C-a v1 vs
v2 compares the v1-trained/v1-tested outputs with the v2-trained/v2-tested
outputs. The guide requests a train/test transfer matrix and cross-variant tau;
it does not prescribe 16 invariance rows. Comparing an output with itself gives
trivial agreement/tau of one and is omitted.

These are the only two fine-tuning presentation tables. Full-year prediction
tables and duplicate diagonal/all-cell views are not generated. Historical
numerical records remain available in the saved summaries. Training time, peak
allocated GPU memory and edge-removal totals are reported in the export's method
notes when recorded metadata are available.

The original single-checkpoint reporter still requires identical weights within
each evaluation row. The matrix reporter explicitly handles comparisons across
independently trained models and checks that initialization, targets, batch
order and budgets agree. Presentation tables omit compiled and repeat controls.

## 4. Overleaf

Package the combined saved report:

```bash
python script/export_overleaf.py \
  --run results/motif_finetune_matrix_imdb_lp_matrix_s1024 \
  --output results/overleaf/motif_imdb_finetune_matrix_s1024
```

LP tables are always included; the old `--include-lp` flag is accepted for
compatibility. Upload the generated ZIP to Overleaf and compile `main.tex` there.
The exporter uses saved summaries, so it can also run locally after pulling the text reports;
raw score arrays and checkpoints can remain on the HPC.

You can combine this matrix with IMDb and DBLP zero-shot reports in the same
project. Fine-tuning tables use a
separate `matrix/` subdirectory and distinct LaTeX labels.

Movie-Director can later use `--task movie_director` and its v1/v3 family, producing
a 2-by-2 experiment. Choose its own run tag, for example `director_matrix`.
