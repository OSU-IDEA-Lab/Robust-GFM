# Slurm for this project: sbatch, monitoring, and how our scripts work

This guide is for someone who already uses `srun` interactively (load modules, activate a venv,
run Python by hand) and wants to use `sbatch`. It is specific to the OSU CoE HPC
(account `eecs`; partitions `dgx2` (V100), `dgxh` (H100/H200), and others). See
https://it.engineering.oregonstate.edu/hpc/slurm-howto.

---

## 1. `srun` vs `sbatch`: the mental model

| | `srun --pty bash` (what you did) | `sbatch script.sbatch` (what we do now) |
|---|---|---|
| What you get | an interactive shell on a GPU node | a job that runs a script with nobody watching |
| Who types the commands | you, live | the script: the same commands, written down once |
| If your SSH/VPN drops | the job dies | **keeps running** |
| Where output goes | your terminal | a log file (`#SBATCH -o ...`) |
| Running many at once | one terminal per job | one command submits 3, 9, 100 jobs (arrays) |
| Reproducibility | whatever you happened to type | the script *is* the record |

**An sbatch script is a shell script that contains what you'd type after `srun --pty bash`.**
The `module load ...`, `source .venv/bin/activate` and `python ...` lines go into a file.
The resource request (`-A eecs -p dgxh --gres=gpu:1 ...`) moves from the `srun` command line
into `#SBATCH` comment lines at the top of that file.

Your usual interactive command:
```bash
srun -A eecs -p dgxh -c 2 -n 2 --gres=gpu:1 --constraint=h100 --mem=64G -t 1-00:00:00 \
     --mail-type=BEGIN --mail-user=tinw@oregonstate.edu --pty bash
# ...then by hand:
module load python/3.10 cuda/12.1
source .venv/bin/activate
python train.py --seed 1
```
The same thing as a batch job (`job.sbatch`):
```bash
#!/bin/bash
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x-%j.out          # %x = job name, %j = job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

module load python/3.10 cuda/12.1
source .venv/bin/activate
python train.py --seed 1
```
Submit with `sbatch job.sbatch`. It prints a job id and returns immediately. Slurm queues the
job, starts it on a node when resources are free, runs the script there, and writes everything
it prints to the log file.

> `#SBATCH` lines are comments to bash but instructions to Slurm. They must come **before** the
> first real command. Anything you can pass to `srun` you can put there.

---

## 2. How this project's scripts are organised

```
slurm/
  env.sh                 # shared setup: modules, venv, CUDA/Triton caches (sourced by every job)
  finetune_seeds.sbatch  # IMDb/DBLP fine-tuning matrix, one array task per lab seed
  zeroshot.sbatch        # IMDb/DBLP frozen zero-shot evaluation
  wordnet.sbatch         # WN18RR zero-shot or fine-tuning (MODE=eval / MODE=finetune)
  status.sh              # progress overview (section 4)
```

`slurm/env.sh` is the "what I used to type after srun" part, written once:
```bash
module load python/3.10 gcc/12.2 cuda/12.1
cd "$SLURM_SUBMIT_DIR"            # the directory you ran sbatch from (the repo root)
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=...    # detected from the GPU the job landed on
export TORCH_EXTENSIONS_DIR=...    # one compiled-extension cache per GPU type
...
```
Each `.sbatch` file then just runs `source slurm/env.sh` followed by the Python command.
**Python code is not wrapped or changed for Slurm.** The scripts in `script/` are run exactly
as you would run them by hand.

### Passing settings into a job
Two mechanisms, both used here:

1. **Override `#SBATCH` options on the command line.** The command line wins over the file:
   ```bash
   sbatch -p dgx2 --constraint=v100 -t 2:00:00 slurm/finetune_seeds.sbatch
   ```
2. **Environment variables via `--export`.** The script reads them:
   ```bash
   sbatch --export=ALL,DATASET=dblp,RUN_TAG=seeds slurm/finetune_seeds.sbatch
   ```
   `ALL` keeps your normal environment and adds the listed variables. Values **cannot contain
   spaces or commas**. For something like `ARGS="--steps 2 --val-limit 16"`, set it in front
   of `sbatch` instead:
   ```bash
   ARGS="--steps 2 --val-limit 16" sbatch --export=ALL,MODE=finetune slurm/wordnet.sbatch
   ```

### Job arrays: one script, many runs
`#SBATCH --array=0-2` submits 3 copies of the same script. Each copy gets its own
`$SLURM_ARRAY_TASK_ID` (0, 1, 2), and the script uses it to pick a seed:
```bash
SEEDS=(1566911444 20241017 20251017)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
The copies run in parallel when GPUs are free, each on **one GPU**. Logs are
`logs/<name>-<arrayjobid>_<taskid>.out`. `wordnet.sbatch --array=0-8` maps task `i` to
variant `i/3` and seed `i%3` (9 runs). You can submit a subset with `--array=6-8`.

### Dependencies: run B only if A succeeded
```bash
SMOKE=$(sbatch --parsable ... --export=ALL,SMOKE=1 slurm/finetune_seeds.sbatch)   # --parsable prints just the id
sbatch --dependency=afterok:$SMOKE slurm/finetune_seeds.sbatch                    # waits for SMOKE to succeed
```
If the smoke test fails, the dependent job shows `(DependencyNeverSatisfied)` and never starts.
Cancel it with `scancel`.

---

## 3. A job's life cycle and states

`sbatch` → **PENDING** (in the queue) → **RUNNING** (on a node) → **COMPLETED** / **FAILED** /
**CANCELLED** / **TIMEOUT** / **OUT_OF_MEMORY**.

Common PENDING reasons (last column of `squeue`):

| Reason | Meaning |
|---|---|
| `Priority` | other jobs are ahead of you; wait |
| `Resources` | you are next, but no node has the requested GPU/CPU/RAM free yet |
| `Dependency` | waiting for the job named in `--dependency` |
| `DependencyNeverSatisfied` | that job failed; this one will never start. Cancel it. |
| `MaxCpuRunMinsPerUser` (or similar `QOS...` limits) | your running jobs together hit the per-user cap of CPUs × remaining time. Lower `-t` (time limit) or `-c` (CPUs). |

---

## 4. Checking progress

### The quick way (this project)
```bash
bash slurm/status.sh             # all my jobs
bash slurm/status.sh 21955721    # one job, every array task
```
It prints three things:
* the queue;
* the last 24 h of finished and running jobs with their state and elapsed time;
* the **latest progress line of each log**: `Step 300/1000 ...` while training,
  `Saved ...: MRR=...` when an evaluation is done, or a `Traceback`/`Error` if it failed.

### The underlying commands
| Command | What it shows |
|---|---|
| `squeue -u $USER` (or the cluster's `sq`) | your pending and running jobs, time used, node, or why they are pending |
| `sacct -j <id> -X -o JobID,State,Elapsed,NodeList` | state and runtime of finished jobs too (squeue forgets them) |
| `showjob <id>` / `scontrol show job <id>` | everything about one job: resources, node, limits, paths, dependency |
| `seff <id>` | after a job ends: how much of the requested CPU and memory it actually used |
| `tail -f logs/<name>-<id>.out` | watch the job's output live (Ctrl-C stops watching, not the job) |
| `grep Step logs/motif-wn-ft-21955721_8.out \| tail -1` | the latest training step only |
| `nodestat dgxh` | free GPUs/CPUs/RAM on a partition's nodes, to guess what starts soon |
| `srun --jobid=<id> --overlap nvidia-smi` | run a command *inside* a running job's allocation, e.g. check GPU use and memory |

"How far along is it?" therefore comes from the **log**. The scripts print
`Step k/1000` with elapsed seconds, so remaining time is about
`(1000 − k) × seconds per step`, plus the final test evaluation.

---

## 5. Changing or stopping jobs

```bash
scancel 21955721           # cancel a job (all array tasks)
scancel 21955721_8         # cancel one array task
scancel -u $USER           # cancel everything you own (careful)
scontrol update jobid=21955721 TimeLimit=03:00:00    # lower a limit (raising usually needs admin)
scontrol update jobid=21955721 Features=h200          # change --constraint while it is still PENDING
scontrol hold 21955721 / scontrol release 21955721    # pause or resume a pending job
```

---

## 6. Pitfalls we ran into (and the rules that came out of them)

1. **Don't edit code while jobs that use it are running.** Slurm copies the `.sbatch` file at
   submission, but **not** the Python files. Jobs read `script/*.py` from disk when each
   step starts. Our fine-tuning runner records a hash of the source files and refused to mix
   evaluations from edited code, which is how a mid-run edit was caught (Movie–Director rerun
   as `md_seeds2`). Editing an `.sbatch` file after submitting is safe.
2. **Paths are relative to where you ran `sbatch`.** Submit from the repo root. `env.sh` does
   `cd "$SLURM_SUBMIT_DIR"`, and `logs/` must exist before the job starts.
3. **Ask for realistic time limits.** A 2-day limit × 8 CPUs × many array tasks hits the
   per-user CPU-minutes cap, and tasks queue one at a time. Our IMDb/DBLP runs take under
   20 min, WN18RR v1/v2 about 5 min, and WN18RR v3 fine-tuning about 1 h 20 min.
4. **Pin the GPU type for everything that goes in one table.** `--constraint=v100`, `h100`
   or `h200` (`vram80g` matches both H100 80 GB and H200). `env.sh` detects the GPU and keeps
   a separate compiled-extension cache per type. The first job on a new GPU type spends a few
   minutes compiling.
5. **One GPU per run.** `--gres=gpu:1`, and the code never uses a second device. Multi-GPU
   would change the batching and numerics compared with the single-GPU baselines.
6. **Smoke-test first.** A 2-step or 16-query run (`SMOKE=1`, or `ARGS="--limit 16"`) catches
   crashes and out-of-memory errors in seconds, before a multi-hour array starts. Chain the
   real run with `--dependency=afterok`.

---

## 7. Cheat sheet for this repo

```bash
# IMDb Movie-Year / DBLP fine-tuning, 3 seeds on V100
sbatch -p dgx2 --constraint=v100 slurm/finetune_seeds.sbatch
sbatch -p dgx2 --constraint=v100 --export=ALL,DATASET=dblp slurm/finetune_seeds.sbatch

# IMDb Movie-Director (task + data override)
MD=DATASET=imdb,TASK=movie_director,DATA=rai-datasets/preprocessed/IMDB_movie_director
sbatch --export=ALL,$MD,RUN_TAG=md_full slurm/zeroshot.sbatch
sbatch -p dgx2 --constraint=v100 --export=ALL,$MD,RUN_TAG=md_seeds2 slurm/finetune_seeds.sbatch

# WN18RR on one H200: zero-shot (3 variants), fine-tuning (3 variants x 3 seeds)
sbatch -p dgxh --constraint=h200 --array=0-2 --export=ALL,MODE=eval,RUN_TAG=full slurm/wordnet.sbatch
sbatch -p dgxh --constraint=h200 --array=0-8 --export=ALL,MODE=finetune,RUN_TAG=seeds slurm/wordnet.sbatch

# Progress
bash slurm/status.sh

# Tables (CPU, seconds; run on the submit node after jobs finish), see docs/SEED_TABLES.md
python script/report_seeds.py --dataset wordnet --zeroshot ... --finetune ... --baseline-tex ...
```

### Interactive debugging is still `srun`
For poking around (testing an import, checking memory), your old workflow is still the right
tool:
```bash
srun -A eecs -p dgx2 --constraint=v100 --gres=gpu:1 -c 4 --mem=32G -t 1:00:00 --pty bash
source slurm/env.sh      # same environment the batch jobs use (set SLURM_SUBMIT_DIR=$PWD first if needed)
python script/wordnet_lp.py eval --variant v1 --limit 16 --run-tag debug
```
