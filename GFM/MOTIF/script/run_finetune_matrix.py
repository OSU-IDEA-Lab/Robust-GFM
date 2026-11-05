"""Train one MOTIF per physical IMDb/DBLP variant and evaluate every train/test cell."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from lp_data import ROOT, PREPROCESSED_ROOT, TASK_VARIANTS, DBLP_TASK_VARIANTS, file_hash, write_json

DATASET_TASKS = {"imdb": TASK_VARIANTS, "dblp": DBLP_TASK_VARIANTS}


def matrix_root(output, tag, seed, dataset="imdb"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Run tag must contain only letters, digits, underscores and hyphens")
    return output / f"motif_finetune_matrix_{dataset}_lp_{tag}_s{seed}"


def record_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def run_logged(command, path, env):
    print("Running: " + " ".join(map(str, command)), flush=True)
    with path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Process failed with exit code {code}; see {path}")


def compute_contract(args, variants):
    model_source = {p.relative_to(ROOT / "upstream").as_posix(): file_hash(p) for p in sorted((ROOT / "upstream/motif").rglob("*.py"))}
    scripts = (("finetune_dblp_lp.py", "finetune_imdb_lp.py") if args.dataset == "dblp" else
               ("finetune_imdb_lp.py",)) + ("eval_lp.py", "lp_data.py", "lp_metrics.py")
    training_source = {name: file_hash(ROOT / "script" / name) for name in scripts}
    return {
        "dataset": args.dataset, "task": args.task, "variants": list(variants), "seed": args.seed,
        "initial_checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
        "export_manifest_sha256": file_hash(args.data / "manifest.json"),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "adversarial_temperature": args.adversarial_temperature, "max_grad_norm": args.max_grad_norm,
        "validate_every": args.validate_every, "val_limit": args.val_limit,
        "evaluation_batch_size": args.eval_batch_size, "evaluation_limit": args.eval_limit,
        "model_source": model_source, "training_source": training_source,
    }


def completed_training(path, variant, contract):
    summary = json.loads((path / "training_summary.json").read_text())
    trained = summary["training_contract"]
    shared = ("dataset", "task", "seed", "initial_checkpoint_sha256", "config_sha256",
              "export_manifest_sha256", "steps", "batch_size", "lr", "adversarial_temperature",
              "max_grad_norm", "validate_every", "val_limit", "model_source", "training_source")
    if any(trained[key] != contract[key] for key in shared):
        raise ValueError(f"Training settings/code differ from this matrix: {path}")
    if (trained["train_variant"] != variant or trained["train_view"] != "physical"
            or summary["completed_steps"] != contract["steps"]):
        raise ValueError(f"Incomplete or incorrect training run: {path}")
    checkpoint = path / "best.pth"
    return {"checkpoint_sha256": file_hash(checkpoint),
            "training_summary_sha256": file_hash(path / "training_summary.json")}


def completed_cell(path, train, test, checkpoint_hash, contract):
    meta = json.loads((path / "metrics.json").read_text())
    evaluated = meta["comparison_contract"]
    trained = evaluated["fine_tuning"]["training_contract"]
    adapter_source = {name: contract["training_source"][name]
                      for name in ("eval_lp.py", "lp_data.py", "lp_metrics.py")}
    if (meta["view"] != "physical" or meta["variant"] != test or meta["repeat"]
            or evaluated["evaluation_stage"] != "finetuned" or evaluated["task"] != contract["task"]
            or evaluated["dataset"] != contract["dataset"] or trained["train_variant"] != train
            or evaluated["checkpoint_sha256"] != checkpoint_hash or evaluated["seed"] != contract["seed"]
            or evaluated["batch_size"] != contract["evaluation_batch_size"]
            or evaluated["limit"] != contract["evaluation_limit"]
            or evaluated["split"] != "test"
            or evaluated["export_manifest_sha256"] != contract["export_manifest_sha256"]
            or evaluated["config_sha256"] != contract["config_sha256"]
            or evaluated["model_source"] != contract["model_source"]
            or evaluated["adapter_source"] != adapter_source):
        raise ValueError(f"Evaluation cell differs from this matrix: {path}")
    if file_hash(path / "scores.npz") != meta["scores_sha256"]:
        raise ValueError(f"Score checksum differs: {path}")
    return {"metrics_sha256": file_hash(path / "metrics.json"), "scores_sha256": meta["scores_sha256"]}


def run(args):
    variants = DATASET_TASKS[args.dataset][args.task]
    source = json.loads((args.data / "manifest.json").read_text())
    if source["format"] != f"motif_{args.dataset}_lp_v1" or args.task not in source["tasks"]:
        raise ValueError("Selected dataset/task is absent from the portable export")
    task_manifest = args.data / args.task / "manifest.json"
    if file_hash(task_manifest) != source["tasks"][args.task]:
        raise ValueError("Task manifest checksum differs from the portable export")
    task_meta = json.loads(task_manifest.read_text())
    if any(variant not in task_meta["variants"] for variant in variants):
        raise ValueError("Not all physical variants are present in the export")
    contract = compute_contract(args, variants)
    root = matrix_root(args.output, args.run_tag, args.seed, args.dataset)
    manifest_path = root / "experiment.json"
    if root.exists():
        if not args.resume:
            raise FileExistsError(f"Matrix exists: {root}; use --resume or a fresh --run-tag")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["format"] != "motif_finetune_matrix_experiment_v1"
                or manifest["run_tag"] != args.run_tag or manifest["compute_contract"] != contract):
            raise ValueError("Matrix compute settings/code changed; use a fresh --run-tag")
    else:
        root.mkdir(parents=True)
        manifest = {"format": "motif_finetune_matrix_experiment_v1", "status": "running",
                    "run_tag": args.run_tag, "compute_contract": contract, "training": {}, "evaluations": {},
                    "evaluation_roots": {v: f"motif_finetuned_{args.dataset}_lp_{args.run_tag}_train_{v}_s{args.seed}" for v in variants}}
        record_json(manifest_path, manifest)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "PYTHONUNBUFFERED": "1"}
    for train in variants:
        training_dir = (args.output / f"motif_finetune_{args.dataset}_lp_{args.run_tag}" / args.task /
                        f"train_{train}" / f"seed{args.seed}")
        checkpoint = training_dir / "best.pth"
        if training_dir.exists() and not args.resume:
            raise FileExistsError(f"Training output exists: {training_dir}; use --resume or a fresh tag")
        if not training_dir.exists():
            command = [sys.executable, str(ROOT / "script" / f"finetune_{args.dataset}_lp.py"),
                       "--task", args.task, "--train-variant", train, "--data", str(args.data),
                       "--checkpoint", str(args.checkpoint), "--config", str(args.config), "--output", str(args.output),
                       "--device", args.device, "--seed", str(args.seed), "--steps", str(args.steps),
                       "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                       "--adversarial-temperature", str(args.adversarial_temperature),
                       "--max-grad-norm", str(args.max_grad_norm), "--validate-every", str(args.validate_every),
                       "--val-limit", str(args.val_limit), "--run-tag", args.run_tag]
            run_logged(command, logs / f"train_{train}.log", env)
        try:
            training = completed_training(training_dir, train, contract)
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError(f"Cannot reuse training run {training_dir}: {error}. "
                               "Incomplete optimizer runs are not resumed; completed variants can be reused with --resume.") from error
        if train in manifest["training"] and manifest["training"][train] != training:
            raise ValueError(f"Saved training checkpoint/summary changed: {training_dir}")
        manifest["training"][train] = training
        record_json(manifest_path, manifest)
        for test in variants:
            evaluation_root = args.output / manifest["evaluation_roots"][train]
            cell_dir = evaluation_root / args.task / "physical" / test
            if cell_dir.exists() and not args.resume:
                raise FileExistsError(f"Evaluation exists: {cell_dir}; use --resume or a fresh tag")
            if not cell_dir.exists():
                command = [sys.executable, str(ROOT / "script/eval_lp.py"), "--single", "--stage", "finetuned",
                           "--dataset", args.dataset, "--task", args.task, "--view", "physical", "--variant", test,
                           "--data", str(args.data), "--output", str(args.output), "--checkpoint", str(checkpoint),
                           "--config", str(args.config), "--device", args.device, "--seed", str(args.seed),
                           "--batch-size", str(args.eval_batch_size), "--limit", str(args.eval_limit),
                           "--run-tag", f"{args.run_tag}_train_{train}_s{args.seed}"]
                run_logged(command, logs / f"train_{train}_test_{test}.log", env)
            cell = completed_cell(cell_dir, train, test, training["checkpoint_sha256"], contract)
            previous = manifest["evaluations"].setdefault(train, {}).get(test)
            if previous is not None and previous != cell:
                raise ValueError(f"Saved evaluation metadata changed: {cell_dir}")
            manifest["evaluations"][train][test] = cell
            record_json(manifest_path, manifest)
    manifest["status"] = "inference_complete"
    record_json(manifest_path, manifest)
    if not args.skip_report:
        command = [sys.executable, str(ROOT / "script/report_finetune_matrix.py"), "--matrix", str(root),
                   "--bootstrap", str(args.bootstrap), "--bootstrap-seed", str(args.bootstrap_seed)]
        run_logged(command, logs / "report.log", env)
        manifest["status"] = "complete"
        record_json(manifest_path, manifest)
    print(f"Matrix results: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_TASKS, default="imdb")
    parser.add_argument("--task", choices=tuple(TASK_VARIANTS) + tuple(DBLP_TASK_VARIANTS))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream/ckpts/motif_3g.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "upstream/config/transductive/MOTIF_inference.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--run-tag", default="matrix")
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed training/evaluation outputs")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    args.task = args.task or next(iter(DATASET_TASKS[args.dataset]))
    if args.task not in DATASET_TASKS[args.dataset]:
        parser.error("Task does not belong to the selected dataset")
    args.data = args.data or PREPROCESSED_ROOT / args.dataset.upper()
    if min(args.steps, args.batch_size, args.eval_batch_size, args.validate_every) < 1 or min(args.val_limit, args.eval_limit) < 0:
        parser.error("Require positive budgets/batch sizes and nonnegative query limits")
    if (not args.device.startswith("cuda") or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not math.isfinite(args.adversarial_temperature) or args.adversarial_temperature < 0
            or args.bootstrap < 0 or args.bootstrap == 1):
        parser.error("Invalid device, optimizer settings or bootstrap count")
    for name in ("data", "checkpoint", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()
