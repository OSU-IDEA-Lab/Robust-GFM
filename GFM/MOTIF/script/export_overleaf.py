"""Package existing result tables for Overleaf without loading scores or models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"imdb": "IMDb", "dblp": "DBLP"}
TASKS = {"movie_year": "Movie--Year", "movie_director": "Movie--Director",
         "paper_conference": "Paper--Conference"}
# Accuracy and Micro-F1 equal Hits@1 when all 20 conferences are ranked, so only macro metrics add information.
CLASS_COLUMNS = (("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
                 ("macro_f1", "Macro-F1"))
CLASS_NOTE = ("Top-1 conference prediction; precision, recall and F1 are macro-averaged over all 20 conferences. "
              "Accuracy and Micro-F1 equal Hits@1 in the ranking table and are omitted. "
              "Top-score ties use the smallest global conference ID. ")
PAIR_COLUMNS = (("prediction_agreement", "Agreement@1"), ("tau_b", r"Kendall-$\tau_b$"),
                ("delta_mrr", r"$\Delta$MRR"))
LP_COLUMNS = (("mrr", "MRR"), ("hits@1", "Hits@1"), ("hits@3", "Hits@3"), ("hits@10", "Hits@10"))
LEGACY_MATRIX_TABLES = ("classification_diagonal.tex", "classification_all.tex", "transfer_accuracy.tex",
                        "transfer_macro_f1.tex", "per_variant_comparisons.tex", "transfer_comparisons.tex",
                        "link_prediction_diagonal.tex", "link_prediction_all.tex", "transfer_mrr.tex")


def latex_escape(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def metric_cell(value):
    estimate, error = value["estimate"], value["bootstrap_se"]
    digits = 6 if any(number is not None and 0 < abs(number) < 0.00005
                      for number in (estimate, error)) else 4
    if error is None:
        return f"${estimate:.{digits}f}$"
    return f"${estimate:.{digits}f} \\pm {error:.{digits}f}$"


def grid_table(headers, rows, caption, label):
    row_end = " " + chr(92) * 2
    lines = ["% Requires adjustbox.", r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\renewcommand{\arraystretch}{1.15}", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{adjustbox}{max width=\linewidth}",
             "\\begin{tabular}{|" + "|".join("l" for _ in headers) + "|}", r"\hline",
             " & ".join(r"\textbf{" + name + "}" for name in headers) + row_end, r"\hline"]
    for row in rows:
        lines.extend([" & ".join(row) + row_end, r"\hline"])
    lines.extend([r"\end{tabular}", r"\end{adjustbox}",
                  "\\caption{" + caption + "}", "\\label{" + label + "}", r"\end{table}", ""])
    return "\n".join(lines)


def presentation_table(record, dataset, task, kind, bootstrap):
    physical = sorted((name.split("/", 1)[1], result)
                      for name, result in record["performance"].items() if name.startswith("physical/"))
    if not physical:
        raise ValueError(f"No physical variants in {dataset}/{task}")
    contract = record["comparison_contract"]
    stage = contract.get("evaluation_stage", "zeroshot")
    method = latex_escape(contract["model"] + (" (FT)" if stage == "finetuned" else ""))
    dataset_name = DATASETS[dataset]
    task_name = TASKS[task]
    queries = physical[0][1]["num_queries"]
    label = f"tab:motif-{dataset}-{task.replace('_', '-')}-{kind}"
    setting = "fine-tuned" if stage == "finetuned" else "zero-shot"
    context = f"Preliminary {setting} MOTIF results on {dataset_name} {task_name}; {queries:,} {contract['split']} queries. "
    if stage == "finetuned":
        trained = contract["fine_tuning"]["training_contract"]
        context += f"Fine-tuned on variant {latex_escape(trained['train_variant'])}; checkpoint selected by validation MRR. "
    uncertainty = (f"Values are estimates $\\pm$ bootstrap standard error ({bootstrap:,} query resamples)."
                   if bootstrap else "Values are point estimates.")
    if kind == "invariance":
        headers = ["Method", "Comparison"] + [name for _, name in PAIR_COLUMNS]
        rows = [[method, latex_escape(name)] + [metric_cell(result["metrics"][key]) for key, _ in PAIR_COLUMNS]
                for name, result in sorted(record["physical_pairs"].items())]
        caption = (context + "The checkpoint is fixed across variants. Both rankings use the same 20 candidates per query. "
                   "Agreement@1 compares predictions; Kendall tau-b compares rankings. "
                   "Delta MRR is the second variant minus the first. " + uncertainty)
    elif kind == "link_prediction":
        headers = ["Dataset", "Method", "Variant"] + [name for _, name in LP_COLUMNS]
        rows = [[dataset_name, method, latex_escape(variant)] +
                [metric_cell(result["metrics"][key]) for key, _ in LP_COLUMNS] for variant, result in physical]
        protocol = "Tail-only link prediction over the original 20 candidates; score ties use average rank. "
        caption = context + protocol + uncertainty
    elif kind == "conference_prediction":
        if any(result["classification_classes"] != 20 or result["benchmark_candidates"] != 20 for _, result in physical):
            raise ValueError("Conference prediction requires the same 20 conferences for prediction and ranking")
        headers = ["Dataset", "Method", "Variant"] + [name for _, name in CLASS_COLUMNS]
        rows = [[dataset_name, method, latex_escape(variant)] +
                [metric_cell(result["metrics"][key]) for key, _ in CLASS_COLUMNS] for variant, result in physical]
        caption = context + CLASS_NOTE + uncertainty
    else:
        raise ValueError("Unsupported presentation table; full-type classification is not reported")
    return grid_table(headers, rows, caption, label)


def matrix_tables(summary, include_lp=True):
    """Two IMDb tables or three DBLP tables; include_lp is a compatibility flag."""
    if summary["dataset"] == "dblp":
        return dblp_matrix_tables(summary)
    variants = summary["variants"]
    task, dataset = summary["task"], summary["dataset"]
    first = summary["cells"][variants[0]][variants[0]]
    context = (f"Preliminary fine-tuned MOTIF on {DATASETS[dataset]} {TASKS[task]}; "
               f"seed {summary['seed']}, {first['num_queries']:,} {summary['split']} queries. ")
    trained = summary["shared_training_contract"]
    context += f"Each variant uses {trained['steps']:,} optimizer steps; validation MRR selects its checkpoint. "
    if trained["val_limit"] or summary["shared_evaluation_contract"]["limit"]:
        context += "Limited-query debugging run. "
    uncertainty = (f"Values are estimates $\\pm$ query-bootstrap standard error "
                   f"({summary['bootstrap']:,} resamples), not variation across training seeds."
                   if summary["bootstrap"] else "Values are point estimates.")
    prefix = f"tab:motif-{dataset}-{task.replace('_', '-')}-matrix"
    protocol = ("Each query ranks the correct year against the same 19 fixed negative years. " if task == "movie_year" else
                "Each query ranks the correct target against the same 19 fixed negative targets. ")
    rows = [["C-a" if train == test else "C-b", train, test] +
            [metric_cell(summary["cells"][train][test]["metrics"][key]) for key, _ in LP_COLUMNS]
            for train in variants for test in variants]
    performance = grid_table(
        ["Regime", "Fine-tuned on", "Test graph"] + [label for _, label in LP_COLUMNS], rows,
        "Prediction performance and cross-representation transfer. " + context + protocol +
        "Each row evaluates one trained checkpoint on one test graph. "
        "Matching train/test variants give matched-variant evaluation (C-a); "
        "other cells give cross-representation transfer (C-b). Score ties use average rank. " + uncertainty,
        prefix + "-transfer-performance")
    invariance = grid_table(
        ["Regime", "Checkpoint trained on", "Test graphs compared"] + [name for _, name in PAIR_COLUMNS], matrix_pair_rows(summary),
        context + protocol + "C-a compares separately trained models, each on its own variant; weights and graphs both change. "
        "C-b groups hold the selected training checkpoint fixed, including its matched-variant reference. "
        "Agreement@1 compares predictions; Kendall tau-b compares within-query rankings. "
        "Delta MRR is the second test variant minus the first. " + uncertainty,
        prefix + "-invariance")
    return {"transfer_performance": performance, "invariance": invariance}


def matrix_pair_rows(summary):
    """Label comparisons of independently trained models separately from fixed weights."""
    rows = [["C-a", latex_escape(pair.replace(" vs ", " and ") + " (separate)"), latex_escape(pair)] +
            [metric_cell(result["metrics"][key]) for key, _ in PAIR_COLUMNS]
            for pair, result in sorted(summary["diagonal_pairs"].items())]
    rows.extend(["C-b", train, latex_escape(pair)] +
                [metric_cell(result["metrics"][key]) for key, _ in PAIR_COLUMNS]
                for train in summary["variants"]
                for pair, result in sorted(summary["within_checkpoint_pairs"][train].items()))
    return rows


def dblp_matrix_tables(summary):
    """All metrics share the same 20 conference entities; no extra NC task."""
    variants = summary["variants"]
    if any(cell["classification_classes"] != 20 or cell["benchmark_candidates"] != 20
           for row in summary["cells"].values() for cell in row.values()):
        raise ValueError("DBLP tables require the same 20 conferences for prediction and ranking")
    first = summary["cells"][variants[0]][variants[0]]
    trained = summary["shared_training_contract"]
    context = (f"Preliminary fine-tuned MOTIF on DBLP Paper--Conference; seed {summary['seed']}, "
               f"{first['num_queries']:,} {summary['split']} queries; {trained['steps']:,} optimizer steps per variant. ")
    if trained["val_limit"] or summary["shared_evaluation_contract"]["limit"]:
        context += "Limited-query debugging run. "
    uncertainty = (f"Values are estimates $\\pm$ query-bootstrap SE ({summary['bootstrap']:,} resamples), "
                   "not variation across training seeds." if summary["bootstrap"] else "Values are point estimates.")
    protocol = ("All metrics use the same 20 conferences. C-a means matching train/test variants; "
                "C-b means different train/test variants. ")
    files = {}
    for name, columns in (("conference_prediction", CLASS_COLUMNS), ("link_prediction", LP_COLUMNS)):
        rows = [["C-a" if train == test else "C-b", train, test] +
                [metric_cell(summary["cells"][train][test]["metrics"][key]) for key, _ in columns]
                for train in variants for test in variants]
        notes = (CLASS_NOTE if name == "conference_prediction" else
                 "MRR and Hits use average ranks for score ties. Hits@1 equals top-1 accuracy when scores are untied. ")
        files[name] = grid_table(["Regime", "Train", "Test"] + [label for _, label in columns], rows,
                                 context + protocol + notes + uncertainty, "tab:motif-dblp-matrix-" + name.replace("_", "-"))
    files["invariance"] = grid_table(
        ["Regime", "Checkpoint trained on", "Test graphs compared"] + [label for _, label in PAIR_COLUMNS], matrix_pair_rows(summary),
        context + "All comparisons use the same 20 conferences per query. C-a compares separately trained models, "
        "each tested on its own variant; the checkpoint column names both training variants. Weights and graphs both change. C-b groups hold the selected "
        "checkpoint fixed across test variants, including its matched-variant reference. Delta MRR is the second "
        "test variant minus the first. " + uncertainty, "tab:motif-dblp-matrix-invariance")
    return files


def matrix_notes(summary):
    """Methods/cost notes without adding more tables."""
    trained = summary["shared_training_contract"]
    notes = [f"Each variant starts independently from the released checkpoint and uses {trained['steps']:,} optimizer steps. "
             f"Validation MRR selects each checkpoint. Training seed: {summary['seed']}."]
    if trained.get("gpu"):
        notes.append("Hardware: " + trained["gpu"] + ".")
    costs = []
    for variant in summary["variants"]:
        run = summary["training"].get(variant, {})
        if "total_seconds" in run and "peak_gpu_bytes" in run:
            costs.append(f"{variant}: {run['total_seconds']:.2f} s / {run['peak_gpu_bytes'] / 2**30:.3f} GiB")
    if costs:
        notes.append("Recorded training time / peak allocated GPU memory: " + "; ".join(costs) + ".")
    removed = [summary["training"].get(v, {}).get("total_removed_target_edges",
                summary["training"].get(v, {}).get("total_removed_training_edges")) for v in summary["variants"]]
    if removed and all(value is not None for value in removed):
        if len(set(removed)) == 1:
            notes.append(f"Batch target-edge and inverse removals: {removed[0]:,} per training run "
                         f"({removed[0] / trained['steps']:.3f} per update on average).")
        else:
            notes.append("Batch target-edge and inverse removals, totals per training run: " +
                         "; ".join(f"{v}: {count:,} ({count / trained['steps']:.3f} per update on average)"
                                   for v, count in zip(summary["variants"], removed)) + ".")
    context_removed = [summary["training"].get(v, {}).get("total_removed_context_edges") for v in summary["variants"]]
    if all(value is not None for value in context_removed):
        notes.append("Unsupported Area-context edges and inverses removed, totals per training run: " +
                     "; ".join(f"{v}: {count:,}" for v, count in zip(summary["variants"], context_removed)) +
                     ". Shared Area edges remain when other training papers still support them.")
    notes.append("This is a single-seed preliminary experiment. Error bars are query-bootstrap standard errors. "
                 "The guide's repeated-training noise floor and multi-seed mean/standard deviation are not yet available.")
    return "\n\n".join(latex_escape(note) for note in notes) + "\n"


def load_matrix_summary(run_root):
    summary = json.loads((run_root / "tables/summary.json").read_text(encoding="utf-8"))
    # Older saved summaries omitted costs; recover only recorded metadata if present.
    manifest_path = run_root / "experiment.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["compute_contract"]["dataset"] != summary["dataset"]
                or manifest["compute_contract"]["task"] != summary["task"]
                or manifest["compute_contract"]["seed"] != summary["seed"]):
            raise ValueError("Matrix summary and experiment identify different tasks/seeds")
        for variant in summary["variants"]:
            path = (run_root.parent / f"motif_finetune_{summary['dataset']}_lp_{manifest['run_tag']}" / summary["task"] /
                    f"train_{variant}" / f"seed{summary['seed']}" / "training_summary.json")
            if path.is_file():
                training = json.loads(path.read_text(encoding="utf-8"))
                trained = training["training_contract"]
                if (trained["dataset"] != summary["dataset"] or trained["task"] != summary["task"] or trained["seed"] != summary["seed"]
                        or trained["train_variant"] != variant):
                    raise ValueError(f"Training metadata identify a different run: {path}")
                for key in ("total_seconds", "peak_gpu_bytes", "total_removed_training_edges"):
                    summary["training"][variant][key] = training[key]
                for key in ("total_removed_target_edges", "total_removed_context_edges"):
                    if key in training:
                        summary["training"][variant][key] = training[key]
    return summary


def collect_runs(run_roots, include_lp=True):
    entries, files = {}, {}
    kinds = ("link_prediction", "invariance")
    for run_root in run_roots:
        directory = run_root / "tables"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("format") == "motif_finetune_matrix_v1":
            summary = load_matrix_summary(run_root)
            dataset, task = summary["dataset"], summary["task"]
            key = dataset, task, "finetuned_matrix"
            if key in entries:
                raise ValueError(f"Duplicate matrix run for {dataset}/{task}")
            tables = matrix_tables(summary, include_lp)
            entries[key] = {"run_root": run_root, "kinds": tuple(tables), "folder": "matrix",
                            "label": "fine-tuning matrix", "notes": matrix_notes(summary)}
            for kind, content in tables.items():
                files[Path("tables") / dataset / task / "matrix" / f"{kind}.tex"] = content
            continue
        for task, record in summary["tasks"].items():
            dataset = record["comparison_contract"]["dataset"]
            if dataset not in DATASETS or task not in TASKS:
                raise ValueError(f"Unsupported dataset/task: {dataset}/{task}")
            stage = record["comparison_contract"].get("evaluation_stage", "zeroshot")
            folder = ("finetuned_" + record["comparison_contract"]["fine_tuning"]["training_contract"]["train_variant"]
                      if stage == "finetuned" else "")
            key = dataset, task, folder or "zeroshot"
            if key in entries:
                raise ValueError(f"Choose one run per dataset/task/setting; duplicate {dataset}/{task}/{stage}")
            # Top-1 metrics are meaningful only when the ranked candidates are every possible answer (DBLP).
            task_kinds = ("link_prediction", "conference_prediction", "invariance") if dataset == "dblp" else kinds
            entries[key] = {"run_root": run_root, "kinds": task_kinds, "folder": folder,
                            "label": folder.replace("_", " ") if folder else "zero-shot"}
            for kind in task_kinds:
                path = Path("tables") / dataset / task / folder / f"{kind}.tex"
                text = presentation_table(record, dataset, task, kind, summary["bootstrap"])
                if folder:
                    text = text.replace(f"tab:motif-{dataset}-{task.replace('_', '-')}-",
                                        f"tab:motif-{dataset}-{task.replace('_', '-')}-{folder.replace('_', '-')}-")
                files[path] = text
    if not entries:
        raise ValueError("No task tables found")
    return entries, kinds, files


def main_document(entries, kinds):
    lines = [r"\documentclass[11pt,a4paper]{article}",
             r"\usepackage[margin=1in]{geometry}",
             r"\usepackage{adjustbox}", r"\usepackage{placeins}",
             r"\begin{document}", ""]
    first_dataset = True
    for dataset, label in DATASETS.items():
        tasks = [task for task in TASKS if any(key[:2] == (dataset, task) for key in entries)]
        if not tasks:
            continue
        if not first_dataset:
            lines.append(r"\clearpage")
        first_dataset = False
        for task in tasks:
            for key, entry in entries.items():
                if key[:2] != (dataset, task):
                    continue
                lines.append(f"\\section*{{{label}: {TASKS[task]} ({entry['label']})}}")
                if entry.get("notes"):
                    lines.extend([entry["notes"], ""])
                for kind in entry["kinds"]:
                    path = (Path("tables") / dataset / task / entry["folder"] / f"{kind}.tex").as_posix()
                    lines.extend([f"\\input{{{path}}}", r"\FloatBarrier"])
                lines.append("")
    return "\n".join(lines + [r"\end{document}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Result run directory; repeat for other datasets or tasks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/overleaf/motif_zeroshot_tables")
    parser.add_argument("--include-lp", action="store_true", help="Compatibility flag; LP tables are always included")
    args = parser.parse_args()
    output = args.output.resolve()
    archive = output.parent / (output.name + ".zip")
    if output.exists() or archive.exists():
        parser.error("Output directory or ZIP already exists; choose a fresh --output")
    try:
        entries, kinds, files = collect_runs([path.resolve() for path in args.run], args.include_lp)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    files[Path("main.tex")] = main_document(entries, kinds)
    files[Path("README.md")] = (
        "# MOTIF Overleaf project\n\n"
        "Upload this ZIP as a new Overleaf project, with main.tex as the main document.\n"
        "Edit main.tex to choose which tables appear. Dataset/task folders keep filenames distinct.\n\n"
        "The presentation copies include physical variants and their pairwise comparisons only.\n"
        "They retain the numbers and uncertainty from the saved reports; no metrics were recomputed.\n\n"
        "IMDb uses the guide's fixed 20 candidates; DBLP ranks all 20 conferences. IMDb fine-tuning has two tables. "
        "DBLP fine-tuning has three: top-1 conference metrics, ranking metrics, and C-a/C-b labelled comparisons.\n"
        "Error bars are query-bootstrap SE; multi-seed training variability is not yet measured.\n\n"
        "## Source runs\n\n" + "\n".join(
            f"- {DATASETS[dataset]} / {TASKS[task].replace('--', '-')} / {entry['label']}: `{entry['run_root'].as_posix()}`"
            for (dataset, task, _), entry in entries.items()) + "\n"
    )
    output.mkdir(parents=True)
    for relative, content in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for relative in sorted(files):
            stream.write(output / relative, arcname=relative.as_posix())
    print(f"Overleaf project: {output}\nUpload ZIP: {archive}")


if __name__ == "__main__":
    main()
