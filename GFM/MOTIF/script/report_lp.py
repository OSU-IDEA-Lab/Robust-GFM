"""Export MOTIF performance and all-pairs invariance as JSON, CSV, Markdown and LaTeX."""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from eval_lp import result_root
from lp_data import ROOT, TASK_VARIANTS, DBLP_TASK_VARIANTS, array_hash, file_hash, write_json
from lp_metrics import check_alignment, compare_predictions, point_metrics, summarize_performance

LP_COLUMNS = (("mrr", "MRR"), ("hits@1", "Hits@1"), ("hits@3", "Hits@3"), ("hits@10", "Hits@10"))
PAIR_COLUMNS = (("prediction_agreement", "Agreement@1"), ("tau_b", "Kendall $\\tau_b$"),
                ("delta_mrr", "$\\Delta$MRR"))
TASK_LABELS = {"movie_year": "Movie-Year", "movie_director": "Movie-Director", "paper_conference": "Paper-Conference"}


def cell(value, latex=False):
    point, se = value["estimate"], value["bootstrap_se"]
    if se is None:
        return f"{point:.4f}"
    return f"${point:.4f} \\pm {se:.4f}$" if latex else f"{point:.4f} ± {se:.4f}"


def latex_escape(text):
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def table(title, rows, columns, label, notes="", bootstrap=1000):
    uncertainty = (f"Values are estimates plus/minus bootstrap standard error from {bootstrap} query resamples."
                   if bootstrap else "Values are point estimates; bootstrap uncertainty was not computed.")
    caption = " ".join((title + ".", notes, uncertainty)).strip()
    md = [f"### {title}", "", notes, "", "| Input | " + " | ".join(name for _, name in columns) + " |",
          "|---|" + "---:|" * len(columns)]
    row_end = " " + chr(92) * 2
    tex = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "r" * len(columns) + "}",
           "\\toprule", "Input & " + " & ".join(name for _, name in columns) + row_end, "\\midrule"]
    for name, values in rows:
        md.append("| " + name + " | " + " | ".join(cell(values[key]) for key, _ in columns) + " |")
        tex.append(latex_escape(name) + " & " + " & ".join(cell(values[key], True) for key, _ in columns) + row_end)
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\caption{" + latex_escape(caption) + "}",
                "\\label{" + label + "}", "\\end{table}", ""])
    return md + [""], tex


def csv_row(base, metrics):
    result = dict(base)
    for key, value in metrics.items():
        result[key] = value["estimate"]
        result[key + "_bootstrap_se"] = value["bootstrap_se"]
        interval = value["ci95"] or (None, None)
        result[key + "_ci95_low"], result[key + "_ci95_high"] = interval
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(root, bootstrap=1000, seed=0):
    summaries, performance_csv, invariance_csv = {}, [], []
    observed_stages = set()
    md = ["# Preliminary MOTIF evaluation", "",
          f"Uncertainty: estimate ± bootstrap SE; {bootstrap} query resamples, seed {seed}. Percentile 95% CIs are in JSON/CSV.",
          "LP uses each query's original positive + 19 fixed negatives, average ranks for ties.",
          "All presentation metrics use the benchmark candidate sets specified in guide B2.",
          "Invariance uses the original 20 candidates. Identical score vectors have tau=1; undefined nonidentical constant-vector tau is assigned 0 and counted in JSON.", ""]
    latex = {"link_prediction": [], "invariance": [], "controls": []}
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "physical").is_dir()):
        runs, contract, reference_arrays = {}, None, None
        for path in sorted(directory.glob("*/*/metrics.json")):
            meta = json.loads(path.read_text())
            if contract is None:
                contract = meta["comparison_contract"]
            elif meta["comparison_contract"] != contract:
                raise ValueError(f"Incompatible checkpoint, query protocol, precision, hardware or code: {path}")
            if file_hash(path.with_name("scores.npz")) != meta["scores_sha256"]:
                raise ValueError(f"Score file checksum differs: {path}")
            with np.load(path.with_name("scores.npz")) as source:
                arrays = {key: source[key].copy() for key in source.files}
            if array_hash(arrays["positives"], arrays["candidates"], arrays["query_index"], arrays["candidate_pool"]) != contract["query_sha256"]:
                raise ValueError(f"Query content differs from inference contract: {path}")
            if reference_arrays is None:
                reference_arrays = arrays
            else:
                check_alignment(reference_arrays, arrays)
            calculated = point_metrics(arrays["scores"], arrays["positives"], arrays["candidate_pool"], arrays["full_type_scores"])
            if any(not np.isclose(value, meta[key], atol=1e-12, rtol=0) for key, value in calculated.items()):
                raise ValueError(f"Stored metrics disagree with raw scores: {path}")
            runs[(meta["view"], meta["variant"], meta["repeat"])] = (meta, arrays)
        if not runs:
            continue
        dataset, task = contract["dataset"], contract["task"]
        stage = contract.get("evaluation_stage", "zeroshot")
        observed_stages.add(stage)
        title = f"MOTIF / {'IMDb' if dataset == 'imdb' else 'DBLP'} / {TASK_LABELS[task]}"
        n_queries = len(reference_arrays["positives"])
        setting = "frozen zero-shot" if stage == "zeroshot" else "fine-tuned checkpoint"
        context = f"Preliminary {setting} evaluation on the {contract.get('split', 'test')} split; {n_queries} queries."
        if stage == "finetuned":
            trained = contract["fine_tuning"]["training_contract"]
            context += f" Checkpoint fine-tuned on {trained['train_view']}/{trained['train_variant']}, selected by validation MRR."
        if contract.get("limit", 0):
            context += " Limited-query smoke run."
        lp_protocol = ("Tail-only ranking over the true year plus 19 fixed frequency-matched negative years per query."
                       if task == "movie_year" else
                       "Tail-only ranking over the true director plus 19 fixed uniform negative directors per query."
                       if task == "movie_director" else "Tail-only ranking over all 20 conferences.")
        records, rows = {}, []
        for (view, variant, repeat), (meta, arrays) in runs.items():
            if repeat:
                continue
            values = summarize_performance(arrays, bootstrap, seed)
            name = f"{view}/{variant}"
            records[name] = {"metrics": values, "num_queries": len(arrays["positives"]),
                             "benchmark_candidates": meta["benchmark_candidates"],
                             "classification_classes": meta["full_type_candidates"]}
            rows.append((name, values))
            performance_csv.append(csv_row({"model": "MOTIF", "dataset": dataset, "task": task,
                                           "view": view, "variant": variant, "queries": len(arrays["positives"])},
                                          {key: values[key] for key, _ in LP_COLUMNS}))
        part, tex = table(title + " (20-candidate link prediction)", rows, LP_COLUMNS,
                          f"tab:motif-{task}-link_prediction", context + " " + lp_protocol + " Tied scores use average rank.", bootstrap)
        md.extend(part)
        latex["link_prediction"].extend(tex)
        physical = sorted(v for view, v, repeat in runs if view == "physical" and not repeat)
        pairs, pair_rows = {}, []
        for first, second in combinations(physical, 2):
            comparison = compare_predictions(runs[("physical", first, False)][1], runs[("physical", second, False)][1], bootstrap, seed)
            name = f"{first} vs {second}"
            pairs[name] = comparison
            pair_rows.append((name, comparison["metrics"]))
            invariance_csv.append(csv_row({"model": "MOTIF", "dataset": dataset, "task": task,
                                          "first": first, "second": second, "queries": comparison["num_queries"],
                                          "undefined_tau_queries": comparison["undefined_tau_queries"]},
                                         {key: value for key, value in comparison["metrics"].items()
                                          if key != "full_type_prediction_agreement"}))
        pair_protocol = ("All columns use the same original 20 candidates per query. Agreement@1 compares predicted entity IDs; "
                         "Kendall tau-b compares candidate rankings within each query and is then averaged over queries. "
                         "Delta MRR is the second variant minus the first. "
                         "Representation differences must be interpreted against the reference-repeat noise floor.")
        part, tex = table(title + " (20-candidate invariance)", pair_rows, PAIR_COLUMNS, f"tab:motif-{task}-invariance",
                          context + " " + pair_protocol, bootstrap)
        md.extend(part)
        latex["invariance"].extend(tex)
        repeat = None
        if ("physical", "v1", False) in runs and ("physical", "v1", True) in runs:
            repeat = compare_predictions(runs[("physical", "v1", False)][1], runs[("physical", "v1", True)][1], bootstrap, seed)
            part, tex = table(title + " (inference noise floor)", [("v1 vs v1 repeat", repeat["metrics"])], PAIR_COLUMNS, f"tab:motif-{task}-noise",
                              context + " Independent inference processes with identical graph and checkpoint; original 20 candidates per query.", bootstrap)
            md.extend(part)
            latex["controls"].extend(tex)
        compiled = [(v, run) for (view, v, rep), run in runs.items() if view == "compiled" and not rep]
        compiled_inputs = (len({run[0]["graph_sha256"] for _, run in compiled}) == 1) if compiled else None
        compiled_outputs = all(np.array_equal(compiled[0][1][1]["full_type_scores"], run[1]["full_type_scores"]) for _, run in compiled) if compiled else None
        family = TASK_VARIANTS[task] if dataset == "imdb" else DBLP_TASK_VARIANTS[task]
        expected_views = ("physical",) if stage == "finetuned" else ("physical", "compiled")
        missing = [f"{view}/{v}" for view in expected_views for v in family if (view, v, False) not in runs]
        if repeat is None and stage == "zeroshot":
            missing.append("physical/v1__repeat")
        md.extend([f"Reference repeat bitwise identical: {repeat['bitwise_identical'] if repeat else 'not run'}.",
                   f"Compiled inputs identical: {compiled_inputs}; compiled full-type scores bitwise identical: {compiled_outputs}.",
                   "Coverage: " + ("complete" if not missing else "partial; missing " + ", ".join(missing)), ""])
        summaries[task] = {"comparison_contract": contract, "performance": records, "physical_pairs": pairs,
                           "reference_repeat": repeat, "compiled_inputs_identical": compiled_inputs,
                           "compiled_full_type_outputs_identical": compiled_outputs, "missing_runs": missing}
    if not summaries:
        raise ValueError("No inference outputs found")
    if len(observed_stages) != 1:
        raise ValueError("Do not combine zero-shot and fine-tuned runs in one result directory")
    output = root / "tables"
    output.mkdir(exist_ok=True)
    (output / "classification.tex").unlink(missing_ok=True)
    write_json(output / "summary.json", {"evaluation_stage": next(iter(observed_stages)), "report_status": "preliminary",
                                         "uncertainty": "bootstrap_standard_error", "bootstrap": bootstrap,
                                         "bootstrap_seed": seed, "tasks": summaries})
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for name, lines in latex.items():
        (output / f"{name}.tex").write_text("% Requires \\usepackage{booktabs}\n" + "\n".join(lines), encoding="utf-8")
    write_csv(output / "performance.csv", performance_csv)
    write_csv(output / "invariance.csv", invariance_csv)
    print(f"Wrote {output}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imdb", "dblp"), required=True)
    parser.add_argument("--stage", choices=("zeroshot", "finetuned"), default="zeroshot")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.bootstrap == 1:
        parser.error("Bootstrap must be zero or at least two")
    build_report(result_root(args.results, args.dataset, args.run_tag, args.stage), args.bootstrap, args.bootstrap_seed)


if __name__ == "__main__":
    main()
