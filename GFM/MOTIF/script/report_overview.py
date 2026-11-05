"""One-row-per-method overview across all variants, from the seed_summary_* tables.

For each dataset/task and method:
  * average metric over variants          (METRICS.md §6: "the mean of the variant rows")
  * average |difference| of the metric over all variant pairs  (mean of |M_u - M_v| on the variant means)
  * average Kendall-tau over all variant pairs                  (METRICS.md §4.1: "the mean of the pair rows")

MOTIF averages and tau are taken within each training seed first, then reported as mean ±
population std over seeds (zero-shot/PT has one run). |Delta| is computed from the variant
means for every method, because the lab's baselines only provide per-variant means; taking it
per seed for MOTIF alone would add training noise to MOTIF's number only. Baseline rows are
exact averages of those means, without a ±.

    python script/report_overview.py
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from lp_data import ROOT, write_json

LP = [("imdb_movie_year", "IMDb Movie--Year"), ("imdb_movie_director", "IMDb Movie--Director"),
      ("dblp_paper_conference", "DBLP Paper--Conference"), ("wordnet_link_prediction", "WN18RR")]
NC = [("nc_imdb", "IMDb genre"), ("nc_freebase", "Freebase BOOK")]
LP_METRICS = (("mrr", "MRR"), ("hits@1", "Hits@1"))
NC_METRICS = (("accuracy", "Accuracy"),)


def overview(block, variants, metrics):
    """Per-method averages; per-seed when 'runs' are stored, else from the variant means."""
    pairs = list(combinations(variants, 2))
    per_seed = all("runs" in block["performance"][v][metrics[0][0]] for v in variants)
    if per_seed:
        def series(cell):
            return np.asarray(cell["runs"], dtype=float)
    else:
        def series(cell):
            return np.asarray([cell["mean"]], dtype=float)
    row = {}
    for key, _ in metrics:
        values = np.stack([series(block["performance"][v][key]) for v in variants])  # [variants, seeds]
        row[key] = values.mean(0)
        means = {v: block["performance"][v][key]["mean"] for v in variants}
        row[f"delta_{key}"] = np.mean([abs(means[a] - means[b]) for a, b in pairs])
    row["tau"] = np.stack([series(block["invariance"][f"{a} vs {b}"]["tau"]) for a, b in pairs]).mean(0)
    return {k: {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=0)) if per_seed and not k.startswith("delta_") else None,
                "runs": np.atleast_1d(v).tolist()} for k, v in row.items()}, per_seed


def fmt(value, latex=True):
    if value["std"] is None:
        return f"${value['mean']:.4f}$" if latex else f"{value['mean']:.4f}"
    return f"${value['mean']:.4f} \\pm {value['std']:.4f}$" if latex else f"{value['mean']:.4f} ± {value['std']:.4f}"


def table(rows, metrics, label, caption):
    columns = [(k, n) for k, n in metrics] + [(f"delta_{k}", f"$|\\Delta|$ {n}") for k, n in metrics] + [("tau", "Kendall-$\\tau$")]
    spec = "||l||l||" + "|".join("c" for _ in metrics) + "||" + "|".join("c" for _ in metrics) + "||c||"
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(f"\\textbf{{{n}}}" for n in ["Dataset", "Method"] + [n for _, n in columns]) + " \\\\",
             "\\hline\\hline"]
    previous = None
    for dataset, method, values in rows:
        if previous is not None and dataset != previous:
            lines.append("\\hline")
        lines.extend([" & ".join([dataset, method] + [fmt(values[k]) for k, _ in columns]) + " \\\\", "\\hline"])
        previous = dataset
    lines.extend(["\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""])
    return "\n".join(lines), columns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/seed_summary_overview")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = ("Averages and Kendall-$\\tau$ for MOTIF are taken within each training seed, then mean $\\pm$ "
              "population std over the three seeds; ZS/PT is a single frozen run. $|\\Delta|$ is the mean absolute "
              "difference between the per-variant means over all variant pairs, computed the same way for every "
              "method (no $\\pm$). Baseline rows are exact averages of the lab's per-variant means, whose per-seed "
              "values are unavailable, so they carry no $\\pm$. Kendall-$\\tau$ is the mean over all variant pairs.")
    outputs, report, record = [], ["# Overview: averages over all variants", ""], {}
    for kind, datasets, metrics, title in (("lp", LP, LP_METRICS, "Link prediction"),
                                           ("nc", NC, NC_METRICS, "Node classification")):
        rows = []
        for key, name in datasets:
            path = args.results / f"seed_summary_{key}" / "summary.json"
            if not path.exists():
                print(f"skipping {key}: no {path}")
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            # Baselines first, then the frozen MOTIF row, then fine-tuned (summary.json keys are sorted).
            order = sorted(summary["methods"], key=lambda m: (m.startswith("MOTIF"), m.endswith("(FT)")))
            for method, block in ((m, summary["methods"][m]) for m in order):
                values, _ = overview(block, summary["variants"], metrics)
                rows.append((name, method, values))
                record.setdefault(key, {})[method] = values
        if not rows:
            continue
        caption = (f"{title}: averages over all representation variants per dataset. " + common
                   + (" WN18RR MOTIF (PT) was pretrained on WN18RR." if kind == "lp" else
                      " There is no CMPNN NC baseline; baselines use node features, MOTIF uses training labels as edges."))
        latex, columns = table(rows, metrics, f"tab:overview_{kind}", caption)
        (args.output / f"overview_{kind}.tex").write_text(latex, encoding="utf-8")
        outputs.append(args.output / f"overview_{kind}.tex")
        report += [f"## {title}", "", "| Dataset | Method | " + " | ".join(n.replace("$|\\Delta|$", "|Δ|")
                   .replace("Kendall-$\\tau$", "Kendall-τ") for _, n in columns) + " |",
                   "|---|---|" + "---:|" * len(columns)]
        report += ["| " + " | ".join([d.replace("--", "–"), m] + [fmt(v[k], latex=False) for k, _ in columns]) + " |"
                   for d, m, v in rows]
        report.append("")
    report.append(common.replace("$\\pm$", "±").replace("$|\\Delta|$", "|Δ|").replace("Kendall-$\\tau$", "Kendall-τ"))
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output / "summary.json", record)
    print("Wrote " + ", ".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
