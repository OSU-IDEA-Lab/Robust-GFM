"""Seed-averaged MOTIF (ZS)/(FT) tables in the lab format, optionally beside a baseline.

Follows SCR/INV-GNN-12D9/docs/results/METRICS.md: every cell is mean ± population std
(ddof=0) over training seeds. FT rows are the matched-variant (diagonal) models, i.e. the
"original" arm: one model per variant and seed. Kendall-tau compares two of those models on
the same queries and candidates within one seed. Zero-shot inference has no training seed,
so its single run is reported with std 0.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from lp_data import ROOT, TASK_VARIANTS, DBLP_TASK_VARIANTS, write_json
from lp_metrics import average_ranks, binary_metrics, check_alignment, classification_metrics, mean_tau_b, top1_tail
from report_lp import write_csv

METRICS = (("mrr", "MRR"), ("hits@1", "Hits@1"), ("hits@3", "Hits@3"),
           ("precision", "Precision"), ("recall", "Recall"))
DATASETS = {"imdb": TASK_VARIANTS, "dblp": DBLP_TASK_VARIANTS, "wordnet": {"link_prediction": ("v1", "v2", "v3")}}
NAMES = {"imdb": "IMDb", "dblp": "DBLP", "wordnet": "WN18RR"}
# Baseline .tex block suffixes and column orders, from SCR docs/results/*_lp.tex.
BASELINE_TASK_LABELS = {"movie_year": " / Movie--Year (MY)", "movie_director": " / Movie--Director (MD)",
                        "paper_conference": "", "link_prediction": ""}
BASELINE_RESULT_COLUMNS = ("precision", "recall", "f1", "hits@1", "hits@3", "mrr")
BASELINE_TAU_COLUMNS = ("tau", "tau@1", "tau@3")
# Node classification (script/nc_motif.py), compared with the lab's NC baselines (docs/results/*_nc.tex).
NC_METRICS = (("accuracy", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"),
              ("micro_f1", "Micro-F1"), ("macro_f1", "Macro-F1"))
NC_DATASETS = {"nc_imdb": ("IMDb", ("v1", "v2", "v3", "v4")), "nc_freebase": ("Freebase", ("v1", "v2", "v3"))}


def load_scores(path):
    with np.load(path) as source:
        return {key: source[key].copy() for key in source.files}


def performance(arrays):
    # WordNet ranks come from filtered full-entity ranking; its score rows are sampled candidates.
    ranks = arrays["ranks"] if "ranks" in arrays else average_ranks(arrays["scores"])
    return {"mrr": float(np.mean(1 / ranks)), "hits@1": float(np.mean(ranks <= 1)),
            "hits@3": float(np.mean(ranks <= 3)),
            **binary_metrics(arrays["scores"], arrays.get("candidates"), arrays.get("positives"))}


def wordnet_runs(roots, variants, zeroshot=False):
    """{seed: {variant: arrays}} from script/wordnet_lp.py eval outputs. Each seed reads its own
    CMPNN-style candidate rows; zero-shot has one model, so its spread is candidate sampling only."""
    runs = {}
    for root in roots:
        for variant in variants:
            directory = root / "link_prediction" / "physical" / variant
            meta = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if meta["limit"]:
                raise ValueError(f"Smoke-test evaluation (limit {meta['limit']}): {directory}")
            scores = load_scores(directory / "scores.npz")
            for seed in (meta["candidate_seeds"] if zeroshot else [meta["seed"]]):
                if variant in runs.setdefault(seed, {}):
                    raise ValueError(f"Duplicate seed {seed} for {variant}: {directory}")
                runs[seed][variant] = {"scores": scores[f"scores_{seed}"], "ranks": scores["ranks"]}
    return runs


def aggregate(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "runs": values.tolist()}


def finetuned_runs(roots, dataset, task, variants):
    """Diagonal score arrays per training seed, from run_finetune_matrix.py outputs."""
    runs, shared = {}, None
    for root in roots:
        manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        contract = manifest["compute_contract"]
        if (contract["dataset"], contract["task"], contract["variants"]) != (dataset, task, list(variants)):
            raise ValueError(f"Matrix is not {dataset}/{task} with variants {variants}: {root}")
        normalized = {key: value for key, value in contract.items() if key != "seed"}
        if shared is None:
            shared = normalized
        elif normalized != shared:
            raise ValueError(f"Fine-tuning settings or code differ from the first matrix: {root}")
        if contract["seed"] in runs:
            raise ValueError(f"Duplicate training seed {contract['seed']}: {root}")
        runs[contract["seed"]] = {}
        for variant in variants:
            if variant not in manifest["evaluations"].get(variant, {}):
                raise ValueError(f"Missing matched-variant evaluation {variant}: {root}")
            directory = root.parent / manifest["evaluation_roots"][variant] / task / "physical" / variant
            runs[contract["seed"]][variant] = load_scores(directory / "scores.npz")
    return runs


def nc_performance(arrays):
    classes = np.arange(arrays["scores"].shape[1])
    return classification_metrics(arrays["y"], top1_tail(arrays["scores"], classes), classes)


def nc_runs(roots, variants, zeroshot=False):
    """{seed: {variant: arrays}} from script/nc_motif.py eval outputs (zero-shot: one pseudo-seed)."""
    runs, nodes = {}, None
    for root in roots:
        for variant in variants:
            directory = root / "physical" / variant
            meta = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if meta["limit"]:
                raise ValueError(f"Smoke-test evaluation (limit {meta['limit']}): {directory}")
            arrays = load_scores(directory / "scores.npz")
            if nodes is None:
                nodes = arrays["nodes"]
            if not np.array_equal(arrays["nodes"], nodes):
                raise ValueError(f"Test nodes are not aligned: {directory}")
            seed = "zero-shot" if zeroshot else meta["seed"]
            if variant in runs.setdefault(seed, {}):
                raise ValueError(f"Duplicate seed {seed} for {variant}: {directory}")
            runs[seed][variant] = arrays
    return runs


def summarize(runs, variants, measure=performance, metrics=METRICS):
    """runs: {seed: {variant: arrays}} -> per-variant and per-pair mean ± std over seeds."""
    rows = {v: {key: aggregate([measure(runs[s][v])[key] for s in runs]) for key, _ in metrics}
            for v in variants}
    pairs = {}
    for first, second in combinations(variants, 2):
        taus = [mean_tau_b(runs[s][first]["scores"], runs[s][second]["scores"]) for s in runs]
        pairs[f"{first} vs {second}"] = {"tau": aggregate([t for t, _ in taus]),
                                         "undefined_tau_queries": [u for _, u in taus]}
    return {"seeds": sorted(runs), "performance": rows, "invariance": pairs}


def parse_baseline(path, model, task, arm="original", result_columns=BASELINE_RESULT_COLUMNS,
                   tau_columns=BASELINE_TAU_COLUMNS):
    """Read one model/arm block from the RESULTS and INVARIANCE sections of a lab .tex file.
    A header may carry a parenthesised note (e.g. SlotGAT's "original (upstream stock ...)");
    the first such block is used."""
    text = path.read_text(encoding="utf-8")
    header = f"% ---- {model} / {arm}{BASELINE_TASK_LABELS.get(task, '')}"

    def block(section):
        lines = text.split(f"%% ################################ {section}", 1)[1].splitlines()
        start = next((i for i, line in enumerate(lines)
                      if line.strip() == header or line.strip().startswith(header + " (")), None)
        if start is None:
            raise ValueError(f"'{header}' not found in the {section} section of {path}")
        rows = []
        for line in lines[start + 1:]:
            if line.startswith("%") or not line.strip():
                if rows or line.startswith(("%%", "% ----")):
                    break
                continue
            cells = [cell.strip() for cell in line.rstrip("\\ ").split("&") if cell.strip()]
            rows.append((cells[0], [tuple(map(float, cell.split("$\\pm$"))) for cell in cells[1:]]))
        return rows

    def variant(label):
        return "v" + re.search(r"(\d+)\s*$", label).group(1)

    def pair(label):
        first, second = re.split(r"\s+vs\.?\s+", label)
        return f"{variant(first)} vs {variant(second)}"

    performance_rows = {variant(label): {key: {"mean": mean, "std": std}
                                         for key, (mean, std) in zip(result_columns, values)}
                        for label, values in block("RESULTS")}
    pairs = {pair(label): {key: {"mean": mean, "std": std} for key, (mean, std) in zip(tau_columns, values)}
             for label, values in block("INVARIANCE")}
    return {"source": str(path), "arm": arm, "performance": performance_rows, "invariance": pairs}


def cell(value, latex=True):
    if value is None:
        return "--"
    return (f"${value['mean']:.4f} \\pm {value['std']:.4f}$" if latex
            else f"{value['mean']:.4f} ± {value['std']:.4f}")


def latex_table(groups, columns, row_keys, label, caption, first_header):
    # Ranking and binary-classification columns are boxed separately, as in assets/table format.png.
    spec = "||l||l||c|c|c||c|c||" if columns in (METRICS, NC_METRICS) else "||l||l||" + "|".join("c" for _ in columns) + "||"
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(f"\\textbf{{{name}}}" for name in ("Method", first_header) + tuple(n for _, n in columns))
             + " \\\\", "\\hline\\hline"]
    for index, (method, rows) in enumerate(groups):
        if index:
            lines.append("\\hline")
        for key in row_keys:
            if key not in rows:
                continue
            values = [cell(rows[key].get(metric)) for metric, _ in columns]
            lines.extend([" & ".join([method, key] + values) + " \\\\", "\\hline"])
    lines.extend(["\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""])
    return "\n".join(lines)


def markdown_table(groups, columns, row_keys, first_header):
    lines = ["| Method | " + first_header + " | " + " | ".join(n for _, n in columns) + " |",
             "|---|---|" + "---:|" * len(columns)]
    for method, rows in groups:
        lines.extend("| " + " | ".join([method, key] + [cell(rows[key].get(m), latex=False) for m, _ in columns])
                     + " |" for key in row_keys if key in rows)
    return lines


def write_outputs(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    dataset, task, variants = summary["dataset"], summary["task"], summary["variants"]
    pairs = [f"{a} vs {b}" for a, b in combinations(variants, 2)]
    performance_groups = [(name, block["performance"]) for name, block in summary["methods"].items()]
    tau_groups = [(name, block["invariance"]) for name, block in summary["methods"].items()]
    seeds = summary["finetune_seeds"]
    wordnet = dataset == "wordnet"
    zeroshot_note = ("MOTIF (PT) is the released checkpoint, which was pretrained on WN18RR (v1's graph), so it is not "
                     "zero-shot; its spread is over the three seeds' candidate draws (ranking metrics are exact). "
                     "MOTIF ran on one H200, CMPNN on V100. "
                     if wordnet else "zero-shot has no training seed (std 0). ")
    note = (f"Mean $\\pm$ population std over training seeds ({', '.join(map(str, seeds))}); "
            + zeroshot_note) if seeds else "Zero-shot only. "
    name = NAMES[dataset] + ("" if wordnet else " " + task.replace("_", "--").title())
    protocol = ("MRR and Hits use filtered ranking over all 40,943 entities, head and tail queries; "
                "Precision/Recall use the positive plus 100 seeded candidates per query, "
                if wordnet else "Ranking over 20 candidates per query; Precision/Recall pool all (query, candidate) pairs, ")
    (output / "performance.tex").write_text(latex_table(
        performance_groups, METRICS, variants, f"tab:{dataset}_{task}_seed_performance",
        f"{name} link prediction. {protocol}positive at $\\sigma(s) \\geq 0.5$. {note}", "Variant"),
        encoding="utf-8")
    (output / "invariance.tex").write_text(latex_table(
        tau_groups, (("tau", "Kendall-$\\tau$"),), pairs, f"tab:{dataset}_{task}_seed_invariance",
        f"{name} Kendall-$\\tau_b$ between candidate rankings of two variants' models on the same queries, "
        f"averaged over queries. {note}", "Comparison"), encoding="utf-8")
    report = [f"# {name}: seed-averaged results", "",
              note.replace("$\\pm$", "±") + "FT rows use the model trained and tested on the same variant.", "",
              "## Performance", "", *markdown_table(performance_groups, METRICS, variants, "Variant"), "",
              "## Invariance", "", *markdown_table(tau_groups, (("tau", "Kendall-τ"),), pairs, "Comparison"), ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_csv(output / "performance.csv", [
        {"method": method, "variant": v, **{f"{k}_{s}": rows[v][k][s] for k, _ in METRICS for s in ("mean", "std")
                                            if k in rows[v]}}
        for method, rows in performance_groups for v in variants if v in rows])
    write_csv(output / "invariance.csv", [
        {"method": method, "comparison": p, "tau_mean": rows[p]["tau"]["mean"], "tau_std": rows[p]["tau"]["std"]}
        for method, rows in tau_groups for p in pairs if p in rows])
    write_json(output / "summary.json", summary)
    print(f"Wrote seed-averaged tables: {output}")


def write_nc_outputs(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    dataset, variants = summary["dataset"], summary["variants"]
    pairs = [f"{a} vs {b}" for a, b in combinations(variants, 2)]
    performance_groups = [(name, block["performance"]) for name, block in summary["methods"].items()]
    tau_groups = [(name, block["invariance"]) for name, block in summary["methods"].items()]
    name, seeds = NC_DATASETS[dataset][0], summary["finetune_seeds"]
    baselines = [m for m, block in summary["methods"].items() if block.get("source") != "motif"]
    note = ((f"Mean $\\pm$ population std over training seeds ({', '.join(map(str, seeds))}); "
             "zero-shot has no training seed (std 0). ") if seeds else "Zero-shot only. ")
    protocol = ("MOTIF classifies by link prediction to one node per class; only training nodes are "
                "linked to their class, so it sees training labels as graph edges and no node features. "
                + (f"Baselines ({', '.join(baselines)}: lab original arm) " if baselines else "")
                + ("use plot-keyword features. " if dataset == "nc_imdb" and baselines else
                   "use learned type embeddings. " if baselines else ""))
    (output / "performance.tex").write_text(latex_table(
        performance_groups, NC_METRICS, variants, f"tab:{dataset}_seed_performance",
        f"{name} node classification (macro Precision/Recall). {protocol}{note}", "Variant"), encoding="utf-8")
    (output / "invariance.tex").write_text(latex_table(
        tau_groups, (("tau", "Kendall-$\\tau$"),), pairs, f"tab:{dataset}_seed_invariance",
        f"{name} Kendall-$\\tau_b$ between the class scores of two variants' models for the same test node, "
        f"averaged over nodes. {note}", "Comparison"), encoding="utf-8")
    report = [f"# {name} node classification: seed-averaged results", "", (protocol + note).replace("$\\pm$", "±"), "",
              "## Performance", "", *markdown_table(performance_groups, NC_METRICS, variants, "Variant"), "",
              "## Invariance", "", *markdown_table(tau_groups, (("tau", "Kendall-τ"),), pairs, "Comparison"), ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_csv(output / "performance.csv", [
        {"method": method, "variant": v, **{f"{k}_{s}": rows[v][k][s] for k, _ in NC_METRICS for s in ("mean", "std")
                                            if k in rows[v]}}
        for method, rows in performance_groups for v in variants if v in rows])
    write_csv(output / "invariance.csv", [
        {"method": method, "comparison": p, "tau_mean": rows[p]["tau"]["mean"], "tau_std": rows[p]["tau"]["std"]}
        for method, rows in tau_groups for p in pairs if p in rows])
    write_json(output / "summary.json", summary)
    print(f"Wrote seed-averaged tables: {output}")


def nc_main(args):
    variants = list(NC_DATASETS[args.dataset][1])
    methods = {}
    for model in (args.baseline if args.baseline_tex else []):
        methods[model] = parse_baseline(args.baseline_tex, model, "nc",
                                        result_columns=[k for k, _ in NC_METRICS], tau_columns=("tau",))
    sources = []
    if args.zeroshot:
        sources.append(("MOTIF (ZS)", nc_runs([args.zeroshot], variants, zeroshot=True)))
    finetuned = nc_runs(args.finetune, variants) if args.finetune else {}
    if finetuned:
        sources.append(("MOTIF (FT)", finetuned))
    for method, runs in sources:
        methods[method] = {"source": "motif", **summarize(runs, variants, nc_performance, NC_METRICS)}
    summary = {"format": "motif_nc_seed_summary_v1", "dataset": args.dataset, "variants": variants,
               "finetune_seeds": sorted(finetuned), "aggregation": "mean and population std (ddof=0) over seeds",
               "zeroshot": str(args.zeroshot) if args.zeroshot else None,
               "finetune": [str(path) for path in args.finetune], "methods": methods}
    write_nc_outputs(summary, args.output or ROOT / "results" / f"seed_summary_{args.dataset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS) + list(NC_DATASETS), default="imdb")
    parser.add_argument("--task")
    parser.add_argument("--zeroshot", type=Path, help="A report_lp/eval_lp run root, e.g. results/motif_zeroshot_imdb_lp_full")
    parser.add_argument("--finetune", type=Path, nargs="*", default=[],
                        help="One run_finetune_matrix.py root (WordNet: wordnet_lp.py finetuned eval root) per training seed")
    parser.add_argument("--baseline-tex", type=Path, help="Lab results file, e.g. SCR/.../docs/results/IMDB_lp.tex")
    parser.add_argument("--baseline", nargs="+", default=["CMPNN"],
                        help="Model name(s) of the baseline block(s); NC tables accept several, e.g. RGCN SlotGAT")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.zeroshot and not args.finetune:
        parser.error("Give --zeroshot and/or --finetune")
    if args.dataset in NC_DATASETS:
        nc_main(args)
        return
    task = args.task or next(iter(DATASETS[args.dataset]))
    variants = list(DATASETS[args.dataset][task])
    if not args.zeroshot and not args.finetune:
        parser.error("Give --zeroshot and/or --finetune")

    methods, reference = {}, None
    for model in (args.baseline if args.baseline_tex else []):
        methods[model] = parse_baseline(args.baseline_tex, model, task)
    sources = []
    wordnet = args.dataset == "wordnet"
    if args.zeroshot:
        zeroshot = (wordnet_runs([args.zeroshot], variants, zeroshot=True) if wordnet else
                    {"zero-shot": {v: load_scores(args.zeroshot / task / "physical" / v / "scores.npz")
                                   for v in variants}})
        # The released checkpoint was pretrained on WN18RR, so its frozen rows are not zero-shot there.
        sources.append(("MOTIF (PT)" if wordnet else "MOTIF (ZS)", zeroshot))
    finetuned = ((wordnet_runs(args.finetune, variants) if wordnet else
                  finetuned_runs(args.finetune, args.dataset, task, variants)) if args.finetune else {})
    if finetuned:
        sources.append(("MOTIF (FT)", finetuned))
    for method, runs in sources:
        for arrays in (a for seed in runs.values() for a in seed.values()):
            if reference is None:
                reference = arrays
            if not wordnet:  # WordNet rows are aligned by construction: same split order and seeded draws.
                check_alignment(reference, arrays)
        methods[method] = {"source": "motif", **summarize(runs, variants)}
    summary = {"format": "motif_seed_summary_v1", "dataset": args.dataset, "task": task, "variants": variants,
               "finetune_seeds": sorted(finetuned), "aggregation": "mean and population std (ddof=0) over seeds",
               "zeroshot": str(args.zeroshot) if args.zeroshot else None,
               "finetune": [str(path) for path in args.finetune], "methods": methods}
    output = args.output or ROOT / "results" / f"seed_summary_{args.dataset}_{task}"
    write_outputs(summary, output)


if __name__ == "__main__":
    main()
