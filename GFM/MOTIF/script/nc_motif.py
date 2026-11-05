"""Node classification with MOTIF as link prediction to class nodes (IMDb genre, Freebase BOOK).

Each class becomes a node; every TRAINING node gets one edge `node_to_class` to its class
node. A query (node, node_to_class, ?) ranks the C class nodes; the top one is the
prediction. Validation and test nodes have no class edge. Metrics follow SCR
docs/results/METRICS.md section 2 (accuracy, macro precision/recall, micro/macro-F1), and
Kendall-tau compares the C class scores per node between variants.

Labels, splits and variant graphs are the lab's own:
  imdb:     models/RGCN/preprocess_IMDB_rgcn_structural.py --tasks nc bundles (v1-v4)
  freebase: data_tierA/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3} with the
            shared split Freebase_unchanged/train_val_test_idx.npz (global BOOK node ids)

    python script/nc_motif.py export --dataset imdb --bundle-root rai-datasets/preprocessed/_build/IMDB_nc/IMDB/nc
    python script/nc_motif.py eval --dataset imdb --variant v1 --run-tag full
    python script/nc_motif.py finetune --dataset imdb --variant v1 --seed 1566911444 --run-tag seeds
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

from eval_lp import build_message_graph, state_hash
from finetune_imdb_lp import batch_schedule, make_batch, save_checkpoint, validate
from lp_data import ROOT, array_hash, file_hash, forward_edges, write_json
from lp_metrics import classification_metrics, top1_tail

DATA_ROOT = ROOT / "rai-datasets/preprocessed/NC"
GNN_ROOT = ROOT.parents[1] / "GNN"
FREEBASE_DIR = ROOT / "rai-datasets/data_tierA/dataset_variant_3hops_filtered"
FREEBASE_SPLIT = ROOT / "rai-datasets/data_tierA/Freebase_unchanged/train_val_test_idx.npz"
VARIANTS = {"imdb": {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4"},
            "freebase": {"v1": "unchanged", "v2": "exact_2", "v3": "exact_3"}}
LABEL_RELATION = "node_to_class"


# ----------------------------------------------------------------------------- export
def imdb_source(bundle_root):
    sys.path.insert(0, str(GNN_ROOT / "common"))
    from inv_rgcn.data import load_graph_data

    graphs, reference = {}, None
    for variant in VARIANTS["imdb"]:
        data = load_graph_data(bundle_root / variant)
        if reference is None:
            reference = data
        if data.supervision_hash != reference.supervision_hash or data.num_nodes != reference.num_nodes:
            raise ValueError(f"{variant}: labels/split/nodes differ from v1")
        edges, names = forward_edges(data.physical_edge_index.numpy(), data.physical_edge_type.numpy(),
                                     data.physical_relation_names)
        graphs[variant] = (edges, names)
    y = reference.y.numpy()
    split = {s: np.flatnonzero(getattr(reference, f"{s}_mask").numpy()) for s in ("train", "val", "test")}
    return graphs, reference.num_nodes, {s: (n, y[n]) for s, n in split.items()}, \
        {"bundle_root": str(bundle_root), "supervision_hash": reference.supervision_hash}


def freebase_source(_):
    import pandas as pd

    rows = [line.rstrip("\n").split("\t") for line in open(FREEBASE_DIR / "unchanged/label.dat", encoding="utf-8")]
    labels = {int(r[0]): int(r[3]) for r in rows if int(r[2]) == 0}
    stored = np.load(FREEBASE_SPLIT)
    split = {s: np.sort(stored[f"{s}_idx"]) for s in ("train", "val", "test")}
    if set(np.concatenate(list(split.values())).tolist()) != set(labels):
        raise ValueError("The shared Freebase split must cover exactly the labelled BOOK nodes")
    num_nodes = sum(1 for _ in open(FREEBASE_DIR / "unchanged/node.dat", encoding="utf-8"))
    graphs = {}
    for variant, name in VARIANTS["freebase"].items():
        directory = FREEBASE_DIR / name
        other = {int(r[0]): int(r[3]) for r in (l.rstrip("\n").split("\t") for l in open(directory / "label.dat", encoding="utf-8"))
                 if int(r[2]) == 0}
        if other != labels:
            raise ValueError(f"{name}: labels differ from unchanged")
        frame = pd.read_csv(directory / "link.dat", sep="\t", header=None, usecols=[0, 1, 2],
                            dtype=np.int32, engine="c")
        edges = frame.to_numpy()[:, [0, 1, 2]]
        del frame
        relations = int(edges[:, 2].max()) + 1
        graphs[variant] = (edges, [f"relation_{r}" for r in range(relations)])
        print(f"read {name}: {len(edges):,} edges, {relations} relations", flush=True)
    return graphs, num_nodes, {s: (n, np.asarray([labels[int(i)] for i in n])) for s, n in split.items()}, \
        {"split_file": str(FREEBASE_SPLIT), "split_sha256": file_hash(FREEBASE_SPLIT)}


def run_export(args):
    output = args.output / args.dataset
    if output.exists():
        raise FileExistsError(f"Export exists: {output}")
    graphs, num_nodes, split, provenance = (imdb_source if args.dataset == "imdb" else freebase_source)(args.bundle_root)
    classes = np.unique(split["train"][1])
    if not all(np.isin(split[s][1], classes).all() for s in ("val", "test")) or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Classes must be 0..C-1 and all present in training")
    class_nodes = num_nodes + classes  # Fresh nodes appended after every original node.
    stage = Path(tempfile.mkdtemp(prefix="nc-export-", dir=args.output))
    manifest = {"format": "motif_nc_v1", "dataset": args.dataset, "num_nodes": int(num_nodes + len(classes)),
                "num_original_nodes": int(num_nodes), "class_nodes": class_nodes.tolist(),
                "split_counts": {s: int(len(split[s][0])) for s in split}, "provenance": provenance,
                "label_edges": "training_nodes_only", "variants": {}}
    train_edges = np.column_stack((split["train"][0], class_nodes[split["train"][1]], np.zeros(len(split["train"][0]))))
    for variant, (edges, names) in graphs.items():
        label = len(names)
        held_out = np.concatenate((split["val"][0], split["test"][0]))
        if np.isin(edges[:, :2], class_nodes).any():
            raise ValueError("Source graph already touches class node ids")
        full = np.concatenate((edges.astype(np.int64), train_edges.astype(np.int64) + [0, 0, label]))
        if np.isin(full[full[:, 2] == label, 0], held_out).any():
            raise ValueError("Leakage: a validation/test node has a class edge")
        np.save(stage / f"{variant}.npy", full.astype(np.int32))
        manifest["variants"][variant] = {"source_name": VARIANTS[args.dataset][variant], "relation_names": names + [LABEL_RELATION],
                                         "label_relation": label, "edges": int(len(full)), "edges_sha256": file_hash(stage / f"{variant}.npy")}
        print(f"exported {variant}: {len(full):,} edges ({len(train_edges):,} label edges), {label + 1} relations", flush=True)
    np.savez(stage / "labels.npz", class_nodes=class_nodes, **{f"{s}_{k}": v for s, (n, y) in split.items()
                                                               for k, v in (("nodes", n), ("y", y))})
    manifest["labels_sha256"] = file_hash(stage / "labels.npz")
    write_json(stage / "manifest.json", manifest)
    stage.rename(output)
    print(f"NC export: {output}")


# ----------------------------------------------------------------------------- shared
def load_task(args):
    directory = args.data / args.dataset
    manifest = json.loads((directory / "manifest.json").read_text())
    meta = manifest["variants"][args.variant]
    path = directory / f"{args.variant}.npy"
    edges = np.load(path, mmap_mode="r") if meta["edges"] > 50_000_000 else np.load(path)
    with np.load(directory / "labels.npz") as source:
        labels = {k: source[k].copy() for k in source.files}
    return manifest, meta, np.asarray(edges), labels


def queries(labels, split, limit=0):
    nodes, y = labels[f"{split}_nodes"], labels[f"{split}_y"]
    if limit:
        nodes, y = nodes[:limit], y[:limit]
    classes = labels["class_nodes"]
    # Candidates: the true class first (training loss convention), then the others in class order.
    candidates = np.stack([np.concatenate(([classes[c]], np.delete(classes, c))) for c in y])
    return np.column_stack((nodes, classes[y])), candidates, y


def class_scores(torch, model, graph, nodes, classes, relation, batch_size, device):
    scores = []
    with torch.no_grad():
        for start in range(0, len(nodes), batch_size):
            chunk = np.column_stack((nodes[start:start + batch_size], np.zeros(len(nodes[start:start + batch_size]))))
            batch = make_batch(torch, chunk.astype(np.int64), np.tile(classes, (len(chunk), 1)), relation, device)
            scores.append(model(graph, batch).float().cpu().numpy())
    return np.concatenate(scores)


# ----------------------------------------------------------------------------- eval
def run_eval(args):
    from wordnet_lp import setup

    torch, device, model, fine_tuning = setup(args)
    manifest, meta, edges, labels = load_task(args)
    stage = "finetuned" if fine_tuning else "zeroshot"
    if fine_tuning and fine_tuning["training_contract"]["dataset"] != f"nc_{args.dataset}":
        raise ValueError("Checkpoint was fine-tuned on a different task")
    tag = f"_s{fine_tuning['training_contract']['seed']}" if fine_tuning else ""
    output = args.output / f"motif_{stage}_nc_{args.dataset}_{args.run_tag}{tag}" / "physical" / args.variant
    if output.exists():
        raise FileExistsError(f"Run exists: {output}")
    model.requires_grad_(False)
    before = state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph = build_message_graph(edges, meta["relation_names"], manifest["num_nodes"], device)
    lift_seconds = time.perf_counter() - started
    nodes, y = labels["test_nodes"][:args.limit or None], labels["test_y"][:args.limit or None]
    classes = labels["class_nodes"]
    scores = class_scores(torch, model, graph, nodes, classes, meta["label_relation"], args.batch_size, device)
    if state_hash(model) != before:
        raise RuntimeError("Checkpoint state changed during frozen inference")
    predicted = top1_tail(scores, np.arange(len(classes)))  # Ties go to the smallest class id.
    metrics = classification_metrics(y, predicted, np.arange(len(classes)))
    record = {"model": "MOTIF", "dataset": f"nc_{args.dataset}", "variant": args.variant, "evaluation_stage": stage,
              **metrics, "num_queries": len(nodes), "limit": args.limit, "classes": len(classes),
              "checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
              "export_manifest_sha256": file_hash(args.data / args.dataset / "manifest.json"),
              "graph_edges_with_inverses": graph.num_edges, "lift_hyperedges": graph.relation_hypergraph.num_edges,
              "lift_seconds": lift_seconds, "total_seconds": time.perf_counter() - started,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(device), "gpu": torch.cuda.get_device_name(device),
              "parameters_sha256": before, "seed": fine_tuning["training_contract"]["seed"] if fine_tuning else None}
    if fine_tuning:
        record["training_contract"] = fine_tuning["training_contract"]
        record["selected_step"] = fine_tuning["selected_step"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motif-nc-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        np.savez_compressed(staged / "scores.npz", scores=scores, y=y, nodes=nodes, predicted=predicted)
        record["scores_sha256"] = file_hash(staged / "scores.npz")
        write_json(staged / "metrics.json", record)
        staged.rename(output)
    print(f"Saved {output}: accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")


# ----------------------------------------------------------------------------- finetune
def run_finetune(args):
    from wordnet_lp import setup

    torch, device, model, fine_tuning = setup(args)
    if fine_tuning:
        raise ValueError("Start each experiment from the released pretrained checkpoint")
    from motif import tasks

    manifest, meta, edges, labels = load_task(args)
    run_dir = args.output / f"motif_finetune_nc_{args.dataset}_{args.run_tag}" / f"train_{args.variant}" / f"seed{args.seed}"
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    native_builder = tasks.build_relation_hypergraph

    def build_lift(graph, *builder_args, **builder_kwargs):  # Same scalar-count fix as zero-shot inference.
        graph = native_builder(graph, *builder_args, **builder_kwargs)
        graph.relation_hypergraph.num_nodes = int(graph.relation_hypergraph.num_nodes)
        return graph

    tasks.build_relation_hypergraph = build_lift
    relation = meta["label_relation"]
    positives, candidates, _ = queries(labels, "train")
    val_pos, val_candidates, _ = queries(labels, "val", args.val_limit)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    initial_hash = state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph = build_message_graph(edges, meta["relation_names"], manifest["num_nodes"], device)
    contract = {
        "dataset": f"nc_{args.dataset}", "train_variant": args.variant, "seed": args.seed,
        "initial_checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
        "export_manifest_sha256": file_hash(args.data / args.dataset / "manifest.json"),
        "training_nodes": len(positives), "classes": candidates.shape[1],
        "edge_removal": "native_exact_batch_label_edge_and_inverse", "selection_metric": "validation_accuracy",
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "adversarial_temperature": args.adversarial_temperature, "max_grad_norm": args.max_grad_norm,
        "validate_every": args.validate_every, "val_limit": args.val_limit,
        "precision": "float32_tf32_disabled", "gpu": torch.cuda.get_device_name(device),
        "model_source": {p.relative_to(ROOT / "upstream").as_posix(): file_hash(p) for p in sorted((ROOT / "upstream/motif").rglob("*.py"))},
        "training_source": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/eval_lp.py",
                                                        ROOT / "script/finetune_imdb_lp.py")},
    }
    run_dir.mkdir(parents=True)
    write_json(run_dir / "training_config.json", contract)
    initial = validate(torch, model, graph, val_pos, val_candidates, relation, args.batch_size, device)
    print(f"Initial validation accuracy={initial['hits@1']:.4f} MRR={initial['mrr']:.4f}", flush=True)
    best, best_step = float("-inf"), None
    schedule, schedule_hash = batch_schedule(positives, args.batch_size, args.seed, f"nc_{args.dataset}"), hashlib.sha256()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=("step", "loss", "grad_norm_before_clip", "validation_accuracy",
                                                     "validation_mrr", "seconds"))
        writer.writeheader()
        for step in range(1, args.steps + 1):
            indices = next(schedule)
            schedule_hash.update(array_hash(positives[indices]).encode())
            model.train()
            batch = make_batch(torch, positives[indices], candidates[indices], relation, device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(graph, batch)
            labels_ = torch.zeros_like(scores)
            labels_[:, 0] = 1
            losses = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels_, reduction="none")
            weights = torch.ones_like(scores)
            with torch.no_grad():
                weights[:, 1:] = torch.softmax(scores[:, 1:] / args.adversarial_temperature, dim=-1)
            loss = ((losses * weights).sum(-1) / weights.sum(-1)).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "loss": float(loss.item()), "grad_norm_before_clip": float(norm.item()),
                      "seconds": time.perf_counter() - started}
            if step % args.validate_every == 0 or step == args.steps:
                result = validate(torch, model, graph, val_pos, val_candidates, relation, args.batch_size, device)
                record.update({"validation_accuracy": result["hits@1"], "validation_mrr": result["mrr"]})
                if result["hits@1"] > best:  # Ties keep the earlier checkpoint.
                    best, best_step = result["hits@1"], step
                    save_checkpoint(torch, run_dir / "best.pth", model, {
                        "training_contract": contract, "selected_step": step, "validation_accuracy": best,
                        "initial_parameters_sha256": initial_hash, "parameters_sha256": state_hash(model)})
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}, validation accuracy={result['hits@1']:.4f}, "
                      f"best step={best_step}, {record['seconds']:.0f}s", flush=True)
            elif step == 1 or step % args.log_every == 0:
                print(f"Step {step}/{args.steps}: loss={record['loss']:.4f}, {record['seconds']:.0f}s", flush=True)
            writer.writerow(record)
            history.flush()
    if state_hash(model) == initial_hash:
        raise RuntimeError("Model parameters did not change during fine-tuning")
    write_json(run_dir / "training_summary.json", {
        "training_contract": contract, "initial_validation": initial, "best_step": best_step,
        "best_validation_accuracy": best, "completed_steps": args.steps, "schedule_sha256": schedule_hash.hexdigest(),
        "total_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)})
    print(f"Saved fine-tuned checkpoint: {run_dir / 'best.pth'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("export", "eval", "finetune"))
    parser.add_argument("--dataset", choices=VARIANTS, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--bundle-root", type=Path, help="export imdb: the lab's IMDB/nc bundle directory")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream/ckpts/motif_3g.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "upstream/config/transductive/MOTIF_inference.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="eval: first N test nodes (smoke tests only)")
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("Invalid run tag")
    if args.mode != "export" and args.variant not in VARIANTS[args.dataset]:
        parser.error(f"--variant must be one of {list(VARIANTS[args.dataset])}")
    if args.mode == "export" and args.dataset == "imdb" and args.bundle_root is None:
        parser.error("export --dataset imdb needs --bundle-root")
    args.output = (args.output or (DATA_ROOT.parent if args.mode == "export" else ROOT / "results")).resolve()
    if args.mode == "export":
        args.output = args.output / "NC"
        args.output.mkdir(parents=True, exist_ok=True)
    for name in ("data", "checkpoint", "config"):
        setattr(args, name, getattr(args, name).resolve())
    if args.bundle_root:
        args.bundle_root = args.bundle_root.resolve()
    {"export": run_export, "eval": run_eval, "finetune": run_finetune}[args.mode](args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "upstream"))
    main()
