"""Frozen scoring of released or fine-tuned MOTIF checkpoints on portable LP exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from lp_data import ROOT, PREPROCESSED_ROOT, array_hash, file_hash, full_target_negatives, write_json
from lp_metrics import average_ranks, point_metrics


def result_root(output, dataset, tag, stage="zeroshot"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Run tag must contain only letters, digits, underscores and hyphens")
    if stage not in ("zeroshot", "finetuned"):
        raise ValueError("Unknown evaluation stage")
    return output / f"motif_{stage}_{dataset}_lp_{tag}"


def query_subset(positives, limit, seed, task):
    if limit <= 0 or limit >= len(positives):
        return np.arange(len(positives), dtype=np.int64)
    keys = [hashlib.sha256(f"{seed}|{task}|{h}|{t}".encode()).digest() for h, t in positives]
    return np.sort(sorted(range(len(keys)), key=keys.__getitem__)[:limit]).astype(np.int64)


def build_message_graph(edges, names, num_nodes, device):
    """Use the released Lift implementation on context edges with one inverse copy."""
    import torch
    from torch_geometric.data import Data
    from motif import tasks

    forward = torch.as_tensor(edges, dtype=torch.long, device=device)
    count = len(names)
    graph = Data(edge_index=torch.cat((forward[:, :2].t(), forward[:, [1, 0]].t()), dim=1),
                 edge_type=torch.cat((forward[:, 2], forward[:, 2] + count)),
                 num_nodes=int(num_nodes), num_relations=2 * count, device=device)
    graph = tasks.build_relation_hypergraph(graph)
    # The builder uses a scalar tensor for this count. RelHCNet increments its
    # local count for padding; a Python int prevents mutation of the stored graph.
    graph.relation_hypergraph.num_nodes = int(graph.relation_hypergraph.num_nodes)
    return graph


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(array_hash(value.detach().cpu().numpy()).encode())
    return result.hexdigest()


def load_inputs(args):
    root = json.loads((args.data / "manifest.json").read_text())
    if root["format"] != f"motif_{args.dataset}_lp_v1":
        raise ValueError("Wrong portable dataset format")
    directory = args.data / args.task
    if file_hash(directory / "manifest.json") != root["tasks"][args.task]:
        raise ValueError("Task manifest hash differs")
    meta = json.loads((directory / "manifest.json").read_text())
    for name in ("queries.npz", f"{args.variant}.npz"):
        if file_hash(directory / name) != meta["files"][name]:
            raise ValueError(f"Input checksum differs: {name}")
    with np.load(directory / "queries.npz") as source:
        positives = source[f"{args.split}_pos"].copy()
        pool = (source["candidate_pool"].copy() if "candidate_pool" in source else
                np.flatnonzero(source["node_type"] == (4 if args.task == "movie_year" else 1)))
        negatives = (full_target_negatives(positives, pool) if args.dataset == "dblp" else
                     source[f"{args.split}_neg"].copy())
    index = query_subset(positives, args.limit, args.seed, args.task)
    positives, negatives = positives[index], negatives[index]
    candidates = np.column_stack((positives[:, 1], negatives))
    with np.load(directory / f"{args.variant}.npz") as source:
        edges = source[args.view].copy()
    if array_hash(edges) != meta["variants"][args.variant]["edges_sha256"][args.view]:
        raise ValueError("Effective graph checksum differs")
    if not np.isin(candidates, pool).all() or np.any(pool[:-1] >= pool[1:]):
        raise ValueError("Invalid full-type candidate pool")
    return meta, edges, positives, candidates, pool, index


def worker(args):
    import torch
    import torch_geometric
    import triton
    from motif import util
    from motif.models import MOTIF

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("MOTIF's Triton inference requires a visible CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    output = result_root(args.output, args.dataset, args.run_tag, args.stage) / args.task / args.view / (args.variant + ("__repeat" if args.repeat else ""))
    if output.exists():
        raise FileExistsError(f"Run exists: {output}; choose a new --run-tag")
    meta, edges, positives, candidates, pool, index = load_inputs(args)
    cfg = util.load_config(str(args.config), context={"dataset": "custom", "epochs": 0,
                           "bpe": None, "gpus": [device.index or 0], "ckpt": str(args.checkpoint)})
    model = MOTIF(cfg.model.relation_model, cfg.model.entity_model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    fine_tuning = checkpoint.get("fine_tuning")
    if (args.stage == "finetuned") != (fine_tuning is not None):
        raise ValueError("Checkpoint/stage mismatch: released weights use zeroshot; adapted weights use finetuned")
    if fine_tuning is not None:
        trained = fine_tuning["training_contract"]
        if trained["dataset"] != args.dataset or trained["task"] != args.task:
            raise ValueError("Fine-tuned checkpoint dataset/task differs from evaluation")
        if trained["config_sha256"] != file_hash(args.config):
            raise ValueError("Fine-tuning and evaluation model configurations differ")
        if trained["export_manifest_sha256"] != file_hash(args.data / "manifest.json"):
            raise ValueError("Fine-tuning and evaluation dataset exports differ")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    parameters_before = state_hash(model)
    if fine_tuning is not None and parameters_before != fine_tuning["parameters_sha256"]:
        raise ValueError("Fine-tuned checkpoint weights differ from their recorded hash")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    graph = build_message_graph(edges, meta["relation_names"], meta["num_nodes"], device)
    torch.cuda.synchronize(device)
    lift_seconds = time.perf_counter() - started
    scores = []
    target = meta["target_relation"]
    with torch.no_grad():
        for start in range(0, len(positives), args.batch_size):
            heads = torch.as_tensor(positives[start:start + args.batch_size, 0], device=device)
            batch = torch.empty((len(heads), len(pool), 3), dtype=torch.long, device=device)
            batch[:, :, 0] = heads[:, None]
            batch[:, :, 1] = torch.as_tensor(pool, device=device)[None, :]
            batch[:, :, 2] = target
            # Native forward: relation encoder -> entity encoder -> decoder.
            scores.append(model(graph, batch).cpu().numpy())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    full = np.concatenate(scores)
    selected = np.take_along_axis(full, np.searchsorted(pool, candidates), axis=1)
    calculated = point_metrics(selected, positives, pool, full)
    parameters_after = state_hash(model)
    if parameters_after != parameters_before:
        raise RuntimeError("Checkpoint state changed during frozen inference")
    contract = {"model": "MOTIF", "dataset": args.dataset, "task": args.task,
                "evaluation_stage": args.stage,
                "checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
                "export_manifest_sha256": file_hash(args.data / "manifest.json"),
                "query_sha256": array_hash(positives, candidates, index, pool),
                "relation_names": meta["relation_names"], "seed": args.seed, "split": args.split,
                "batch_size": args.batch_size, "limit": args.limit, "precision": "float32_tf32_disabled",
                "rank_convention": "1+better+tied/2", "top1_ties": "smallest_global_tail_id",
                "macro_labels": "all_target_type_entities_zero_division_0",
                "torch": torch.__version__, "pyg": torch_geometric.__version__, "triton": triton.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "model_source": {p.relative_to(ROOT / "upstream").as_posix(): file_hash(p) for p in sorted((ROOT / "upstream/motif").rglob("*.py"))},
                "adapter_source": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "script/lp_metrics.py", ROOT / "script/lp_data.py")}}
    if fine_tuning is not None:
        contract["fine_tuning"] = fine_tuning
    metrics = {"model": "MOTIF", "dataset": args.dataset, "task": args.task,
               "variant": args.variant, "view": args.view, "repeat": args.repeat,
               "comparison_contract": contract, "num_queries": len(positives),
               "benchmark_candidates": candidates.shape[1], "full_type_candidates": len(pool),
               "graph_sha256": array_hash(edges), "graph_edges_with_inverses": graph.num_edges,
               "lift_hyperedges": graph.relation_hypergraph.num_edges,
               "lift_relation_nodes": graph.relation_hypergraph.num_nodes,
               "lift_seconds": lift_seconds, "total_seconds_including_first_jit": elapsed,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
               "parameters_before": parameters_before, "parameters_after": parameters_after,
               **calculated}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motif-run-", dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        np.savez_compressed(stage / "scores.npz", scores=selected, full_type_scores=full,
                            candidate_pool=pool, candidates=candidates, positives=positives,
                            query_index=index, ranks=average_ranks(selected))
        metrics["scores_sha256"] = file_hash(stage / "scores.npz")
        write_json(stage / "metrics.json", metrics)
        stage.rename(output)
    print(f"Saved {output}: MRR={calculated['mrr']:.4f}, full-type accuracy={calculated['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imdb", "dblp"), required=True)
    parser.add_argument("--stage", choices=("zeroshot", "finetuned"), default="zeroshot")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "upstream/ckpts/motif_3g.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "upstream/config/transductive/MOTIF_inference.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-tag")
    parser.add_argument("--task")
    parser.add_argument("--view", choices=("physical", "compiled"))
    parser.add_argument("--variant")
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("Batch size must be positive and limit nonnegative")
    args.data = (args.data or PREPROCESSED_ROOT / args.dataset.upper()).resolve()
    args.output, args.checkpoint, args.config = (p.resolve() for p in (args.output, args.checkpoint, args.config))
    args.run_tag = args.run_tag or (f"smoke{args.limit}" if args.limit else "full")
    result_root(args.output, args.dataset, args.run_tag, args.stage)
    if args.single:
        if not all((args.task, args.view, args.variant)):
            parser.error("A single run needs task, view and variant")
        worker(args)
        return
    manifest = json.loads((args.data / "manifest.json").read_text())
    tasks = [args.task] if args.task else list(manifest["tasks"])
    for task in tasks:
        if task not in manifest["tasks"]:
            parser.error(f"Task absent from export: {task}")
        meta = json.loads((args.data / task / "manifest.json").read_text())
        variants = [args.variant] if args.variant else list(meta["variants"])
        if any(v not in meta["variants"] for v in variants):
            parser.error("Variant absent from export")
        runs = [(view, variant, False) for view in ([args.view] if args.view else ("physical", "compiled")) for variant in variants]
        if not args.no_repeat and (args.view is None or args.view == "physical"):
            runs.append(("physical", "v1", True))
        for view, variant, repeat in runs:
            command = [sys.executable, str(Path(__file__).resolve()), "--single", "--dataset", args.dataset,
                       "--data", str(args.data), "--output", str(args.output), "--checkpoint", str(args.checkpoint),
                       "--config", str(args.config), "--device", args.device, "--batch-size", str(args.batch_size),
                       "--seed", str(args.seed), "--split", args.split, "--limit", str(args.limit),
                       "--run-tag", args.run_tag, "--task", task, "--view", view, "--variant", variant]
            if repeat:
                command.append("--repeat")
            if args.stage != "zeroshot":
                command.extend(["--stage", args.stage])
            env = {**os.environ, "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "upstream"))
    main()
