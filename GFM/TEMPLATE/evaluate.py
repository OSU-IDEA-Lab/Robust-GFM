"""Zero-shot link-prediction evaluation of one checkpoint on one representation variant (template).

Everything model-independent is done here: loading the shared format, skipping unanswerable queries, filtering
against the base truth set, both rank conventions, the 51 candidate scores and the output files required by the
contract in GFM/README.md. A new GFM only implements the two functions marked MODEL:

  load_model(ckpt, device)                      -> model
  score(model, graph, queries, direction)       -> float tensor [B, num_graph_nodes]

`graph` is whatever `prepare_graph` returns for the variant (MODEL as well; default: the raw triples).

    python evaluate.py --dataset-dir ../data/WN18RR --variant no_changes --ckpt <checkpoint> --out <dir> --limit 64
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import numpy as np
import torch

from kg_format import check_prefix, file_sha256, load_base, load_variant, write_json


# --------------------------------------------------------------------------------------------- MODEL: fill in

def load_model(ckpt, device):
    """Load the pretrained checkpoint, in eval mode, on `device`."""
    raise NotImplementedError


def prepare_graph(variant, device):
    """Turn the variant into the model's input once per run. `variant["graph"]` is an int array [E, 3] of
    (h, r, t) over len(variant["entities"]) nodes and len(variant["relations"]) relations. Add inverse edges here
    only if the model expects them (see docs/EVALUATION_GUIDE.md §A3)."""
    return {"triples": torch.as_tensor(variant["graph"], dtype=torch.long, device=device),
            "num_nodes": len(variant["entities"]), "num_relations": len(variant["relations"])}


def score(model, graph, queries, direction):
    """Scores of every node of the variant graph as the answer.

    queries: long tensor [B, 3] of (h, r, t) in base ids. direction 0 = predict t from (h, r);
    direction 1 = predict h from (t, r). Returns a float tensor [B, graph["num_nodes"]]."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------- shared: do not change

def answer_index(base):
    """Known answers per (anchor, relation) and direction, from the base truth set only (never the variant)."""
    allt = np.concatenate([base["train"], base["valid"], base["test"], base["filter_extra"]]).tolist()
    tails, heads = {}, {}
    for h, r, t in allt:
        tails.setdefault((h, r), []).append(t)
        heads.setdefault((t, r), []).append(h)
    return tails, heads


def answer_mask(index, queries, direction, num_entities, device):
    """True for every known answer of each query other than the one being ranked."""
    mask = torch.zeros(len(queries), num_entities, dtype=torch.bool, device=device)
    for i, (h, r, t) in enumerate(queries.tolist()):
        anchor, target = (h, t) if direction == 0 else (t, h)
        known = index[direction].get((anchor, r), [])
        if known:
            mask[i, known] = True
        mask[i, target] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (smoke test)")
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = load_base(args.dataset_dir)
    variant = load_variant(args.dataset_dir, args.variant)
    check_prefix(variant, base)
    N0 = len(base["entities"])
    queries = base[args.split]
    negatives = np.load(os.path.join(base["dir"], f"negatives_{args.split}.npy"))
    unanswerable = set(variant["manifest"].get("unanswerable_query_relations", []))
    keep = np.array([r not in unanswerable for r in queries[:, 1]])
    qidx = np.nonzero(keep)[0]
    if args.limit:
        qidx = qidx[: args.limit]

    t0 = time.time()
    graph = prepare_graph(variant, device)
    model = load_model(args.ckpt, device)
    index = answer_index(base)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_setup = time.time() - t0

    nq = len(qidx)
    rank_pess = np.zeros((nq, 2), dtype=np.int64)
    rank_tie = np.zeros((nq, 2), dtype=np.float64)
    cand_scores = np.zeros((nq, 2, 1 + negatives.shape[2]), dtype=np.float32)
    t1 = time.time()
    with torch.no_grad():
        for s in range(0, nq, args.batch_size):
            ids = qidx[s: s + args.batch_size]
            batch = torch.as_tensor(queries[ids], dtype=torch.long, device=device)
            for d in (0, 1):
                pred = score(model, graph, batch, d)[:, :N0]        # base entities only
                pos = batch[:, 2] if d == 0 else batch[:, 0]
                p = pred.gather(-1, pos.unsqueeze(-1))
                other = ~answer_mask(index, batch, d, N0, device)    # candidates: not another known answer
                better = ((pred > p) & other).sum(-1)
                tied = ((pred == p) & other).sum(-1) - 1            # the answer itself is not a tie
                rank_pess[s: s + len(ids), d] = (1 + better + tied).cpu().numpy()
                rank_tie[s: s + len(ids), d] = (1 + better + tied / 2).cpu().numpy()
                cand = torch.cat([pos.unsqueeze(-1), torch.as_tensor(negatives[ids, d], dtype=torch.long,
                                                                     device=device)], dim=1)
                cand_scores[s: s + len(ids), d] = pred.gather(-1, cand).float().cpu().numpy()
    t_eval = time.time() - t1

    def summary(r):
        r = r.reshape(-1)
        return {"mrr": float(np.mean(1.0 / r)), "hits@1": float(np.mean(r <= 1)),
                "hits@3": float(np.mean(r <= 3)), "hits@10": float(np.mean(r <= 10)), "mr": float(np.mean(r))}

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "scores.npz"), query_index=qidx, rank_pess=rank_pess,
                        rank_tie=rank_tie, cand_scores=cand_scores)
    metrics = {
        "dataset_dir": os.path.basename(os.path.normpath(args.dataset_dir)), "variant": args.variant,
        "split": args.split, "checkpoint": os.path.basename(args.ckpt), "checkpoint_sha256": file_sha256(args.ckpt),
        "variant_hashes": variant["manifest"]["hashes"], "base_hashes": base["manifest"]["hashes"],
        "num_queries_evaluated": int(nq), "num_queries_in_split": int(len(queries)),
        "num_queries_skipped_unanswerable": int((~keep).sum()), "limit": args.limit,
        "pessimistic": summary(rank_pess), "tie_aware": summary(rank_tie),
        "seconds_setup": round(t_setup, 1), "seconds_eval": round(t_eval, 1),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    write_json(os.path.join(args.out, "metrics.json"), metrics)
    print(json.dumps({k: metrics[k] for k in ("variant", "checkpoint", "num_queries_evaluated", "tie_aware",
                                              "seconds_eval", "peak_gpu_mib")}, indent=1))


if __name__ == "__main__":
    main()
