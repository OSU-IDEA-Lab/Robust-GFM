"""Zero-shot (no training) ULTRA evaluation of one checkpoint on one representation variant.

For every query (h, r, t) of the chosen split and both directions (predict t from (h, r); predict h from
(t, r)), ULTRA scores every node of the variant's graph. We keep the scores of the ORIGINAL entities only
(nodes a transformation adds, e.g. reified fact nodes, are never candidates) and save:

  rank_pess   filtered rank as ULTRA computes it (ties count against the answer), for comparability with
              published ULTRA numbers
  rank_tie    tie-aware filtered rank (1 + #better + #tied / 2), used for MRR and delta MRR in the summary
  cand_scores scores of [true answer, 50 fixed filtered negatives] per query and direction -- the vectors
              Kendall tau is computed on (same negatives on every variant, base/negatives_<split>.npy)

Outputs: <out>/scores.npz, <out>/metrics.json. Queries on relations that the variant made unanswerable
(reified away) are skipped and recorded.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# shared data format (GFM/common), optional private extras (pip install --target pydeps) and the untouched
# upstream checkout (setup_upstream.sh)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "pydeps"))
sys.path.insert(0, os.path.join(HERE, "upstream"))
os.environ["PATH"] = os.path.join(HERE, "pydeps", "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(HERE, "torch_extensions"))

import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.data import Data

# upstream ultra/layers.py parses torch_geometric.__version__ as integers, which fails on versions such as
# "2.8.0.post1". Keep only the numeric release part (the upstream checkout stays untouched).
torch_geometric.__version__ = ".".join(torch_geometric.__version__.split(".")[:3])

from kg_format import check_prefix, file_sha256, load_base, load_variant, write_json


def build_graph(variant, device):
    from ultra.tasks import build_relation_graph
    g = torch.as_tensor(variant["graph"], dtype=torch.long)
    R = len(variant["relations"])
    edge_index = torch.stack([g[:, 0], g[:, 2]])
    edge_type = g[:, 1]
    # ULTRA convention: relations 0..R-1 forward, R..2R-1 their (unnamed) inverses
    data = Data(edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),
                edge_type=torch.cat([edge_type, edge_type + R]),
                num_nodes=len(variant["entities"]), num_relations=2 * R)
    data = build_relation_graph(data)
    return data.to(device)


def load_model(ckpt, device):
    from ultra.models import Ultra
    with open(os.path.join(HERE, "upstream", "config", "transductive", "inference.yaml")) as f:
        text = f.read()
    # the template's jinja fields are irrelevant here; only the model block is read
    model_cfg = yaml.safe_load("\n".join(l for l in text.splitlines() if "{{" not in l))["model"]
    model = Ultra(rel_model_cfg=dict(model_cfg["relation_model"]), entity_model_cfg=dict(model_cfg["entity_model"]))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def filter_data(base, device):
    allt = np.concatenate([base["train"], base["valid"], base["test"], base["filter_extra"]])
    t = torch.as_tensor(allt, dtype=torch.long)
    return Data(edge_index=torch.stack([t[:, 0], t[:, 2]]), edge_type=t[:, 1],
                num_nodes=len(base["entities"])).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (smoke test)")
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ultra import tasks

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
    data = build_graph(variant, device)
    filt = filter_data(base, device)
    model = load_model(args.ckpt, device)
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
            batch = torch.as_tensor(queries[ids][:, [0, 2, 1]], dtype=torch.long, device=device)  # (h, t, r)
            t_batch, h_batch = tasks.all_negative(data, batch)
            preds = (model(data, t_batch)[:, :N0], model(data, h_batch)[:, :N0])
            t_mask, h_mask = tasks.strict_negative_mask(filt, batch)
            pos_h, pos_t, _ = batch.t()
            for d, (pred, mask, pos) in enumerate(((preds[0], t_mask, pos_t), (preds[1], h_mask, pos_h))):
                rank_pess[s: s + len(ids), d] = tasks.compute_ranking(pred, pos, mask).cpu().numpy()
                p = pred.gather(-1, pos.unsqueeze(-1))
                better = ((pred > p) & mask).sum(-1)
                tied = ((pred == p) & mask).sum(-1)
                rank_tie[s: s + len(ids), d] = (1 + better + tied / 2).cpu().numpy()
                neg = torch.as_tensor(negatives[ids, d], dtype=torch.long, device=device)
                cand = torch.cat([pos.unsqueeze(-1), neg], dim=1)
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
        "dataset_dir": os.path.basename(os.path.normpath(args.dataset_dir)), "variant": args.variant, "split": args.split,
        "checkpoint": os.path.basename(args.ckpt), "checkpoint_sha256": file_sha256(args.ckpt),
        "variant_hashes": variant["manifest"]["hashes"], "base_hashes": base["manifest"]["hashes"],
        "num_queries_evaluated": int(nq), "num_queries_in_split": int(len(queries)),
        "num_queries_skipped_unanswerable": int((~keep).sum()), "limit": args.limit,
        "num_graph_nodes": int(data.num_nodes), "num_graph_edges_with_inverses": int(data.num_edges),
        "pessimistic": summary(rank_pess), "tie_aware": summary(rank_tie),
        "seconds_setup": round(t_setup, 1), "seconds_eval": round(t_eval, 1),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    write_json(os.path.join(args.out, "metrics.json"), metrics)
    print(json.dumps({k: metrics[k] for k in ("variant", "checkpoint", "num_queries_evaluated", "pessimistic",
                                              "seconds_eval", "peak_gpu_mib")}, indent=1))


if __name__ == "__main__":
    main()
