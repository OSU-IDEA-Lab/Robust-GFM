"""Zero-shot SCR on one original label-entity graph (no parameter updates).

Node classification and the IMDb / DBLP link tasks are posed as link prediction
against a closed candidate list: genres, years, directors, or the 20 venues.
The message graph is raw/train.txt. Held-out queries are raw/valid.txt and
raw/test.txt. The predicted label is the argmax over candidates.txt when that
file exists, otherwise over the training tails of the test relation.

Outputs <out>/scores.npz and <out>/metrics.json.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
KG_ROOT = os.path.join(HERE, "kg-datasets")


def setup_paths():
    if not os.path.isdir(os.path.join(UPSTREAM, "ultra")):
        raise SystemExit("Missing upstream/ultra. Run bash setup_upstream.sh first.")
    sys.path.insert(0, UPSTREAM)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(HERE, ".torch_extensions"))
    install_moe_placeholder()
    from torch_geometric.inspector import Inspector
    if not hasattr(Inspector, "distribute"):  # renamed in PyG 2.5; SCR targets 2.4
        Inspector.distribute = Inspector.collect_param_data


def install_moe_placeholder():
    """SCR imports ultra.moe but does not ship it. The released checkpoint never builds that layer."""
    import types
    import torch.nn as nn

    class MoE(nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ultra.moe is not shipped with SCR; MoE layers are unavailable")

    module = types.ModuleType("ultra.moe")
    module.MoE = MoE
    sys.modules["ultra.moe"] = module


def read_triples(raw_dir):
    """Same entity/relation ordering as FoundationNodeKG.load_file over train, valid, test."""
    ent, rel, out = {}, {}, {}
    for split in ("train", "valid", "test"):
        rows = []
        with open(os.path.join(raw_dir, f"{split}.txt")) as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                for e in (h, t):
                    ent.setdefault(e, len(ent))
                rel.setdefault(r, len(rel))
                rows.append((ent[h], ent[t], rel[r]))
        out[split] = rows
    return ent, rel, out


def label_candidates(raw_dir, ent, trip):
    """The label relation is the test relation. Candidates are candidates.txt, else its training tails."""
    label_rel = trip["test"][0][2]
    path = os.path.join(raw_dir, "candidates.txt")
    if os.path.isfile(path):
        with open(path) as f:
            return label_rel, sorted(ent[line.strip()] for line in f if line.strip())
    return label_rel, sorted({t for _, t, r in trip["train"] if r == label_rel})


def scr_graph_and_model(name, raw_dir, ckpt, device, k, p):
    import torch
    from ultra import datasets, tasks, util
    from ultra.models import SCR, GraphEncoding

    cls = type(name, (datasets.FoundationNodeKG,), {"name": name})
    processed = os.path.join(KG_ROOT, cls.base_name, name, "processed")
    if os.path.isdir(processed):
        import shutil
        shutil.rmtree(processed)
    test_data = cls(root=KG_ROOT)[2]
    test_data = tasks.build_relation_graph(test_data)
    test_data = tasks.build_relation_types(test_data)

    cfg = util.load_config(os.path.join(UPSTREAM, "config", "eval_all.yaml"), context={})
    cfg.model["entity_model"]["feat_type"] = "end"
    if cfg.model.entity_model["self_loop"]:
        test_data = tasks.build_selfloop_edges(test_data)
    test_data = test_data.to(device)

    feat_path = os.path.join(raw_dir, "node_feats.pt")
    node_features = torch.load(feat_path) if os.path.isfile(feat_path) else None
    encoder = GraphEncoding(cfg.model["entity_model"]["feat_dim"], os.path.join(UPSTREAM, "resource") + os.sep,
                            {}, k=3, p=0.8)
    graph = encoder.build_semantic_graph(test_data, name, "bert", "test", node_feature=node_features,
                                         ifload=False, k=k, p=p)

    model = SCR(rel_model_cfg=cfg.model.relation_model, entity_model_cfg=cfg.model.entity_model)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    model = model.to(device).eval()
    ent, rel, trip = read_triples(raw_dir)
    return ent, rel, trip, graph, model, dict(missing=len(missing), unexpected=len(unexpected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="folder under kg-datasets/FoundationNodeKG")
    ap.add_argument("--ckpt", default=os.path.join(UPSTREAM, "ckpts", "scr_3g.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--k", type=int, default=None, help="SCR semantic neighbours (default: GraphEncoding's 3)")
    ap.add_argument("--p", type=float, default=None, help="SCR similarity threshold (default: 0.8)")
    ap.add_argument("--seed", type=int, default=1024,
                    help="SCR's feature unifier is a randomized SVD, so its output is seed-dependent")
    args = ap.parse_args()
    setup_paths()

    import random

    import numpy as np
    import torch
    from sklearn.metrics import f1_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir = os.path.join(KG_ROOT, "FoundationNodeKG", args.dataset, "raw")
    t0 = time.time()
    ent, rel, trip, graph, model, load_info = scr_graph_and_model(
        args.dataset, raw_dir, args.ckpt, device, args.k, args.p)
    prep_sec = time.time() - t0

    label_rel, label_ids = label_candidates(raw_dir, ent, trip)
    id2ent = {v: k for k, v in ent.items()}
    genre_names = [id2ent[i] for i in label_ids]
    label_col = {name: j for j, name in enumerate(genre_names)}
    genre_ids = torch.tensor(label_ids, device=device)

    os.makedirs(args.out, exist_ok=True)
    arrays, metrics = {}, dict(model="scr", dataset=args.dataset, ckpt=os.path.abspath(args.ckpt),
                               seed=args.seed, prep_sec=round(prep_sec, 2), state_dict_load=load_info)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for split in args.splits:
        q = torch.tensor(trip[split], dtype=torch.long, device=device)
        assert (q[:, 2] == label_rel).all()
        scores = []
        t1 = time.time()
        with torch.no_grad():
            for s in range(0, len(q), args.batch_size):
                b = q[s:s + args.batch_size]
                h = b[:, 0:1].expand(-1, len(genre_ids))
                t_batch = torch.stack([h, genre_ids.expand_as(h), b[:, 2:3].expand_as(h)], dim=-1)
                out = model(graph, t_batch)
                scores.append((out[0] if isinstance(out, tuple) else out).float().cpu())
        scores = torch.cat(scores).numpy()
        movie = np.array([id2ent[i] for i in q[:, 0].tolist()])
        label = np.array([label_col[id2ent[i]] for i in q[:, 1].tolist()])
        pred = scores.argmax(1)
        arrays.update({f"{split}_node": movie, f"{split}_label": label, f"{split}_scores": scores})
        metrics[split] = dict(n=int(len(q)), accuracy=float((pred == label).mean()),
                              macro_f1=float(f1_score(label, pred, average="macro")),
                              pred_hist=np.bincount(pred, minlength=len(genre_ids)).tolist(),
                              sec=round(time.time() - t1, 2))
        print(f"scr {args.dataset} {split}: acc={metrics[split]['accuracy']:.4f} "
              f"macroF1={metrics[split]['macro_f1']:.4f} pred_hist={metrics[split]['pred_hist']}")
    if device.type == "cuda":
        metrics["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    np.savez(os.path.join(args.out, "scores.npz"), label_names=np.array(genre_names), **arrays)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
