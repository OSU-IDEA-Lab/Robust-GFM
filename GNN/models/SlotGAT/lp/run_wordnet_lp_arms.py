#!/usr/bin/env python3
"""SlotGAT × WordNet (WN18RR family) link prediction — the four arms, one contract.

    --arm originals     SlotGAT on one variant's PHYSICAL graph            (no_changes | all_inverse_edges | transitive_edges)
    --arm universal     SlotGAT on the universal PHYSICAL graph            (universal_edges = the union)
    --arm invariant     SlotGAT on the compiled SEMANTIC program of a variant (identical tensors on every variant)
    --arm augmentation  ONE SlotGAT trained jointly on the three variants' physical graphs; evaluated on each

Inputs are the RGCN WordNet structural bundles (``RGCN_structural/WORDNET/universal_parity_full/full/
intersection/<variant>/graph.pt``, the same program CMPNN consumes): physical edges typed by the variant's
own relation names, semantic edges typed by the 2,087 semantic relations (11 base + 11 inverse + 2,065
dependency shortcuts), and the official supervision (86,835 / 3,034 / 3,134 base-relation triples).
The two relation lists are aligned index-wise, so one shared edge-type vocabulary serves every arm.

Model / training contract: deterministic parity SlotGAT
(lp_parity_model.py), entity table 128-d, 2 layers × 4 heads + 1-head output, concat-of-layers +
LayerNorm readout, diagonal DistMult over the 11 task relations, 256 uniform negatives per positive,
cross-entropy, Adam lr 1e-3 wd 0, batch 2048, up to 400 epochs, validation every 5, patience 8.
Message graph = the arm's edges + a synthetic reverse edge where the graph has no physical reverse
pair + one self loop per entity. Training edges are NOT masked from the message graph (Q01 > Q00).

Outputs (RGCN WordNet result layout):
    <out-root>/<arm>/full/intersection/<variant>/seed_<s>/{metrics.json, test_scores.npz, history.json, best_model.pt}
plus the gauntlet material: input hashes (edges, supervision, negatives-RNG), per-evaluation parameter hashes,
final checkpoint hash. Evaluation = RGCN/wordnet_structural_eval.evaluate_ranking (imported read-only):
filtered full-entity ranks both directions + the fixed 50-negative candidate sets (seed 1566911444).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RGCN_DIR = HERE.parent.parent.parent / "common"
for p in (str(HERE), str(RGCN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
from lp_parity_model import DistMultDiag, MessagePlan, ParitySlotGATLPEncoder  # noqa: E402
from wordnet_structural_eval import build_filter_dicts, evaluate_ranking  # noqa: E402  (RGCN, read-only)

BUNDLE_ROOT = Path(os.environ.get("INV_GNN_DATA", str(HERE.parent / "data"))) / "preprocessed/RGCN_structural/WORDNET/universal_parity_full/full/intersection"
PHYSICAL_VARIANTS = ("no_changes", "all_inverse_edges", "transitive_edges")
SEEDS = (1566911444, 20241017, 20251017)


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_hash(*modules) -> str:
    h = hashlib.sha256()
    for m in modules:
        for k, v in m.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- data
class Bundle:
    def __init__(self, variant: str, graph: str):
        d = torch.load(BUNDLE_ROOT / variant / "graph.pt", map_location="cpu", weights_only=False)
        assert d["format_version"] == "wordnet_rgcn_conditional_universal_parity_v2", d["format_version"]
        self.variant, self.graph = variant, graph
        self.num_entities = int(d["num_entities"])
        self.relation_names = list(d["semantic_relation_names"]) if graph == "semantic" else list(d["physical_relation_names"])
        assert len(list(d["semantic_relation_names"])) == len(list(d["physical_relation_names"]))
        key = "semantic" if graph == "semantic" else "physical"
        ei, et = d[f"{key}_edge_index"], d[f"{key}_edge_type"]
        edges = torch.stack([ei[0], ei[1], et], 1).to(torch.int64)
        # canonical order (rel, dst, src), duplicate-free — one fixed order for every variant
        order = np.lexsort((edges[:, 0].numpy(), edges[:, 1].numpy(), edges[:, 2].numpy()))
        self.edges = edges[torch.from_numpy(order)].contiguous()
        assert not bool((self.edges[1:] == self.edges[:-1]).all(1).any()), "duplicate edges"
        self.task_relation_names = list(d["task_relation_names"])
        self.train_pos = torch.as_tensor(d["task_train_pos"], dtype=torch.int64)   # (h, task_r, t)
        self.val_pos = torch.as_tensor(d["val_pos"], dtype=torch.int64)
        self.test_pos = torch.as_tensor(d["test_pos"], dtype=torch.int64)
        self.stored_hashes = {"physical": str(d["physical_hash"]), "semantic": str(d["semantic_hash"]), "supervision": str(d["supervision_hash"])}
        self.edge_hash = sha(self.edges.numpy())
        self.supervision_hash = sha(self.train_pos.numpy(), self.val_pos.numpy(), self.test_pos.numpy())


def build_plan(b: Bundle, device) -> tuple[MessagePlan, dict]:
    """Edges + synthetic reverse (type = R + rel) where no physical reverse pair exists + self loops (type 2R)."""
    R = len(b.relation_names)
    e = b.edges.numpy()
    pairs = set(map(tuple, e[:, :2].tolist()))
    rev = [(t, h, R + r) for h, t, r in e.tolist() if h != t and (t, h) not in pairs]
    rev = np.asarray(rev, dtype=np.int64).reshape(-1, 3)
    n = b.num_entities
    loops = np.stack([np.arange(n), np.arange(n), np.full(n, 2 * R)], 1)
    all_e = np.concatenate([e, rev, loops], 0)
    src = torch.from_numpy(all_e[:, 0]).to(device); dst = torch.from_numpy(all_e[:, 1]).to(device); ch = torch.from_numpy(all_e[:, 2]).to(device)
    stats = {"edges": int(len(e)), "synthetic_reverse": int(len(rev)), "self_loops": n, "messages": int(len(all_e)),
             "num_edge_types": 2 * R + 1, "relation_types_present": int(len(np.unique(e[:, 2]))), "message_hash": sha(all_e)}
    return MessagePlan(src, dst, ch, n), stats


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(enc, dec, plan, positives, tail_f, head_f, n, device, save):
    enc.eval(); dec.eval()
    ent = enc(plan)
    return evaluate_ranking(dec, ent, positives.numpy(), tail_filters=tail_f, head_filters=head_f, num_entities=n,
                            device=device, eval_batch_size=128, binary_k=50, candidate_seed=1566911444, save_candidates=save)


def lp_loss(logits: torch.Tensor, mode: str, zeros: torch.Tensor) -> torch.Tensor:
    """logits [B, 1 + K] with the positive in column 0."""
    if mode == "ce":
        return F.cross_entropy(logits, zeros)
    return -(F.logsigmoid(logits[:, 0]).mean() + F.logsigmoid(-logits[:, 1:]).mean())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["originals", "universal", "invariant", "augmentation"], required=True)
    ap.add_argument("--variant", default=None, help="originals/invariant: one physical variant; ignored for universal/augmentation")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=HERE.parent / "results/wn18rr_lp")
    # contract (defaults = Q01)
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--edge-feats", type=int, default=64)
    ap.add_argument("--feat-drop", type=float, default=0.2); ap.add_argument("--attn-drop", type=float, default=0.2)
    ap.add_argument("--slope", type=float, default=0.05); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--readout", choices=["concat_ln", "last_raw"], default="concat_ln")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=2048); ap.add_argument("--negatives", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400, help="epoch cap (augmentation: super-epochs = epochs // n_variants)")
    ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--stop-after", type=int, default=0, help="smoke: stop after this many epochs")
    ap.add_argument("--loss", choices=["ce", "neg_logsigmoid"], default="ce",
                    help="ce = softmax over (positive + negatives) [study default]; neg_logsigmoid = the per-candidate binary objective "
                         "-(logsigmoid(pos).mean() + logsigmoid(-neg).mean()) that RGCN / CMPNN / MAGNN use (decision boundary at logit 0, "
                         "so the tables' precision/recall/F1 at threshold 0 are meaningful)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # determinism: every CUDA op with a deterministic alternative uses it; aggregation is bucketed (no atomics)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arm == "originals":
        assert args.variant in PHYSICAL_VARIANTS, args.variant
        variants, graph = [args.variant], "physical"
    elif args.arm == "universal":
        variants, graph = ["universal_edges"], "physical"
    elif args.arm == "invariant":
        assert args.variant in PHYSICAL_VARIANTS, args.variant
        variants, graph = [args.variant], "semantic"
    else:
        variants, graph = list(PHYSICAL_VARIANTS), "physical"
    run_dirs = {v: args.out_root / args.arm / "full" / "intersection" / v / f"seed_{args.seed}" for v in variants}
    if all((d / "metrics.json").exists() for d in run_dirs.values()) and not args.force:
        print(f"[skip] {args.arm} {variants} seed {args.seed}: metrics.json exists"); return
    tag = f"{args.arm}/{'+'.join(variants)}/s{args.seed}"

    bundles = [Bundle(v, graph) for v in variants]
    ref = bundles[0]
    for b in bundles[1:]:
        assert b.supervision_hash == ref.supervision_hash and b.num_entities == ref.num_entities
    n = ref.num_entities
    plans, plan_stats = {}, {}
    for b in bundles:
        plans[b.variant], plan_stats[b.variant] = build_plan(b, device)
        print(f"[{tag}] {b.variant} {graph}: edge_hash {b.edge_hash[:12]} (stored {b.stored_hashes[graph][:12]}) {plan_stats[b.variant]}", flush=True)
    R = len(ref.relation_names)
    tail_f, head_f = build_filter_dicts(ref.train_pos.numpy(), ref.val_pos.numpy(), ref.test_pos.numpy())

    enc = ParitySlotGATLPEncoder(num_entities=n, num_edge_types=2 * R + 1, hidden=args.hidden, out_dim=args.out_dim,
                                 num_layers=args.layers, num_heads=args.heads, edge_feats=args.edge_feats, feat_drop=args.feat_drop,
                                 attn_drop=args.attn_drop, slope=args.slope, alpha=args.alpha, readout=args.readout).to(device)
    dec = DistMultDiag(len(ref.task_relation_names), enc.embedding_dim).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    n_params = sum(p.numel() for p in params); param_bytes = sum(p.numel() * p.element_size() for p in params)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    init_hash = state_hash(enc, dec)
    print(f"[{tag}] entities {n} task relations {len(ref.task_relation_names)} edge types {2*R+1} params {n_params:,} "
          f"emb_dim {enc.embedding_dim} init_hash {init_hash[:12]} supervision_hash {ref.supervision_hash[:12]}", flush=True)

    train = ref.train_pos.to(device)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    n_var = len(variants)
    super_epochs = args.epochs // n_var if args.arm == "augmentation" else args.epochs
    eval_every = max(1, args.eval_every // n_var) if args.arm == "augmentation" else args.eval_every
    history, best, best_state, bad = [], None, None, 0
    peak_train = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    for epoch in range(1, super_epochs + 1):
        enc.train(); dec.train()
        losses = []
        for v in variants:                       # augmentation: one full pass per variant per super-epoch
            plan = plans[v]
            perm = rng.permutation(len(train))
            for s in range(0, len(perm), args.batch_size):
                bidx = torch.from_numpy(perm[s:s + args.batch_size]).to(device)
                pos = train[bidx]
                B, K = len(bidx), args.negatives
                neg = pos.repeat_interleave(K, 0)
                corrupt_head = torch.rand(B * K, device=device, generator=gen) < 0.5
                rand = torch.randint(0, n, (B * K,), device=device, generator=gen)
                neg[:, 0] = torch.where(corrupt_head, rand, neg[:, 0])
                neg[:, 2] = torch.where(corrupt_head, neg[:, 2], rand)
                ent = enc(plan)
                logits = torch.cat([dec.score(ent, pos).view(B, 1), dec.score(ent, neg).view(B, K)], 1)
                loss = lp_loss(logits, args.loss, torch.zeros(B, dtype=torch.long, device=device))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(float(loss))
        if device.type == "cuda":
            peak_train = max(peak_train, torch.cuda.max_memory_allocated(device))

        if epoch % eval_every == 0 or epoch == super_epochs or (args.stop_after and epoch >= args.stop_after):
            vals = {v: evaluate(enc, dec, plans[v], ref.val_pos, tail_f, head_f, n, device, save=False)[0] for v in variants}
            score = float(np.mean([m["filtered_MRR"] for m in vals.values()]))
            rec = {"epoch": epoch, "variant_epochs": epoch * n_var, "train_loss": float(np.mean(losses)),
                   "val": vals, "val_selection_mrr": score, "param_hash": state_hash(enc, dec), "elapsed_sec": time.time() - t0}
            history.append(rec)
            print(f"[{tag}] ep {epoch:4d} loss {rec['train_loss']:.4f} val fMRR " +
                  " ".join(f"{v}={m['filtered_MRR']:.4f}" for v, m in vals.items()) +
                  f" | H@10 {np.mean([m['Hits@10'] for m in vals.values()]):.4f} | params {rec['param_hash'][:12]} | {rec['elapsed_sec']:.0f}s", flush=True)
            if best is None or score > best["val_selection_mrr"]:
                best, bad = rec, 0
                best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()}, {k: v.detach().clone() for k, v in dec.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[{tag}] early stop at epoch {epoch}", flush=True); break
        if args.stop_after and epoch >= args.stop_after:
            break
    train_time = time.time() - t0
    epochs_completed = epoch

    # ---- final evaluation with the best model, per variant, in the RGCN layout
    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    ckpt_hash = state_hash(enc, dec)
    for v in variants:
        d = run_dirs[v]; d.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time()
        test_m, test_art = evaluate(enc, dec, plans[v], ref.test_pos, tail_f, head_f, n, device, save=True)
        inf_time = time.time() - t1
        peak_inf = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        np.savez(d / "test_scores.npz", **test_art)
        torch.save({"encoder": best_state[0], "decoder": best_state[1], "args": vars(args), "variant": v, "arm": args.arm}, d / "best_model.pt")
        b = next(bb for bb in bundles if bb.variant == v)
        metrics = {
            "format_version": "slotgat_wordnet_lp_arms_v1", "model": "SlotGAT (parity, deterministic)", "arm": args.arm, "graph": graph,
            "variant": v, "seed": args.seed, "variants_trained_on": variants, "config": vars(args) | {"out_root": str(args.out_root)},
            "best_epoch": best["epoch"], "best_val_selection_mrr": best["val_selection_mrr"], "val_metrics": best["val"][v],
            "test_metrics": test_m, "epochs_completed": epochs_completed * n_var, "super_epochs_completed": epochs_completed,  # epochs_completed = variant-epochs (RGCN results convention)
            "train_time_sec": train_time, "inference_time_sec": inf_time,
            "parameter_count": n_params, "parameter_bytes": param_bytes,
            "peak_train_gpu_bytes": int(peak_train), "peak_inference_gpu_bytes": int(peak_inf),
            "num_message_relations": 2 * R + 1, "num_decoder_relations": len(ref.task_relation_names),
            "num_training_positives": int(len(train)), "train_objective": "task_only",
            "graph_path": str(BUNDLE_ROOT / v / "graph.pt"), "message_graph": plan_stats[v],
            "edge_hash": b.edge_hash, "stored_physical_hash": b.stored_hashes["physical"], "stored_semantic_hash": b.stored_hashes["semantic"],
            "supervision_hash": ref.supervision_hash, "init_hash": init_hash, "checkpoint_hash": ckpt_hash,
            "run_dir": str(d), "hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (d / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{tag}] {v}: TEST fMRR {test_m['filtered_MRR']:.4f} H@1 {test_m['Hits@1']:.4f} H@3 {test_m['Hits@3']:.4f} "
              f"H@10 {test_m['Hits@10']:.4f} | best ep {best['epoch']} | ckpt {ckpt_hash[:12]} | {train_time/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
