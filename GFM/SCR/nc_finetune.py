"""Fine-tune SCR on one original variant, then test on that same variant.

Targets are the training label triples in train.txt. Inverse, shortcut, and
other structural edges stay in the message graph and are not supervision.
Before each step the batch's target edges and the model's inverse copies are
removed, so the answer is not visible to that step. The removed-edge count
must be twice the batch size.

Checkpoint selection is validation accuracy for node classification and
validation MRR for link prediction (--select mrr).
"""

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nc_zero_shot as zs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="folder under kg-datasets/FoundationNodeKG")
    ap.add_argument("--ckpt", default=os.path.join(zs.UPSTREAM, "ckpts", "scr_3g.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--adversarial-temperature", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=0, help="steps between validation checks (0 = once per epoch)")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--select", choices=["acc", "mrr"], default="acc")
    args = ap.parse_args()
    zs.setup_paths()

    import random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    raw_dir = os.path.join(zs.KG_ROOT, "FoundationNodeKG", args.dataset, "raw")

    t0 = time.time()
    ent, rel, trip, graph, model, _ = zs.scr_graph_and_model(args.dataset, raw_dir, args.ckpt, device, None, None)
    prep_sec = time.time() - t0

    label_rel, label_ids = zs.label_candidates(raw_dir, ent, trip)
    id2ent = {v: k for k, v in ent.items()}
    label_names = [id2ent[i] for i in label_ids]
    label_col = {lid: j for j, lid in enumerate(label_ids)}
    candidates = torch.tensor(label_ids, device=device)
    num_base_rel = int(graph.num_relations) // 2

    train_q = torch.tensor([t for t in trip["train"] if t[2] == label_rel], dtype=torch.long)
    queries = {s: torch.tensor(trip[s], dtype=torch.long, device=device) for s in ("valid", "test")}

    src, dst, et = graph.edge_index[0].cpu(), graph.edge_index[1].cpu(), graph.edge_type.cpu()
    n = int(graph.num_nodes)
    key = (src * n + dst) * (2 * num_base_rel + 1) + et
    edge_pos = {int(k): i for i, k in enumerate(key.tolist())}
    target_edges = []
    for h, t, r in train_q.tolist():
        fwd = edge_pos[(h * n + t) * (2 * num_base_rel + 1) + r]
        inv = edge_pos[(t * n + h) * (2 * num_base_rel + 1) + r + num_base_rel]
        target_edges.append((fwd, inv))
    target_edges = torch.tensor(target_edges, dtype=torch.long)

    for name, p in model.named_parameters():
        frozen = args.freeze_backbone and ("relation_model" in name or "entity_model.layers" in name)
        p.requires_grad = not frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    param_mib = (sum(p.numel() * p.element_size() for p in model.parameters())
                 + sum(b.numel() * b.element_size() for b in model.buffers())) / 2 ** 20

    def score(data, batch):
        h = batch[:, 0:1].expand(-1, len(candidates))
        t_batch = torch.stack([h, candidates.expand_as(h), batch[:, 2:3].expand_as(h)], dim=-1)
        out = model(data, t_batch)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        q = queries[split]
        s = torch.cat([score(graph, q[i:i + args.eval_batch_size]).float().cpu()
                       for i in range(0, len(q), args.eval_batch_size)]).numpy()
        label = np.array([label_col[int(t)] for t in q[:, 1].tolist()])
        model.train()
        rank = 1 + (s > s[np.arange(len(label)), label][:, None]).sum(1)
        return s, label, dict(acc=float((s.argmax(1) == label).mean()), mrr=float((1 / rank).mean()))

    steps_per_epoch = (len(train_q) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    eval_every = args.eval_every or steps_per_epoch
    keep = torch.ones(graph.edge_index.shape[1], dtype=torch.bool, device=device)
    target_edges = target_edges.to(device)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    curve, losses, removed = [], [], []
    best = dict(val=-1.0, step=0, state=None)
    step, peak_train = 0, 0.0
    t_train = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_q), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            batch = train_q[idx].to(device)
            drop = target_edges[idx.to(device)].flatten()
            keep[drop] = False
            data = copy.copy(graph)
            data.edge_index, data.edge_type = graph.edge_index[:, keep], graph.edge_type[keep]
            removed.append(int((~keep).sum()))
            keep[drop] = True

            pred = score(data, batch)
            target = torch.zeros_like(pred)
            pos = torch.tensor([label_col[int(t)] for t in batch[:, 1].tolist()], device=device)
            target[torch.arange(len(pos), device=device), pos] = 1
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            with torch.no_grad():
                logits = pred / args.adversarial_temperature if args.adversarial_temperature > 0 \
                    else torch.zeros_like(pred)
                weight = torch.softmax(logits.masked_fill(target.bool(), float("-inf")), dim=-1)
                weight = weight + target
            loss = ((loss * weight).sum(-1) / weight.sum(-1)).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))
            step += 1

            if step % eval_every == 0 or step == total_steps:
                peak_train = max(peak_train, torch.cuda.max_memory_allocated() / 2 ** 20)
                _, _, val = evaluate("valid")
                torch.cuda.reset_peak_memory_stats()
                curve.append(dict(step=step, epoch=epoch + 1, train_loss=float(np.mean(losses[-eval_every:])),
                                  val_acc=val["acc"], val_mrr=val["mrr"]))
                print(f"step {step}/{total_steps} loss {curve[-1]['train_loss']:.4f} val_acc {val['acc']:.4f} "
                      f"val_mrr {val['mrr']:.4f}", flush=True)
                if val[args.select] > best["val"]:
                    best.update(val=val[args.select], step=step,
                                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    torch.cuda.synchronize()
    train_sec = time.time() - t_train

    model.load_state_dict(best["state"])
    torch.cuda.reset_peak_memory_stats()
    arrays, metrics = {}, {}
    for split in ("valid", "test"):
        t1 = time.time()
        s, label, res = evaluate(split)
        pred = s.argmax(1)
        arrays.update({f"{split}_node": np.array([id2ent[i] for i in queries[split][:, 0].tolist()]),
                       f"{split}_label": label, f"{split}_scores": s})
        metrics[split] = dict(n=int(len(label)), accuracy=res["acc"], mrr=res["mrr"],
                              macro_f1=float(f1_score(label, pred, average="macro")),
                              pred_hist=np.bincount(pred, minlength=len(label_ids)).tolist(),
                              sec=round(time.time() - t1, 2))
        print(f"scr finetuned {args.dataset} {split}: acc={res['acc']:.4f} mrr={res['mrr']:.4f} "
              f"macroF1={metrics[split]['macro_f1']:.4f}")
    peak_infer = torch.cuda.max_memory_allocated() / 2 ** 20

    assert set(removed) == {2 * args.batch_size} | ({2 * (len(train_q) % args.batch_size)} - {0}), \
        f"target-edge removal mismatch: {sorted(set(removed))}"
    metrics.update(
        model="scr", dataset=args.dataset, ckpt=os.path.abspath(args.ckpt), seed=args.seed, setting="finetune",
        epochs=args.epochs, steps=total_steps, steps_per_epoch=steps_per_epoch, batch_size=args.batch_size,
        lr=args.lr, freeze_backbone=args.freeze_backbone, eval_every=eval_every,
        select=args.select, best_step=best["step"], best_val=best["val"], train_wall_sec=round(train_sec, 1),
        prep_sec=round(prep_sec, 2), param_mib=round(param_mib, 3),
        trainable_params=int(sum(p.numel() for p in trainable)),
        peak_train_gpu_mib=round(peak_train, 1), peak_gpu_mib=round(peak_infer, 1),
        removed_edges_per_batch=sorted(set(removed)), curve=curve,
    )
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "scores.npz"), label_names=np.array(label_names), **arrays)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
