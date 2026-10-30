#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import copy
import csv
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from inv_rgcn.data import load_graph_data
from inv_rgcn.model import ConditionalRGCN, LinkPredictor
from inv_rgcn.metrics import classification_metrics, binary_metrics, ranking_metrics, full_ranking_metrics
from inv_rgcn.utils import set_deterministic, state_dict_hash, gradient_hash, json_dump
from rgcn_resource_metrics import gpu_reset, gpu_peak, model_size


def _graph_of(args) -> str:
    """Read --graph defensively.

    tests/test_invariance.py deliberately calls run_nc(None, ...) -- the NC path
    took no argparse options originally. Session 4 hit exactly this with --norm
    and lost a job to `AttributeError: 'NoneType' object has no attribute 'norm'`.
    """
    return getattr(args, "graph", "semantic") or "semantic"


def message_graph(data, which: str):
    """Return (edge_index, edge_type, edge_weight) for the requested graph.

    ``semantic``  the compiled conditional program -- the INVARIANT arm.
    ``physical``  the variant's own raw graph, uncompiled -- lets this SAME
                  runner, architecture and hyperparameters produce the
                  per-variant ("originals") and union ("universal") arms.

    Why this exists: the universal-graph and originals points previously came
    from a different codebase than the invariant arm, so any gap between them
    mixed METHOD with ARCHITECTURE. On IMDb-MD the invariant model has 3.81M
    parameters against the other runner's ~4.79M, which is not a controlled
    comparison. With this flag every arm is the same model on a different graph.
    """
    if which == "physical":
        w = getattr(data, "physical_edge_weight", None)
        return data.physical_edge_index, data.physical_edge_type, w
    return data.semantic_edge_index, data.semantic_edge_type, data.semantic_edge_weight

DEFAULTS = {
    ("IMDB", "nc"): dict(in_dim=128, hidden_dim=128, out_dim=128, layers=2, bases=8, dropout=0.3, lr=2e-3, weight_decay=1e-5, emb_reg=0.0, epochs=200, patience=20),
    ("IMDB", "movie_link"): dict(in_dim=128, hidden_dim=256, out_dim=256, layers=3, bases=16, dropout=0.1, lr=2e-3, weight_decay=1e-5, emb_reg=1e-6, epochs=200, patience=15),
    ("IMDB", "movie_director"): dict(in_dim=128, hidden_dim=256, out_dim=256, layers=3, bases=16, dropout=0.1, lr=2e-3, weight_decay=1e-5, emb_reg=1e-6, epochs=200, patience=15),
    ("IMDB", "movie_year"): dict(in_dim=128, hidden_dim=256, out_dim=256, layers=3, bases=16, dropout=0.1, lr=2e-3, weight_decay=1e-5, emb_reg=1e-6, epochs=200, patience=15),
    ("DBLP", "paper_conference"): dict(in_dim=128, hidden_dim=256, out_dim=256, layers=3, bases=16, dropout=0.1, lr=2e-3, weight_decay=1e-5, emb_reg=1e-6, epochs=200, patience=15),
}


def cfg_value(args, name, defaults):
    value = getattr(args, name)
    return defaults[name] if value is None else value


def build_encoder(data, cfg, device, emb_init: float = 0.0, norm: str = "mean"):
    fixed = data.x.to(device) if data.x is not None else None
    enc = ConditionalRGCN(
        num_nodes=data.num_nodes,
        num_relations=len(data.semantic_relation_names),
        in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"], out_dim=cfg["out_dim"],
        num_layers=cfg["layers"], num_bases=cfg["bases"], dropout=cfg["dropout"],
        input_features=fixed,
        norm=norm,
    ).to(device)
    if emb_init > 0 and enc.node_embedding is not None:
        # Deterministic given the run seed (set_deterministic ran before this).
        with torch.no_grad():
            enc.node_embedding.weight.normal_(0.0, emb_init)
    return enc


def flatten_negatives(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return flat negative pairs and per-query candidate tails."""
    is_tail_matrix = (
        neg.ndim == 2
        and neg.shape[0] == pos.shape[0]
        and (neg.shape[1] != 2 or not torch.equal(neg[:, 0], pos[:, 0]))
    )
    if is_tail_matrix:
        # [Q,K] global tail ids, including the otherwise ambiguous K=2 case.
        heads = pos[:, 0:1].expand_as(neg)
        flat = torch.stack([heads.reshape(-1), neg.reshape(-1)], dim=1)
        groups = [neg[i] for i in range(len(pos))]
        return flat, groups
    if neg.ndim != 2 or neg.shape[1] != 2:
        raise ValueError(f"Unsupported negative tensor shape {tuple(neg.shape)}")
    # Flat pair list. Group by query head; if repeated heads have multiple positives,
    # every positive uses the same candidate tail set, matching filtered ranking.
    by_head: dict[int, list[int]] = {}
    for h, t in neg.tolist(): by_head.setdefault(int(h), []).append(int(t))
    groups = []
    for h, _ in pos.tolist():
        tails = by_head.get(int(h), [])
        if not tails:
            # Fall back to the full flat set only when no head-aligned negatives exist.
            tails = [int(t) for t in neg[:, 1].tolist()]
        groups.append(torch.tensor(sorted(set(tails)), dtype=torch.long, device=pos.device))
    return neg, groups


# Target node type whose ENTIRE population is the candidate set under the
# ``full_target_type`` protocol. DBLP paper-conference has only 20 conferences,
# so sampled negatives are both unnecessary and (as shipped) badly malformed:
# the shared-split NPZ stores ONE flat negative per positive, which after
# grouping by head gives 955 queries a single candidate while the 990 queries
# with no head-aligned negative fall back to the full 20-venue set. Ranking
# against every conference removes the sampling question entirely and matches
# the WordNet full-entity headline protocol.
FULL_RANK_TARGET_TYPE = {
    ("DBLP", "paper_conference"): "conference",
    # Movie-Year has only 91 candidates, so full ranking costs nothing and is the
    # most honest protocol. It is OPT-IN via --eval-protocol full_target_type; the
    # default sampled protocol uses the frequency-matched negatives baked into the
    # preprocessed splits.
    ("IMDB", "movie_year"): "year",
}


def target_type_candidates(data, dataset: str, task: str) -> torch.Tensor | None:
    """All node ids of the ranked destination type, or None if not configured."""
    type_name = FULL_RANK_TARGET_TYPE.get((dataset, task))
    if type_name is None:
        return None
    counts = data.metadata.get("num_nodes_by_type") or {}
    offsets = data.metadata.get("offsets") or {}
    if type_name not in counts or type_name not in offsets:
        raise ValueError(
            f"{dataset}/{task}: metadata lacks node-type layout for '{type_name}'; "
            "cannot build the full candidate set"
        )
    start = int(offsets[type_name])
    return torch.arange(start, start + int(counts[type_name]), dtype=torch.long)


def known_true_tails(data) -> dict[int, set[int]]:
    """Every (head -> tails) pair observed in any split.

    Used only to REMOVE other correct answers from a query's candidate list --
    the standard filtered-ranking convention (WN18RR, FB15k-237). It never adds
    an edge to the message graph, so it is not leakage.
    """
    truth: dict[int, set[int]] = {}
    for key in ("train_pos", "val_pos", "test_pos"):
        arr = data.task_data.get(key)
        if arr is None:
            continue
        for head, tail in arr.tolist():
            truth.setdefault(int(head), set()).add(int(tail))
    return truth


@torch.no_grad()
def full_rank_scores(predictor, h, pos, candidates, truth):
    """Rank each positive against EVERY candidate tail, filtered.

    Returns (ranks, score_matrix). ``score_matrix`` keeps the unfiltered scores
    for every candidate so invariance audits still compare dense, identically
    shaped outputs across variants.
    """
    heads = pos[:, 0]
    cand = candidates.to(pos.device)
    num_q, num_c = heads.numel(), cand.numel()
    pair_heads = heads.repeat_interleave(num_c)
    pair_tails = cand.repeat(num_q)
    scores = predictor(h, torch.stack([pair_heads, pair_tails], dim=1)).view(num_q, num_c)

    cand_list = cand.tolist()
    cand_pos = {c: i for i, c in enumerate(cand_list)}
    ranks = np.empty(num_q, dtype=np.float64)
    row_scores = scores.detach().cpu().numpy()
    for i, (head, tail) in enumerate(pos.tolist()):
        col = cand_pos.get(int(tail))
        if col is None:
            raise ValueError(f"positive tail {tail} is not of the ranked node type")
        row = row_scores[i].copy()
        for other in truth.get(int(head), ()):  # filtered protocol
            j = cand_pos.get(int(other))
            if j is not None and j != col:
                row[j] = -np.inf
        # Ties use the AVERAGED rank (1 + #strictly_better + #tied/2). This side previously used the PESSIMISTIC rank while the baselines used an optimistic stable sort, so the invariant arm was scored under a stricter rule than the models it is compared against. All three conventions agree exactly when scores are distinct.
        better = int((row > row[col]).sum())
        tied = int((row == row[col]).sum())          # includes the positive itself
        ranks[i] = 1.0 + better + (tied - 1) / 2.0
    return ranks, row_scores


def score_ranking(encoder, predictor, data, pos, neg, device, args):
    # `args` MUST be a parameter. The --graph refactor replaced the
    # direct data.semantic_* access here with message_graph(data, _graph_of(args)),
    # but score_ranking has no args in scope, so every LP run crashed with
    # NameError at test-time ranking. The other seven call sites
    # are inside functions that do have args; this was the only one.
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        h = encoder(*message_graph(data, _graph_of(args)))
        pos_logits = predictor(h, pos)
        _flat, groups = flatten_negatives(pos, neg)
        neg_rows=[]
        max_k=max(len(g) for g in groups)
        for (head, _), tails in zip(pos.tolist(), groups):
            if len(tails)<max_k:
                tails=torch.cat([tails, tails[-1:].repeat(max_k-len(tails))])
            pairs=torch.stack([torch.full_like(tails, int(head)), tails],dim=1)
            neg_rows.append(predictor(h,pairs))
        neg_logits=torch.stack(neg_rows,dim=0)
    return h, pos_logits, neg_logits



def lp_loss(pos_logit: torch.Tensor, neg_logit: torch.Tensor, mode: str) -> torch.Tensor:
    """Link-prediction loss.

    ``pairwise`` is the ORIGINAL baseline's loss (run_IMDB_rgcn_lp.pairwise_loss):
    positives and negatives each contribute their own mean, so the K:1 negative
    imbalance does not dominate the gradient. ``pooled_bce`` is the previous
    behaviour -- one mean over the concatenation -- kept for reproducibility.
    """
    if mode == "pairwise":
        return -(F.logsigmoid(pos_logit).mean() + F.logsigmoid(-neg_logit).mean())
    logits = torch.cat([pos_logit, neg_logit])
    labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def val_mrr(predictor, h: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mean reciprocal rank of each positive against its own candidate tails."""
    _flat, groups = flatten_negatives(pos, neg)
    rr = 0.0
    for (head, tail), tails in zip(pos.tolist(), groups):
        if not len(tails):
            continue
        pairs = torch.stack([torch.full_like(tails, int(head)), tails], dim=1)
        s_neg = predictor(h, pairs)
        s_pos = predictor(h, pos.new_tensor([[int(head), int(tail)]]))
        rr += 1.0 / (int((s_neg > s_pos).sum()) + 1)
    return rr / max(len(pos), 1)


def run_nc(args, data, cfg, device, run_dir):
    # tests/test_invariance.py calls this with args=None (the NC path took no argparse
    # options before --norm was added), so read it defensively and fall back to the
    # argparse default.
    encoder=build_encoder(data,cfg,device,norm=getattr(args,"norm","mean") or "mean")
    classifier=nn.Linear(cfg["out_dim"], int(data.metadata.get("num_classes",3))).to(device)
    model=nn.ModuleDict({"encoder":encoder,"classifier":classifier})
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    best=-float("inf");best_state=None;bad=0;trace=[];first_audit={}
    gpu_reset(device);t0=time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train();opt.zero_grad(set_to_none=True)
        h=encoder(*message_graph(data, _graph_of(args)))
        logits=classifier(h)
        loss=F.cross_entropy(logits[data.train_mask],data.y[data.train_mask])
        loss.backward()
        if epoch==0:
            first_audit={"initial_logits_max_abs":float(logits.detach().abs().max()),"gradient_hash":gradient_hash(model)}
        opt.step()
        model.eval()
        with torch.no_grad():
            h=encoder(*message_graph(data, _graph_of(args)));logits=classifier(h)
            val=classification_metrics(logits[data.val_mask],data.y[data.val_mask])["micro_f1"]
        trace.append({"epoch":epoch+1,"train_loss":float(loss.item()),"val_micro_f1":val,"state_hash":state_dict_hash(model.state_dict())})
        if val>best+1e-12:
            best=val;bad=0;best_state=copy.deepcopy(model.state_dict())
        else:
            bad+=1
            if bad>=cfg["patience"]: break
    train_sec=time.perf_counter()-t0
    _peak_train=gpu_peak(device)   # training window closes BEFORE the test forward
    model.load_state_dict(best_state);model.eval()
    # Separate inference window: peak train AND peak inference GPU are both recorded.
    gpu_reset(device);_t_inf=time.perf_counter()
    with torch.no_grad():
        h=encoder(*message_graph(data, _graph_of(args)));logits=classifier(h)
        metrics=classification_metrics(logits[data.test_mask],data.y[data.test_mask])
        test_logits=logits[data.test_mask].cpu().numpy();test_ids=torch.where(data.test_mask)[0].cpu().numpy()
    _inf_sec=time.perf_counter()-_t_inf;_peak_inf=gpu_peak(device)
    np.savez(run_dir/"scores.npz",ids=test_ids,scores=test_logits)
    torch.save(model.state_dict(),run_dir/"checkpoint.pt")
    best_epoch=int(np.argmax([t["val_micro_f1"] for t in trace]))+1
    metrics.update({"train_time_sec":train_sec,"epochs":len(trace),"best_epoch":best_epoch,"best_val":best,
                    "select_on":"val_micro_f1",
                    "inference_time_sec":_inf_sec,
                    "peak_training_gpu_mib":_peak_train["peak_gpu_mib"],
                    "peak_training_reserved_gpu_mib":_peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib":_peak_inf["peak_gpu_mib"],
                    "peak_inference_reserved_gpu_mib":_peak_inf["peak_gpu_reserved_mib"]})
    metrics.update(model_size(model))
    return model,metrics,trace,first_audit


def run_lp(args,data,cfg,device,run_dir):
    encoder=build_encoder(data,cfg,device,emb_init=args.emb_init,norm=args.norm)
    predictor=LinkPredictor(cfg["out_dim"],mode=args.decoder).to(device)
    model=nn.ModuleDict({"encoder":encoder,"predictor":predictor})
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    td={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.task_data.items()}
    train_pos,train_neg=td["train_pos"],td["train_neg"]
    val_pos,val_neg=td["val_pos"],td["val_neg"]
    test_pos,test_neg=td["test_pos"],td["test_neg"]
    flat_train_neg,_=flatten_negatives(train_pos,train_neg)
    flat_val_neg,_=flatten_negatives(val_pos,val_neg)
    # Full-candidate protocol: rank against every node of the target type.
    candidates=None
    if args.eval_protocol=="full_target_type":
        candidates=target_type_candidates(data,args.dataset,args.task)
        if candidates is None:
            raise ValueError(
                f"--eval-protocol full_target_type is not configured for "
                f"{args.dataset}/{args.task}; add it to FULL_RANK_TARGET_TYPE"
            )
        candidates=candidates.to(device)
        truth=known_true_tails(data)
    best=-float("inf");best_state=None;bad=0;best_epoch=0;trace=[];first_audit={}
    gpu_reset(device);t0=time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train();opt.zero_grad(set_to_none=True)
        h=encoder(*message_graph(data, _graph_of(args)))
        pos_log=predictor(h,train_pos);neg_log=predictor(h,flat_train_neg)
        loss=lp_loss(pos_log,neg_log,args.loss)
        if cfg["emb_reg"]:
            loss=loss+cfg["emb_reg"]*h.pow(2).mean()
        loss.backward()
        if epoch==0:
            with torch.no_grad():
                _init=torch.cat([pos_log.detach(),neg_log.detach()]).abs().max()
            first_audit={"initial_logits_max_abs":float(_init),"gradient_hash":gradient_hash(model)}
        opt.step()
        model.eval()
        with torch.no_grad():
            h=encoder(*message_graph(data, _graph_of(args)))
            vp=predictor(h,val_pos);vn=predictor(h,flat_val_neg)
            vl=float(lp_loss(vp,vn,args.loss))
            # Ranking quality on validation. BCE/pairwise loss bottoms out long
            # before ranking does (DBLP: val loss minimal at epoch ~6 while train
            # loss keeps falling to 0.33 through epoch 21), so stopping on loss
            # restores a barely-trained checkpoint. Select on MRR instead.
            if candidates is not None:
                vranks,_=full_rank_scores(predictor,h,val_pos,candidates,truth)
                vmrr=float(np.mean(1.0/vranks))
            else:
                vmrr=float(val_mrr(predictor,h,val_pos,val_neg))
        trace.append({"epoch":epoch+1,"train_loss":float(loss.item()),"val_loss":float(vl),
                      "val_mrr":vmrr,"state_hash":state_dict_hash(model.state_dict())})
        score = vmrr if args.select_on=="val_mrr" else -vl
        if score>best+1e-12:
            best=score;bad=0;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        else:
            bad+=1
            if bad>=cfg["patience"]: break
    train_sec=time.perf_counter()-t0
    _peak_train=gpu_peak(device)
    model.load_state_dict(best_state)
    gpu_reset(device);_t_inf=time.perf_counter()
    if candidates is not None:
        encoder.eval();predictor.eval()
        with torch.no_grad():
            h=encoder(*message_graph(data, _graph_of(args)))
            ranks,cand_matrix=full_rank_scores(predictor,h,test_pos,candidates,truth)
            pos_log=predictor(h,test_pos)
        p=pos_log.cpu().numpy()
        # Binary metrics need a negative pool; use every non-true candidate.
        cand_ids=candidates.tolist()
        neg_vals=[]
        for row,(head,_tail) in zip(cand_matrix,test_pos.tolist()):
            true_set=truth.get(int(head),set())
            neg_vals.extend(v for c,v in zip(cand_ids,row) if int(c) not in true_set)
        neg_flat=np.asarray(neg_vals,dtype=np.float64)
        labels=np.concatenate([np.ones(len(p)),np.zeros(neg_flat.size)])
        all_logits=np.concatenate([p,neg_flat])
        metrics=binary_metrics(all_logits,labels)
        metrics.update(full_ranking_metrics(ranks))
        metrics.update({"eval_protocol":"full_target_type",
                        "num_candidates":int(candidates.numel())})
        candidate_scores=cand_matrix
        np.savez(run_dir/"scores.npz",queries=test_pos[:,0].cpu().numpy(),
                 positive_tails=test_pos[:,1].cpu().numpy(),scores=candidate_scores,
                 candidate_ids=np.asarray(cand_ids,dtype=np.int64),ranks=ranks)
    else:
        h,pos_log,neg_log=score_ranking(encoder,predictor,data,test_pos,test_neg,device,args)
        p=pos_log.cpu().numpy();n=neg_log.cpu().numpy()
        labels=np.concatenate([np.ones(len(p)),np.zeros(n.size)]);all_logits=np.concatenate([p,n.reshape(-1)])
        metrics=binary_metrics(all_logits,labels);metrics.update(ranking_metrics(p,n))
        metrics.update({"eval_protocol":"sampled","num_candidates":int(n.shape[1])+1})
        candidate_scores=np.concatenate([p[:,None],n],axis=1)
        np.savez(run_dir/"scores.npz",queries=test_pos[:,0].cpu().numpy(),positive_tails=test_pos[:,1].cpu().numpy(),scores=candidate_scores)
    _inf_sec=time.perf_counter()-_t_inf;_peak_inf=gpu_peak(device)
    metrics.update({"train_time_sec":train_sec,"epochs":len(trace),"best_epoch":best_epoch,
                    "select_on":args.select_on,"best_val_score":best,
                    # params + peak GPU are recorded beside the task metrics.
                    "inference_time_sec":_inf_sec,
                    "peak_training_gpu_mib":_peak_train["peak_gpu_mib"],
                    "peak_training_reserved_gpu_mib":_peak_train["peak_gpu_reserved_mib"],
                    "peak_inference_gpu_mib":_peak_inf["peak_gpu_mib"],
                    "peak_inference_reserved_gpu_mib":_peak_inf["peak_gpu_reserved_mib"]})
    metrics.update(model_size(model))
    torch.save(model.state_dict(),run_dir/"checkpoint.pt")
    return model,metrics,trace,first_audit


def main():
    ap=argparse.ArgumentParser(description="Train architecture-level invariant conditional RGCN")
    ap.add_argument("--data-root",default="data/preprocessed/RGCN_structural")
    ap.add_argument("--dataset",choices=("IMDB","DBLP"),required=True)
    ap.add_argument("--task",required=True)
    ap.add_argument("--variant",required=True)
    ap.add_argument("--seed",type=int,default=1566911444)
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-root",default="results/RGCN_structural")
    ap.add_argument("--decoder",choices=("dot","distmult"),default="dot")
    ap.add_argument("--loss",choices=("pairwise","pooled_bce"),default="pairwise",
                    help="pairwise matches the original baseline and is the default; "
                         "pooled_bce reproduces the earlier collapsed runs")
    ap.add_argument("--select-on",dest="select_on",choices=("val_mrr","val_loss"),default="val_mrr",
                    help="checkpoint-selection/early-stopping metric")
    ap.add_argument("--norm",choices=("mean","sum"),default="mean",
                    help="Message aggregation over in-edges of a relation. 'mean' divides "
                         "by destination degree (previous behaviour); 'sum' matches DGL "
                         "RelGraphConv's default and preserves the popularity prior on "
                         "high-degree destination types such as DBLP conferences.")
    ap.add_argument("--eval-protocol",dest="eval_protocol",
                    choices=("sampled","full_target_type"),default="sampled",
                    help="'sampled' ranks against the stored negatives; 'full_target_type' "
                         "ranks against EVERY node of the ranked destination type (filtered). "
                         "Required for DBLP, whose stored negatives are malformed.")
    ap.add_argument("--graph",choices=("semantic","physical"),default="semantic",
                    help="Which graph to propagate on. 'semantic' = the compiled "
                         "conditional program (the invariant arm). 'physical' = the "
                         "variant's own raw graph, which lets this same runner and "
                         "architecture produce the originals and universal arms so all "
                         "three are comparable.")
    ap.add_argument("--emb-init",dest="emb_init",type=float,default=0.1,
                    help="std of the node-embedding init. Xavier over an (N,128) table "
                         "gives std~0.011, which combined with the layer's destination-degree "
                         "normalisation drives every ReLU dead; h then collapses to the final "
                         "layer's zero bias, so every score is 0, the loss pins at ln 2 and the "
                         "bias gradient (2*bias) is exactly 0 -- a permanently stable saddle. "
                         "Set <=0 to restore xavier_uniform_.")
    for name,typ in (("in_dim",int),("hidden_dim",int),("out_dim",int),("layers",int),("bases",int),("epochs",int),("patience",int),("dropout",float),("lr",float),("weight_decay",float),("emb_reg",float)):
        ap.add_argument("--"+name.replace("_","-"),dest=name,type=typ,default=None)
    args=ap.parse_args();key=(args.dataset,args.task)
    # Progress lines so a hang can be placed (a run once blocked for 16 h with no output;
    # the log could not tell "stuck before data load" from "stuck at interpreter exit").
    import socket, datetime
    print(f"[run] start {datetime.datetime.now().isoformat(timespec='seconds')} host={socket.gethostname()} "
          f"task={args.task} variant={args.variant} seed={args.seed} graph={getattr(args,'graph','semantic')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if key not in DEFAULTS: raise SystemExit(f"Unsupported dataset/task {key}; supported={list(DEFAULTS)}")
    defaults=DEFAULTS[key];cfg={k:cfg_value(args,k,defaults) for k in defaults}
    set_deterministic(args.seed,strict=True)
    device=torch.device(args.device)
    data_path=Path(args.data_root)/args.dataset/args.task/args.variant
    data=load_graph_data(data_path).to(device)
    print(f"[run] data loaded and on {device} {datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)
    run_dir=Path(args.output_root)/args.dataset/args.task/args.variant/f"seed_{args.seed}";run_dir.mkdir(parents=True,exist_ok=True)
    if args.task=="nc": model,metrics,trace,first=run_nc(args,data,cfg,device,run_dir)
    else: model,metrics,trace,first=run_lp(args,data,cfg,device,run_dir)
    contract={"norm":args.norm,"eval_protocol":args.eval_protocol,"decoder":args.decoder,
              "loss":args.loss,"select_on":args.select_on,"emb_init":args.emb_init}
    summary={"dataset":args.dataset,"task":args.task,"variant":args.variant,"seed":args.seed,"config":cfg,
             "contract":contract,
             "physical_hash":data.physical_hash,"semantic_hash":data.semantic_hash,"feature_hash":data.feature_hash,
             "supervision_hash":data.supervision_hash,
             "checkpoint_hash":state_dict_hash(model.state_dict()),"metrics":metrics,"first_step_audit":first,"run_dir":str(run_dir)}
    json_dump(summary,run_dir/"metrics.json");json_dump(trace,run_dir/"trace.json")
    print(json.dumps(summary,indent=2,sort_keys=True))
    print(f"[run] complete {datetime.datetime.now().isoformat(timespec='seconds')} -> {run_dir}", flush=True)

if __name__=="__main__": main()
