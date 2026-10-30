#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from inv_rgcn.data import ConditionalGraphData, canonicalize_edges, save_graph_data
from inv_rgcn.utils import set_deterministic, parse_str_list

SEED = 1566911444
VARIANTS = ("v1", "v2", "v3", "universal")
# 'universal' is the union of v1/v2/v3's Area placements, compiled through this
# same pipeline so the universal-graph arm can be trained by the SAME runner and
# architecture as the invariant arm (run_rgcn_structural.py --graph physical).
# Without it the universal point comes from a different codebase and any
# invariant-vs-universal gap mixes method with architecture.
REL_NAMES = [
    "author_to_paper", "paper_to_author",
    "paper_to_term", "term_to_paper",
    "paper_to_conference", "conference_to_paper",
    "paper_to_area", "area_to_paper",
    "conference_to_area", "area_to_conference",
    "author_to_area", "area_to_author",
]
REL = {name: i for i, name in enumerate(REL_NAMES)}
NODE_TYPES = {"author": 0, "paper": 1, "term": 2, "conference": 3, "area": 4}


def read_tables(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(raw_dir)
    al = pd.read_csv(root / "author_label.txt", sep="\t", names=["author_id", "label", "author_name"], header=None, encoding="utf-8")
    pa = pd.read_csv(root / "paper_author.txt", sep="\t", names=["paper_id", "author_id"], header=None, encoding="utf-8")
    pc = pd.read_csv(root / "paper_conf.txt", sep="\t", names=["paper_id", "conf_id"], header=None, encoding="utf-8")
    pt = pd.read_csv(root / "paper_term.txt", sep="\t", names=["paper_id", "term_id"], header=None, encoding="utf-8")
    return al, pa, pc, pt


def filter_tables(al, pa, pc, pt, paper_subset=None, min_conf=0):
    pa = pa[pa["author_id"].isin(set(al["author_id"]))].reset_index(drop=True)
    papers = set(pa["paper_id"])
    pc = pc[pc["paper_id"].isin(papers)].reset_index(drop=True)
    pt = pt[pt["paper_id"].isin(papers)].reset_index(drop=True)
    if min_conf > 0:
        keep_conf = pc["conf_id"].value_counts().loc[lambda x: x >= min_conf].index
        pc = pc[pc["conf_id"].isin(keep_conf)].reset_index(drop=True)
        papers = set(pc["paper_id"])
        pa = pa[pa["paper_id"].isin(papers)].reset_index(drop=True)
        pt = pt[pt["paper_id"].isin(papers)].reset_index(drop=True)
    if paper_subset is not None:
        papers = set(paper_subset) & set(pc["paper_id"])
        pa = pa[pa["paper_id"].isin(papers)].reset_index(drop=True)
        pc = pc[pc["paper_id"].isin(papers)].reset_index(drop=True)
        pt = pt[pt["paper_id"].isin(papers)].reset_index(drop=True)
    return al, pa, pc, pt


def paper_area(al: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    a2r = al.set_index("author_id")["label"]
    pr = pa.assign(area_id=pa["author_id"].map(a2r))[["paper_id", "area_id"]].dropna().drop_duplicates()
    counts = pr.groupby("paper_id")["area_id"].nunique()
    bad = set(counts[counts != 1].index)
    if bad:
        pr = pr[~pr["paper_id"].isin(bad)]
    return pr.drop_duplicates("paper_id").reset_index(drop=True)


def make_maps(pa, pc, pt, pr):
    ids = {
        "author": sorted(pa["author_id"].unique()),
        "paper": sorted(pc["paper_id"].unique()),
        "term": sorted(pt["term_id"].unique()),
        "conference": sorted(pc["conf_id"].unique()),
        "area": sorted(pr["area_id"].unique()),
    }
    offsets = {}
    cur = 0
    maps = {}
    for name in ("author", "paper", "term", "conference", "area"):
        offsets[name] = cur
        maps[name] = {raw: cur + i for i, raw in enumerate(ids[name])}
        cur += len(ids[name])
    node_type = np.empty(cur, dtype=np.int64)
    for name in ("author", "paper", "term", "conference", "area"):
        lo, hi = offsets[name], offsets[name] + len(ids[name])
        node_type[lo:hi] = NODE_TYPES[name]
    return ids, maps, offsets, node_type


def load_or_create_splits(shared_npz: str, pc: pd.DataFrame, seed: int, neg_k: int) -> tuple[dict[str, np.ndarray], list]:
    if shared_npz and Path(shared_npz).exists():
        z = np.load(shared_npz)
        splits = {k: np.asarray(z[k]) for k in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
        subset = z["paper_subset"].tolist() if "paper_subset" in z.files else None
        return splits, subset
    rng = np.random.default_rng(seed)
    pos = pc[["paper_id", "conf_id"]].drop_duplicates().to_numpy()
    order = rng.permutation(len(pos)); pos = pos[order]
    n_val, n_test = max(1, int(.1*len(pos))), max(1, int(.1*len(pos)))
    val, test, train = pos[:n_val], pos[n_val:n_val+n_test], pos[n_val+n_test:]
    confs = np.asarray(sorted(pc["conf_id"].unique()))
    truth = defaultdict(set)
    for p, c in pos: truth[p].add(c)
    def neg(arr, salt):
        rr = np.random.default_rng(seed + salt); out=[]
        for p, _ in arr:
            valid = np.asarray([c for c in confs if c not in truth[p]])
            choices = rr.choice(valid, size=neg_k, replace=len(valid)<neg_k)
            out.extend([(p, c) for c in choices])
        return np.asarray(out)
    return {
        "train_pos": train, "val_pos": val, "test_pos": test,
        "train_neg": neg(train, 1), "val_neg": neg(val, 2), "test_neg": neg(test, 3),
    }, None


def remap_splits(raw_splits: dict[str, np.ndarray], maps: dict) -> dict[str, torch.Tensor]:
    out = {}
    for key, arr in raw_splits.items():
        df = pd.DataFrame(arr, columns=["paper_id", "conf_id"])
        df["paper_id"] = df["paper_id"].map(maps["paper"])
        df["conf_id"] = df["conf_id"].map(maps["conference"])
        df = df.dropna()
        out[key] = torch.tensor(df.to_numpy(dtype=np.int64), dtype=torch.long)
    return out


def add(store, rel, u, v): store[rel].add((int(u), int(v)))


def build_base_edges(pa, pt, train_pos_global, maps):
    base = defaultdict(set)
    for p, a in pa[["paper_id","author_id"]].itertuples(index=False):
        if p in maps["paper"] and a in maps["author"]:
            P, A = maps["paper"][p], maps["author"][a]
            add(base,"author_to_paper",A,P); add(base,"paper_to_author",P,A)
    for p, t in pt[["paper_id","term_id"]].itertuples(index=False):
        if p in maps["paper"] and t in maps["term"]:
            P, T = maps["paper"][p], maps["term"][t]
            add(base,"paper_to_term",P,T); add(base,"term_to_paper",T,P)
    for P, C in train_pos_global.tolist():
        add(base,"paper_to_conference",P,C); add(base,"conference_to_paper",C,P)
    return base


def build_area_certificate(pa, pc, pr, maps, support_papers: set) -> list[dict]:
    authors = defaultdict(list)
    confs = defaultdict(list)
    for p,a in pa[["paper_id","author_id"]].itertuples(index=False): authors[p].append(a)
    for p,c in pc[["paper_id","conf_id"]].itertuples(index=False): confs[p].append(c)
    area = dict(pr[["paper_id","area_id"]].itertuples(index=False))
    records=[]
    for p in sorted(support_papers):
        if p not in maps["paper"] or p not in area or area[p] not in maps["area"]: continue
        rec={"paper": maps["paper"][p], "area": maps["area"][area[p]],
             "authors": sorted({maps["author"][a] for a in authors[p] if a in maps["author"]}),
             "conferences": sorted({maps["conference"][c] for c in confs[p] if c in maps["conference"]})}
        records.append(rec)
    return records


def build_physical(base, certificate, variant):
    raw = defaultdict(set, {k:set(v) for k,v in base.items()})
    for rec in certificate:
        P,R=rec["paper"],rec["area"]
        if variant=="v1":
            add(raw,"paper_to_area",P,R); add(raw,"area_to_paper",R,P)
        elif variant=="v2":
            for C in rec["conferences"]:
                add(raw,"conference_to_area",C,R); add(raw,"area_to_conference",R,C)
        elif variant=="v3":
            for A in rec["authors"]:
                add(raw,"author_to_area",A,R); add(raw,"area_to_author",R,A)
        elif variant=="universal":
            # DBLP* -- Area on Paper AND Conference AND Author at once.
            add(raw,"paper_to_area",P,R); add(raw,"area_to_paper",R,P)
            for C in rec["conferences"]:
                add(raw,"conference_to_area",C,R); add(raw,"area_to_conference",R,C)
            for A in rec["authors"]:
                add(raw,"author_to_area",A,R); add(raw,"area_to_author",R,A)
        else: raise ValueError(variant)
    return raw


def _adj(edges, relation):
    out=defaultdict(set)
    for u,v in edges.get(relation,set()): out[int(u)].add(int(v))
    return out


def build_paper_area_pair_filter(certificate):
    """Build a context-only, block-local Paper--Area support map.

    DBLP v2 stores Area evidence on Conference nodes. A conference can occur in
    papers from several areas, so a global P->C->Area traversal is generally
    one-to-many even though each transformation block (one supported paper) has
    exactly one licensed Area. The pair filter records that training-only block
    certificate. It is used only to select semantic witnesses and is never added
    to the physical or semantic RGCN edge sets.
    """
    out = defaultdict(set)
    for rec in certificate:
        out[int(rec["paper"])].add(int(rec["area"]))
    return {paper: set(areas) for paper, areas in out.items()}


def compile_semantic_from_contexts(base, raw, support_papers, variant, paper_area_filter=None):
    """Recover the transformation closure from one distinct physical graph.

    The physical context proposes Area candidates:

    * v1: Paper -> Area
    * v2: Paper -> Conference -> Area
    * v3: Paper -> Author -> Area

    Candidate paths are then intersected with the training-only, block-local
    Paper--Area pair filter. This is required for v2 because a shared Conference
    may be connected to several Areas across different paper blocks. The filter
    does not become a model edge; it only identifies which physical witness
    belongs to the current paper's invertible block.
    """
    if paper_area_filter is None:
        paper_area_filter = {}
    sem=defaultdict(set,{k:set(v) for k,v in base.items()})
    p_to_c=_adj(base,"paper_to_conference")
    p_to_a=_adj(base,"paper_to_author")
    p_to_r=_adj(raw,"paper_to_area")
    c_to_r=_adj(raw,"conference_to_area")
    a_to_r=_adj(raw,"author_to_area")
    audit={
        "support_papers":len(support_papers),
        "variant":variant,
        "context_matches":0,
        "emitted_messages":0,
        "pair_filter":"paper-area-from-author-label",
        "pair_filter_scope":"training-paper-conference-support",
        "ambiguous_raw_contexts":0,
        "raw_candidates_rejected_by_pair_filter":0,
    }
    for paper in sorted(map(int,support_papers)):
        if variant=="v1":
            raw_candidates=set(p_to_r.get(paper,set()))
            context="P-Area"
        elif variant=="v2":
            raw_candidates=set()
            for conf in p_to_c.get(paper,set()):
                raw_candidates.update(c_to_r.get(conf,set()))
            context="P-Conference-Area"
        elif variant=="v3":
            raw_candidates=set()
            for author in p_to_a.get(paper,set()):
                raw_candidates.update(a_to_r.get(author,set()))
            context="P-Author-Area"
        elif variant=="universal":
            # DBLP* carries all three placements at once, so every route is
            # available; their union is the raw candidate set and the block-local
            # pair filter selects the licensed Area exactly as for v2/v3.
            raw_candidates=set(p_to_r.get(paper,set()))
            for conf in p_to_c.get(paper,set()):
                raw_candidates.update(c_to_r.get(conf,set()))
            for author in p_to_a.get(paper,set()):
                raw_candidates.update(a_to_r.get(author,set()))
            context="P-Area+P-Conference-Area+P-Author-Area"
        else:
            raise ValueError(variant)

        licensed=set(map(int,paper_area_filter.get(paper,set())))
        # Backward compatibility for simple, intrinsically unambiguous contexts.
        # Real DBLP v2 preprocessing always supplies the explicit pair filter.
        if not licensed and len(raw_candidates)==1:
            licensed=set(raw_candidates)
        if len(licensed)!=1:
            raise RuntimeError(
                f"Invalid or missing block-local Paper-Area certificate for paper={paper}: "
                f"licensed_areas={sorted(licensed)}, "
                f"raw_area_candidates={sorted(raw_candidates)}"
            )
        candidates=raw_candidates & licensed
        if len(raw_candidates)>1:
            audit["ambiguous_raw_contexts"] += 1
        audit["raw_candidates_rejected_by_pair_filter"] += len(raw_candidates-candidates)
        if len(candidates)!=1:
            raise RuntimeError(
                f"Block-local context mismatch for paper={paper}, variant={variant}, "
                f"context={context}, raw_area_candidates={sorted(raw_candidates)}, "
                f"licensed_areas={sorted(licensed)}, filtered_candidates={sorted(candidates)}"
            )

        area=next(iter(candidates)); audit["context_matches"]+=1
        add(sem,"paper_to_area",paper,area);add(sem,"area_to_paper",area,paper)
        for conf in p_to_c.get(paper,set()):
            add(sem,"conference_to_area",conf,area);add(sem,"area_to_conference",area,conf)
        for author in p_to_a.get(paper,set()):
            add(sem,"author_to_area",author,area);add(sem,"area_to_author",area,author)
        audit["emitted_messages"] += 2 + 2*len(p_to_c.get(paper,set())) + 2*len(p_to_a.get(paper,set()))
    return sem,audit


def compile_semantic_from_certificate(base, raw, certificate, variant):
    """Full-transductive fallback when held-out P-C context is intentionally allowed."""
    sem=defaultdict(set,{k:set(v) for k,v in base.items()})
    audit={"certificate_records":len(certificate),"variant":variant,"context_matches":0,"emitted_messages":0}
    for rec in certificate:
        paper,area=rec["paper"],rec["area"]
        if variant=="v1": ok=(paper,area) in raw["paper_to_area"]
        elif variant=="v2": ok=all((c,area) in raw["conference_to_area"] for c in rec["conferences"])
        else: ok=all((a,area) in raw["author_to_area"] for a in rec["authors"])
        if not ok: raise RuntimeError(f"Missing physical context for certificate record {rec}")
        audit["context_matches"]+=1
        add(sem,"paper_to_area",paper,area);add(sem,"area_to_paper",area,paper)
        for conf in rec["conferences"]:
            add(sem,"conference_to_area",conf,area);add(sem,"area_to_conference",area,conf)
        for author in rec["authors"]:
            add(sem,"author_to_area",author,area);add(sem,"area_to_author",area,author)
        audit["emitted_messages"] += 2 + 2*len(rec["conferences"]) + 2*len(rec["authors"])
    return sem,audit


def to_tensors(edges):
    src=[];dst=[];typ=[]
    for rel in REL_NAMES:
        for u,v in sorted(edges.get(rel,set())):
            src.append(u);dst.append(v);typ.append(REL[rel])
    return canonicalize_edges(torch.tensor([src,dst],dtype=torch.long),torch.tensor(typ,dtype=torch.long))


def main():
    ap=argparse.ArgumentParser(description="Build physically distinct DBLP variants with invariant conditional RGCN messages")
    ap.add_argument("--raw-dir",default="data/raw/DBLP")
    ap.add_argument("--shared-npz",default="data/preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz")
    ap.add_argument("--out-root",default="data/preprocessed/RGCN_structural")
    ap.add_argument("--variants",default=",".join(VARIANTS))
    ap.add_argument("--mapping-scope",choices=("train","full"),default="train",
                    help="train: transformation certificates use training P-C blocks only (no held-out topology); full: reproduce full transductive variant construction")
    ap.add_argument("--min-conf",type=int,default=0)
    ap.add_argument("--neg-k",type=int,default=10)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args(); set_deterministic(args.seed)
    variants=parse_str_list(args.variants)
    if set(variants)-set(VARIANTS): raise SystemExit(f"Unknown variants: {set(variants)-set(VARIANTS)}")

    al,pa,pc,pt=read_tables(args.raw_dir)
    raw_splits,paper_subset=load_or_create_splits(args.shared_npz,pc,args.seed,args.neg_k)
    al,pa,pc,pt=filter_tables(al,pa,pc,pt,paper_subset,args.min_conf)
    pr=paper_area(al,pa)
    valid=set(pr["paper_id"]); pa=pa[pa["paper_id"].isin(valid)];pc=pc[pc["paper_id"].isin(valid)];pt=pt[pt["paper_id"].isin(valid)]
    ids,maps,offsets,node_type=make_maps(pa,pc,pt,pr)
    splits=remap_splits(raw_splits,maps)
    base=build_base_edges(pa,pt,splits["train_pos"],maps)
    if args.mapping_scope=="train":
        # Convert global training paper ids back to raw ids for certificate selection.
        inverse_p={g:r for r,g in maps["paper"].items()}
        support={inverse_p[int(g)] for g in splits["train_pos"][:,0].tolist() if int(g) in inverse_p}
    else:
        support=set(pc["paper_id"])
    # Variant construction uses the known transformation blocks. In train scope,
    # the invariant compiler itself recovers each block from the physical context.
    cert=build_area_certificate(pa,pc,pr,maps,support)
    support_global={maps["paper"][p] for p in support if p in maps["paper"]}
    paper_area_filter=build_paper_area_pair_filter(cert)
    task_data={k:v for k,v in splits.items()}
    # Provenance: which shared-split file produced this store. The DBLP partition
    # is not recoverable from the store itself (ids are remapped), and two
    # different partitions of the same pair universe exist.
    # Recording it here makes the split auditable.
    _npz=str(args.shared_npz) if args.shared_npz else ""
    _npz_sha=""
    if _npz and Path(_npz).exists():
        import hashlib as _hashlib
        _npz_sha=_hashlib.sha256(Path(_npz).read_bytes()).hexdigest()
    sem_hashes=[]
    for variant in variants:
        raw=build_physical(base,cert,variant)
        if args.mapping_scope=="train":
            sem,audit=compile_semantic_from_contexts(
                base,raw,support_global,variant,paper_area_filter
            )
            audit["compiler"]="raw_context_with_block_local_pair_filter"
        else:
            sem,audit=compile_semantic_from_certificate(base,raw,cert,variant)
            audit["compiler"]="transformation_certificate"
        pidx,ptyp,_=to_tensors(raw); sidx,styp,sw=to_tensors(sem)
        data=ConditionalGraphData(
            variant=variant,num_nodes=len(node_type),node_type=torch.tensor(node_type),
            physical_edge_index=pidx,physical_edge_type=ptyp,physical_relation_names=REL_NAMES,
            semantic_edge_index=sidx,semantic_edge_type=styp,semantic_edge_weight=sw,semantic_relation_names=REL_NAMES,
            x=None,y=None,task_data=task_data,context_audit=audit,
            metadata={"dataset":"DBLP","task":"paper_conference","mapping_scope":args.mapping_scope,
                      "num_nodes_by_type":{k:len(ids[k]) for k in ids},"offsets":offsets,
                      "message_semantics":"set","paper_conference_edges":"train_only",
                      "shared_npz":_npz,"shared_npz_sha256":_npz_sha,
                      "certificate_required":args.mapping_scope=="full",
                      "compiler_mode":"raw_context_with_block_local_pair_filter" if args.mapping_scope=="train" else "transformation_certificate",
                      "pair_filter_model_input":False,
                      "certificate_note":"Train scope matches the variant-specific raw path and uses a training-only Paper-Area pair filter derived from author labels to select the correct local block. The filter is context-only and is never propagated. Full scope permits the dataset transformation certificate because held-out P-C context is not in the message graph."}
        )
        out=Path(args.out_root)/"DBLP"/"paper_conference"/variant;save_graph_data(data,out)
        sem_hashes.append(data.semantic_hash)
        print(f"[DBLP/{variant}] physical={data.physical_hash[:12]} semantic={data.semantic_hash[:12]} Eraw={ptyp.numel()} Esem={styp.numel()}")
    if len(set(sem_hashes))!=1: raise RuntimeError(f"Semantic program mismatch: {sem_hashes}")
    print("DBLP preprocessing complete: raw hashes differ; semantic hashes match.")

if __name__=="__main__": main()
