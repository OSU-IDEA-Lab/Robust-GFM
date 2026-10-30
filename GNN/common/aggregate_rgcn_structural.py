#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def row_kendall(a: np.ndarray, b: np.ndarray) -> float:
    vals=[]
    for x,y in zip(a,b):
        if np.array_equal(x,y):
            vals.append(1.0)
            continue
        tau,_=kendalltau(x,y,nan_policy="omit")
        if not np.isnan(tau): vals.append(float(tau))
    return float(np.mean(vals)) if vals else float("nan")


def load_runs(root: Path):
    runs=[]
    for p in root.rglob("metrics.json"):
        obj=json.loads(p.read_text());obj["metrics_path"]=str(p);obj["path"]=p.parent;runs.append(obj)
    return runs


def main():
    ap=argparse.ArgumentParser(description="Aggregate conditional RGCN performance and invariance")
    ap.add_argument("--results-root",default="results/RGCN_structural")
    ap.add_argument("--out-dir",default="results/RGCN_structural/aggregate")
    args=ap.parse_args();root=Path(args.results_root);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    runs=load_runs(root)
    if not runs: raise SystemExit(f"No metrics.json under {root}")
    groups=defaultdict(list)
    # Other experiment families (e.g. WORDNET structural) share this results root
    # with a different metrics schema; aggregate only records with our keys.
    skipped=sum(1 for r in runs if not all(k in r for k in ("dataset","task","variant","metrics")))
    if skipped: print(f"[aggregate] skipping {skipped} metrics.json with foreign schema")
    runs=[r for r in runs if all(k in r for k in ("dataset","task","variant","metrics"))]
    if not runs: raise SystemExit("No structural-LP metrics.json records found")
    for r in runs: groups[(r["dataset"],r["task"],r["variant"])].append(r)
    rows=[]
    for (ds,task,var),items in sorted(groups.items()):
        metric_names=sorted({k for x in items for k,v in x["metrics"].items() if isinstance(v,(int,float))})
        row={"dataset":ds,"task":task,"variant":var,"runs":len(items),"semantic_hash_count":len({x["semantic_hash"] for x in items}),"physical_hash_count":len({x["physical_hash"] for x in items})}
        for m in metric_names:
            vals=np.asarray([x["metrics"].get(m,np.nan) for x in items],dtype=float)
            row[m+"_mean"]=float(np.nanmean(vals));row[m+"_std"]=float(np.nanstd(vals,ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"per_variant.csv",index=False)

    # Task-level spread across variant means.
    spreads=[]
    by_task=defaultdict(list)
    for row in rows: by_task[(row["dataset"],row["task"])].append(row)
    for (ds,task),items in sorted(by_task.items()):
        rec={"dataset":ds,"task":task}
        means=[k for k in items[0] if k.endswith("_mean")]
        for key in means:
            vals=[x.get(key,np.nan) for x in items];rec[key.replace("_mean","_spread")]=float(np.nanmax(vals)-np.nanmin(vals))
        spreads.append(rec)
    pd.DataFrame(spreads).to_csv(out/"spread.csv",index=False)

    # Same-seed pairwise output agreement.
    by_dst_seed=defaultdict(dict)
    for r in runs: by_dst_seed[(r["dataset"],r["task"],r["seed"])][r["variant"]]=r
    inv=[]
    for (ds,task,seed),variants in sorted(by_dst_seed.items()):
        for va,vb in combinations(sorted(variants),2):
            ra,rb=variants[va],variants[vb]
            za=np.load(ra["path"]/"scores.npz");zb=np.load(rb["path"]/"scores.npz")
            a,b=za["scores"],zb["scores"]
            id_keys=[k for k in ("ids","queries","positive_tails") if k in za.files or k in zb.files]
            ids_equal=all(k in za.files and k in zb.files and np.array_equal(za[k],zb[k]) for k in id_keys)
            shape_equal=a.shape==b.shape
            tau=row_kendall(a,b) if shape_equal and ids_equal else float("nan")
            maxdiff=float(np.max(np.abs(a-b))) if shape_equal and ids_equal else float("inf")
            ta=json.loads((ra["path"]/"trace.json").read_text());tb=json.loads((rb["path"]/"trace.json").read_text())
            trace_equal=len(ta)==len(tb) and all(x.get("state_hash")==y.get("state_hash") for x,y in zip(ta,tb))
            inv.append({"dataset":ds,"task":task,"seed":seed,"variant_a":va,"variant_b":vb,
                        "raw_graphs_different":ra["physical_hash"]!=rb["physical_hash"],
                        "semantic_program_equal":ra["semantic_hash"]==rb["semantic_hash"],
                        "feature_equal":ra.get("feature_hash")==rb.get("feature_hash"),
                        "supervision_equal":ra.get("supervision_hash")==rb.get("supervision_hash"),
                        "ids_equal":ids_equal,"score_shape_equal":shape_equal,"max_abs_score_diff":maxdiff,
                        "kendall_tau":tau,"checkpoint_hash_equal":ra["checkpoint_hash"]==rb["checkpoint_hash"],
                        "training_trace_equal":trace_equal,
                        "gradient_hash_equal":ra.get("first_step_audit",{}).get("gradient_hash")==rb.get("first_step_audit",{}).get("gradient_hash")})
    pd.DataFrame(inv).to_csv(out/"invariance.csv",index=False)
    summary={"num_runs":len(runs),"per_variant":rows,"spread":spreads,"invariance":inv}
    (out/"aggregate.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str))
    print(f"Wrote {out/'per_variant.csv'}, {out/'spread.csv'}, {out/'invariance.csv'}")

if __name__=="__main__": main()
