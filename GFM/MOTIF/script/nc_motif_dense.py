"""Run script/nc_motif.py with a dense relation-hypergraph builder, for graphs too large for cuSPARSE.

On Freebase v3 (exact_3, 447M edges plus inverses) the released `motif.tasks.build_relation_hypergraph`
fails in `torch.sparse.mm` ("cusparseSpGEMM_workEstimation: insufficient resources"). The hypergraph
only depends on which (relation, relation[, relation]) patterns occur, never on the counts, so the
same edges can be built from dense node-by-relation incidence matrices, streaming the edges in
chunks. `build_relation_hypergraph_dense` returns the same edge_index/edge_type, in the same order
(lexicographic, as `coalesce()` sorts them); `--check` verifies that against the released builder.

    python script/nc_motif_dense.py eval --dataset freebase --variant v3 ...   # same arguments as nc_motif.py
    python script/nc_motif_dense.py --check --dataset freebase --variant v1    # compare with the released builder
"""
from __future__ import annotations

import sys

import torch
from torch_geometric.data import Data

from lp_data import ROOT

sys.path.insert(0, str(ROOT / "upstream"))
from motif import tasks  # noqa: E402

CHUNK = 1 << 23  # edges per matmul chunk; 8M x num_rels float32 per operand


def incidence(nodes, types, num_nodes, num_rels):
    """Dense [num_nodes, num_rels] bool: node takes part in at least one edge of the relation."""
    out = torch.zeros(num_nodes, num_rels, dtype=torch.bool, device=nodes.device)
    out[nodes, types] = True
    return out


def build_relation_hypergraph_dense(graph, max_arity=3):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = int(graph.num_nodes), int(graph.num_relations)
    device = graph.device
    edge_index, edge_type = edge_index.to(device), edge_type.to(device)
    head, tail = edge_index[0], edge_index[1]
    Eh = incidence(head, edge_type, num_nodes, num_rels)
    Et = incidence(tail, edge_type, num_nodes, num_rels)
    # Same check as the released builder: every tail node has a nonzero tail degree.
    assert Et.any(1)[tail].all()

    # 3-length paths through edge (m, r2, n): r1 meets m, r3 meets n (as head h or tail t).
    paths = {name: torch.zeros(num_rels, num_rels, num_rels, device=device) for name in ("tfh", "tft", "hft", "hfh")}
    for r2 in range(num_rels):
        edges = (edge_type == r2).nonzero().squeeze(1)
        for start in range(0, len(edges), CHUNK):
            chunk = edges[start:start + CHUNK]
            m, n = head[chunk], tail[chunk]
            tm, hm, tn, hn = Et[m].float(), Eh[m].float(), Et[n].float(), Eh[n].float()
            paths["tfh"][:, r2] += tm.T @ hn
            paths["tft"][:, r2] += tm.T @ tn
            paths["hft"][:, r2] += hm.T @ tn
            paths["hfh"][:, r2] += hm.T @ hn
    # Counts are sums of 0/1 products, so > 0 is exact even where float32 rounds large counts.
    path_3 = [(paths[name] > 0).nonzero() for name in ("tfh", "tft", "hft", "hfh")]
    edge_index_path_3 = torch.cat([p.T for p in path_3], dim=1)
    edge_type_path_3 = torch.cat([torch.full((len(p),), 3 + i, dtype=torch.long, device=device) for i, p in enumerate(path_3)])

    Ehf, Etf = Eh.float(), Et.float()
    path_2 = [((a.T @ b) > 0).nonzero() for a, b in ((Ehf, Ehf), (Etf, Etf), (Ehf, Etf))]  # hh, tt, ht
    edge_index_path_2 = torch.cat([p.T for p in path_2], dim=1)
    edge_type_path_2 = torch.cat([torch.full((len(p),), i, dtype=torch.long, device=device) for i, p in enumerate(path_2)])
    expanded_binary_edge_index = torch.full((max_arity, edge_index_path_2.size(1)), -1, dtype=torch.long, device=device)
    expanded_binary_edge_index[:2] = edge_index_path_2

    graph.relation_hypergraph = Data(
        edge_index=torch.cat([expanded_binary_edge_index, edge_index_path_3], dim=1),
        edge_type=torch.cat([edge_type_path_2, edge_type_path_3]),
        num_nodes=torch.tensor(num_rels, device=device),
        num_relations=3 + 4,
    )
    return graph


def check(argv):
    """Build both hypergraphs for one exported variant and require identical edges."""
    import argparse
    import json

    from eval_lp import build_message_graph
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", default=ROOT / "rai-datasets/preprocessed/NC")
    args = parser.parse_args(argv)
    root = f"{args.data}/{args.dataset}"
    manifest = json.load(open(f"{root}/manifest.json"))
    edges = np.load(f"{root}/{args.variant}.npy")
    names = manifest["variants"][args.variant]["relation_names"]
    device = torch.device("cuda")
    reference = build_message_graph(edges, names, manifest["num_nodes"], device).relation_hypergraph
    tasks.build_relation_hypergraph = build_relation_hypergraph_dense
    dense = build_message_graph(edges, names, manifest["num_nodes"], device).relation_hypergraph
    same = torch.equal(reference.edge_index, dense.edge_index) and torch.equal(reference.edge_type, dense.edge_type)
    print(f"{args.dataset} {args.variant}: {reference.edge_index.size(1)} vs {dense.edge_index.size(1)} hyperedges, "
          f"identical={same}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        check(sys.argv[2:])
    else:
        import nc_motif

        tasks.build_relation_hypergraph = build_relation_hypergraph_dense
        nc_motif.main()
