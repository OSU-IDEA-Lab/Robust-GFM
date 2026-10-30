"""CMPNN with (a) order-independent aggregation and (b) alias-aware easy-edge masking.

Everything else — query boundary, DistMult messages, PNA statistics, LayerNorm/short-cut,
the ``undirected(add_inverse=True)`` doubling, ``negative_sample_to_tail`` and the MLP
head — is the repository model, untouched.
"""
from __future__ import annotations

from typing import Optional

import torch
from torchdrug import data
from torchdrug.layers import functional

from cmpnn.model import CMPNN

from .conv import DeterministicRelationalConv


class InvariantCMPNN(CMPNN):
    """``CMPNN`` whose layers are :class:`DeterministicRelationalConv`.

    ``alias`` is an optional ``[num_node]`` LongTensor mapping each node to its bijective
    copy (Link ↔ Movie on IMDb; identity elsewhere). During training the query edge is
    masked together with the same fact expressed on the copy node, so no training-only
    path to the label survives.
    """

    def __init__(self, *args, deterministic: bool = True, alias: Optional[torch.Tensor] = None,
                 label_derived_relations: Optional[list] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic
        # relation ids (in the graph's vocabulary, BEFORE CMPNN adds inverses) whose
        # (head, *) edges are masked with the query -- see bundles.LABEL_DERIVED_RELATIONS
        self.label_derived_relations = list(label_derived_relations or [])
        if deterministic:
            new_layers = torch.nn.ModuleList()
            for old in self.layers:
                conv = DeterministicRelationalConv(
                    input_dim=old.input_dim, output_dim=old.output_dim, num_relation=old.num_relation,
                    query_input_dim=old.query_input_dim, message_func=old.message_func,
                    aggregate_func=old.aggregate_func, layer_norm=old.layer_norm is not None,
                    activation=old.activation, dependent=old.dependent, set_boundary=old.set_boundary,
                    rgcn=old.rgcn, num_bases=old.num_bases, has_readout=old.has_readout,
                    readout_type=old.readout_type, query_specific_readout=old.query_specific_readout,
                )
                conv.load_state_dict(old.state_dict())   # identical initial parameters
                new_layers.append(conv)
            self.layers = new_layers
        self.register_buffer("alias", alias.clone() if alias is not None else None, persistent=False)

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.alias is None and not self.label_derived_relations:
            return super().remove_easy_edges(graph, h_index, t_index, r_index)
        alias = self.alias if self.alias is not None else torch.arange(graph.num_node, device=graph.device)
        a_h = alias[h_index]
        a_t = alias[t_index]
        if self.remove_one_hop:
            # every edge between the endpoints, either direction, any relation -- and the same on
            # the alias copies (IMDb Link <-> Movie)
            heads = torch.cat([h_index, t_index, a_h, t_index, h_index, a_t], dim=-1)
            tails = torch.cat([t_index, h_index, t_index, a_h, a_t, h_index], dim=-1)
            rels = -torch.ones_like(heads)
        else:
            # WordNet contract (remove_one_hop=False, as in the stock runner): only the exact query
            # triple (h, t, r) is masked -- plus the same triple on the alias copies. This runs BEFORE
            # graph.undirected(add_inverse=True), so masking the forward triple removes both
            # directions, exactly as the stock branch does. With an identity alias (WordNet) the
            # copies coincide with the original; graph.match tolerates the duplicate patterns.
            if r_index is None:
                raise ValueError("remove_one_hop=False masking needs r_index")
            heads = torch.cat([h_index, a_h, h_index], dim=-1)
            tails = torch.cat([t_index, t_index, a_t], dim=-1)
            rels = torch.cat([r_index, r_index, r_index], dim=-1)
        for rid in self.label_derived_relations:            # (head, *, rid): the head's label-derived edges
            hq = h_index[:, :1]
            heads = torch.cat([heads, hq], dim=-1)
            tails = torch.cat([tails, -torch.ones_like(hq)], dim=-1)
            rels = torch.cat([rels, torch.full_like(hq, int(rid))], dim=-1)
        pattern = torch.stack([heads, tails, rels], dim=-1).flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)


def build_graph(edges: torch.Tensor, num_node: int, num_relation: int, device) -> data.Graph:
    """torchdrug Graph from an ``[E, 3]`` (src, dst, rel) edge table in canonical order."""
    g = data.Graph(edges.to(torch.long), num_node=int(num_node), num_relation=int(num_relation))
    return g.to(device)


def build_model(num_relation: int, alias: Optional[torch.Tensor], *, input_dim=32, hidden_dim=32,
                num_layers=6, deterministic=True, message_func="distmult", aggregate_func="pna",
                dependent=False, gradient_checkpoint=False, label_derived_relations=None,
                remove_one_hop=True, chunk_edges=None) -> InvariantCMPNN:
    """The baseline runners' architecture, deterministic conv.

    IMDb / DBLP baselines: PNA aggregation, remove_one_hop=True (run_CMPNN_IMDB_md.py,
    run_CMPNN_DBLP_pc.py). WordNet baseline: SUM aggregation, remove_one_hop=False
    (run_CMPNN_wordnet_lp.py) -- and sum is what makes the 1.6M-message program affordable.
    """
    m = InvariantCMPNN(
        input_dim=input_dim, hidden_dims=[hidden_dim] * num_layers, num_relation=num_relation,
        message_func=message_func, aggregate_func=aggregate_func, short_cut=True, layer_norm=True,
        dependent=dependent, set_boundary=True, remove_one_hop=remove_one_hop, activation="relu",
        initialization="Query", deterministic=deterministic, alias=alias,
        label_derived_relations=label_derived_relations,
    )
    for layer in m.layers:
        layer.gradient_checkpoint = gradient_checkpoint   # (the repo layer's own flag; its import is broken)
        layer.chunk_edges = chunk_edges                    # deterministic chunked reduction (WordNet)
    return m
