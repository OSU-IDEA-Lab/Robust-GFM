"""remove_easy_edges with remove_one_hop=False (WordNet contract): identity alias == stock mask; a real alias adds
exactly the copy triples of the batch; remove_one_hop=True unchanged. Run: python tests/test_remove_easy_edges_one_hop_false.py (CPU)."""
import torch, sys
sys.path.insert(0, ".")
from torchdrug import data
from cmpnn.model import CMPNN
from inv_cmpnn.model import InvariantCMPNN
torch.manual_seed(0); n, R = 300, 4
el = torch.unique(torch.cat([torch.randint(0, n, (3000, 2)), torch.randint(0, R, (3000, 1))], 1), dim=0)
g = data.Graph(el, num_node=n, num_relation=R)
B, K = 16, 5
pos = el[torch.randperm(len(el))[:B]]                                   # real triples as positives
h = pos[:, 0:1].repeat(1, K); t = torch.cat([pos[:, 1:2], torch.randint(0, n, (B, K - 1))], 1); r = pos[:, 2:3].repeat(1, K)
kw = dict(input_dim=8, hidden_dims=[8, 8], num_relation=R, message_func="distmult", aggregate_func="sum",
          short_cut=True, layer_norm=True, dependent=False, set_boundary=True, activation="relu", initialization="Query")
stock = CMPNN(remove_one_hop=False, **kw)
ident = InvariantCMPNN(remove_one_hop=False, deterministic=False, alias=torch.arange(n), **kw)
m_stock = stock.remove_easy_edges(g, h, t, r).edge_list; m_ident = ident.remove_easy_edges(g, h, t, r).edge_list
assert torch.equal(m_stock, m_ident), "identity alias must reproduce the stock mask"
print(f"identity alias == stock mask: {len(el)} -> {len(m_stock)} edges ({len(el)-len(m_stock)} masked, {B} positives in batch)")
alias = torch.arange(n); alias[:100] = torch.arange(100, 200); alias[100:200] = torch.arange(0, 100)   # nodes 0-99 <-> 100-199 copies
al = InvariantCMPNN(remove_one_hop=False, deterministic=False, alias=alias, **kw)
m_al = al.remove_easy_edges(g, h, t, r).edge_list
kept = {tuple(x) for x in m_al.tolist()}; kept_id = {tuple(x) for x in m_ident.tolist()}
assert kept <= kept_id, "alias mask must be a subset (superset of removed edges)"
extra = kept_id - kept
batch = [(int(hh), int(tt), int(rr)) for hh, tt, rr in zip(h.flatten().tolist(), t.flatten().tolist(), r.flatten().tolist())]   # positives AND negatives, as stock masks
want = {(int(alias[hh]), tt, rr) for hh, tt, rr in batch} | {(hh, int(alias[tt]), rr) for hh, tt, rr in batch}
assert extra <= want and all((e in {tuple(x) for x in el.tolist()}) for e in extra), "extra removals must be exactly the copy triples present in the graph"
print(f"alias: {len(extra)} additional copy-triple removals, all of the expected form")
# remove_one_hop=True path unchanged vs the pre-fix implementation semantics: masks (h,t,*) and (t,h,*)
one = InvariantCMPNN(remove_one_hop=True, deterministic=False, alias=torch.arange(n), **kw)
s1 = CMPNN(remove_one_hop=True, **kw)
assert torch.equal(one.remove_easy_edges(g, h, t, r).edge_list, s1.remove_easy_edges(g, h, t, r).edge_list)
print("remove_one_hop=True with identity alias == stock: OK")
print("ALL OK")
