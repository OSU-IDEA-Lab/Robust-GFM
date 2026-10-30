"""Unit tests for inv_cmpnn (synthetic graphs, CPU, no data files).

    cd models/CMPNN && PYTHONPATH=. python -m unittest tests.test_inv_cmpnn -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inv_cmpnn.bundles import _forward_half_pairs, _select_forward, canonical_edge_order  # noqa: E402
from inv_cmpnn.conv import DeterministicRelationalConv  # noqa: E402
from inv_cmpnn.model import InvariantCMPNN, build_graph  # noqa: E402
from cmpnn.layer import GeneralizedRelationalConv  # noqa: E402
from cmpnn.model import CMPNN  # noqa: E402


class ForwardHalfTests(unittest.TestCase):
    def test_pairs_imdb_style(self):
        names = ["movie_to_director", "director_to_movie", "movie_to_actor", "actor_to_movie", "movie_to_year"]
        fwd, rev = _forward_half_pairs(names)
        self.assertEqual(fwd, [0, 2, 4])
        self.assertEqual(rev, {1: 0, 3: 2})

    def test_pairs_wordnet_style(self):
        names = ["base::_hypernym", "inverse::_hypernym__inv", "dependency::abc"]
        fwd, rev = _forward_half_pairs(names)
        self.assertEqual(fwd, [0, 2])
        self.assertEqual(rev, {1: 0})

    def test_select_forward_drops_mirrors_and_sorts(self):
        names = ["a_to_b", "b_to_a"]
        edges = torch.tensor([[2, 1, 0], [0, 1, 0], [1, 2, 1], [1, 0, 1]])  # forward + exact mirrors
        out, fnames, _ = _select_forward(edges, names)
        self.assertEqual(fnames, ["a_to_b"])
        self.assertEqual(out.tolist(), [[0, 1, 0], [2, 1, 0]])   # sorted by (rel, dst, src)

    def test_select_forward_rejects_unmirrored_reverse(self):
        names = ["a_to_b", "b_to_a"]
        edges = torch.tensor([[0, 1, 0], [5, 0, 1]])   # reverse edge with no forward partner
        with self.assertRaises(ValueError):
            _select_forward(edges, names)

    def test_select_forward_rejects_duplicates(self):
        names = ["a_to_b"]
        edges = torch.tensor([[0, 1, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            _select_forward(edges, names)

    def test_canonical_order_is_permutation_invariant(self):
        e = torch.tensor([[3, 1, 0], [0, 2, 1], [1, 1, 0], [2, 0, 1]])
        p = torch.randperm(4)
        a = e[canonical_edge_order(e[:, 0], e[:, 1], e[:, 2])]
        b = e[p][canonical_edge_order(e[p][:, 0], e[p][:, 1], e[p][:, 2])]
        self.assertTrue(torch.equal(a, b))


def _toy(seed=0, n=30, e=90, R=3):
    g = torch.Generator().manual_seed(seed)
    el = torch.cat([torch.randint(0, n, (e, 2), generator=g), torch.randint(0, R, (e, 1), generator=g)], 1)
    el = torch.unique(el, dim=0)
    return el, n, R


class DeterministicConvTests(unittest.TestCase):
    def test_matches_stock_conv_numerically(self):
        el, n, R = _toy()
        torch.manual_seed(0)
        stock = CMPNN(input_dim=8, hidden_dims=[8, 8], num_relation=R, aggregate_func="pna", dependent=False)
        det = InvariantCMPNN(input_dim=8, hidden_dims=[8, 8], num_relation=R, aggregate_func="pna", dependent=False, deterministic=True)
        det.load_state_dict(stock.state_dict())
        self.assertTrue(all(isinstance(l, DeterministicRelationalConv) for l in det.layers))
        self.assertTrue(all(type(l) is GeneralizedRelationalConv for l in stock.layers))
        g = build_graph(el, n, R, "cpu")
        h = torch.full((4, 5), 3); t = torch.randint(0, n, (4, 5)); r = torch.zeros(4, 5, dtype=torch.long)
        torch.manual_seed(1); a = stock(g, h, t, r)
        torch.manual_seed(1); b = det(g, h, t, r)
        self.assertTrue(torch.allclose(a, b, atol=1e-5), (a - b).abs().max())

    def test_bitwise_stable_under_edge_permutation(self):
        """Same multiset of edges in a different order -> identical outputs bitwise."""
        el, n, R = _toy()
        torch.manual_seed(0)
        det = InvariantCMPNN(input_dim=8, hidden_dims=[8, 8], num_relation=R, aggregate_func="pna", dependent=False, deterministic=True)
        h = torch.full((4, 5), 3); t = torch.randint(0, n, (4, 5)); r = torch.zeros(4, 5, dtype=torch.long)
        outs = []
        for perm_seed in (0, 1):
            perm = torch.randperm(el.shape[0], generator=torch.Generator().manual_seed(perm_seed))
            e = el[perm]
            e = e[canonical_edge_order(e[:, 0], e[:, 1], e[:, 2])]   # the loader's canonical order
            torch.manual_seed(1); outs.append(det(build_graph(e, n, R, "cpu"), h, t, r))
        self.assertTrue(torch.equal(outs[0], outs[1]))


class MaskingTests(unittest.TestCase):
    def test_alias_and_label_derived_edges_are_masked(self):
        # nodes: 0 movie, 1 director, 2 link(=alias of 0), 3 area
        # rels: 0 movie_to_director, 1 link_to_director, 2 movie_to_area, 3 other
        el = torch.tensor([[0, 1, 0], [2, 1, 1], [0, 3, 2], [3, 1, 3], [0, 2, 3]])
        alias = torch.tensor([2, 1, 0, 3])
        m = InvariantCMPNN(input_dim=4, hidden_dims=[4], num_relation=4, aggregate_func="pna", dependent=False,
                           deterministic=True, alias=alias, label_derived_relations=[2], remove_one_hop=True)
        g = build_graph(el, 4, 4, "cpu")
        h = torch.tensor([[0, 0]]); t = torch.tensor([[1, 3]]); r = torch.tensor([[0, 0]])
        gm = m.remove_easy_edges(g, h, t, r)
        kept = set(map(tuple, gm.edge_list.tolist()))
        self.assertNotIn((0, 1, 0), kept)   # query edge
        self.assertNotIn((2, 1, 1), kept)   # alias copy link->director
        self.assertNotIn((0, 3, 2), kept)   # label-derived movie_to_area
        self.assertIn((3, 1, 3), kept)      # general fact stays
        self.assertIn((0, 2, 3), kept)      # movie->link stays

    def test_no_alias_falls_back_to_stock(self):
        el, n, R = _toy()
        m = InvariantCMPNN(input_dim=4, hidden_dims=[4], num_relation=R, aggregate_func="pna", dependent=False, deterministic=True, remove_one_hop=True)
        g = build_graph(el, n, R, "cpu")
        H, T = int(el[0, 0]), int(el[0, 1])
        h = torch.tensor([[H, H]]); t = torch.tensor([[T, T]]); r = torch.tensor([[int(el[0, 2])] * 2])
        gm = m.remove_easy_edges(g, h, t, r)
        # stock remove_one_hop semantics: every (H,T,*) and (T,H,*) edge goes, nothing else
        expected = int((((el[:, 0] == H) & (el[:, 1] == T)) | ((el[:, 0] == T) & (el[:, 1] == H))).sum())
        self.assertEqual(int(gm.num_edge), int(g.num_edge) - expected)


if __name__ == "__main__":
    unittest.main()
