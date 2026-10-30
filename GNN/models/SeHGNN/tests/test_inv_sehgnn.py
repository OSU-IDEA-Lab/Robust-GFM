"""Unit tests for inv_sehgnn: compiler == stock hg_propagate_feat_dgl; cap determinism; mask == stock model."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from inv_sehgnn.channels import ChannelSpec, compile_channel, full_type_paths, row_normalize, typed_union  # noqa: E402
from magnn_compile_common import binarize  # noqa: E402


def toy_graph(seed=0):
    """Types 0 (target, 30), 1 (20), 2 (10); relations 0->1, 1->2, 0->2 (random)."""
    rng = np.random.default_rng(seed)
    counts = [30, 20, 10]; n = sum(counts)
    node_type = np.repeat(np.arange(3), counts)
    off = np.cumsum([0] + counts)
    mats, rel_types = {}, {}
    for name, (a, b), p in (("r01", (0, 1), 0.15), ("r12", (1, 2), 0.2), ("r02", (0, 2), 0.05)):
        block = rng.random((counts[a], counts[b])) < p
        r, c = np.where(block)
        mats[name] = binarize(sp.coo_matrix((np.ones(len(r), dtype=np.int8), (r + off[a], c + off[b])), shape=(n, n)))
        rel_types[name] = (a, b)
    return n, node_type, counts, off, mats, rel_types


class TestCompilerVsDGL(unittest.TestCase):
    def test_matches_hg_propagate_feat_dgl(self):
        import dgl
        sys.path.insert(0, str(_HERE))
        from utils.tools import hg_propagate_feat_dgl
        n, node_type, counts, off, mats, rel_types = toy_graph()
        K = 3
        names = {0: "A", 1: "B", 2: "C"}
        # DGL heterograph with both directions per relation (as load_imdb_graph / HGB loaders do)
        edges = {}
        for name, m in mats.items():
            a, b = rel_types[name]; coo = m.tocoo()
            edges[(names[a], f"{names[a]}-{names[b]}", names[b])] = (coo.row - off[a], coo.col - off[b])
            edges[(names[b], f"{names[b]}-{names[a]}", names[a])] = (coo.col - off[b], coo.row - off[a])
        g = dgl.heterograph(edges, num_nodes_dict={names[t]: counts[t] for t in range(3)})
        X = {t: np.random.default_rng(t + 1).random((counts[t], 5)).astype(np.float32) for t in range(3)}
        for t in range(3):
            g.nodes[names[t]].data[names[t]] = torch.from_numpy(X[t])
        g = hg_propagate_feat_dgl(g, "A", K, K + 1, [], echo=False)
        ref = {k: v.numpy() for k, v in g.nodes["A"].data.items()}
        adj = typed_union(mats, rel_types, n)
        targets = np.arange(counts[0])
        ours = {}
        for path, steps in full_type_paths(adj, 0, K):
            key = "".join(names[t] for t in path)
            R, _ = compile_channel(ChannelSpec(key, path, steps), targets, node_type, cap=0, cap_seed=0)
            ours[key] = R @ X[path[-1]]
        self.assertEqual(set(ref), set(ours))
        for k in ref:
            np.testing.assert_allclose(ours[k], ref[k], rtol=1e-5, atol=1e-6, err_msg=k)

    def test_cap_is_deterministic_and_unbiased(self):
        n, node_type, counts, off, mats, rel_types = toy_graph(1)
        adj = typed_union(mats, rel_types, n)
        targets = np.arange(counts[0])
        path = [0, 1, 0]; steps = [adj[(0, 1)], adj[(1, 0)]]
        exact, a0 = compile_channel(ChannelSpec("ABA", path, steps), targets, node_type, cap=0, cap_seed=0)
        s1, a1 = compile_channel(ChannelSpec("ABA", path, steps), targets, node_type, cap=3, cap_seed=7)
        s2, _ = compile_channel(ChannelSpec("ABA", path, steps), targets, node_type, cap=3, cap_seed=7)
        self.assertEqual((s1 != s2).nnz, 0)
        self.assertGreater(a1["sampled_rows"], 0)
        # a big cap is exact again; a sampled row keeps its mass on the exact support
        big, ab = compile_channel(ChannelSpec("ABA", path, steps), targets, node_type, cap=10**6, cap_seed=7)
        self.assertEqual((big != exact).nnz, 0)
        for i in range(counts[0]):
            sup_e = set(exact[i].indices); sup_s = set(s1[i].indices)
            self.assertTrue(sup_s <= sup_e)
            if len(sup_e):
                self.assertAlmostEqual(float(s1[i].sum()), 1.0, places=5)

    def test_mask_all_ones_equals_stock_forward(self):
        from inv_sehgnn.model import SeHGNN
        torch.manual_seed(0)
        keys = ["M", "MA", "MAM"]
        feats = {k: torch.randn(7, 6) for k in keys}
        m = SeHGNN("IMDB", 8, 8, 3, keys, [], "M", 0.0, 0.0, 0.0, 2, 2, "none", data_size={k: 6 for k in keys}, num_heads=1).eval()
        a = m(None, feats, {}, None)
        b = m(None, feats, {}, torch.ones(3))
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
