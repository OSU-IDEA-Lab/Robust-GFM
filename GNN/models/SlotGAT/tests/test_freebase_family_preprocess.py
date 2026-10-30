"""preprocess_family: one semantic program over {unchanged, exact_2, exact_3, union}."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freebase_block_conditional import FreebaseBlockContextPlan, preprocess_family  # noqa: E402


def _write_variant(directory: Path, info: str, edges, node_lines, labels):
    directory.mkdir()
    (directory / "node.dat").write_text("".join(node_lines))
    (directory / "label.dat").write_text(labels)
    (directory / "info.dat").write_text(info)
    (directory / "link.dat").write_text("".join(f"{u}\t{v}\t{r}\n" for u, v, r in edges))


class FamilyPreprocessTests(unittest.TestCase):
    def test_family_shares_one_program_and_declares_shortcuts_per_variant(self):
        rng = np.random.RandomState(0)
        nA, nB, nC = 3, 4, 4
        offB, offC = nA, nA + nB
        # base relations: 0 A-r-B, 1 B-s-C, 2 C-t-B
        r = sp.csr_matrix((rng.rand(nA, nB) > 0.4).astype(np.int64))
        s_ = sp.csr_matrix((rng.rand(nB, nC) > 0.4).astype(np.int64))
        t = sp.csr_matrix((rng.rand(nC, nB) > 0.5).astype(np.int64))
        rs = ((r @ s_) > 0).astype(np.int64)            # A -> C, 2-hop
        rst = ((rs @ t) > 0).astype(np.int64)           # A -> B, 3-hop
        self.assertGreater(rs.nnz, 0); self.assertGreater(rst.nnz, 0)

        def pairs(mat, off_src, off_dst, rel):
            coo = sp.coo_matrix(mat)
            return [(int(u) + off_src, int(v) + off_dst, rel) for u, v in zip(coo.row, coo.col)]

        base = pairs(r, 0, offB, 0) + pairs(s_, offB, offC, 1) + pairs(t, offC, offB, 2)
        two = pairs(rs, 0, offC, 3)
        three = pairs(rst, 0, offB, 4)
        node_lines = [f"{i}\tn{i}\t{ty}\n" for i, ty in enumerate([0] * nA + [1] * nB + [2] * nC)]
        labels = "0\tn0\t0\t0\n1\tn1\t0\t1\n2\tn2\t0\t2\n"
        base_info = ("node.dat\nTYPE\tMEANING\n0\tA\n1\tB\n2\tC\nlink.dat\nLINK\tSTART\tEND\tMEANING\n"
                     "0\t0\t1\tA-r-B\n1\t1\t2\tB-s-C\n2\t2\t1\tC-t-B\nlabel.dat\nTYPE\tCLASS\tMEANING\n")
        two_line = "3\t0\t2\tA-r-B-s-C\n"; three_line = "4\t0\t1\tA-r-B-s-C-t-B\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_variant(root / "unchanged", base_info, base, node_lines, labels)
            _write_variant(root / "exact_2", base_info.replace("label.dat\n", two_line + "label.dat\n"), base + two, node_lines, labels)
            _write_variant(root / "exact_3", base_info.replace("label.dat\n", three_line + "label.dat\n"), base + three, node_lines, labels)
            _write_variant(root / "union", base_info.replace("label.dat\n", two_line + three_line + "label.dat\n"), base + two + three, node_lines, labels)
            reg2 = root / "exact_2.jsonl"; reg3 = root / "exact_3.jsonl"
            reg2.write_text(json.dumps({"new_relation_id": 3, "new_relation_name": "A-r-B-s-C", "composition": "A-r-B ; B-s-C",
                                        "hop_count": 2, "start_type": "A", "end_type": "C", "node_type_path": "A ; B ; C",
                                        "num_new_edges": int(rs.nnz), "unique_endpoint_pairs": int(rs.nnz), "skipped_reason": ""}) + "\n")
            reg3.write_text(json.dumps({"new_relation_id": 3, "new_relation_name": "A-r-B-s-C-t-B", "composition": "A-r-B ; B-s-C ; C-t-B",
                                        "hop_count": 3, "start_type": "A", "end_type": "B", "node_type_path": "A ; B ; C ; B",
                                        "num_new_edges": int(rst.nnz), "unique_endpoint_pairs": int(rst.nnz), "skipped_reason": ""}) + "\n")
            out = root / "out"
            audit = preprocess_family(
                unchanged_dir=root / "unchanged",
                changed_dirs={"exact_2": root / "exact_2", "exact_3": root / "exact_3", "union_exact_2_3": root / "union"},
                registry_paths=[reg2, reg3], output_root=out,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["family"], ["unchanged", "exact_2", "exact_3", "union_exact_2_3"])
            plans = {v: FreebaseBlockContextPlan.load(out / v) for v in audit["family"]}
            self.assertEqual(len({p.semantic_program_hash for p in plans.values()}), 1)
            self.assertEqual(len({p.context_plan_hash for p in plans.values()}), 4)
            self.assertEqual(len({p.raw_graph_hash for p in plans.values()}), 4)
            self.assertEqual(plans["unchanged"].physical_shortcut_names, None)
            self.assertEqual(plans["exact_2"].physical_shortcut_names, ("A-r-B-s-C",))
            self.assertEqual(plans["exact_3"].physical_shortcut_names, ("A-r-B-s-C-t-B",))
            self.assertEqual(plans["union_exact_2_3"].physical_shortcut_names, ("A-r-B-s-C", "A-r-B-s-C-t-B"))
            for p in plans.values():
                p.verify_runtime_context()
                self.assertEqual(p.num_channels, 2 * 3 + 2 * 2 + 1)
            # the 3-hop mapping walks three channels forward
            three_hop = next(m for m in plans["exact_3"].block_relations if m.name == "A-r-B-s-C-t-B")
            self.assertEqual(len(three_hop.forward_steps), 3)
            self.assertEqual(three_hop.endpoint_count, int(rst.nnz))
            # a variant that carries a 2-hop shortcut it does not declare must be rejected
            plans["exact_3"].physical_shortcut_fingerprints["A-r-B-s-C"] = {"count": 1, "sum64": 0, "xor64": 0}
            with self.assertRaises(ValueError):
                plans["exact_3"].verify_runtime_context()


if __name__ == "__main__":
    unittest.main()
