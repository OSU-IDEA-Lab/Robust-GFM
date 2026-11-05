"""CPU checks of real Lift construction, metric definitions and report artifacts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT / "script"))
from eval_lp import build_message_graph, load_inputs, main as evaluate_main, query_subset
from lp_data import array_hash, file_hash, write_json
from lp_metrics import (average_ranks, classification_metrics, compare_predictions,
                        point_metrics, summarize_performance, top1_tail)
from report_lp import build_report


def fixture(full=None):
    pool = np.array([10, 11, 12])
    positives = np.array([[0, 10], [1, 11], [2, 12]])
    full = np.array([[3., 2., 1.], [1., 3., 2.], [1., 2., 3.]]) if full is None else full
    candidates = np.array([[p] + list(np.resize(pool[pool != p], 19)) for p in positives[:, 1]])
    return {"scores": np.take_along_axis(full, np.searchsorted(pool, candidates), axis=1),
            "positives": positives, "candidate_pool": pool, "candidates": candidates,
            "query_index": np.arange(3), "full_type_scores": full}


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_sklearn_with_absent_classes_and_ties(self):
        labels = np.array([10, 11, 12, 13])
        y, pred = np.array([10, 10, 11, 12]), np.array([10, 11, 11, 11])
        actual = classification_metrics(y, pred, labels)
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
        self.assertEqual(actual["accuracy"], accuracy_score(y, pred))
        np.testing.assert_allclose([actual[k] for k in ("precision_macro", "recall_macro", "macro_f1")], [p, r, f])
        self.assertEqual(actual["micro_f1"], actual["accuracy"])

    def test_top1_ties_do_not_favor_positive_first(self):
        self.assertEqual(top1_tail(np.ones((1, 3)), np.array([[12, 11, 10]]))[0], 10)
        np.testing.assert_array_equal(average_ranks(np.ones((1, 20))), [10.5])

    def test_all_pair_alignment_and_undefined_tau_convention(self):
        a = fixture()
        same = compare_predictions(a, fixture(), bootstrap=20)
        self.assertEqual(same["metrics"]["tau_b"]["estimate"], 1)
        self.assertTrue(same["bitwise_identical"])
        b = fixture(np.zeros((3, 3)))
        result = compare_predictions(a, b, bootstrap=20)
        self.assertEqual(result["undefined_tau_queries"], 3)
        self.assertEqual(result["metrics"]["tau_b"]["estimate"], 0)
        b["candidates"][0, 0] = 99
        with self.assertRaisesRegex(ValueError, "alignment"):
            compare_predictions(a, b)

    def test_bootstrap_is_reproducible_and_not_seed_variance(self):
        a = fixture(np.array([[3., 2., 1.], [3., 2., 1.], [1., 2., 3.]]))
        first = summarize_performance(a, bootstrap=50, seed=3)
        self.assertEqual(first, summarize_performance(a, bootstrap=50, seed=3))
        self.assertAlmostEqual(first["accuracy"]["estimate"], 2 / 3)
        self.assertGreater(first["accuracy"]["bootstrap_se"], 0)
        self.assertEqual(len(first["accuracy"]["ci95"]), 2)

    def test_query_selection_is_keyed_to_facts_not_file_positions(self):
        pos = np.array([[i, i + 10] for i in range(30)])
        idx = query_subset(pos, 10, 1024, "movie_year")
        reverse = pos[::-1]
        rev_idx = query_subset(reverse, 10, 1024, "movie_year")
        self.assertEqual(set(map(tuple, pos[idx])), set(map(tuple, reverse[rev_idx])))

    def test_real_motif_lift_is_built_per_graph_with_one_inverse_expansion(self):
        import torch
        names = ["movie_to_director", "movie_to_link", "movie_to_year"]
        first = np.array([[0, 2, 0], [0, 1, 1], [0, 3, 2]])
        second = first.copy()
        second[0, 0] = 1
        a = build_message_graph(first, names, 4, torch.device("cpu"))
        b = build_message_graph(second, names, 4, torch.device("cpu"))
        self.assertEqual(a.num_edges, 6)
        self.assertEqual(a.num_relations, 6)
        self.assertIsInstance(a.relation_hypergraph.num_nodes, int)
        self.assertEqual(a.relation_hypergraph.num_nodes, 6)
        self.assertFalse(torch.equal(a.relation_hypergraph.edge_index, b.relation_hypergraph.edge_index))
        np.testing.assert_array_equal(a.edge_type.numpy(), [0, 1, 2, 3, 4, 5])

    def test_export_loader_rejects_corrupt_graph_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "movie_year"
            task.mkdir()
            pool = np.arange(3, 23)
            queries = {"node_type": np.array([0, 0, 0] + [4] * 20)}
            for split, head in zip(("train", "val", "test"), (0, 1, 2)):
                queries[f"{split}_pos"] = np.array([[head, 3]])
                queries[f"{split}_neg"] = pool[None, 1:]
            np.savez(task / "queries.npz", **queries)
            edges = np.array([[0, 3, 0]])
            np.savez(task / "v1.npz", physical=edges, compiled=edges)
            meta = {"files": {p.name: file_hash(p) for p in task.glob("*.npz")},
                    "variants": {"v1": {"edges_sha256": {"physical": array_hash(edges)}}}}
            write_json(task / "manifest.json", meta)
            write_json(root / "manifest.json", {"format": "motif_imdb_lp_v1", "tasks": {"movie_year": file_hash(task / "manifest.json")}})
            args = SimpleNamespace(data=root, dataset="imdb", task="movie_year", variant="v1", split="test", limit=0, seed=1024, view="physical")
            _, actual, pos, candidates, actual_pool, _ = load_inputs(args)
            np.testing.assert_array_equal(actual, edges)
            np.testing.assert_array_equal(actual_pool, pool)
            self.assertEqual(candidates.shape, (1, 20))
            (task / "v1.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inputs(args)

    def test_four_variant_report_has_six_pairs_and_latex_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "movie_year"
            a = fixture()
            contract = {"dataset": "imdb", "task": "movie_year", "checkpoint": "synthetic-test-only",
                        "query_sha256": array_hash(a["positives"], a["candidates"], a["query_index"], a["candidate_pool"])}
            for view in ("physical", "compiled"):
                for variant in ("v1", "v2", "v3", "v4"):
                    directory = task / view / variant
                    directory.mkdir(parents=True)
                    arrays = fixture()
                    meta = {"comparison_contract": contract, "view": view, "variant": variant, "repeat": False,
                            "benchmark_candidates": 20, "full_type_candidates": 3,
                            "graph_sha256": "compiled" if view == "compiled" else variant,
                            **point_metrics(arrays["scores"], arrays["positives"], arrays["candidate_pool"], arrays["full_type_scores"])}
                    np.savez(directory / "scores.npz", **arrays)
                    meta["scores_sha256"] = file_hash(directory / "scores.npz")
                    write_json(directory / "metrics.json", meta)
            import shutil
            repeat = task / "physical/v1__repeat"
            shutil.copytree(task / "physical/v1", repeat)
            meta["view"], meta["variant"], meta["repeat"], meta["graph_sha256"] = "physical", "v1", True, "v1"
            write_json(repeat / "metrics.json", meta)
            summary = build_report(root, bootstrap=20)
            self.assertEqual(len(summary["movie_year"]["physical_pairs"]), 6)
            self.assertEqual(summary["movie_year"]["missing_runs"], [])
            self.assertTrue(summary["movie_year"]["reference_repeat"]["bitwise_identical"])
            self.assertTrue(summary["movie_year"]["compiled_full_type_outputs_identical"])
            tex = (root / "tables/invariance.tex").read_text()
            self.assertIn("\\pm", tex)
            self.assertIn("Kendall $\\tau_b$", tex)
            self.assertIn("v3 vs v4", tex)
            self.assertIn(chr(92) * 2 + "\n", tex)
            self.assertIn("Preliminary frozen zero-shot", tex)
            self.assertIn("20 candidates per query", tex)
            self.assertIn("20 query resamples", tex)
            classification = (root / "tables/classification.tex").read_text()
            self.assertIn("actual calendar year among all 3", classification)
            self.assertIn("Precision and recall are macro-averaged", classification)
            lp = (root / "tables/link_prediction.tex").read_text()
            self.assertIn("19 fixed frequency-matched negative years", lp)
            before = (root / "tables/summary.json").read_bytes()
            build_report(root, bootstrap=20)
            self.assertEqual(before, (root / "tables/summary.json").read_bytes())

    def test_dispatch_uses_separate_processes_and_a_reference_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            families = {"movie_year": ("v1", "v2", "v3", "v4"), "movie_director": ("v1", "v3")}
            write_json(root / "manifest.json", {"tasks": {task: "unused-for-dispatch" for task in families}})
            for task, variants in families.items():
                (root / task).mkdir()
                write_json(root / task / "manifest.json", {"variants": {v: {} for v in variants}})
            with patch.object(sys, "argv", ["eval_lp.py", "--dataset", "imdb", "--data", str(root), "--limit", "64"]), patch("eval_lp.subprocess.run") as calls:
                evaluate_main()
            self.assertEqual(calls.call_count, 14)
            commands = [call.args[0] for call in calls.call_args_list]
            self.assertEqual(sum("--repeat" in command for command in commands), 2)
            self.assertTrue(all("--single" in command for command in commands))
            self.assertTrue(all(command[command.index("--run-tag") + 1] == "smoke64" for command in commands))
            for call in calls.call_args_list:
                self.assertEqual(call.kwargs["env"]["PYTHONHASHSEED"], "0")


if __name__ == "__main__":
    unittest.main()
