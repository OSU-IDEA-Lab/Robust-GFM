"""Experimental data-contract checks; no MOTIF or CUDA imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from lp_data import (SOURCE_ROOT, audit_dblp_heldout, audit_heldout, file_hash, forward_edges,
                     validate_full_target_queries, validate_queries, verify_raw_files)


class PreprocessingTests(unittest.TestCase):
    def test_vendored_builders_match_recorded_scr_sources(self):
        manifest = json.loads((ROOT / "tests/PROVENANCE.json").read_text())
        for name, expected in manifest["vendored_files"].items():
            with self.subTest(name=name):
                path = ROOT / "rai-datasets/SHA256SUMS" if name == "data/SHA256SUMS" else SOURCE_ROOT / name
                self.assertEqual(file_hash(path), expected)

    def test_inverse_selection_preserves_empty_relation_slots(self):
        names = ["movie_to_year", "year_to_movie", "movie_to_actor", "actor_to_movie"]
        edges, vocab = forward_edges(np.array([[0, 3], [3, 0]]), np.array([0, 1]), names)
        np.testing.assert_array_equal(edges, [[0, 3, 0]])
        self.assertEqual(vocab, ["movie_to_year", "movie_to_actor"])

    def test_asymmetric_and_duplicate_edges_are_rejected(self):
        names = ["movie_to_year", "year_to_movie"]
        with self.assertRaisesRegex(ValueError, "Reverse edges"):
            forward_edges(np.array([[0], [3]]), np.array([0]), names)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            forward_edges(np.array([[0, 0], [3, 3]]), np.array([0, 0]), names)

    def test_imdb_link_copies_of_heldout_labels_are_rejected(self):
        queries = {"val_pos": np.array([[0, 6]]), "test_pos": np.array([[1, 7]])}
        with self.assertRaisesRegex(ValueError, "link_to_year"):
            audit_heldout(np.array([[3, 6, 1]]), ["movie_to_year", "link_to_year"],
                          queries, "movie_year", 0, 3)

    def test_repeated_imdb_negatives_survive_but_wrong_types_and_split_overlap_fail(self):
        queries = {}
        for split, head in zip(("train", "val", "test"), (0, 1, 2)):
            queries[f"{split}_pos"] = np.array([[head, 3]])
            queries[f"{split}_neg"] = np.full((1, 19), 4)
        types = np.array([0, 0, 0, 4, 4])
        validate_queries(queries, types, 4)
        queries["test_pos"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_queries(queries, types, 4)
        queries["test_pos"][0, 0] = 2
        queries["val_neg"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "target node type"):
            validate_queries(queries, types, 4)

    def test_dblp_full_pool_is_verified_independent_of_negative_storage_order(self):
        pool = np.arange(3, 23)
        types = np.array([1, 1, 1] + [3] * 20)
        queries = {}
        for split, head in zip(("train", "val", "test"), (0, 1, 2)):
            queries[f"{split}_pos"] = np.array([[head, 3]])
            queries[f"{split}_neg"] = np.array([[head, t] for t in pool[1:][::-1]])
        result = validate_full_target_queries(queries, types, pool)
        np.testing.assert_array_equal(result["test_neg"], pool[None, 1:])
        queries["test_neg"][0, 1] = 3
        with self.assertRaisesRegex(ValueError, "full target complement"):
            validate_full_target_queries(queries, types, pool)

    def test_dblp_rejects_heldout_targets_and_paper_specific_area_context(self):
        queries = {"val_pos": np.array([[0, 6]]), "test_pos": np.array([[1, 7]])}
        names = ["paper_to_conference", "paper_to_area"]
        with self.assertRaisesRegex(ValueError, "Held-out label"):
            audit_dblp_heldout(np.array([[0, 6, 0]]), names, queries)
        with self.assertRaisesRegex(ValueError, "Area context"):
            audit_dblp_heldout(np.array([[1, 8, 1]]), names, queries)

    def test_raw_hash_guard_rejects_changed_data_and_accepts_renamed_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "renamed.csv"
            raw.write_text("canonical input\n")
            digest = file_hash(raw)
            (root / "SHA256SUMS").write_text(f"{digest}  raw/IMDB/movie_metadata.csv\n")
            names = {"renamed.csv": "movie_metadata.csv"}
            self.assertEqual(verify_raw_files(root / "SHA256SUMS", root, "IMDB", names, names), {"renamed.csv": digest})
            raw.write_text("changed input\n")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                verify_raw_files(root / "SHA256SUMS", root, "IMDB", names, names)


if __name__ == "__main__":
    unittest.main()
