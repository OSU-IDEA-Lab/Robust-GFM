"""CPU checks of the lab-format binary metrics, tau convention and baseline parsing."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from lp_metrics import binary_metrics, mean_tau_b
from report_seeds import aggregate, parse_baseline

BASELINE = r"""%% ################################ RESULTS ################################
% ---- CMPNN / original / Movie--Year (MY)
IMDb1 & 0.0506 $\pm$ 0.0015 & 0.4492 $\pm$ 0.3210 & 0.0637 $\pm$ 0.0403 & 0.0647 $\pm$ 0.0110 & 0.1595 $\pm$ 0.0120 & 0.1887 $\pm$ 0.0115 \\
& & & IMDb2 & 0.0567 $\pm$ 0.0136 & 0.2286 $\pm$ 0.1861 & 0.0612 $\pm$ 0.0292 & 0.0687 $\pm$ 0.0062 & 0.1635 $\pm$ 0.0083 & 0.1928 $\pm$ 0.0058 \\

% ---- CMPNN / canonical / Movie--Year (MY)
IMDb1 & 9 $\pm$ 9 & 9 $\pm$ 9 & 9 $\pm$ 9 & 9 $\pm$ 9 & 9 $\pm$ 9 & 9 $\pm$ 9 \\

%% ################################ INVARIANCE ################################
% ---- CMPNN / original / Movie--Year (MY)
IMDb1 vs. IMDb2 & 0.2224 $\pm$ 0.3067 & 0.1552 $\pm$ 0.1795 & 0.1695 $\pm$ 0.2491 \\
"""


class SeedReportTests(unittest.TestCase):
    def test_binary_metrics_pool_all_pairs_at_logit_zero(self):
        scores = np.array([[0.5, -1., 2.], [-0.1, 0., -3.]])
        candidates, positives = np.array([[7, 8, 9], [8, 7, 9]]), np.array([[0, 7], [1, 8]])
        labels = (candidates == positives[:, 1:]).ravel()
        predicted = (1 / (1 + np.exp(-scores)) >= 0.5).ravel()
        actual = binary_metrics(scores, candidates, positives)
        self.assertAlmostEqual(actual["precision"], precision_score(labels, predicted))
        self.assertAlmostEqual(actual["recall"], recall_score(labels, predicted))
        with self.assertRaises(ValueError):  # A negative equal to the true tail would be a second positive.
            binary_metrics(scores, np.array([[7, 7, 9], [8, 7, 9]]), positives)

    def test_tau_excludes_undefined_queries_and_counts_identical_as_one(self):
        a = np.array([[1., 2., 3.], [1., 1., 1.], [3., 2., 1.]])
        b = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
        self.assertEqual(mean_tau_b(a, b), (0.0, 1))
        self.assertEqual(mean_tau_b(a[1:2], a[1:2]), (1.0, 0))

    def test_population_std_and_baseline_block_parsing(self):
        self.assertAlmostEqual(aggregate([1., 3.])["std"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "IMDB_lp.tex"
            path.write_text(BASELINE, encoding="utf-8")
            parsed = parse_baseline(path, "CMPNN", "movie_year")
        self.assertEqual(set(parsed["performance"]), {"v1", "v2"})
        self.assertEqual(parsed["performance"]["v2"]["mrr"], {"mean": 0.1928, "std": 0.0058})
        self.assertEqual(parsed["performance"]["v1"]["recall"], {"mean": 0.4492, "std": 0.3210})
        self.assertEqual(parsed["invariance"]["v1 vs v2"]["tau"], {"mean": 0.2224, "std": 0.3067})


if __name__ == "__main__":
    unittest.main()
