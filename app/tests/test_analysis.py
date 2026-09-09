from __future__ import annotations

import unittest

from evaluation.analyze_evaluation import _classification_metrics, _mcnemar_exact


class AnalysisTests(unittest.TestCase):
    def test_classification_metrics_include_failures_as_errors(self) -> None:
        metrics = _classification_metrics([
            ("eligible", "eligible"),
            ("review_required", "__failure__"),
            ("temporary_hold", "temporary_hold"),
        ])
        self.assertEqual(metrics["candidate_count"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3, places=6)
        self.assertEqual(metrics["by_class"]["review_required"]["fn"], 1)

    def test_mcnemar_reports_discordant_pairs(self) -> None:
        result = _mcnemar_exact(
            [True, False, False, True, False],
            [True, True, True, False, False],
        )
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["guarded_only_correct"], 2)
        self.assertEqual(result["discordant_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
