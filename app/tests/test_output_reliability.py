from __future__ import annotations

import random
import unittest

from evaluation.analyze_output_reliability import (
    _classification_metrics,
    _draw_stratified_indices,
    _retained_successful_review,
    _stratified_groups,
)


class OutputReliabilityAnalysisTests(unittest.TestCase):
    def test_conditional_metrics_exclude_missing_outputs(self) -> None:
        conditional = _classification_metrics([
            ("eligible", "eligible"),
            ("review_required", "review_required"),
            ("temporary_hold", "temporary_hold"),
        ])
        penalized = _classification_metrics([
            ("eligible", "eligible"),
            ("review_required", "review_required"),
            ("temporary_hold", "temporary_hold"),
            ("eligible", "__failure__"),
            ("review_required", "__failure__"),
            ("temporary_hold", "__failure__"),
        ])
        self.assertEqual(conditional["macro_f1"], 1.0)
        self.assertAlmostEqual(penalized["macro_f1"], 2 / 3, places=6)

    def test_conditional_metrics_are_not_estimable_without_predictions(self) -> None:
        self.assertIsNone(_classification_metrics([]))

    def test_failure_abstention_is_not_counted_as_a_false_positive(self) -> None:
        metrics = _classification_metrics([
            ("eligible", "__failure__"),
            ("review_required", "review_required"),
            ("temporary_hold", "temporary_hold"),
        ])
        self.assertEqual(metrics["by_class"]["eligible"]["fn"], 1)
        self.assertEqual(metrics["by_class"]["review_required"]["fp"], 0)
        self.assertEqual(metrics["by_class"]["temporary_hold"]["fp"], 0)

    def test_failed_pipeline_record_is_excluded_from_end_to_end_predictions(self) -> None:
        case = {"case_id": "case-1", "recipients": [{"recipient_id": 1}]}
        review = {
            "case_id": "case-1",
            "assessments": [
                {
                    "recipient_id": 1,
                    "state": "eligible",
                    "evidence_codes": ["no_current_concern"],
                }
            ],
        }
        self.assertIsNone(
            _retained_successful_review(
                {"status": "failure", "review": review},
                case,
            )
        )
        self.assertIsNotNone(
            _retained_successful_review(
                {"status": "success", "review": review},
                case,
            )
        )

    def test_stratified_resample_preserves_each_design_cell_size(self) -> None:
        cases = [
            {"organ_type": "kidney", "scenario_family": "a"},
            {"organ_type": "kidney", "scenario_family": "a"},
            {"organ_type": "kidney", "scenario_family": "b"},
            {"organ_type": "liver", "scenario_family": "a"},
        ]
        groups = _stratified_groups(cases)
        sampled = _draw_stratified_indices(random.Random(17), groups)
        self.assertEqual(len(sampled), len(cases))
        source_cell_sizes = sorted(len(group) for group in groups)
        sampled_cell_sizes = sorted(
            sum(index in group for index in sampled) for group in groups
        )
        self.assertEqual(sampled_cell_sizes, source_cell_sizes)


if __name__ == "__main__":
    unittest.main()
