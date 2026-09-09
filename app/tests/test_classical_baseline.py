from __future__ import annotations

import unittest

from evaluation.run_classical_baseline import (
    _feature_union,
    _logistic_regression,
    _predict_evidence_sets,
)


class ClassicalBaselineTests(unittest.TestCase):
    def test_pinned_vectorizer_and_classifier_fit_together(self) -> None:
        notes = [
            "active infection remains under treatment",
            "current infection remains active",
            "specialist review is pending",
            "further workup remains pending",
            "no current concern is present",
            "the prior infection has resolved",
        ]
        states = [
            "temporary_hold",
            "temporary_hold",
            "review_required",
            "review_required",
            "eligible",
            "eligible",
        ]
        features = _feature_union()
        matrix = features.fit_transform(notes)
        model = _logistic_regression()
        model.fit(matrix, states)
        self.assertEqual(len(model.predict(matrix)), len(notes))

    def test_evidence_predictions_are_restricted_to_predicted_state(self) -> None:
        classes = [
            "active_infection",
            "no_current_concern",
            "specialist_clearance_pending",
        ]
        predictions = _predict_evidence_sets(
            ["temporary_hold"],
            [[0.8, 0.95, 0.9]],
            classes,
        )
        self.assertEqual(predictions, [["active_infection"]])

    def test_evidence_prediction_uses_best_allowed_code_below_threshold(self) -> None:
        classes = [
            "active_infection",
            "no_current_concern",
            "resolved_condition",
        ]
        predictions = _predict_evidence_sets(
            ["eligible"],
            [[0.99, 0.2, 0.4]],
            classes,
        )
        self.assertEqual(predictions, [["resolved_condition"]])


if __name__ == "__main__":
    unittest.main()
