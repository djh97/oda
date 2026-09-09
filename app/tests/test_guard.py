from __future__ import annotations

import unittest

from src.decision_guard import GuardAbstention, GuardError, apply_guarded_policy
from src.schemas import NoteReviewBatch


def _ranked() -> list[dict]:
    return [
        {"recipient_id": 1, "selectable": True, "rank": 1, "score": 90},
        {"recipient_id": 2, "selectable": True, "rank": 2, "score": 80},
        {"recipient_id": 3, "selectable": True, "rank": 3, "score": 70},
        {"recipient_id": 4, "selectable": False, "rank": None, "score": 99},
    ]


def _review(states: dict[int, str]) -> NoteReviewBatch:
    codes = {
        "eligible": ["no_current_concern"],
        "review_required": ["specialist_clearance_pending"],
        "temporary_hold": ["active_infection"],
    }
    return NoteReviewBatch.model_validate({
        "case_id": "case-1",
        "assessments": [
            {"recipient_id": recipient_id, "state": state, "evidence_codes": codes[state]}
            for recipient_id, state in states.items()
        ],
    })


class GuardTests(unittest.TestCase):
    def test_guard_preserves_baseline_when_top_is_not_held(self) -> None:
        decision = apply_guarded_policy(
            7,
            _ranked(),
            _review({1: "review_required", 2: "eligible", 3: "eligible", 4: "temporary_hold"}),
            [1, 2, 3, 4],
        )
        self.assertEqual(decision.primary_recipient_id, 1)
        self.assertEqual(decision.backup_recipient_id, 2)
        self.assertFalse(decision.overrode_baseline)
        self.assertFalse(decision.backup_changed)
        self.assertEqual(decision.decision_source, "deterministic_guard")

    def test_guard_skips_only_temporary_holds(self) -> None:
        decision = apply_guarded_policy(
            7,
            _ranked(),
            _review({1: "temporary_hold", 2: "review_required", 3: "eligible", 4: "eligible"}),
            [1, 2, 3, 4],
        )
        self.assertEqual((decision.primary_recipient_id, decision.backup_recipient_id), (2, 3))
        self.assertTrue(decision.overrode_baseline)
        self.assertTrue(decision.backup_changed)
        self.assertIn("[1]", decision.decision_reason)

    def test_guard_records_backup_only_displacement(self) -> None:
        decision = apply_guarded_policy(
            7,
            _ranked(),
            _review({1: "eligible", 2: "temporary_hold", 3: "eligible", 4: "temporary_hold"}),
            [1, 2, 3, 4],
        )
        self.assertEqual((decision.primary_recipient_id, decision.backup_recipient_id), (1, 3))
        self.assertFalse(decision.overrode_baseline)
        self.assertTrue(decision.backup_changed)
        self.assertIn("[2]", decision.decision_reason)
        self.assertNotIn("4", decision.decision_reason)

    def test_guard_fails_on_missing_assessment(self) -> None:
        with self.assertRaises(GuardError):
            apply_guarded_policy(
                7,
                _ranked(),
                _review({1: "eligible", 2: "eligible", 3: "eligible"}),
                [1, 2, 3, 4],
            )
        with self.assertRaisesRegex(GuardError, "unique positive"):
            apply_guarded_policy(
                7,
                _ranked(),
                _review({1: "eligible", 2: "eligible", 3: "eligible", 4: "eligible"}),
                [1, 2, 3, 3],
            )
        duplicate_ranked = [*_ranked(), {"recipient_id": 1, "selectable": True}]
        with self.assertRaisesRegex(GuardError, "baseline order contains duplicate"):
            apply_guarded_policy(
                7,
                duplicate_ranked,
                _review({1: "eligible", 2: "eligible", 3: "eligible", 4: "eligible"}),
                [1, 2, 3, 4],
            )

    def test_guard_abstains_if_fewer_than_two_nonheld_candidates(self) -> None:
        with self.assertRaises(GuardAbstention):
            apply_guarded_policy(
                7,
                _ranked(),
                _review({1: "temporary_hold", 2: "temporary_hold", 3: "eligible", 4: "eligible"}),
                [1, 2, 3, 4],
            )


if __name__ == "__main__":
    unittest.main()
