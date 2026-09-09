from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation.generate_synthetic_data as generator_cli
from evaluation.manage_fine_tuning import validate_development_data
from evaluation.synthetic_dataset import (
    DATASET_DIR,
    PROTOCOL,
    _note_component_pool,
    generate_case,
)
from src.schemas import NoteReviewBatch


class DatasetTests(unittest.TestCase):
    def test_development_dataset_passes_all_invariants(self) -> None:
        report = validate_development_data()
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["errors"], [])
        expected_states = set(PROTOCOL["note_taxonomy"]["states"])
        for summary in report["split_summaries"].values():
            for counts in summary["readiness_by_payload_position"].values():
                self.assertTrue(expected_states.issubset(counts))
            for counts in summary["readiness_by_recipient_id_suffix"].values():
                self.assertTrue(expected_states.issubset(counts))

    def test_case_generation_is_deterministic(self) -> None:
        left = generate_case("validation", 2026090702, 11, "kidney", "top1_temporary_hold")
        right = generate_case("validation", 2026090702, 11, "kidney", "top1_temporary_hold")
        self.assertEqual(left, right)

        multiple = generate_case(
            "validation",
            2026090702,
            12,
            "kidney",
            "top2_temporary_hold",
        )
        baseline = multiple["reference"]["baseline_order"]
        by_id = {item["recipient_id"]: item for item in multiple["recipients"]}
        self.assertEqual(
            [by_id[recipient_id]["reference_note_state"] for recipient_id in baseline[:2]],
            ["temporary_hold", "temporary_hold"],
        )
        self.assertEqual(multiple["reference"]["primary_recipient_id"], baseline[2])

    def test_fine_tuning_assistant_uses_runtime_schema(self) -> None:
        path = DATASET_DIR / "fine_tuning_training.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(next(handle)) for _ in range(10)]
        for row in rows:
            self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
            NoteReviewBatch.model_validate_json(row["messages"][2]["content"])
            user_payload = json.loads(row["messages"][1]["content"])
            self.assertNotIn("reference_note_state", json.dumps(user_payload))

    def test_split_counts_match_protocol(self) -> None:
        for split, settings in PROTOCOL["dataset"]["splits"].items():
            path = DATASET_DIR / f"{split}_cases.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), int(settings["cases"]))

    def test_note_component_pools_are_disjoint_across_study_splits(self) -> None:
        splits = tuple(PROTOCOL["dataset"]["splits"])
        for index, left in enumerate(splits):
            for right in splits[index + 1:]:
                self.assertFalse(_note_component_pool(left).intersection(_note_component_pool(right)))

    def test_generation_is_blocked_after_fine_tuning_or_test_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine_tuning_state = root / "fine_tuning_job.json"
            test_lock = root / "test_lock.json"
            with (
                patch.object(generator_cli, "FINE_TUNING_STATE_PATH", fine_tuning_state),
                patch.object(generator_cli, "TEST_LOCK_PATH", test_lock),
                patch.object(
                    generator_cli,
                    "assert_test_seed_not_retired",
                    side_effect=ValueError("retired test seed"),
                ),
                self.assertRaisesRegex(ValueError, "retired test seed"),
            ):
                generator_cli.assert_generation_is_not_frozen()

            with (
                patch.object(generator_cli, "FINE_TUNING_STATE_PATH", fine_tuning_state),
                patch.object(generator_cli, "TEST_LOCK_PATH", test_lock),
                patch.object(
                    generator_cli,
                    "assert_test_seed_not_retired",
                    return_value=2026090791,
                ),
            ):
                generator_cli.assert_generation_is_not_frozen()
                fine_tuning_state.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "generation is frozen"):
                    generator_cli.assert_generation_is_not_frozen()
                fine_tuning_state.unlink()
                test_lock.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "generation is frozen"):
                    generator_cli.assert_generation_is_not_frozen()


if __name__ == "__main__":
    unittest.main()
