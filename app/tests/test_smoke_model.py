from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import evaluation.smoke_test_model as smoke
from evaluation.synthetic_dataset import DATASET_DIR
from src.llm_client import NoteReviewResult
from src.schemas import ModelRunMetadata, NoteReviewBatch


class ModelSmokeTests(unittest.TestCase):
    def test_smoke_uses_validation_case_and_writes_no_test_lock(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        review = NoteReviewBatch.model_validate({
            "case_id": case["case_id"],
            "assessments": [
                {
                    "recipient_id": recipient["recipient_id"],
                    "state": recipient["reference_note_state"],
                    "evidence_codes": recipient["reference_evidence_codes"],
                }
                for recipient in case["recipients"]
            ],
        })

        def fake_caller(**kwargs: object) -> NoteReviewResult:
            self.assertEqual(kwargs["case_id"], case["case_id"])
            return NoteReviewResult(
                review=review,
                metadata=ModelRunMetadata(
                    model_id=smoke.MODEL_SETTINGS["base_model_snapshot"],
                    latency_ms=1.0,
                ),
                raw_text=review.model_dump_json(),
            )

        blocked_caller = MagicMock()
        with (
            patch.object(
                smoke,
                "assert_test_seed_not_retired",
                side_effect=ValueError("retired test seed"),
            ),
            self.assertRaisesRegex(ValueError, "retired test seed"),
        ):
            smoke.run_smoke_test(
                smoke.MODEL_SETTINGS["base_model_snapshot"],
                "untuned_preflight",
                "unused-key",
                caller=blocked_caller,
            )
        blocked_caller.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch.object(smoke, "OUTPUT_DIR", output_dir),
                patch.object(smoke, "validate_development_data", return_value={"valid": True}),
                patch.object(smoke, "assert_test_seed_not_retired", return_value=2026090791),
            ):
                output = smoke.run_smoke_test(
                    smoke.MODEL_SETTINGS["base_model_snapshot"],
                    "untuned_preflight",
                    "unused-key",
                    caller=fake_caller,
                )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(record["state_correct_count"], 10)
        self.assertEqual(record["evidence_exact_count"], 10)
        self.assertIn("not a test-set outcome", record["purpose"])
        self.assertIn("smoke_test", record["implementation_sha256"])
        self.assertEqual(record["status"], "success")

    def test_smoke_retains_raw_output_from_schema_failure(self) -> None:
        error = RuntimeError("invalid schema")
        error.raw_text = "{invalid-json"
        error.model_run_metadata = ModelRunMetadata(
            model_id=smoke.MODEL_SETTINGS["base_model_snapshot"],
            latency_ms=2.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch.object(smoke, "OUTPUT_DIR", output_dir),
                patch.object(smoke, "validate_development_data", return_value={"valid": True}),
                patch.object(smoke, "assert_test_seed_not_retired", return_value=2026090791),
            ):
                output = smoke.run_smoke_test(
                    smoke.MODEL_SETTINGS["base_model_snapshot"],
                    "untuned_preflight",
                    "unused-key",
                    caller=MagicMock(side_effect=error),
                )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "failure")
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["raw_model_output"], "{invalid-json")
        self.assertEqual(record["model_run"]["latency_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
