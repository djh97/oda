from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation.run_model_evaluation as runner
from evaluation.synthetic_dataset import DATASET_DIR
from src.llm_client import NoteReviewResult
from src.schemas import ModelRunMetadata, NoteReviewBatch


class EvaluationLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            cls.case = json.loads(next(handle))

    def setUp(self) -> None:
        complete = {
            name: "a" * 64
            for name in (
                "generation_manifest",
                "training_cases",
                "validation_cases",
                "fine_tuning_training",
                "fine_tuning_validation",
                "fine_tuning_access_check",
                "fine_tuning_job",
                "classical_validation_diagnostics",
                "untuned_smoke_test",
                "fine_tuned_smoke_test",
            )
        }
        patcher = patch.object(runner, "_pretest_artifact_hashes", return_value=complete)
        patcher.start()
        self.addCleanup(patcher.stop)
        terminal_patcher = patch.object(runner, "_validate_pretest_terminal_records")
        terminal_patcher.start()
        self.addCleanup(terminal_patcher.stop)
        condition_patcher = patch.object(runner, "_validate_condition_model")
        condition_patcher.start()
        self.addCleanup(condition_patcher.stop)
        seed_patcher = patch.object(
            runner,
            "assert_test_seed_not_retired",
            return_value=2026090791,
        )
        seed_patcher.start()
        self.addCleanup(seed_patcher.stop)

    def _fake_result(self) -> NoteReviewResult:
        review_value = {
            "case_id": self.case["case_id"],
            "assessments": [
                {
                    "recipient_id": recipient["recipient_id"],
                    "state": recipient["reference_note_state"],
                    "evidence_codes": recipient["reference_evidence_codes"],
                }
                for recipient in self.case["recipients"]
            ],
        }
        review = NoteReviewBatch.model_validate(review_value)
        return NoteReviewResult(
            review=review,
            metadata=ModelRunMetadata(model_id="test-model", latency_ms=1.0),
            raw_text=review.model_dump_json(),
        )

    def test_lock_is_created_immediately_before_first_model_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"

            def fake_call(**_: object) -> NoteReviewResult:
                self.assertTrue(lock_path.exists())
                return self._fake_result()

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", side_effect=fake_call),
            ):
                output = runner.evaluate(
                    "test-model",
                    "test-condition",
                    "test-key",
                    confirm_test_lock=True,
                )
                locked = json.loads(lock_path.read_text(encoding="utf-8"))
                runner.validate_test_lock(locked)
                tampered = {**locked, "requirements_lock_sha256": "0" * 64}
                with self.assertRaisesRegex(RuntimeError, "requirements_lock_sha256"):
                    runner.validate_test_lock(tampered)

            self.assertTrue(lock_path.exists())
            config = json.loads((root / "test_condition_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["test_lock"]["test_file_sha256"], runner._sha256(test_path))
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"
            config_path = root / "test_condition_config.json"
            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", return_value=self._fake_result()),
            ):
                lock_path.write_text(
                    json.dumps({**runner._test_lock_value(), "locked_at_utc": "2026-09-07T00:00:00Z"}),
                    encoding="utf-8",
                )
                config_path.write_text(
                    json.dumps({
                        "condition": "test-condition",
                        "model_id_requested": "test-model",
                        "test_file_sha256": runner._sha256(test_path),
                        "evaluation_in_progress": True,
                    }),
                    encoding="utf-8",
                )
                runner.evaluate("test-model", "test-condition", "test-key")

            recovered = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(recovered["test_lock"]["test_file_sha256"], runner._sha256(test_path))

    def test_preflight_config_error_does_not_create_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"
            (root / "test_condition_config.json").write_text(
                json.dumps({
                    "condition": "test-condition",
                    "model_id_requested": "different-model",
                    "test_file_sha256": runner._sha256(test_path),
                }),
                encoding="utf-8",
            )

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
            ):
                with self.assertRaisesRegex(RuntimeError, "different condition, model, or test file"):
                    runner.evaluate("test-model", "test-condition", "test-key")

            self.assertFalse(lock_path.exists())

    def test_first_test_run_requires_explicit_lock_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(
                    runner,
                    "validate_existing",
                    return_value={"valid": True},
                ) as validate_dataset,
                patch.object(runner, "_load_jsonl", wraps=runner._load_jsonl) as load_jsonl,
            ):
                with self.assertRaisesRegex(RuntimeError, "explicit authorization"):
                    runner.evaluate("test-model", "test-condition", "test-key")

            self.assertFalse(lock_path.exists())
            self.assertFalse((root / "test_condition_config.json").exists())
            validate_dataset.assert_not_called()
            load_jsonl.assert_not_called()

    def test_lock_creation_refuses_missing_pretest_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"
            with (
                patch.object(
                    runner,
                    "assert_test_seed_not_retired",
                    side_effect=ValueError("retired test seed"),
                ),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                self.assertRaisesRegex(ValueError, "retired test seed"),
            ):
                runner._enforce_test_lock(create=True)
            self.assertFalse(lock_path.exists())

            missing = dict(runner._pretest_artifact_hashes())
            missing["fine_tuned_smoke_test"] = None
            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "_pretest_artifact_hashes", return_value=missing),
                self.assertRaisesRegex(RuntimeError, "fine_tuned_smoke_test"),
            ):
                runner._enforce_test_lock(create=True)
            self.assertFalse(lock_path.exists())

    def test_failed_attempt_remains_final_when_run_is_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", side_effect=RuntimeError("simulated failure")) as call,
            ):
                output = runner.evaluate(
                    "test-model",
                    "test-condition",
                    "test-key",
                    confirm_test_lock=True,
                )
                runner.evaluate("test-model", "test-condition", "test-key")

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(call.call_count, 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", side_effect=KeyboardInterrupt()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.evaluate(
                        "test-model",
                        "test-condition",
                        "test-key",
                        confirm_test_lock=True,
                    )

            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review") as resumed_call,
            ):
                output = runner.evaluate("test-model", "test-condition", "test-key")

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(resumed_call.call_count, 0)
            self.assertEqual(records[0]["error_type"], "InterruptedAttempt")
            config = json.loads((root / "test_condition_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["evaluation_in_progress"])
            self.assertEqual(config["attempted_case_count"], 1)

    def test_provider_model_mismatch_is_retained_as_a_failed_first_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"
            result = self._fake_result()
            mismatched = NoteReviewResult(
                review=result.review,
                metadata=ModelRunMetadata(model_id="different-model", latency_ms=1.0),
                raw_text=result.raw_text,
            )
            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", return_value=mismatched),
            ):
                output = runner.evaluate(
                    "test-model",
                    "test-condition",
                    "test-key",
                    confirm_test_lock=True,
                )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failure")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertIn("different model identifier", record["error"])

    def test_model_failure_retains_available_raw_response_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            test_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
            lock_path = root / "test_lock.json"
            error = RuntimeError("invalid schema")
            error.raw_text = "{invalid-json"
            error.model_run_metadata = ModelRunMetadata(
                model_id="test-model", latency_ms=3.0
            )
            with (
                patch.object(runner, "TEST_PATH", test_path),
                patch.object(runner, "OUTPUT_DIR", root),
                patch.object(runner, "TEST_LOCK_PATH", lock_path),
                patch.object(runner, "validate_existing", return_value={"valid": True}),
                patch.object(runner, "call_note_review", side_effect=error),
            ):
                output = runner.evaluate(
                    "test-model",
                    "test-condition",
                    "test-key",
                    confirm_test_lock=True,
                )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failure")
            self.assertEqual(record["raw_model_output"], "{invalid-json")
            self.assertEqual(record["model_run"]["latency_ms"], 3.0)

    def test_attempt_ledger_must_follow_locked_case_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            runner._append_jsonl(
                path,
                {
                    "event": "attempt_started",
                    "case_id": "case-2",
                    "condition": "condition",
                    "model_id_requested": "model",
                    "started_at_utc": "2026-09-07T00:00:00Z",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "locked test order"):
                runner._attempted_case_ids(
                    path,
                    condition="condition",
                    model_id="model",
                    expected_order=["case-1", "case-2"],
                )


class EvaluationConditionTests(unittest.TestCase):
    def test_untuned_condition_requires_the_pinned_base_model(self) -> None:
        base_model = runner.load_protocol()["model_evaluation"]["base_model_snapshot"]
        runner._validate_condition_model("untuned", base_model)
        with self.assertRaisesRegex(RuntimeError, "preserved expected model"):
            runner._validate_condition_model("untuned", "another-model")

    def test_fine_tuned_condition_requires_verified_local_adapter(self) -> None:
        state = {"adapter": {"model_id": "local:test:model"}}
        with patch.object(runner, "validate_local_training_state", return_value=state):
            runner._validate_condition_model("fine_tuned", "local:test:model")
            with self.assertRaisesRegex(RuntimeError, "preserved expected model"):
                runner._validate_condition_model("fine_tuned", "local:other:model")

        with self.assertRaisesRegex(ValueError, "condition must be"):
            runner._validate_condition_model("exploratory", "model")


if __name__ == "__main__":
    unittest.main()
