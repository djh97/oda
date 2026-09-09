from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation.analyze_evaluation as analysis
import evaluation.compare_model_conditions as comparison
from evaluation.synthetic_dataset import DATASET_DIR
from src.decision_guard import apply_guarded_policy
from src.policy import rank_recipients_baseline
from src.schemas import NoteReviewBatch


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_attempts(path: Path, cases: list[dict], condition: str, model_id: str) -> None:
    _write_jsonl(
        path,
        [
            {
                "event": "attempt_started",
                "case_id": case["case_id"],
                "condition": condition,
                "model_id_requested": model_id,
                "started_at_utc": "2026-09-07T00:00:00Z",
            }
            for case in cases
        ],
    )


def _perfect_record(case: dict, condition: str) -> dict:
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
    guarded = apply_guarded_policy(
        int(case["donor"]["donor_id"]),
        rank_recipients_baseline(case["donor"], case["recipients"]),
        review,
        [int(recipient["recipient_id"]) for recipient in case["recipients"]],
    )
    return {
        "case_id": case["case_id"],
        "condition": condition,
        "model_id_requested": f"model-{condition}",
        "organ_type": case["organ_type"],
        "scenario_family": case["scenario_family"],
        "status": "success",
        "attempt_started_at_utc": "2026-09-07T00:00:00Z",
        "completed_at_utc": "2026-09-07T00:00:01Z",
        "review": review.model_dump(mode="json"),
        "raw_model_output": review.model_dump_json(),
        "guarded_decision": guarded.model_dump(mode="json"),
        "model_run": {
            "model_id": f"model-{condition}",
            "latency_ms": 10,
            "input_tokens": 100,
            "output_tokens": 50,
        },
    }


class AnalysisPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            self.cases = [json.loads(next(handle)) for _ in range(8)]

    def test_complete_oracle_run_produces_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            records_path = root / "oracle_raw.jsonl"
            _write_jsonl(test_path, self.cases)
            _write_jsonl(records_path, [_perfect_record(case, "oracle") for case in self.cases])
            config_path = root / "oracle_config.json"
            attempt_path = root / "oracle_attempts.jsonl"
            _write_attempts(attempt_path, self.cases, "oracle", "model-oracle")
            test_lock = {
                "test_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                "analyzer_sha256": hashlib.sha256(analysis.ANALYSIS_PATH.read_bytes()).hexdigest(),
            }
            (root / "test_lock.json").write_text(json.dumps(test_lock), encoding="utf-8")
            config_path.write_text(
                json.dumps({
                    "condition": "oracle",
                    "model_id_requested": "model-oracle",
                    "test_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                    "raw_file_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                    "test_case_count": len(self.cases),
                    "raw_record_count": len(self.cases),
                    "successful_record_count": len(self.cases),
                    "failed_record_count": 0,
                    "attempt_log_file": str(attempt_path),
                    "attempted_case_count": len(self.cases),
                    "attempt_log_sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                    "evaluation_in_progress": False,
                    "test_lock": test_lock,
                }),
                encoding="utf-8",
            )

            with (
                patch.object(analysis, "TEST_PATH", test_path),
                patch.object(analysis, "BOOTSTRAP_RESAMPLES", 20),
                patch.object(analysis, "validate_test_lock"),
            ):
                summary = analysis.analyze(records_path)
                summary_path = root / "oracle_summary.json"
                recorded_bytes = summary_path.read_bytes()
                recomputed = analysis.analyze(records_path, write_outputs=False)

            self.assertEqual(summary["classification"]["macro_f1"], 1.0)
            self.assertEqual(recorded_bytes, summary_path.read_bytes())
            self.assertEqual(
                {key: value for key, value in summary.items() if key != "generated_at_utc"},
                {key: value for key, value in recomputed.items() if key != "generated_at_utc"},
            )
            self.assertEqual(summary["case_metrics"]["guarded_top1_conformant"]["proportion"], 1.0)
            self.assertEqual(summary["case_metrics"]["guarded_primary_unsafe"]["count"], 0)
            self.assertEqual(summary["case_metrics"]["unsafe_or_missing_primary"]["count"], 0)
            self.assertEqual(summary["classification"]["all_ten_states_exact"]["count"], 8)
            self.assertEqual(set(summary["by_note_style"]), {
                "context_first",
                "delayed_finding",
                "finding_first",
                "interleaved",
            })
            self.assertEqual(set(summary["by_note_complexity"]), {
                "contrast",
                "direct",
                "multi_evidence",
                "multi_evidence_contrast",
            })
            self.assertEqual(
                sum(item["case_count"] for item in summary["by_candidate_pool_difficulty"].values()),
                len(self.cases),
            )
            kidney = summary["by_organ"][self.cases[0]["organ_type"]]
            self.assertIn("count", kidney["guarded_top1_conformance"])
            self.assertIn("denominator", kidney["unsafe_or_missing_primary"])

    def test_paired_comparison_detects_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            left_path = root / "left.jsonl"
            right_path = root / "right.jsonl"
            _write_jsonl(test_path, self.cases)
            _write_jsonl(
                left_path,
                [
                    {
                        "case_id": case["case_id"],
                        "condition": "failed",
                        "model_id_requested": "failed-model",
                        "status": "failure",
                        "attempt_started_at_utc": "2026-09-07T00:00:00Z",
                        "completed_at_utc": "2026-09-07T00:00:01Z",
                        "error_type": "SyntheticFailure",
                    }
                    for case in self.cases
                ],
            )
            _write_jsonl(right_path, [_perfect_record(case, "oracle") for case in self.cases])
            test_lock = {
                "test_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                "comparison_sha256": hashlib.sha256(comparison.COMPARISON_PATH.read_bytes()).hexdigest(),
            }
            (root / "test_lock.json").write_text(json.dumps(test_lock), encoding="utf-8")
            for path, condition, model in (
                (left_path, "failed", "failed-model"),
                (right_path, "oracle", "model-oracle"),
            ):
                attempt_path = path.with_name(path.stem + "_attempts.jsonl")
                _write_attempts(attempt_path, self.cases, condition, model)
                successful_count = len(self.cases) if condition == "oracle" else 0
                path.with_name(path.stem + "_config.json").write_text(
                    json.dumps({
                        "condition": condition,
                        "model_id_requested": model,
                        "test_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                        "raw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "test_case_count": len(self.cases),
                        "raw_record_count": len(self.cases),
                        "successful_record_count": successful_count,
                        "failed_record_count": len(self.cases) - successful_count,
                        "attempt_log_file": str(attempt_path),
                        "attempted_case_count": len(self.cases),
                        "attempt_log_sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                        "evaluation_in_progress": False,
                        "test_lock": test_lock,
                    }),
                    encoding="utf-8",
                )

            with (
                patch.object(comparison, "TEST_PATH", test_path),
                patch.object(comparison, "OUTPUT_DIR", root),
                patch.object(comparison, "BOOTSTRAP_RESAMPLES", 20),
                patch.object(comparison, "validate_test_lock"),
            ):
                summary = comparison.compare(left_path, right_path, "failed", "oracle")
                recomputed = comparison.compare(
                    left_path,
                    right_path,
                    "failed",
                    "oracle",
                    write_output=False,
                )

            self.assertGreater(summary["delta_right_minus_left"]["macro_f1"], 0)
            self.assertEqual(summary["right"]["metrics"]["guarded_top1_conformance"], 1.0)
            self.assertEqual(summary["left"]["metrics"]["unsafe_or_missing_primary_rate"], 1.0)
            self.assertEqual(summary["right"]["metrics"]["unsafe_or_missing_primary_rate"], 0.0)
            self.assertIn("unsafe_or_missing_primary", summary["paired_exact_mcnemar"])
            self.assertEqual(
                {key: value for key, value in summary.items() if key != "generated_at_utc"},
                {key: value for key, value in recomputed.items() if key != "generated_at_utc"},
            )
            with self.assertRaises(RuntimeError):
                comparison.compare(left_path, right_path, "failed", "oracle")
            with self.assertRaises(RuntimeError):
                comparison.compare(left_path, right_path, "wrong-label", "oracle")

    def test_comparison_rejects_tampered_success_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.jsonl"
            records = [_perfect_record(case, "oracle") for case in self.cases]
            records[0]["guarded_decision"]["primary_recipient_id"] = records[0][
                "guarded_decision"
            ]["backup_recipient_id"]
            _write_jsonl(records_path, records)
            with self.assertRaisesRegex(RuntimeError, "Guarded decision does not reproduce"):
                comparison._condition_rows(records_path, self.cases)

            records = [_perfect_record(case, "oracle") for case in self.cases]
            records[0]["model_run"]["model_id"] = "different-model"
            _write_jsonl(records_path, records)
            with self.assertRaisesRegex(RuntimeError, "Observed model differs"):
                comparison._condition_rows(records_path, self.cases)

            record = _perfect_record(self.cases[0], "oracle")
            record["attempt_started_at_utc"] = "2026-09-07T00:00:02Z"
            with self.assertRaisesRegex(RuntimeError, "start time differs"):
                comparison.validate_outcome_attempt_provenance(
                    [record],
                    {
                        str(record["case_id"]): {
                            "case_id": record["case_id"],
                            "condition": "oracle",
                            "model_id_requested": "model-oracle",
                            "started_at_utc": "2026-09-07T00:00:00Z",
                        }
                    },
                    require_complete=True,
                )


if __name__ == "__main__":
    unittest.main()
