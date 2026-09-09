from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation.benchmark_offchain as benchmark_module
from evaluation.benchmark_offchain import (
    _append_jsonl,
    _recover_or_validate_benchmark_completion,
    _reconcile_incomplete_attempts,
    _write_json,
    audit_attempt_provenance,
    _validate_source_run,
    benchmark_source_hashes,
)
from evaluation.paper_full_workflow import DEFAULT_PROTOCOL_PATH, DEMO_CASE_PATH, PROTOCOL, _sha256
from evaluation.project_environment import load_authoritative_project_env


class OffchainBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_id = "ft:test-model"
        self.summary = {
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
            "demo_case_sha256": _sha256(DEMO_CASE_PATH),
            "model_id_requested": self.model_id,
            "model": {"model_id": "ft:observed-model"},
            "implementation_sha256": {
                key: value
                for key, value in benchmark_source_hashes().items()
                if key not in {"benchmark", "local_training_state"}
            },
            "model_evidence": {
                "local_lora_training_sha256": benchmark_source_hashes()[
                    "local_training_state"
                ]
            },
        }

    def test_source_run_must_match_every_frozen_benchmark_input(self) -> None:
        benchmark_settings = PROTOCOL["performance_evaluation"]["offchain_benchmark"]
        with (
            patch.object(
                benchmark_module,
                "assert_test_seed_not_retired",
                side_effect=ValueError("retired test seed"),
            ),
            patch.object(
                benchmark_module,
                "load_authoritative_project_env",
            ) as load_environment,
            self.assertRaisesRegex(ValueError, "retired test seed"),
        ):
            benchmark_module.benchmark(
                Path("unused"),
                measured_runs=int(benchmark_settings["measured_runs"]),
                warmup_runs=int(benchmark_settings["warmup_runs"]),
            )
        load_environment.assert_not_called()

        self.assertEqual(
            _validate_source_run(self.summary, self.model_id),
            "ft:observed-model",
        )

        replacements = {
            "protocol_id": "different-protocol",
            "protocol_sha256": "0" * 64,
            "demo_case_sha256": "1" * 64,
            "model_id_requested": "ft:different-model",
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                changed = dict(self.summary)
                changed[field] = replacement
                with self.assertRaises(RuntimeError):
                    _validate_source_run(changed, self.model_id)

        missing_observed_model = dict(self.summary)
        missing_observed_model["model"] = {}
        with self.assertRaises(RuntimeError):
            _validate_source_run(missing_observed_model, self.model_id)

        changed_implementation = dict(self.summary)
        changed_implementation["implementation_sha256"] = {
            **self.summary["implementation_sha256"],
            "policy": "0" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "implementation differs"):
            _validate_source_run(changed_implementation, self.model_id)

        changed_training_state = dict(self.summary)
        changed_training_state["model_evidence"] = {
            **self.summary["model_evidence"],
            "local_lora_training_sha256": "2" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "training state differs"):
            _validate_source_run(changed_training_state, self.model_id)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            _append_jsonl(
                output_dir / "attempts.jsonl",
                {
                    "attempt_id": "measured-001",
                    "event": "started",
                    "phase": "measured",
                    "recorded_at_utc": "2026-09-07T00:00:00Z",
                    "run_number": 1,
                },
            )
            audit = _reconcile_incomplete_attempts(
                output_dir,
                warmup_runs=0,
                measured_runs=1,
            )
            self.assertEqual(audit["records"][0]["status"], "failure")
            self.assertEqual(audit["records"][0]["error_type"], "InterruptedAttempt")
            audit_attempt_provenance(
                output_dir,
                warmup_runs=0,
                measured_runs=1,
                require_complete=True,
            )

            _append_jsonl(
                output_dir / "attempts.jsonl",
                {
                    "attempt_id": "measured-001",
                    "event": "started",
                    "phase": "measured",
                    "recorded_at_utc": "2026-09-07T00:01:00Z",
                    "run_number": 1,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate started"):
                audit_attempt_provenance(
                    output_dir,
                    warmup_runs=0,
                    measured_runs=1,
                    require_complete=True,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / "study.env"
            env_path.write_text("OPENAI_MODEL_ID=file-model\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ODA_DISABLE_DOTENV": "0",
                    "OPENAI_MODEL_ID": "shell-model",
                    "SEPOLIA_RPC_URL": "shell-rpc",
                },
                clear=False,
            ):
                load_authoritative_project_env(env_path)
                self.assertEqual(os.environ["OPENAI_MODEL_ID"], "file-model")
                self.assertNotIn("SEPOLIA_RPC_URL", os.environ)

    def test_attempt_ledger_rejects_out_of_order_or_invalid_time_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            _append_jsonl(
                output_dir / "attempts.jsonl",
                {
                    "attempt_id": "measured-002",
                    "event": "started",
                    "phase": "measured",
                    "recorded_at_utc": "2026-09-07T00:00:00Z",
                    "run_number": 2,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "prespecified attempt order"):
                audit_attempt_provenance(
                    output_dir,
                    warmup_runs=0,
                    measured_runs=2,
                    require_complete=False,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            _append_jsonl(
                output_dir / "attempts.jsonl",
                {
                    "attempt_id": "measured-001",
                    "event": "started",
                    "phase": "measured",
                    "recorded_at_utc": "not-a-time",
                    "run_number": 1,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "valid UTC timestamp"):
                audit_attempt_provenance(
                    output_dir,
                    warmup_runs=0,
                    measured_runs=1,
                    require_complete=False,
                )

    def test_finalizing_state_recovers_one_canonical_completion_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "run-1"
            output_dir.mkdir()
            state_path = root / "run_state.json"
            state = {"status": "finalizing", "run_directory": output_dir.name}
            _write_json(state_path, state)
            pointer = {
                "completed_at_utc": "2026-09-07T00:00:00Z",
                "run_directory": output_dir.name,
                "summary_sha256": "a" * 64,
                "attempt_ledger_sha256": "b" * 64,
                "raw_runs_sha256": "c" * 64,
            }

            with patch(
                "evaluation.benchmark_offchain._expected_completion_pointer",
                return_value=pointer,
            ):
                completed = _recover_or_validate_benchmark_completion(
                    root,
                    output_dir,
                    state_path,
                    state,
                )

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(
                json.loads((root / "completed_run.json").read_text(encoding="utf-8")),
                pointer,
            )
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["status"],
                "completed",
            )


if __name__ == "__main__":
    unittest.main()
