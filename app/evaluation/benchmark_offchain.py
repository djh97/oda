"""Benchmark the recurring encrypted-record, ranking, model, and guard path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluation.artifact_paths import portable_path
from evaluation.paper_full_workflow import (
    APP_DIR,
    CURRENT_OUTPUT_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEMO_CASE_PATH,
    GUARD_PATH,
    IPFS_CLIENT_PATH,
    LLM_CLIENT_PATH,
    LOCAL_LLM_CLIENT_PATH,
    LOCAL_TRAINING_STATE_PATH,
    POLICY_PATH,
    PROTOCOL,
    PROJECT_ENVIRONMENT_PATH,
    SCHEMAS_PATH,
    SECURE_STORAGE_PATH,
    _read_json,
    _required_env,
    _runtime_profile,
    _sha256,
)
from evaluation.project_environment import load_authoritative_project_env
from src.decision_guard import apply_guarded_policy
from src.ipfs_client import fetch_encrypted_json_batch, fetch_encrypted_json_from_ipfs
from src.llm_client import SYSTEM_PROMPT
from src.local_llm_client import LocalNoteReviewClient, validate_local_training_state
from src.policy import assert_protocol_frozen, assert_test_seed_not_retired, rank_recipients_baseline
from src.secure_storage import canonical_json_sha256


SCRIPT_PATH = Path(__file__).resolve()
ARTIFACT_PATHS_PATH = APP_DIR / "evaluation" / "artifact_paths.py"
BENCHMARK_SOURCE_PATHS = {
    "benchmark": SCRIPT_PATH,
    "llm_client": LLM_CLIENT_PATH,
    "local_llm_client": LOCAL_LLM_CLIENT_PATH,
    "local_training_state": LOCAL_TRAINING_STATE_PATH,
    "policy": POLICY_PATH,
    "decision_guard": GUARD_PATH,
    "schemas": SCHEMAS_PATH,
    "secure_storage": SECURE_STORAGE_PATH,
    "ipfs_client": IPFS_CLIENT_PATH,
    "project_environment": PROJECT_ENVIRONMENT_PATH,
    "artifact_paths": ARTIFACT_PATHS_PATH,
}

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: object, *, label: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a valid UTC timestamp") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    values: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path.name} at line {line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"{path.name} line {line_number} is not a JSON object")
            values.append(value)
    return values


def _record_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_source_hashes() -> Dict[str, str]:
    return {name: _sha256(path) for name, path in BENCHMARK_SOURCE_PATHS.items()}


def benchmark_software_environment() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "pydantic",
                "cryptography",
                "requests",
            )
        },
    }


def _attempt_specs(warmup_runs: int, measured_runs: int) -> list[Dict[str, Any]]:
    return [
        *(
            {
                "attempt_id": f"warmup-{run_number:03d}",
                "phase": "warmup",
                "run_number": run_number,
            }
            for run_number in range(1, warmup_runs + 1)
        ),
        *(
            {
                "attempt_id": f"measured-{run_number:03d}",
                "phase": "measured",
                "run_number": run_number,
            }
            for run_number in range(1, measured_runs + 1)
        ),
    ]


def _attempt_identity(value: Mapping[str, Any]) -> tuple[str, str, int]:
    attempt_id = str(value.get("attempt_id", "")).strip()
    phase = str(value.get("phase", "")).strip()
    run_number = value.get("run_number")
    if phase not in {"warmup", "measured"}:
        raise RuntimeError(f"Invalid benchmark phase for attempt {attempt_id or '<missing>'}")
    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number < 1:
        raise RuntimeError(f"Invalid run number for benchmark attempt {attempt_id or '<missing>'}")
    expected_id = f"{phase}-{run_number:03d}"
    if attempt_id != expected_id:
        raise RuntimeError(f"Benchmark attempt identifier {attempt_id!r} should be {expected_id!r}")
    return attempt_id, phase, run_number


def audit_attempt_provenance(
    output_dir: Path,
    *,
    warmup_runs: int,
    measured_runs: int,
    require_complete: bool,
) -> Dict[str, Any]:
    """Validate the append-only first-attempt ledger and its outcome records."""
    expected_specs = _attempt_specs(warmup_runs, measured_runs)
    expected = {str(item["attempt_id"]): item for item in expected_specs}
    ledger_path = output_dir / "attempts.jsonl"
    raw_path = output_dir / "runs.jsonl"
    events = _read_jsonl(ledger_path)
    records = _read_jsonl(raw_path)
    starts: Dict[str, Dict[str, Any]] = {}
    finishes: Dict[str, Dict[str, Any]] = {}
    event_positions: Dict[tuple[str, str], int] = {}
    event_order: list[tuple[str, str]] = []
    previous_event_time: datetime | None = None

    for position, event in enumerate(events):
        attempt_id, phase, run_number = _attempt_identity(event)
        if attempt_id not in expected:
            raise RuntimeError(f"Unexpected benchmark attempt {attempt_id}")
        event_type = str(event.get("event", "")).strip()
        if event_type not in {"started", "finished"}:
            raise RuntimeError(f"Invalid ledger event for benchmark attempt {attempt_id}")
        target = starts if event_type == "started" else finishes
        if attempt_id in target:
            raise RuntimeError(f"Duplicate {event_type} event for benchmark attempt {attempt_id}")
        event_time = _parse_utc(
            event.get("recorded_at_utc"),
            label=f"Benchmark attempt {attempt_id} ledger timestamp",
        )
        if previous_event_time is not None and event_time < previous_event_time:
            raise RuntimeError("Benchmark ledger timestamps are not chronological")
        previous_event_time = event_time
        target[attempt_id] = dict(event)
        event_positions[(attempt_id, event_type)] = position
        event_order.append((attempt_id, event_type))
        if phase != expected[attempt_id]["phase"] or run_number != expected[attempt_id]["run_number"]:
            raise RuntimeError(f"Benchmark attempt {attempt_id} has inconsistent identity fields")

    expected_event_order = [
        (str(spec["attempt_id"]), event_type)
        for spec in expected_specs
        for event_type in ("started", "finished")
    ]
    if event_order != expected_event_order[: len(event_order)]:
        raise RuntimeError("Benchmark ledger events do not follow the prespecified attempt order")

    record_map: Dict[str, Dict[str, Any]] = {}
    record_order: list[str] = []
    for record in records:
        attempt_id, phase, run_number = _attempt_identity(record)
        if attempt_id not in expected:
            raise RuntimeError(f"Unexpected benchmark outcome {attempt_id}")
        if attempt_id in record_map:
            raise RuntimeError(f"Duplicate outcome for benchmark attempt {attempt_id}")
        if phase != expected[attempt_id]["phase"] or run_number != expected[attempt_id]["run_number"]:
            raise RuntimeError(f"Benchmark outcome {attempt_id} has inconsistent identity fields")
        status = str(record.get("status", "")).strip()
        if status not in {"success", "failure"}:
            raise RuntimeError(f"Benchmark outcome {attempt_id} has invalid status")
        started_at = _parse_utc(
            record.get("started_at_utc"),
            label=f"Benchmark outcome {attempt_id} start timestamp",
        )
        completed_at = _parse_utc(
            record.get("completed_at_utc"),
            label=f"Benchmark outcome {attempt_id} completion timestamp",
        )
        if completed_at < started_at:
            raise RuntimeError(f"Benchmark outcome {attempt_id} completed before it started")
        record_map[attempt_id] = dict(record)
        record_order.append(attempt_id)

    expected_record_order = [str(spec["attempt_id"]) for spec in expected_specs]
    if record_order != expected_record_order[: len(record_order)]:
        raise RuntimeError("Benchmark outcomes do not follow the prespecified attempt order")

    for attempt_id, finish in finishes.items():
        if attempt_id not in starts:
            raise RuntimeError(f"Benchmark attempt {attempt_id} finished without a start event")
        if attempt_id not in record_map:
            raise RuntimeError(f"Benchmark attempt {attempt_id} finished without an outcome")
        if event_positions[(attempt_id, "finished")] <= event_positions[(attempt_id, "started")]:
            raise RuntimeError(f"Benchmark attempt {attempt_id} finished before it started")
        record = record_map[attempt_id]
        if finish.get("status") != record.get("status"):
            raise RuntimeError(f"Benchmark attempt {attempt_id} ledger status does not match its outcome")
        if finish.get("recorded_at_utc") != record.get("completed_at_utc"):
            raise RuntimeError(f"Benchmark attempt {attempt_id} completion timestamp is inconsistent")
        if finish.get("record_sha256") != _record_sha256(record):
            raise RuntimeError(f"Benchmark attempt {attempt_id} outcome hash verification failed")

    for attempt_id, record in record_map.items():
        if attempt_id not in starts:
            raise RuntimeError(f"Benchmark outcome {attempt_id} has no start event")
        if record.get("started_at_utc") != starts[attempt_id].get("recorded_at_utc"):
            raise RuntimeError(f"Benchmark attempt {attempt_id} start timestamp is inconsistent")

    if require_complete:
        expected_ids = set(expected)
        if set(starts) != expected_ids or set(record_map) != expected_ids or set(finishes) != expected_ids:
            raise RuntimeError("The benchmark first-attempt ledger is incomplete")

    ordered_records = [record_map[item["attempt_id"]] for item in expected_specs if item["attempt_id"] in record_map]
    return {
        "expected_specs": expected_specs,
        "events": events,
        "records": ordered_records,
        "starts": starts,
        "finishes": finishes,
        "record_map": record_map,
    }


def _append_outcome(output_dir: Path, record: Mapping[str, Any]) -> None:
    raw_path = output_dir / "runs.jsonl"
    ledger_path = output_dir / "attempts.jsonl"
    _append_jsonl(raw_path, record)
    _append_jsonl(
        ledger_path,
        {
            "attempt_id": record["attempt_id"],
            "event": "finished",
            "phase": record["phase"],
            "record_sha256": _record_sha256(record),
            "recorded_at_utc": record["completed_at_utc"],
            "run_number": record["run_number"],
            "status": record["status"],
        },
    )


def _reconcile_incomplete_attempts(
    output_dir: Path,
    *,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    """Retain interrupted first attempts without issuing replacement calls."""
    audit = audit_attempt_provenance(
        output_dir,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        require_complete=False,
    )
    for spec in audit["expected_specs"]:
        attempt_id = str(spec["attempt_id"])
        start = audit["starts"].get(attempt_id)
        record = audit["record_map"].get(attempt_id)
        finish = audit["finishes"].get(attempt_id)
        if start is not None and record is None:
            interrupted = {
                **spec,
                "started_at_utc": start["recorded_at_utc"],
                "status": "failure",
                "completed_at_utc": _utc_now(),
                "total_seconds": None,
                "error_type": "InterruptedAttempt",
                "error": (
                    "The process ended before an outcome was durably recorded; "
                    "the first attempt was retained as a failure."
                ),
            }
            _append_outcome(output_dir, interrupted)
        elif record is not None and finish is None:
            _append_jsonl(
                output_dir / "attempts.jsonl",
                {
                    "attempt_id": attempt_id,
                    "event": "finished",
                    "phase": record["phase"],
                    "record_sha256": _record_sha256(record),
                    "recorded_at_utc": record["completed_at_utc"],
                    "run_number": record["run_number"],
                    "status": record["status"],
                },
            )
    return audit_attempt_provenance(
        output_dir,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        require_complete=False,
    )


def _flat_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    base = {
        "attempt_id": record["attempt_id"],
        "phase": record["phase"],
        "run_number": record["run_number"],
        "started_at_utc": record["started_at_utc"],
        "status": record["status"],
    }
    if record["status"] == "success":
        timings = record["timings_seconds"]
        metadata = record["model_metadata"]
        return {
            **base,
            "encrypted_record_retrieval_seconds": round(float(timings["encrypted_record_retrieval"]), 6),
            "protocol_ranking_seconds": round(float(timings["protocol_ranking"]), 6),
            "model_note_review_seconds": round(float(timings["model_note_review"]), 6),
            "deterministic_guard_seconds": round(float(timings["deterministic_guard"]), 6),
            "total_seconds": round(float(timings["total"]), 6),
            "model_id_observed": metadata["model_id"],
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "primary_conforms": record["reference_primary_conforms"],
            "backup_conforms": record["reference_backup_conforms"],
            "error_type": "",
        }
    total_seconds = record.get("total_seconds")
    return {
        **base,
        "encrypted_record_retrieval_seconds": "",
        "protocol_ranking_seconds": "",
        "model_note_review_seconds": "",
        "deterministic_guard_seconds": "",
        "total_seconds": "" if total_seconds is None else round(float(total_seconds), 6),
        "model_id_observed": "",
        "input_tokens": "",
        "output_tokens": "",
        "primary_conforms": False,
        "backup_conforms": False,
        "error_type": record.get("error_type", "UnknownError"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _describe(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None,
                "min_seconds": None, "max_seconds": None}
    return {
        "n": len(values),
        "mean_seconds": round(statistics.mean(values), 6),
        "median_seconds": round(statistics.median(values), 6),
        "p95_seconds": round(_percentile(values, 0.95), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def _resolve_run_dir(value: Path | None) -> Path:
    workflow_root = (CURRENT_OUTPUT_DIR / "full_workflow").resolve()
    expected_summary_hash = None
    if value is None:
        pointer = _read_json(CURRENT_OUTPUT_DIR / "latest_full_workflow.json")
        candidate = (APP_DIR / str(pointer["run_directory"])).resolve()
        expected_summary_hash = str(pointer.get("run_summary_sha256", "")).strip()
    else:
        candidate = value.resolve()
    try:
        candidate.relative_to(workflow_root)
    except ValueError as exc:
        raise RuntimeError(f"Run directory must be under {workflow_root}") from exc
    if not (candidate / "run_summary.json").exists():
        raise RuntimeError("The selected directory is not a completed full-workflow run")
    if expected_summary_hash and _sha256(candidate / "run_summary.json") != expected_summary_hash:
        raise RuntimeError("The latest-workflow pointer does not match the source run summary")
    return candidate


def _validate_source_run(source_summary: Mapping[str, Any], model_id: str) -> str:
    """Require the benchmark to reuse the canonical run's frozen inputs."""
    expected = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "demo_case_sha256": _sha256(DEMO_CASE_PATH),
        "model_id_requested": model_id,
    }
    descriptions = {
        "protocol_id": "protocol identifier",
        "protocol_sha256": "protocol hash",
        "demo_case_sha256": "demonstration-case hash",
        "model_id_requested": "requested model identifier",
    }
    for field, expected_value in expected.items():
        observed = str(source_summary.get(field, "")).strip()
        if observed != str(expected_value):
            raise RuntimeError(
                f"The source workflow {descriptions[field]} differs from the active benchmark"
            )
    source_implementation = source_summary.get("implementation_sha256")
    expected_implementation = benchmark_source_hashes()
    if not isinstance(source_implementation, Mapping) or any(
        source_implementation.get(name) != value
        for name, value in expected_implementation.items()
        if name not in {"benchmark", "local_training_state"}
    ):
        raise RuntimeError("The source workflow implementation differs from the active benchmark")
    source_model_evidence = source_summary.get("model_evidence")
    if (
        not isinstance(source_model_evidence, Mapping)
        or source_model_evidence.get("local_lora_training_sha256")
        != expected_implementation["local_training_state"]
    ):
        raise RuntimeError("The source workflow training state differs from the active benchmark")
    observed_model = str(source_summary.get("model", {}).get("model_id", "")).strip()
    if not observed_model:
        raise RuntimeError("The source workflow does not record the provider-returned model identifier")
    return observed_model


def _expected_completion_pointer(output_dir: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    ledger_path = output_dir / "attempts.jsonl"
    raw_path = output_dir / "runs.jsonl"
    timings_path = output_dir / "timings.csv"
    summary_path = output_dir / "summary.json"
    hashes_path = output_dir / "artifact_hashes.json"
    required = (ledger_path, raw_path, timings_path, summary_path, hashes_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError("The finalizing off-chain benchmark is missing an artifact")

    audit = audit_attempt_provenance(
        output_dir,
        warmup_runs=int(state["warmup_runs"]),
        measured_runs=int(state["measured_runs"]),
        require_complete=True,
    )
    summary = _read_json(summary_path)
    expected_summary_fields = {
        "source_workflow_run": state["source_workflow_run"],
        "source_run_summary_sha256": state["source_run_summary_sha256"],
        "demo_case_sha256": state["demo_case_sha256"],
        "model_id_requested": state["model_id_requested"],
        "canonical_model_id_observed": state["canonical_model_id_observed"],
        "protocol_id": state["protocol_id"],
        "protocol_sha256": state["protocol_sha256"],
        "implementation_sha256": state["implementation_sha256"],
        "environment": state["software_environment"],
        "warmup_runs_excluded": state["warmup_runs"],
        "measured_runs_requested": state["measured_runs"],
        "measured_runs_attempted": state["measured_runs"],
        "attempt_ledger_event_count": len(audit["events"]),
        "attempt_ledger_sha256": _sha256(ledger_path),
        "raw_runs_sha256": _sha256(raw_path),
    }
    for field, expected in expected_summary_fields.items():
        if summary.get(field) != expected:
            raise RuntimeError(
                f"The finalizing off-chain benchmark summary has a different {field.replace('_', ' ')}"
            )
    _parse_utc(summary.get("generated_at_utc"), label="Benchmark completion timestamp")

    artifacts = _read_json(hashes_path)
    hashed_paths = (ledger_path, raw_path, timings_path, summary_path)
    if set(artifacts) != {path.name for path in hashed_paths}:
        raise RuntimeError("The off-chain benchmark artifact-hash map has missing or extra entries")
    for path in hashed_paths:
        record = artifacts.get(path.name)
        if not isinstance(record, Mapping) or (
            record.get("sha256") != _sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"The off-chain benchmark artifact hash failed for {path.name}")

    return {
        "completed_at_utc": summary["generated_at_utc"],
        "run_directory": output_dir.name,
        "summary_sha256": _sha256(summary_path),
        "attempt_ledger_sha256": _sha256(ledger_path),
        "raw_runs_sha256": _sha256(raw_path),
    }


def _completion_state(
    state: Mapping[str, Any],
    pointer: Mapping[str, Any],
    *,
    status: str,
) -> Dict[str, Any]:
    return {
        **state,
        "status": status,
        "completed_at_utc": pointer["completed_at_utc"],
        "summary_sha256": pointer["summary_sha256"],
        "attempt_ledger_sha256": pointer["attempt_ledger_sha256"],
        "raw_runs_sha256": pointer["raw_runs_sha256"],
    }


def _recover_or_validate_benchmark_completion(
    benchmark_root: Path,
    output_dir: Path,
    state_path: Path,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    pointer = _expected_completion_pointer(output_dir, state)
    pointer_path = benchmark_root / "completed_run.json"
    if pointer_path.exists():
        if _read_json(pointer_path) != pointer:
            raise RuntimeError("The frozen off-chain benchmark pointer is malformed or stale")
    else:
        _write_json(pointer_path, pointer)

    completed = _completion_state(state, pointer, status="completed")
    for field in (
        "completed_at_utc",
        "summary_sha256",
        "attempt_ledger_sha256",
        "raw_runs_sha256",
    ):
        if field in state and state.get(field) != completed[field]:
            raise RuntimeError(f"The off-chain benchmark completion state has a stale {field}")
    if dict(state) != completed:
        _write_json(state_path, completed)
    return completed


def _load_or_create_benchmark_output(
    benchmark_root: Path,
    run_dir: Path,
    *,
    model_id: str,
    source_observed_model: str,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[Path, Path, Dict[str, Any], bool]:
    benchmark_root.mkdir(parents=True, exist_ok=True)
    state_path = benchmark_root / "run_state.json"
    expected = {
        "schema_version": 1,
        "source_workflow_run": portable_path(run_dir),
        "source_run_summary_sha256": _sha256(run_dir / "run_summary.json"),
        "demo_case_sha256": _sha256(DEMO_CASE_PATH),
        "model_id_requested": model_id,
        "canonical_model_id_observed": source_observed_model,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "implementation_sha256": benchmark_source_hashes(),
        "software_environment": benchmark_software_environment(),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
    }
    if state_path.exists():
        state = _read_json(state_path)
        for field, expected_value in expected.items():
            if state.get(field) != expected_value:
                raise RuntimeError(
                    f"The off-chain benchmark state has a different {field.replace('_', ' ')}"
                )
        output_dir = (benchmark_root / str(state.get("run_directory", ""))).resolve()
        try:
            output_dir.relative_to(benchmark_root.resolve())
        except ValueError as exc:
            raise RuntimeError("The off-chain benchmark state leaves its run directory") from exc
        if not output_dir.is_dir():
            raise RuntimeError("The off-chain benchmark run directory is missing")
        pointer_exists = (benchmark_root / "completed_run.json").exists()
        status = state.get("status")
        if status in {"finalizing", "completed"} or pointer_exists:
            completed = _recover_or_validate_benchmark_completion(
                benchmark_root,
                output_dir,
                state_path,
                state,
            )
            return output_dir, state_path, completed, True
        if status != "in_progress":
            raise RuntimeError("The off-chain benchmark state is not resumable")
        return output_dir, state_path, state, False

    if (benchmark_root / "completed_run.json").exists():
        raise RuntimeError("The off-chain benchmark pointer has no matching run state")

    output_dir = benchmark_root / _stamp()
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {
        **expected,
        "status": "in_progress",
        "created_at_utc": _utc_now(),
        "run_directory": output_dir.name,
    }
    _write_json(state_path, state)
    return output_dir, state_path, state, False


def benchmark(run_dir: Path, *, measured_runs: int, warmup_runs: int) -> Path:
    assert_test_seed_not_retired(PROTOCOL)
    assert_protocol_frozen(PROTOCOL)
    if measured_runs < 1 or warmup_runs < 0:
        raise ValueError("measured_runs must be positive and warmup_runs cannot be negative")
    benchmark_settings = PROTOCOL["performance_evaluation"]["offchain_benchmark"]
    if (
        measured_runs != int(benchmark_settings["measured_runs"])
        or warmup_runs != int(benchmark_settings["warmup_runs"])
    ):
        raise RuntimeError(
            "Benchmark run counts must match the prespecified protocol before measurements begin"
        )
    load_authoritative_project_env()
    gateway = _required_env("PINATA_GATEWAY")
    encryption_key = _required_env("OFFCHAIN_ENCRYPTION_KEY")
    training_state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
    model_id = str(training_state["adapter"]["model_id"])
    model_client = LocalNoteReviewClient(model_id, adapter=True)

    case = _read_json(DEMO_CASE_PATH)
    source_summary = _read_json(run_dir / "run_summary.json")
    source_observed_model = _validate_source_run(source_summary, model_id)
    benchmark_root = run_dir / "offchain_benchmark"
    benchmark_pointer = benchmark_root / "completed_run.json"
    expected_donor = _runtime_profile(case["donor"])
    expected_recipients = [_runtime_profile(item) for item in case["recipients"]]
    cids = _read_json(run_dir / "encrypted_profile_cids.json")
    recipient_cids = {int(key): str(value) for key, value in cids["recipients"].items()}
    if set(recipient_cids) != set(range(1, 11)):
        raise RuntimeError("The workflow run does not contain exactly ten recipient CIDs")

    output_dir, state_path, state, already_completed = _load_or_create_benchmark_output(
        benchmark_root,
        run_dir,
        model_id=model_id,
        source_observed_model=source_observed_model,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    if already_completed:
        return output_dir
    raw_path = output_dir / "runs.jsonl"
    ledger_path = output_dir / "attempts.jsonl"
    audit = _reconcile_incomplete_attempts(
        output_dir,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    total_runs = len(audit["expected_specs"])

    for index, spec in enumerate(audit["expected_specs"], start=1):
        attempt_id = str(spec["attempt_id"])
        if attempt_id in audit["finishes"]:
            continue
        started_at = _utc_now()
        _append_jsonl(
            ledger_path,
            {
                **spec,
                "event": "started",
                "recorded_at_utc": started_at,
            },
        )
        base = {
            **spec,
            "started_at_utc": started_at,
        }
        overall_start = time.perf_counter()
        try:
            retrieval_start = time.perf_counter()
            donor = fetch_encrypted_json_from_ipfs(
                gateway,
                str(cids["donor"]),
                encryption_key,
                expected_aad="donor:1",
                timeout=20,
                retries=int(
                    PROTOCOL["performance_evaluation"]["offchain_benchmark"][
                        "ipfs_fetch_retries_per_measurement"
                    ]
                ),
            )
            recipients = fetch_encrypted_json_batch(
                gateway,
                [
                    (recipient_cids[recipient_id], f"recipient:{recipient_id}")
                    for recipient_id in range(1, 11)
                ],
                encryption_key,
                timeout=20,
                retries=int(
                    PROTOCOL["performance_evaluation"]["offchain_benchmark"][
                        "ipfs_fetch_retries_per_measurement"
                    ]
                ),
            )
            retrieval_seconds = time.perf_counter() - retrieval_start
            if canonical_json_sha256(donor) != canonical_json_sha256(expected_donor):
                raise RuntimeError("Retrieved donor profile failed the source-object integrity check")
            if len(recipients) != len(expected_recipients) or any(
                canonical_json_sha256(observed) != canonical_json_sha256(expected)
                for observed, expected in zip(recipients, expected_recipients)
            ):
                raise RuntimeError("At least one retrieved recipient profile failed its integrity check")

            ranking_start = time.perf_counter()
            ranked = rank_recipients_baseline(donor, recipients, PROTOCOL)
            ranking_seconds = time.perf_counter() - ranking_start

            model_start = time.perf_counter()
            model_result = model_client(
                model_id=model_id,
                api_key="",
                case_id=str(case["case_id"]),
                organ_type=str(case["organ_type"]),
                recipients=recipients,
            )
            if model_result.metadata.model_id != source_observed_model:
                raise RuntimeError(
                    "The runtime model identifier differs from the canonical workflow"
                )
            model_seconds = time.perf_counter() - model_start

            guard_start = time.perf_counter()
            guarded = apply_guarded_policy(
                1,
                ranked,
                model_result.review,
                [int(item["recipient_id"]) for item in recipients],
            )
            guard_seconds = time.perf_counter() - guard_start
            total_seconds = time.perf_counter() - overall_start
            record = {
                **base,
                "status": "success",
                "completed_at_utc": _utc_now(),
                "timings_seconds": {
                    "encrypted_record_retrieval": retrieval_seconds,
                    "protocol_ranking": ranking_seconds,
                    "model_note_review": model_seconds,
                    "deterministic_guard": guard_seconds,
                    "total": total_seconds,
                },
                "model_metadata": model_result.metadata.model_dump(mode="json"),
                "parsed_review": model_result.review.model_dump(mode="json"),
                "raw_model_output": model_result.raw_text,
                "guarded_decision": guarded.model_dump(mode="json"),
                "reference_primary_conforms": (
                    guarded.primary_recipient_id == int(case["reference"]["primary_recipient_id"])
                ),
                "reference_backup_conforms": (
                    guarded.backup_recipient_id == int(case["reference"]["backup_recipient_id"])
                ),
            }
        except Exception as exc:
            record = {
                **base,
                "status": "failure",
                "completed_at_utc": _utc_now(),
                "total_seconds": time.perf_counter() - overall_start,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _append_outcome(output_dir, record)
        audit = audit_attempt_provenance(
            output_dir,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            require_complete=False,
        )
        print(
            f"[{index}/{total_runs}] {spec['phase']} run {spec['run_number']}: {record['status']}",
            flush=True,
        )

    audit = audit_attempt_provenance(
        output_dir,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        require_complete=True,
    )
    flat_rows = [_flat_row(record) for record in audit["records"]]

    _write_csv(output_dir / "timings.csv", flat_rows)
    measured = [row for row in flat_rows if row["phase"] == "measured"]
    successful = [row for row in measured if row["status"] == "success"]
    timing_fields = (
        "encrypted_record_retrieval_seconds",
        "protocol_ranking_seconds",
        "model_note_review_seconds",
        "deterministic_guard_seconds",
        "total_seconds",
    )
    summary = {
        "generated_at_utc": _utc_now(),
        "source_workflow_run": portable_path(run_dir),
        "source_run_summary_sha256": _sha256(run_dir / "run_summary.json"),
        "demo_case_sha256": _sha256(DEMO_CASE_PATH),
        "model_id_requested": model_id,
        "canonical_model_id_observed": source_observed_model,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "benchmark_script_sha256": _sha256(SCRIPT_PATH),
        "implementation_sha256": benchmark_source_hashes(),
        "model_request_config": {
            "temperature": PROTOCOL["model_evaluation"]["temperature"],
            "seed": PROTOCOL["model_evaluation"]["seed"],
            "provider": PROTOCOL["model_evaluation"]["provider"],
            "do_sample": PROTOCOL["model_evaluation"]["do_sample"],
            "maximum_new_tokens": PROTOCOL["model_evaluation"]["maximum_new_tokens"],
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "environment": benchmark_software_environment(),
        "warmup_runs_excluded": warmup_runs,
        "measured_runs_requested": measured_runs,
        "measured_runs_attempted": len(measured),
        "measured_runs_successful": len(successful),
        "measured_runs_failed": len(measured) - len(successful),
        "failed_measured_attempts_rerun": False,
        "attempt_ledger_event_count": len(audit["events"]),
        "attempt_ledger_sha256": _sha256(ledger_path),
        "raw_runs_sha256": _sha256(raw_path),
        "replacement_attempt_count": 0,
        "interrupted_attempts_retained_as_failures": sum(
            1 for record in audit["records"] if record.get("error_type") == "InterruptedAttempt"
        ),
        "ipfs_fetch_retries_per_measurement": int(
            PROTOCOL["performance_evaluation"]["offchain_benchmark"][
                "ipfs_fetch_retries_per_measurement"
            ]
        ),
        "all_measured_primary_conform": len(successful) == measured_runs and all(
            bool(row["primary_conforms"]) for row in measured
        ),
        "all_measured_backup_conform": len(successful) == measured_runs and all(
            bool(row["backup_conforms"]) for row in measured
        ),
        "timings": {
            field.removesuffix("_seconds"): _describe([float(row[field]) for row in successful])
            for field in timing_fields
        },
        "token_totals": {
            "input": sum(int(row["input_tokens"]) for row in successful if row["input_tokens"] != ""),
            "output": sum(int(row["output_tokens"]) for row in successful if row["output_tokens"] != ""),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "artifact_hashes.json",
        {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                ledger_path,
                raw_path,
                output_dir / "timings.csv",
                output_dir / "summary.json",
            )
        },
    )
    pointer = _expected_completion_pointer(output_dir, state)
    finalizing = _completion_state(state, pointer, status="finalizing")
    _write_json(state_path, finalizing)
    _write_json(benchmark_pointer, pointer)
    _write_json(state_path, _completion_state(state, pointer, status="completed"))
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    benchmark_settings = PROTOCOL["performance_evaluation"]["offchain_benchmark"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--measured-runs", type=int, default=int(benchmark_settings["measured_runs"]))
    parser.add_argument("--warmup-runs", type=int, default=int(benchmark_settings["warmup_runs"]))
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = _resolve_run_dir(args.run_dir)
    output_dir = benchmark(
        run_dir,
        measured_runs=args.measured_runs,
        warmup_runs=args.warmup_runs,
    )
    print(f"Off-chain benchmark artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
