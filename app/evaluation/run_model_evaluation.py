"""Resumable evaluation of one model on the locked synthetic test split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from evaluation.artifact_paths import portable_path
from evaluation.project_environment import load_authoritative_project_env
from evaluation.synthetic_dataset import DATASET_DIR, validate_existing
from src.decision_guard import apply_guarded_policy
from src.llm_client import SYSTEM_PROMPT, call_note_review
from src.local_llm_client import LocalNoteReviewClient, validate_local_training_state
from src.policy import (
    DEFAULT_PROTOCOL_FREEZE_PATH,
    DEFAULT_PROTOCOL_PATH,
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
    rank_recipients_baseline,
)


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
TEST_PATH = DATASET_DIR / "test_cases.jsonl"
TEST_LOCK_PATH = OUTPUT_DIR / "test_lock.json"
GENERATOR_PATH = APP_DIR / "evaluation" / "synthetic_dataset.py"
EVALUATOR_PATH = Path(__file__).resolve()
ANALYZER_PATH = APP_DIR / "evaluation" / "analyze_evaluation.py"
COMPARISON_PATH = APP_DIR / "evaluation" / "compare_model_conditions.py"
CLASSICAL_BASELINE_PATH = APP_DIR / "evaluation" / "run_classical_baseline.py"
MODEL_API_COST_PATH = APP_DIR / "evaluation" / "build_model_api_cost.py"
LOCAL_TRAINING_SCRIPT_PATH = APP_DIR / "evaluation" / "train_local_lora.py"
LOCAL_TRAINING_STATE_PATH = OUTPUT_DIR.parent / "model" / "local_lora_training.json"
LOCAL_BASE_MODEL_MANIFEST_PATH = OUTPUT_DIR.parent / "model" / "local_base_model_manifest.json"
LOCAL_LORA_PREFLIGHT_PATH = OUTPUT_DIR.parent / "model" / "local_lora_preflight.json"
LOCAL_LORA_VALIDATION_PREFLIGHT_PATH = (
    OUTPUT_DIR.parent / "model" / "local_lora_validation_preflight.json"
)
LOCAL_LORA_INHERITANCE_PATH = (
    OUTPUT_DIR.parent / "model" / "local_lora_checkpoint_inheritance.json"
)
OPENAI_COMPARATOR_ACCESS_PATH = (
    OUTPUT_DIR.parent / "model" / "openai_comparator_access_check.json"
)
PROTOCOL_LINEAGE_PATH = OUTPUT_DIR.parent / "protocol" / "protocol_lineage_v1.2.0.json"
PROTOCOL_AMENDMENT_PATH = OUTPUT_DIR.parent / "protocol" / "protocol_amendment_v1.2.0.json"
SMOKE_TEST_PATH = APP_DIR / "evaluation" / "smoke_test_model.py"
LLM_CLIENT_PATH = APP_DIR / "src" / "llm_client.py"
LOCAL_LLM_CLIENT_PATH = APP_DIR / "src" / "local_llm_client.py"
POLICY_PATH = APP_DIR / "src" / "policy.py"
GUARD_PATH = APP_DIR / "src" / "decision_guard.py"
SCHEMAS_PATH = APP_DIR / "src" / "schemas.py"
ARTIFACT_PATHS_PATH = APP_DIR / "evaluation" / "artifact_paths.py"
REQUIREMENTS_LOCK_PATH = APP_DIR / "requirements-lock.txt"
REQUIREMENTS_ML_PATH = APP_DIR / "requirements-ml.txt"
ALLOWED_CONDITIONS = {"untuned", "fine_tuned", "openai"}


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _pretest_artifact_hashes() -> Dict[str, str | None]:
    model_dir = OUTPUT_DIR.parent / "model"
    paths = {
        "generation_manifest": DATASET_DIR / "generation_manifest.json",
        "training_cases": DATASET_DIR / "training_cases.jsonl",
        "validation_cases": DATASET_DIR / "validation_cases.jsonl",
        "fine_tuning_training": DATASET_DIR / "fine_tuning_training.jsonl",
        "fine_tuning_validation": DATASET_DIR / "fine_tuning_validation.jsonl",
        "local_base_model_manifest": LOCAL_BASE_MODEL_MANIFEST_PATH,
        "local_lora_preflight": LOCAL_LORA_PREFLIGHT_PATH,
        "local_lora_validation_preflight": LOCAL_LORA_VALIDATION_PREFLIGHT_PATH,
        "local_lora_checkpoint_inheritance": LOCAL_LORA_INHERITANCE_PATH,
        "local_lora_training": LOCAL_TRAINING_STATE_PATH,
        "protocol_lineage": PROTOCOL_LINEAGE_PATH,
        "protocol_amendment": PROTOCOL_AMENDMENT_PATH,
        "classical_validation_diagnostics": model_dir / "classical_validation_diagnostics.json",
        "untuned_smoke_test": model_dir / "smoke_tests" / "untuned_preflight.json",
        "fine_tuned_smoke_test": model_dir / "smoke_tests" / "fine_tuned_preflight.json",
        "openai_comparator_access_check": OPENAI_COMPARATOR_ACCESS_PATH,
        "openai_smoke_test": model_dir / "smoke_tests" / "openai_preflight.json",
    }
    return {name: _optional_sha256(path) for name, path in paths.items()}


def _validate_pretest_terminal_records() -> None:
    model_dir = OUTPUT_DIR.parent / "model"
    protocol = load_protocol()
    hosted = protocol["model_evaluation"]["hosted_comparator"]
    access_path = model_dir / "openai_comparator_access_check.json"
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if (
        access.get("status") != "passed"
        or access.get("protocol_id") != protocol["protocol_id"]
        or access.get("protocol_version") != protocol["version"]
        or access.get("model_id_requested") != hosted["model_id"]
        or access.get("model_id_returned") != hosted["model_id"]
        or access.get("test_case_data_opened") is not False
        or access.get("held_out_test_lock_present") is not False
    ):
        raise RuntimeError("The hosted-comparator access preflight did not pass")

    local_state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
    expected = {
        "untuned_preflight": protocol["model_evaluation"]["base_model_snapshot"],
        "fine_tuned_preflight": local_state["adapter"]["model_id"],
        "openai_preflight": hosted["model_id"],
    }
    for condition, model_id in expected.items():
        path = model_dir / "smoke_tests" / f"{condition}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        model_run = record.get("model_run")
        if (
            record.get("status") not in {"success", "failure"}
            or record.get("condition") != condition
            or record.get("case_index") != 0
            or record.get("model_id_requested") != model_id
            or not isinstance(model_run, Mapping)
            or model_run.get("model_id") != model_id
        ):
            raise RuntimeError(f"The {condition} record is not a valid terminal preflight")
        if condition in {"fine_tuned_preflight", "openai_preflight"} and record.get(
            "status"
        ) != "success":
            raise RuntimeError(f"The {condition} record did not pass")
        if record.get("status") == "failure" and (
            not record.get("error_type")
            or not isinstance(record.get("raw_model_output"), str)
            or not record["raw_model_output"].strip()
        ):
            raise RuntimeError(f"The {condition} failure lacks raw diagnostic evidence")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: object, *, label: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a valid UTC timestamp") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_label(value: str) -> str:
    label = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return label or "model"


def _validate_condition_model(condition: str, model_id: str) -> None:
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(ALLOWED_CONDITIONS)}")
    settings = load_protocol()["model_evaluation"]
    if condition == "untuned":
        expected_model = str(settings["base_model_snapshot"])
    elif condition == "fine_tuned":
        state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
        expected_model = str(state["adapter"]["model_id"])
    else:
        expected_model = str(settings["hosted_comparator"]["model_id"])
    if not expected_model or model_id != expected_model:
        raise RuntimeError(f"{condition} evaluation must use its preserved expected model")


def _condition_runtime_settings(condition: str) -> Dict[str, Any]:
    settings = load_protocol()["model_evaluation"]
    if condition == "openai":
        hosted = settings["hosted_comparator"]
        return {
            "provider": hosted["provider"],
            "temperature": hosted["temperature"],
            "seed": hosted["seed"],
            "do_sample": False,
            "maximum_new_tokens": hosted["maximum_completion_tokens"],
            "attempts_per_case_per_condition": hosted["attempts_per_case"],
            "endpoint": hosted["endpoint"],
            "sdk_max_retries": hosted["sdk_max_retries"],
            "store": hosted["store"],
        }
    return {
        "provider": settings["provider"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "do_sample": settings["do_sample"],
        "maximum_new_tokens": settings["maximum_new_tokens"],
        "attempts_per_case_per_condition": settings["attempts_per_case_per_condition"],
        "endpoint": "local_transformers_generate",
        "sdk_max_retries": 0,
        "store": False,
    }


def _existing_case_ids(
    path: Path,
    *,
    expected_order: Iterable[str] | None = None,
) -> set[str]:
    if not path.exists():
        return set()
    rows = _load_jsonl(path)
    case_ids = [str(row.get("case_id")) for row in rows]
    if any(case_id in {"", "None"} for case_id in case_ids):
        raise RuntimeError(f"Existing evaluation output contains a missing case ID: {path}")
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(f"Existing evaluation output contains duplicate case IDs: {path}")
    if expected_order is not None:
        expected = list(expected_order)
        if case_ids != expected[: len(case_ids)]:
            raise RuntimeError("Existing evaluation outcomes do not follow the locked test order")
    return set(case_ids)


def _attempted_case_ids(
    path: Path,
    *,
    condition: str,
    model_id: str,
    expected_order: Iterable[str] | None = None,
) -> tuple[set[str], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return set(), {}
    rows = _load_jsonl(path)
    by_case: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if (
            not case_id
            or row.get("event") != "attempt_started"
            or row.get("condition") != condition
            or row.get("model_id_requested") != model_id
        ):
            raise RuntimeError(f"Invalid attempt-ledger entry in {path}")
        if case_id in by_case:
            raise RuntimeError(f"Duplicate attempted case {case_id} in {path}")
        _parse_utc(
            row.get("started_at_utc"),
            label=f"Attempt ledger timestamp for {case_id}",
        )
        by_case[case_id] = row
    if expected_order is not None:
        expected = list(expected_order)
        observed = list(by_case)
        if observed != expected[: len(observed)]:
            raise RuntimeError("Attempt ledger does not follow the locked test order")
    return set(by_case), by_case


def validate_outcome_attempt_provenance(
    records: Iterable[Mapping[str, Any]],
    attempt_rows: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool,
) -> None:
    record_ids: set[str] = set()
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in record_ids:
            raise RuntimeError("Evaluation outcomes contain a missing or duplicate case ID")
        record_ids.add(case_id)
        attempt = attempt_rows.get(case_id)
        if attempt is None:
            raise RuntimeError(f"Evaluation outcome for {case_id} has no first-attempt record")
        if (
            record.get("condition") != attempt.get("condition")
            or record.get("model_id_requested") != attempt.get("model_id_requested")
        ):
            raise RuntimeError(f"Evaluation outcome provenance differs from its attempt for {case_id}")
        started_at = str(attempt.get("started_at_utc", ""))
        if record.get("attempt_started_at_utc") != started_at:
            raise RuntimeError(f"Evaluation outcome start time differs from its attempt for {case_id}")
        started = _parse_utc(started_at, label=f"Attempt start time for {case_id}")
        completed = _parse_utc(
            record.get("completed_at_utc"),
            label=f"Outcome completion time for {case_id}",
        )
        if completed < started:
            raise RuntimeError(f"Evaluation outcome predates its attempt for {case_id}")
    if require_complete and record_ids != set(attempt_rows):
        raise RuntimeError("Evaluation outcomes do not cover the complete first-attempt ledger")


def _test_lock_value() -> Dict[str, Any]:
    return {
        "protocol_id": load_protocol()["protocol_id"],
        "test_file_sha256": _sha256(TEST_PATH),
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "protocol_freeze_sha256": _sha256(DEFAULT_PROTOCOL_FREEZE_PATH),
        "generator_sha256": _sha256(GENERATOR_PATH),
        "evaluator_sha256": _sha256(EVALUATOR_PATH),
        "analyzer_sha256": _sha256(ANALYZER_PATH),
        "comparison_sha256": _sha256(COMPARISON_PATH),
        "classical_baseline_sha256": _sha256(CLASSICAL_BASELINE_PATH),
        "model_api_cost_script_sha256": _sha256(MODEL_API_COST_PATH),
        "local_training_script_sha256": _sha256(LOCAL_TRAINING_SCRIPT_PATH),
        "smoke_test_sha256": _sha256(SMOKE_TEST_PATH),
        "llm_client_sha256": _sha256(LLM_CLIENT_PATH),
        "local_llm_client_sha256": _sha256(LOCAL_LLM_CLIENT_PATH),
        "policy_sha256": _sha256(POLICY_PATH),
        "guard_sha256": _sha256(GUARD_PATH),
        "schemas_sha256": _sha256(SCHEMAS_PATH),
        "artifact_paths_sha256": _sha256(ARTIFACT_PATHS_PATH),
        "requirements_lock_sha256": _sha256(REQUIREMENTS_LOCK_PATH),
        "requirements_ml_sha256": _sha256(REQUIREMENTS_ML_PATH),
        "protocol_lineage_sha256": _sha256(PROTOCOL_LINEAGE_PATH),
        "protocol_amendment_sha256": _sha256(PROTOCOL_AMENDMENT_PATH),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "pretest_artifact_sha256": _pretest_artifact_hashes(),
    }


def validate_test_lock(locked: Mapping[str, Any]) -> None:
    """Verify every frozen test and source fingerprint against the current tree."""
    current = _test_lock_value()
    recorded = {key: value for key, value in locked.items() if key != "locked_at_utc"}
    if recorded != current:
        changed = sorted(
            key
            for key in set(recorded).union(current)
            if recorded.get(key) != current.get(key)
        )
        raise RuntimeError(
            "The locked test or evaluation implementation changed after the first test call. "
            f"Create a new protocol version and test seed before continuing. Changed: {changed}"
        )


def _enforce_test_lock(*, create: bool = True) -> Dict[str, Any] | None:
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    current = _test_lock_value()
    if TEST_LOCK_PATH.exists():
        with TEST_LOCK_PATH.open("r", encoding="utf-8") as handle:
            locked = json.load(handle)
        validate_test_lock(locked)
        return locked
    if not create:
        return None
    pretest_hashes = current.get("pretest_artifact_sha256", {})
    missing_pretests = sorted(
        name
        for name, value in pretest_hashes.items()
        if not isinstance(value, str) or not value
    )
    if missing_pretests:
        raise RuntimeError(
            "The held-out test cannot be locked before all pretest artifacts exist. Missing: "
            + ", ".join(missing_pretests)
        )
    _validate_pretest_terminal_records()
    locked = {**current, "locked_at_utc": _utc_now()}
    _write_json(TEST_LOCK_PATH, locked)
    return locked


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate(
    model_id: str,
    condition: str,
    api_key: str = "",
    *,
    limit: int | None = None,
    confirm_test_lock: bool = False,
    caller: Callable[..., Any] | None = None,
) -> Path:
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    _validate_condition_model(condition, model_id)
    model_settings = _condition_runtime_settings(condition)
    if condition == "openai" and caller is None and not str(api_key).strip():
        raise RuntimeError("OPENAI_API_KEY is required before locking the hosted comparison")
    selected_caller = caller or call_note_review

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{_safe_label(condition)}_raw.jsonl"
    config_path = OUTPUT_DIR / f"{_safe_label(condition)}_config.json"
    attempt_path = OUTPUT_DIR / f"{_safe_label(condition)}_attempts.jsonl"

    if (output_path.exists() or attempt_path.exists()) and not config_path.exists():
        raise RuntimeError(
            "Evaluation output exists without its configuration file"
        )

    existing_config = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing_config = json.load(handle)
        if (
            existing_config.get("condition") != condition
            or existing_config.get("model_id_requested") != model_id
            or existing_config.get("test_file_sha256") != _sha256(TEST_PATH)
        ):
            raise RuntimeError(
                "Existing evaluation output belongs to a different condition, model, or test file. "
                "Use a new condition label and preserve the existing artifacts."
            )
        previous_incomplete = existing_config.get("evaluation_in_progress") is True
        if not previous_incomplete:
            if (
                existing_config.get("raw_file_sha256")
                and existing_config["raw_file_sha256"] != _sha256(output_path)
            ):
                raise RuntimeError("Existing evaluation raw output no longer matches its recorded hash")
            if (
                existing_config.get("attempt_log_sha256")
                and existing_config["attempt_log_sha256"] != _sha256(attempt_path)
            ):
                raise RuntimeError("Existing attempt ledger no longer matches its recorded hash")
    else:
        previous_incomplete = False

    test_lock = _enforce_test_lock(create=False)
    if test_lock is None and not confirm_test_lock:
        raise RuntimeError(
            "The first held-out evaluation requires explicit authorization. Review the frozen "
            "protocol, then rerun with --confirm-test-lock."
        )
    if test_lock is None:
        test_lock = _enforce_test_lock(create=True)
    if test_lock is None:
        raise RuntimeError("Unable to create or read the frozen test lock")
    if existing_config is not None:
        recorded_lock = existing_config.get("test_lock")
        if recorded_lock is None and previous_incomplete:
            existing_config["test_lock"] = test_lock
            _write_json(config_path, existing_config)
        elif recorded_lock != test_lock:
            raise RuntimeError("Existing evaluation configuration does not match the frozen test lock")

    validation = validate_existing()
    if not validation["valid"]:
        raise RuntimeError("The locked dataset did not pass validation")
    all_cases = _load_jsonl(TEST_PATH)
    cases = list(all_cases)
    if limit is not None:
        cases = cases[:limit]
    expected_case_order = [str(case["case_id"]) for case in all_cases]
    completed = _existing_case_ids(output_path, expected_order=expected_case_order)
    attempted, attempt_rows = _attempted_case_ids(
        attempt_path,
        condition=condition,
        model_id=model_id,
        expected_order=expected_case_order,
    )
    existing_records = _load_jsonl(output_path) if output_path.exists() else []
    validate_outcome_attempt_provenance(
        existing_records,
        attempt_rows,
        require_complete=False,
    )
    expected_case_ids = {str(case["case_id"]) for case in all_cases}
    if not completed.issubset(expected_case_ids):
        raise RuntimeError("Existing evaluation output contains case IDs outside the locked test set")
    if not attempted.issubset(expected_case_ids):
        raise RuntimeError("Attempt ledger contains case IDs outside the locked test set")
    if not completed.issubset(attempted):
        raise RuntimeError("Evaluation output contains a case without a recorded first attempt")
    if output_path.exists():
        for row in _load_jsonl(output_path):
            if row.get("condition") != condition or row.get("model_id_requested") != model_id:
                raise RuntimeError(
                    "Existing evaluation records belong to a different condition or requested model"
                )
    orphaned = attempted.difference(completed)
    if orphaned and not previous_incomplete:
        raise RuntimeError(
            "Attempt ledger contains an unterminated case but the configuration is not marked in progress"
        )

    invocation = {
        "invoked_at_utc": _utc_now(),
        "requested_case_limit": limit,
    }
    if existing_config is None:
        config = {
            "condition": condition,
            "model_id_requested": model_id,
            "protocol_id": load_protocol()["protocol_id"],
            "test_file": portable_path(TEST_PATH),
            "test_file_sha256": _sha256(TEST_PATH),
            "test_case_count": len(all_cases),
            "temperature": model_settings["temperature"],
            "seed": model_settings["seed"],
            "provider": model_settings["provider"],
            "do_sample": model_settings["do_sample"],
            "maximum_new_tokens": model_settings["maximum_new_tokens"],
            "attempts_per_case_per_condition": model_settings["attempts_per_case_per_condition"],
            "endpoint": model_settings["endpoint"],
            "sdk_max_retries": model_settings["sdk_max_retries"],
            "store": model_settings["store"],
            "attempt_log_file": portable_path(attempt_path),
            "software_environment": {
                "python": platform.python_version(),
                "openai": importlib.metadata.version("openai"),
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
                "pydantic": importlib.metadata.version("pydantic"),
            },
            "first_invoked_at_utc": invocation["invoked_at_utc"],
            "test_lock": test_lock,
            "evaluation_invocations": [],
        }
    else:
        config = existing_config
    config.setdefault("evaluation_invocations", []).append(invocation)
    config["last_invoked_at_utc"] = invocation["invoked_at_utc"]
    config["current_invocation_target_case_count"] = len(cases)
    config["evaluation_in_progress"] = True
    _write_json(config_path, config)

    case_by_id = {str(case["case_id"]): case for case in all_cases}
    case_order = {case_id: index for index, case_id in enumerate(case_by_id)}
    for case_id in sorted(orphaned, key=case_order.__getitem__):
        case = case_by_id[case_id]
        attempt = attempt_rows[case_id]
        _append_jsonl(
            output_path,
            {
                "case_id": case_id,
                "condition": condition,
                "model_id_requested": model_id,
                "organ_type": case["organ_type"],
                "scenario_family": case["scenario_family"],
                "status": "failure",
                "attempt_started_at_utc": attempt["started_at_utc"],
                "completed_at_utc": _utc_now(),
                "error_type": "InterruptedAttempt",
                "error": (
                    "The first recorded attempt did not reach a terminal record before the process "
                    "stopped; the case was retained as a failure and was not retried."
                ),
            },
        )
        completed.add(case_id)

    for number, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        attempt_started_at = _utc_now()
        _append_jsonl(
            attempt_path,
            {
                "event": "attempt_started",
                "case_id": case_id,
                "condition": condition,
                "model_id_requested": model_id,
                "started_at_utc": attempt_started_at,
            },
        )
        attempted.add(case_id)
        base = {
            "case_id": case_id,
            "condition": condition,
            "model_id_requested": model_id,
            "organ_type": case["organ_type"],
            "scenario_family": case["scenario_family"],
            "attempt_started_at_utc": attempt_started_at,
        }
        try:
            result = selected_caller(
                model_id=model_id,
                api_key=api_key,
                case_id=case_id,
                organ_type=case["organ_type"],
                recipients=case["recipients"],
                temperature=float(model_settings["temperature"]),
                seed=int(model_settings["seed"]),
            )
            if result.metadata.model_id != model_id:
                raise RuntimeError("Model runtime returned a different model identifier")
            ranked = rank_recipients_baseline(case["donor"], case["recipients"])
            decision = apply_guarded_policy(
                donor_id=int(case["donor"]["donor_id"]),
                ranked=ranked,
                review=result.review,
                expected_recipient_ids=[item["recipient_id"] for item in case["recipients"]],
            )
            row = {
                **base,
                "status": "success",
                "completed_at_utc": _utc_now(),
                "review": result.review.model_dump(mode="json"),
                "guarded_decision": decision.model_dump(mode="json"),
                "model_run": result.metadata.model_dump(mode="json"),
                "raw_model_output": result.raw_text,
            }
        except Exception as exc:
            metadata = getattr(exc, "model_run_metadata", None)
            row = {
                **base,
                "status": "failure",
                "completed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_model_output": getattr(exc, "raw_text", None),
                "model_run": (
                    metadata.model_dump(mode="json") if metadata is not None else None
                ),
            }
        _append_jsonl(output_path, row)
        print(f"[{number}/{len(cases)}] {case_id} {row['status']}", flush=True)
    final_records = _load_jsonl(output_path)
    config["last_completed_at_utc"] = _utc_now()
    config["evaluation_in_progress"] = False
    config["raw_record_count"] = len(final_records)
    config["successful_record_count"] = sum(row.get("status") == "success" for row in final_records)
    config["failed_record_count"] = sum(row.get("status") != "success" for row in final_records)
    config["raw_file_sha256"] = _sha256(output_path)
    config["attempted_case_count"] = len(attempted)
    config["attempt_log_sha256"] = _sha256(attempt_path)
    _write_json(config_path, config)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--condition", choices=sorted(ALLOWED_CONDITIONS), default="fine_tuned")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--confirm-test-lock",
        action="store_true",
        help="Authorize creation of the irreversible held-out test lock if none exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    load_authoritative_project_env()
    settings = load_protocol()["model_evaluation"]
    if args.condition == "untuned":
        expected_model = str(settings["base_model_snapshot"])
        adapter = False
        api_key = ""
        client: Callable[..., Any] = LocalNoteReviewClient(expected_model, adapter=adapter)
    elif args.condition == "fine_tuned":
        training_state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
        expected_model = str(training_state["adapter"]["model_id"])
        adapter = True
        api_key = ""
        client = LocalNoteReviewClient(expected_model, adapter=adapter)
    else:
        expected_model = str(settings["hosted_comparator"]["model_id"])
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from app/.env")
        client = call_note_review
    model_id = str(args.model_id or expected_model).strip()
    if not model_id:
        raise RuntimeError("The frozen local condition has no model identifier")
    path = evaluate(
        model_id,
        args.condition,
        api_key,
        limit=args.limit,
        confirm_test_lock=args.confirm_test_lock,
        caller=client,
    )
    print(f"Saved raw evaluation records to {path}")


if __name__ == "__main__":
    main()
