"""Create, monitor, and archive the supervised fine-tuning job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from evaluation.project_environment import load_authoritative_project_env
from openai import OpenAI

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.synthetic_dataset import DATASET_DIR, validate_cases
from src.policy import (
    DEFAULT_PROTOCOL_PATH,
    PolicyInputError,
    assert_preserved_artifact_lineage,
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
)
from src.llm_client import OPENAI_API_BASE, SYSTEM_PROMPT
from src.schemas import NoteReviewBatch


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "model"
STATE_PATH = OUTPUT_DIR / "fine_tuning_job.json"
ACCESS_CHECK_PATH = OUTPUT_DIR / "fine_tuning_access_check.json"
TRAIN_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
TRAIN_CASES_PATH = DATASET_DIR / "training_cases.jsonl"
VALIDATION_CASES_PATH = DATASET_DIR / "validation_cases.jsonl"
GENERATION_MANIFEST_PATH = DATASET_DIR / "generation_manifest.json"
ACTIVE_PROTOCOL = load_protocol()
MODEL_SETTINGS = ACTIVE_PROTOCOL["model_evaluation"]
DEFAULT_BASE_MODEL = str(MODEL_SETTINGS["base_model_snapshot"])
DEFAULT_EPOCHS = int(MODEL_SETTINGS["fine_tuning"]["epochs"])
OPENAI_FINE_TUNING_DEPRECATION_URL = (
    "https://developers.openai.com/api/docs/deprecations#update-to-openais-self-serve-fine-tuning"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_path_matches(record: Any, expected: Path, *, field: str = "path") -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        return resolve_recorded_path(record.get(field)) == expected.resolve()
    except ValueError:
        return False


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _json_equivalent(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=True, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_validation_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_validation_value(item)
            for key, item in value.items()
            if not str(key).endswith("_at_utc")
        }
    if isinstance(value, list):
        return [_stable_validation_value(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _error_details(exc: Exception) -> Dict[str, Any]:
    details: Dict[str, Any] = {"type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(status_code, int):
        details["status_code"] = status_code
    if request_id:
        details["request_id"] = str(request_id)
    provider_code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if provider_code:
        details["provider_code"] = str(provider_code)
    elif isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping) and error.get("code"):
            details["provider_code"] = str(error["code"])
    return details


def _is_definitive_client_rejection(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in {
        408,
        409,
        425,
        429,
    }


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        raise RuntimeError(f"No fine-tuning state found at {STATE_PATH}")
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_state(state: Dict[str, Any]) -> None:
    _write_json(STATE_PATH, state)


def validate_provider_job(state: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """Check that provider metadata describes the exact prespecified job request."""
    training_file = state.get("training_file", {})
    validation_file = state.get("validation_file", {})
    hyperparameters = job.get("hyperparameters", {})
    method = job.get("method")
    metadata = job.get("metadata")
    checks = {
        "job_id_present": bool(str(job.get("id", "")).strip()),
        "object_is_fine_tuning_job": job.get("object") == "fine_tuning.job",
        "base_model_matches": job.get("model") == state.get("base_model_requested"),
        "training_file_matches": (
            isinstance(training_file, dict)
            and bool(str(training_file.get("id", "")).strip())
            and job.get("training_file") == training_file.get("id")
        ),
        "training_upload_metadata_matches": (
            isinstance(training_file, dict)
            and training_file.get("purpose") == "fine-tune"
            and training_file.get("filename") == TRAIN_PATH.name
            and training_file.get("bytes") == TRAIN_PATH.stat().st_size
        ),
        "validation_file_matches": (
            isinstance(validation_file, dict)
            and bool(str(validation_file.get("id", "")).strip())
            and job.get("validation_file") == validation_file.get("id")
        ),
        "validation_upload_metadata_matches": (
            isinstance(validation_file, dict)
            and validation_file.get("purpose") == "fine-tune"
            and validation_file.get("filename") == VALIDATION_PATH.name
            and validation_file.get("bytes") == VALIDATION_PATH.stat().st_size
        ),
        "seed_matches": job.get("seed") == int(MODEL_SETTINGS["seed"]),
        "epochs_match": (
            isinstance(hyperparameters, dict)
            and hyperparameters.get("n_epochs") == DEFAULT_EPOCHS
        ),
        "supervised_method_matches": (
            method is None or (isinstance(method, dict) and method.get("type") == "supervised")
        ),
        "protocol_metadata_matches": (
            isinstance(metadata, dict)
            and metadata.get("protocol") == ACTIVE_PROTOCOL["protocol_id"]
            and metadata.get("purpose") == "synthetic-note-readiness"
            and metadata.get("local_run_id") == state.get("local_run_id")
        ),
    }
    return {
        "validated_at_utc": _utc_now(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def check_model_access(client: OpenAI, base_model: str) -> Dict[str, Any]:
    """Confirm authenticated access to the fixed base model without uploading data."""
    assert_test_seed_not_retired(ACTIVE_PROTOCOL)
    assert_protocol_frozen(ACTIVE_PROTOCOL)
    if STATE_PATH.exists():
        raise RuntimeError(
            "Fine-tuning state already exists; refusing to replace its bound model-access "
            "record or contact the provider"
        )
    error_details = None
    model = None
    try:
        model = _serialize(client.models.retrieve(base_model))
    except Exception as exc:
        error_details = _error_details(exc)
    model_id = str(model.get("id", "")) if isinstance(model, dict) else ""
    summary = {
        "checked_at_utc": _utc_now(),
        "endpoint": f"{OPENAI_API_BASE}/models/{base_model}",
        "base_model_requested": base_model,
        "model_retrievable": isinstance(model, dict),
        "model_id_returned": model_id or None,
        "model_id_matches": model_id == base_model,
        "model_owned_by": model.get("owned_by") if isinstance(model, dict) else None,
        "model_shutdown_date": model.get("shutdown_date") if isinstance(model, dict) else None,
        "fine_tuning_job_authorization_confirmed": False,
        "authorization_note": (
            "Model retrieval confirms base-model access only. Fine-tuning job authorization "
            "is confirmed only when the job-creation request succeeds."
        ),
        "error_details": error_details,
        "openai_sdk_version": importlib.metadata.version("openai"),
    }
    _write_json(ACCESS_CHECK_PATH, summary)
    return summary


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_development_data() -> Dict[str, Any]:
    """Validate training and validation data without parsing held-out cases."""
    splits = {
        "training": _load_jsonl(TRAIN_CASES_PATH),
        "validation": _load_jsonl(VALIDATION_CASES_PATH),
    }
    report = validate_cases(splits)
    manifest = json.loads(GENERATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    errors = list(report.get("errors", []))
    if not isinstance(manifest, dict):
        raise RuntimeError("Generation manifest is not a JSON object")
    if manifest.get("protocol_id") != ACTIVE_PROTOCOL["protocol_id"]:
        errors.append("Generation manifest protocol metadata differs from the active protocol")
    source_inputs = manifest.get("source_inputs", {})
    protocol_record = source_inputs.get("protocol_json", {}) if isinstance(source_inputs, dict) else {}
    prompt_hash = source_inputs.get("system_prompt_sha256") if isinstance(source_inputs, dict) else None
    inherited_protocol = (
        manifest.get("protocol_version") != ACTIVE_PROTOCOL["version"]
        or not isinstance(protocol_record, dict)
        or protocol_record.get("sha256") != _sha256(DEFAULT_PROTOCOL_PATH)
    )
    if inherited_protocol:
        try:
            assert_preserved_artifact_lineage(
                "generation_manifest",
                GENERATION_MANIFEST_PATH,
                artifact_protocol_version=manifest.get("protocol_version"),
                protocol_source_sha256=(
                    protocol_record.get("sha256")
                    if isinstance(protocol_record, dict)
                    else None
                ),
            )
        except PolicyInputError as exc:
            errors.append(f"Generation manifest protocol lineage is invalid: {exc}")
    if prompt_hash != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest():
        errors.append("Generation manifest source inputs are stale")
    artifacts = manifest.get("artifacts", {})
    development_paths = {
        TRAIN_CASES_PATH.name: TRAIN_CASES_PATH,
        VALIDATION_CASES_PATH.name: VALIDATION_CASES_PATH,
        TRAIN_PATH.name: TRAIN_PATH,
        VALIDATION_PATH.name: VALIDATION_PATH,
    }
    for name, path in development_paths.items():
        record = artifacts.get(name, {}) if isinstance(artifacts, dict) else {}
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            errors.append(f"Generation manifest development artifact is stale: {name}")
    report["manifest_development_artifacts_valid"] = not any(
        error.startswith("Generation manifest") for error in errors
    )
    report["errors"] = errors
    report["valid"] = not errors
    return report


def _validate_fine_tuning_file(path: Path, expected_rows: int) -> Dict[str, Any]:
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path.name} has {len(rows)} rows; expected {expected_rows}")
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise RuntimeError(f"{path.name} row {index} has an invalid message sequence")
        if messages[0].get("content") != SYSTEM_PROMPT:
            raise RuntimeError(f"{path.name} row {index} does not use the frozen system prompt")
        user = json.loads(messages[1]["content"])
        if set(user) != {"case_id", "organ_type", "candidates"}:
            raise RuntimeError(f"{path.name} row {index} exposes unexpected user-payload fields")
        if any(set(item) != {"recipient_id", "medical_notes"} for item in user["candidates"]):
            raise RuntimeError(f"{path.name} row {index} exposes fields beyond ID and note text")
        expected_id_values = [int(item["recipient_id"]) for item in user["candidates"]]
        if len(expected_id_values) != 10 or len(expected_id_values) != len(set(expected_id_values)):
            raise RuntimeError(f"{path.name} row {index} does not contain ten unique candidate IDs")
        assistant = NoteReviewBatch.model_validate_json(messages[2]["content"])
        observed_id_values = [int(item.recipient_id) for item in assistant.assessments]
        if (
            assistant.case_id != user["case_id"]
            or len(observed_id_values) != len(set(observed_id_values))
            or set(observed_id_values) != set(expected_id_values)
        ):
            raise RuntimeError(f"{path.name} row {index} has inconsistent case or recipient IDs")
        if str(user["case_id"]) in case_ids:
            raise RuntimeError(f"{path.name} contains duplicate case ID {user['case_id']}")
        case_ids.add(str(user["case_id"]))
    return {"row_count": len(rows), "case_ids": case_ids, "sha256": _sha256(path)}


def validate_fine_tuning_files() -> Dict[str, Any]:
    split_settings = ACTIVE_PROTOCOL["dataset"]["splits"]
    train = _validate_fine_tuning_file(TRAIN_PATH, int(split_settings["training"]["cases"]))
    validation = _validate_fine_tuning_file(
        VALIDATION_PATH,
        int(split_settings["validation"]["cases"]),
    )
    if train["case_ids"].intersection(validation["case_ids"]):
        raise RuntimeError("Fine-tuning train and validation files contain overlapping cases")
    return {
        "training": {key: value for key, value in train.items() if key != "case_ids"},
        "validation": {key: value for key, value in validation.items() if key != "case_ids"},
    }


def _fine_tuning_request(base_model: str, epochs: int, local_run_id: str) -> Dict[str, Any]:
    return {
        "model": str(base_model),
        "suffix": "oda-note-readiness-v1",
        "seed": int(MODEL_SETTINGS["seed"]),
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": int(epochs),
                    "batch_size": MODEL_SETTINGS["fine_tuning"].get("batch_size", "auto"),
                    "learning_rate_multiplier": MODEL_SETTINGS["fine_tuning"].get(
                        "learning_rate_multiplier", "auto"
                    ),
                }
            },
        },
        "metadata": {
            "protocol": str(ACTIVE_PROTOCOL["protocol_id"]),
            "purpose": "synthetic-note-readiness",
            "local_run_id": local_run_id,
        },
    }


def _request_headers(state: Mapping[str, Any], operation: str) -> Dict[str, str]:
    request_ids = state.get("request_ids")
    if not isinstance(request_ids, Mapping):
        raise RuntimeError("Fine-tuning state has no preserved provider request IDs")
    request_id = str(request_ids.get(operation, ""))
    if not request_id or not request_id.isascii() or len(request_id) > 512:
        raise RuntimeError(f"Fine-tuning state has an invalid {operation} request ID")
    return {
        "Idempotency-Key": request_id,
        "X-Client-Request-Id": request_id,
    }


def _validate_upload_record(record: Any, path: Path, label: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"The {label} upload response is not an object")
    checks = {
        "id": bool(str(record.get("id", "")).strip()),
        "purpose": record.get("purpose") == "fine-tune",
        "filename": record.get("filename") == path.name,
        "bytes": record.get("bytes") == path.stat().st_size,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"The {label} upload response failed validation: {failed}")
    return record


def _validate_resume_state(
    state: Dict[str, Any],
    request: Dict[str, Any],
    development_report: Dict[str, Any],
    file_validation: Dict[str, Any],
) -> None:
    local_run_id = str(state.get("local_run_id", ""))
    try:
        valid_run_id = uuid.UUID(hex=local_run_id).hex == local_run_id
    except ValueError:
        valid_run_id = False
    request_ids = state.get("request_ids")
    expected_request_id_keys = {"training_upload", "validation_upload", "job_creation"}
    valid_request_ids = (
        isinstance(request_ids, dict)
        and set(request_ids) == expected_request_id_keys
        and len(set(str(value) for value in request_ids.values())) == len(expected_request_id_keys)
        and all(
            bool(str(value)) and str(value).isascii() and len(str(value)) <= 512
            for value in request_ids.values()
        )
    )
    protocol = state.get("protocol")
    access = state.get("access_check")
    generation_manifest = state.get("generation_manifest")
    training_dataset = state.get("training_dataset")
    validation_dataset = state.get("validation_dataset")
    job_creation_attempts = state.get("job_creation_attempts")
    valid_job_creation_attempts = (
        isinstance(job_creation_attempts, list)
        and all(
            isinstance(attempt, dict)
            and attempt.get("request_id") == (
                request_ids.get("job_creation") if isinstance(request_ids, dict) else None
            )
            and bool(str(attempt.get("started_at_utc", "")).strip())
            and attempt.get("outcome") in {
                "started",
                "uncertain",
                "provider_response_received",
                "provider_job_recovered",
                "provider_rejected",
            }
            for attempt in job_creation_attempts
        )
    )
    expected = {
        "base_model": state.get("base_model_requested") == DEFAULT_BASE_MODEL,
        "local_run_id": valid_run_id,
        "request_ids": valid_request_ids,
        "request": state.get("request_without_file_ids") == request,
        "protocol": (
            isinstance(protocol, dict)
            and protocol.get("id") == ACTIVE_PROTOCOL["protocol_id"]
            and protocol.get("version") == ACTIVE_PROTOCOL["version"]
            and _recorded_path_matches(protocol, DEFAULT_PROTOCOL_PATH)
            and protocol.get("sha256") == _sha256(DEFAULT_PROTOCOL_PATH)
        ),
        "generation_manifest": (
            isinstance(generation_manifest, dict)
            and _recorded_path_matches(generation_manifest, GENERATION_MANIFEST_PATH)
            and generation_manifest.get("sha256") == _sha256(GENERATION_MANIFEST_PATH)
        ),
        "system_prompt": state.get("system_prompt_sha256")
        == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "training_dataset": (
            isinstance(training_dataset, dict)
            and _recorded_path_matches(training_dataset, TRAIN_PATH)
            and training_dataset.get("sha256") == _sha256(TRAIN_PATH)
        ),
        "validation_dataset": (
            isinstance(validation_dataset, dict)
            and _recorded_path_matches(validation_dataset, VALIDATION_PATH)
            and validation_dataset.get("sha256") == _sha256(VALIDATION_PATH)
        ),
        "development_validation": _json_equivalent(
            _stable_validation_value(state.get("development_data_validation")),
            _stable_validation_value(development_report),
        ),
        "fine_tuning_file_validation": _json_equivalent(
            state.get("fine_tuning_file_validation"),
            file_validation,
        ),
        "job_creation_attempts": valid_job_creation_attempts,
        "access_check": (
            isinstance(access, dict)
            and access.get("model_retrievable") is True
            and access.get("model_id_matches") is True
            and access.get("base_model_requested") == DEFAULT_BASE_MODEL
        ),
    }
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise RuntimeError(
            "Existing fine-tuning state is stale or malformed; refusing a provider call: "
            + ", ".join(failed)
        )
    if not ACCESS_CHECK_PATH.is_file() or json.loads(
        ACCESS_CHECK_PATH.read_text(encoding="utf-8")
    ) != access:
        raise RuntimeError("The preserved model-access check differs from the fine-tuning state")
    for state_key, path, label in (
        ("training_file", TRAIN_PATH, "training file"),
        ("validation_file", VALIDATION_PATH, "validation file"),
    ):
        if state_key in state:
            _validate_upload_record(state[state_key], path, label)


def _ensure_file_upload(
    client: OpenAI,
    state: Dict[str, Any],
    *,
    state_key: str,
    operation: str,
    path: Path,
    label: str,
) -> Dict[str, Any]:
    if state_key in state:
        _validate_upload_record(state[state_key], path, label)
        return state
    state["status"] = f"{operation}_started"
    state[f"{operation}_started_at_utc"] = _utc_now()
    _write_state(state)
    try:
        with path.open("rb") as handle:
            response = client.files.create(
                file=handle,
                purpose="fine-tune",
                extra_headers=_request_headers(state, operation),
            )
        record = _validate_upload_record(_serialize(response), path, label)
    except Exception as exc:
        state["status"] = f"{operation}_uncertain"
        state[f"{operation}_error"] = _error_details(exc)
        _write_state(state)
        raise RuntimeError(
            f"The {label} upload did not return validated metadata; rerun create to replay "
            "the same idempotent request"
        ) from exc
    state[state_key] = record
    state["status"] = f"{operation}_complete"
    state[f"{operation}_completed_at_utc"] = _utc_now()
    state.pop(f"{operation}_error", None)
    _write_state(state)
    return state


def _find_provider_job(client: OpenAI, state: Dict[str, Any]) -> Dict[str, Any] | None:
    metadata_filter = {"local_run_id": str(state["local_run_id"])}
    try:
        first_page = client.fine_tuning.jobs.list(limit=100, metadata=metadata_filter)
        pages = list(first_page.iter_pages()) if hasattr(first_page, "iter_pages") else [first_page]
        jobs = [
            _serialize(job)
            for page in pages
            for job in list(getattr(page, "data", []))
        ]
    except Exception as exc:
        state["status"] = "job_recovery_check_failed"
        state["job_recovery_error"] = _error_details(exc)
        _write_state(state)
        raise RuntimeError(
            "Could not check for an existing fine-tuning job; refusing to risk a duplicate paid job"
        ) from exc
    if any(
        not isinstance(job, dict)
        or not isinstance(job.get("metadata"), dict)
        or job["metadata"].get("local_run_id") != state["local_run_id"]
        for job in jobs
    ):
        state["status"] = "job_recovery_filter_mismatch"
        _write_state(state)
        raise RuntimeError("The provider fine-tuning job filter returned an unexpected record")
    job_ids = [str(job.get("id", "")) for job in jobs]
    state["job_recovery_check"] = {
        "checked_at_utc": _utc_now(),
        "metadata_filter": metadata_filter,
        "page_count": len(pages),
        "matching_job_ids": job_ids,
    }
    state.pop("job_recovery_error", None)
    if len(job_ids) != len(set(job_ids)) or any(not job_id for job_id in job_ids):
        state["status"] = "job_recovery_invalid_ids"
        _write_state(state)
        raise RuntimeError("The provider returned duplicate or empty fine-tuning job IDs")
    if len(jobs) > 1:
        state["status"] = "duplicate_provider_jobs"
        _write_state(state)
        raise RuntimeError("More than one provider job has the preserved local run ID")
    _write_state(state)
    return jobs[0] if jobs else None


def _accept_provider_job(
    state: Dict[str, Any],
    job: Any,
    *,
    evidence: str,
    status: str,
) -> Dict[str, Any]:
    serialized = _serialize(job)
    if not isinstance(serialized, dict):
        raise RuntimeError("The fine-tuning job response is not an object")
    state["job"] = serialized
    state["provider_job_validation"] = validate_provider_job(state, serialized)
    if not state["provider_job_validation"]["all_passed"]:
        state["status"] = "provider_job_mismatch"
        _write_state(state)
        raise RuntimeError("The provider fine-tuning job differs from the prespecified request")
    state["fine_tuning_job_authorization"] = {
        "confirmed": True,
        "confirmed_at_utc": _utc_now(),
        "evidence": evidence,
        "job_id": str(serialized["id"]),
    }
    state["status"] = status
    _write_state(state)
    return state


def _validated_state_for_refresh() -> Dict[str, Any]:
    """Validate all local provenance before retrieving a provider job."""
    state = _load_state()
    local_run_id = str(state.get("local_run_id", ""))
    request = _fine_tuning_request(DEFAULT_BASE_MODEL, DEFAULT_EPOCHS, local_run_id)
    development_report = validate_development_data()
    if not development_report["valid"]:
        raise RuntimeError("Training or validation data failed validation")
    file_validation = validate_fine_tuning_files()
    _validate_resume_state(state, request, development_report, file_validation)
    job = state.get("job")
    if not isinstance(job, dict) or not str(job.get("id", "")).strip():
        raise RuntimeError("The fine-tuning state has no validated provider job ID")
    validation = validate_provider_job(state, job)
    if not validation["all_passed"]:
        raise RuntimeError("The retained provider job differs from the prespecified request")
    return state


def create_job(client: OpenAI, base_model: str, epochs: int) -> Dict[str, Any]:
    assert_test_seed_not_retired(ACTIVE_PROTOCOL)
    assert_protocol_frozen(ACTIVE_PROTOCOL)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if str(base_model) != DEFAULT_BASE_MODEL or int(epochs) != DEFAULT_EPOCHS:
        raise RuntimeError(
            "The requested base model or epoch count differs from the active prespecified protocol"
        )
    development_report = validate_development_data()
    if not development_report["valid"]:
        raise RuntimeError("Training or validation data failed validation; refusing to upload")
    file_validation = validate_fine_tuning_files()

    if STATE_PATH.exists():
        state = _load_state()
        if state.get("status") == "job_creation_rejected":
            raise RuntimeError(
                "The provider definitively rejected this fine-tuning run; refusing another creation request"
            )
        local_run_id = str(state.get("local_run_id", ""))
        request = _fine_tuning_request(base_model, epochs, local_run_id)
        _validate_resume_state(state, request, development_report, file_validation)
        if "job" in state:
            validation = validate_provider_job(state, state["job"])
            if not validation["all_passed"]:
                raise RuntimeError("Existing provider job evidence no longer validates")
            return state
    else:
        access = check_model_access(client, base_model)
        if not access["model_retrievable"] or not access["model_id_matches"]:
            raise RuntimeError(
                f"The study credential cannot retrieve the prespecified base model {base_model}"
            )
        local_run_id = uuid.uuid4().hex
        request = _fine_tuning_request(base_model, epochs, local_run_id)
        request_ids = {
            operation: f"oda-ft-{local_run_id}-{operation.replace('_', '-')}"
            for operation in ("training_upload", "validation_upload", "job_creation")
        }
        state = {
            "created_at_utc": _utc_now(),
            "local_run_id": local_run_id,
            "request_ids": request_ids,
            "base_model_requested": base_model,
            "protocol": {
                "id": ACTIVE_PROTOCOL["protocol_id"],
                "version": ACTIVE_PROTOCOL["version"],
                "path": portable_path(DEFAULT_PROTOCOL_PATH),
                "sha256": _sha256(DEFAULT_PROTOCOL_PATH),
            },
            "generation_manifest": {
                "path": portable_path(GENERATION_MANIFEST_PATH),
                "sha256": _sha256(GENERATION_MANIFEST_PATH),
            },
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "training_dataset": {
                "path": portable_path(TRAIN_PATH),
                "sha256": _sha256(TRAIN_PATH),
            },
            "validation_dataset": {
                "path": portable_path(VALIDATION_PATH),
                "sha256": _sha256(VALIDATION_PATH),
            },
            "development_data_validation": development_report,
            "fine_tuning_file_validation": file_validation,
            "openai_sdk_version": importlib.metadata.version("openai"),
            "access_check": access,
            "request_without_file_ids": request,
            "status": "initializing",
            "job_events": [],
            "job_creation_attempts": [],
        }
        _write_state(state)

    state = _ensure_file_upload(
        client,
        state,
        state_key="training_file",
        operation="training_upload",
        path=TRAIN_PATH,
        label="training file",
    )
    state = _ensure_file_upload(
        client,
        state,
        state_key="validation_file",
        operation="validation_upload",
        path=VALIDATION_PATH,
        label="validation file",
    )

    recovered = _find_provider_job(client, state)
    if recovered is not None:
        for previous_attempt in reversed(state.get("job_creation_attempts", [])):
            if previous_attempt.get("outcome") in {"started", "uncertain"}:
                previous_attempt["outcome"] = "provider_job_recovered"
                previous_attempt["recovered_at_utc"] = _utc_now()
                break
        return _accept_provider_job(
            state,
            recovered,
            evidence="A metadata-filtered provider query recovered the preserved job",
            status="job_recovered",
        )

    attempt = {
        "started_at_utc": _utc_now(),
        "request_id": state["request_ids"]["job_creation"],
        "outcome": "started",
    }
    state["job_creation_attempts"].append(attempt)
    state["status"] = "job_creation_started"
    _write_state(state)
    try:
        job = client.fine_tuning.jobs.create(
            training_file=state["training_file"]["id"],
            validation_file=state["validation_file"]["id"],
            extra_headers=_request_headers(state, "job_creation"),
            **request,
        )
    except Exception as exc:
        attempt["completed_at_utc"] = _utc_now()
        attempt["error_details"] = _error_details(exc)
        definitive = _is_definitive_client_rejection(exc)
        attempt["outcome"] = "provider_rejected" if definitive else "uncertain"
        state["status"] = (
            "job_creation_rejected" if definitive else "job_creation_uncertain"
        )
        state["fine_tuning_job_authorization"] = {
            "confirmed": False,
            "attempted_at_utc": attempt["started_at_utc"],
            "evidence": (
                "The provider returned a definitive client rejection"
                if definitive
                else "No provider job object has yet been returned or recovered"
            ),
        }
        if definitive:
            state["provider_availability"] = {
                "available": False,
                "determined_at_utc": attempt["completed_at_utc"],
                "basis": "Definitive HTTP client rejection during job creation",
                "official_policy_url": OPENAI_FINE_TUNING_DEPRECATION_URL,
            }
        _write_state(state)
        if definitive:
            raise RuntimeError(
                "Fine-tuning job creation was definitively rejected by the provider; "
                "no retry was attempted"
            ) from exc
        raise RuntimeError(
            "Fine-tuning job creation had an uncertain outcome; rerun create to recover by "
            "metadata or replay the same idempotent request"
        ) from exc
    attempt["completed_at_utc"] = _utc_now()
    attempt["outcome"] = "provider_response_received"
    return _accept_provider_job(
        state,
        job,
        evidence="fine_tuning.jobs.create returned a validated job object",
        status="job_created",
    )


def reconcile_job_creation(client: OpenAI) -> Dict[str, Any]:
    """Check for a provider job without replaying the paid creation request."""
    state = _load_state()
    if isinstance(state.get("job"), dict):
        validation = validate_provider_job(state, state["job"])
        if not validation["all_passed"]:
            raise RuntimeError("Existing provider job evidence no longer validates")
        return state
    development_report = validate_development_data()
    file_validation = validate_fine_tuning_files()
    local_run_id = str(state.get("local_run_id", ""))
    request = _fine_tuning_request(DEFAULT_BASE_MODEL, DEFAULT_EPOCHS, local_run_id)
    _validate_resume_state(state, request, development_report, file_validation)
    recovered = _find_provider_job(client, state)
    if recovered is not None:
        for attempt in reversed(state.get("job_creation_attempts", [])):
            if attempt.get("outcome") in {"started", "uncertain"}:
                attempt["outcome"] = "provider_job_recovered"
                attempt["recovered_at_utc"] = _utc_now()
                break
        return _accept_provider_job(
            state,
            recovered,
            evidence="A metadata-filtered provider query recovered the preserved job",
            status="job_recovered",
        )

    attempts = state.get("job_creation_attempts")
    latest = attempts[-1] if isinstance(attempts, list) and attempts else None
    error_details = latest.get("error_details") if isinstance(latest, dict) else None
    status_code = error_details.get("status_code") if isinstance(error_details, dict) else None
    if status_code not in {400, 401, 403, 404, 422}:
        state["status"] = "job_creation_unresolved"
        _write_state(state)
        raise RuntimeError("No provider job was found, but the prior outcome is not definitive")
    latest["outcome"] = "provider_rejected"
    latest["reconciled_at_utc"] = _utc_now()
    latest["reconciliation"] = "No job matched the preserved local_run_id"
    state["status"] = "job_creation_rejected"
    state["fine_tuning_job_authorization"] = {
        "confirmed": False,
        "attempted_at_utc": latest.get("started_at_utc"),
        "evidence": "HTTP client rejection and metadata-filtered provider query found no job",
    }
    state["provider_availability"] = {
        "available": False,
        "determined_at_utc": latest["reconciled_at_utc"],
        "basis": "HTTP client rejection with no matching provider job",
        "official_policy_url": OPENAI_FINE_TUNING_DEPRECATION_URL,
    }
    _write_state(state)
    return state


def refresh_job(client: OpenAI) -> Dict[str, Any]:
    state = _validated_state_for_refresh()
    job_id = state["job"]["id"]
    job = client.fine_tuning.jobs.retrieve(job_id)
    event_page = client.fine_tuning.jobs.list_events(job_id, limit=100)
    pages = list(event_page.iter_pages())
    events = [event for page in pages for event in page.data]
    event_ids = [str(event.id) for event in events]
    final_has_more = bool(getattr(pages[-1], "has_more", False)) if pages else False
    state["last_refreshed_at_utc"] = _utc_now()
    state["job"] = _serialize(job)
    state["job_events"] = [_serialize(event) for event in events]
    state["job_events_pagination"] = {
        "initial_page_limit": 100,
        "page_count": len(pages),
        "event_count": len(events),
        "unique_event_count": len(set(event_ids)),
        "final_page_has_more": final_has_more,
        "complete": bool(pages) and not final_has_more and len(event_ids) == len(set(event_ids)),
    }
    state["provider_job_validation"] = validate_provider_job(state, state["job"])
    if not state["provider_job_validation"]["all_passed"]:
        state["status"] = "provider_job_mismatch"
        _write_state(state)
        raise RuntimeError("The retrieved fine-tuning job metadata differs from the prespecified request")
    _write_state(state)
    return state


def archive_result_files(client: OpenAI, state: Dict[str, Any]) -> Dict[str, Any]:
    result_dir = OUTPUT_DIR / "result_files"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_ids = state.get("job", {}).get("result_files", [])
    if (
        not isinstance(result_ids, list)
        or not result_ids
        or len(result_ids) != len(set(result_ids))
    ):
        raise RuntimeError("The fine-tuning job has malformed or duplicate result-file IDs")
    existing_archive = state.get("archived_result_files")
    if existing_archive is not None:
        if not isinstance(existing_archive, list) or len(existing_archive) != len(result_ids):
            raise RuntimeError("The retained fine-tuning result-file archive is malformed")
        archived_ids: set[str] = set()
        for record in existing_archive:
            if not isinstance(record, dict):
                raise RuntimeError("The retained fine-tuning result-file archive is malformed")
            file_id = str(record.get("file_id", ""))
            try:
                path = resolve_recorded_path(
                    record.get("local_path"),
                    permitted_root=result_dir,
                )
            except ValueError as exc:
                raise RuntimeError("A retained fine-tuning result file has an invalid path") from exc
            metadata = record.get("provider_metadata")
            if (
                file_id in archived_ids
                or file_id not in result_ids
                or not path.is_file()
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != _sha256(path)
                or not isinstance(metadata, dict)
                or metadata.get("id") != file_id
            ):
                raise RuntimeError("A retained fine-tuning result file failed verification")
            archived_ids.add(file_id)
        if archived_ids != set(result_ids):
            raise RuntimeError("The retained fine-tuning result-file IDs differ from the job")
        return state

    archived = []
    for file_id in result_ids:
        if not isinstance(file_id, str) or not file_id:
            raise RuntimeError("The fine-tuning job has an empty result-file ID")
        metadata = client.files.retrieve(file_id)
        response = client.files.content(file_id)
        content = response.read()
        if not isinstance(content, bytes):
            raise RuntimeError(f"Fine-tuning result file {file_id} did not return bytes")
        serialized_metadata = _serialize(metadata)
        if not isinstance(serialized_metadata, dict) or serialized_metadata.get("id") != file_id:
            raise RuntimeError(f"Fine-tuning result file {file_id} returned mismatched metadata")
        provider_bytes = serialized_metadata.get("bytes")
        if provider_bytes is not None and provider_bytes != len(content):
            raise RuntimeError(f"Fine-tuning result file {file_id} returned a mismatched byte count")
        original_name = Path(
            str(serialized_metadata.get("filename", "") or "result.bin")
        ).name
        file_id_digest = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:16]
        path = result_dir / f"{file_id_digest}_{original_name}"
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError(
                    f"Fine-tuning result file {file_id} conflicts with a retained partial download"
                )
        else:
            _write_bytes(path, content)
        archived.append({
            "file_id": file_id,
            "provider_metadata": serialized_metadata,
            "local_path": portable_path(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    state["result_files_archived_at_utc"] = _utc_now()
    state["archived_result_files"] = archived
    _write_state(state)
    return state


def wait_for_job(client: OpenAI, poll_seconds: int) -> Dict[str, Any]:
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be positive")
    while True:
        state = refresh_job(client)
        status = state["job"]["status"]
        print(f"{_utc_now()} fine-tuning job {state['job']['id']} status={status}", flush=True)
        if status in {"succeeded", "failed", "cancelled"}:
            if status == "succeeded":
                state = archive_result_files(client, state)
            return state
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("access", "create", "reconcile", "status", "wait"))
    parser.add_argument("--base-model", default=os.getenv("OPENAI_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.action in {"access", "create", "reconcile"}:
        assert_test_seed_not_retired(ACTIVE_PROTOCOL)
        assert_protocol_frozen(ACTIVE_PROTOCOL)
    load_authoritative_project_env()

    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in app/.env")
    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_API_BASE,
        timeout=180.0,
        max_retries=int(MODEL_SETTINGS["sdk_max_retries"]),
    )
    if args.action == "access":
        access = check_model_access(client, args.base_model)
        print(json.dumps(access, indent=2, sort_keys=True))
        return
    if args.action == "create":
        state = create_job(client, args.base_model, args.epochs)
    elif args.action == "reconcile":
        state = reconcile_job_creation(client)
    elif args.action == "status":
        state = refresh_job(client)
    else:
        state = wait_for_job(client, args.poll_seconds)
    if isinstance(state.get("job"), dict):
        print(json.dumps(state["job"], indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": state.get("status"),
                    "job_present": False,
                    "fine_tuning_job_authorization": state.get(
                        "fine_tuning_job_authorization"
                    ),
                    "provider_availability": state.get("provider_availability"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
