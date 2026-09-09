"""Exercise one validation case before a locked test-set model run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from evaluation.project_environment import load_authoritative_project_env
from evaluation.manage_fine_tuning import validate_development_data
from evaluation.synthetic_dataset import DATASET_DIR
from src.decision_guard import apply_guarded_policy
from src.llm_client import SYSTEM_PROMPT, NoteReviewResult, call_note_review
from src.local_llm_client import LocalNoteReviewClient, validate_local_training_state
from src.policy import (
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
    rank_recipients_baseline,
)


APP_DIR = Path(__file__).resolve().parents[1]
VALIDATION_PATH = DATASET_DIR / "validation_cases.jsonl"
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "model" / "smoke_tests"
LOCAL_TRAINING_STATE_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_training.json"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_SETTINGS = load_protocol()["model_evaluation"]
ALLOWED_CONDITIONS = {"untuned_preflight", "fine_tuned_preflight", "openai_preflight"}
SMOKE_SOURCE_PATHS = {
    "smoke_test": SCRIPT_PATH,
    "llm_client": APP_DIR / "src" / "llm_client.py",
    "policy": APP_DIR / "src" / "policy.py",
    "decision_guard": APP_DIR / "src" / "decision_guard.py",
    "schemas": APP_DIR / "src" / "schemas.py",
    "project_environment": APP_DIR / "evaluation" / "project_environment.py",
    "local_training": APP_DIR / "evaluation" / "train_local_lora.py",
    "local_llm_client": APP_DIR / "src" / "local_llm_client.py",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_label(value: str) -> str:
    label = "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
    return label or "model"


def _write_json(path: Path, value: Dict[str, Any]) -> None:
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


def smoke_source_hashes() -> Dict[str, str]:
    return {name: _sha256(path) for name, path in SMOKE_SOURCE_PATHS.items()}


def _load_validation_case(case_index: int) -> Dict[str, Any]:
    if case_index < 0:
        raise ValueError("case_index cannot be negative")
    with VALIDATION_PATH.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(line for line in handle if line.strip()):
            if index == case_index:
                return json.loads(line)
    raise ValueError(f"Validation case index {case_index} is out of range")


def _validate_condition_model(condition: str, model_id: str) -> None:
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(ALLOWED_CONDITIONS)}")
    if condition == "untuned_preflight":
        expected_model = str(MODEL_SETTINGS["base_model_snapshot"])
    elif condition == "fine_tuned_preflight":
        state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
        expected_model = str(state["adapter"]["model_id"])
    else:
        expected_model = str(MODEL_SETTINGS["hosted_comparator"]["model_id"])
    if not expected_model or model_id != expected_model:
        raise RuntimeError(f"{condition} must use its preserved expected model")


def _condition_runtime_settings(condition: str) -> Dict[str, Any]:
    if condition == "openai_preflight":
        hosted = MODEL_SETTINGS["hosted_comparator"]
        return {
            "temperature": hosted["temperature"],
            "seed": hosted["seed"],
            "provider": hosted["provider"],
            "do_sample": False,
            "maximum_new_tokens": hosted["maximum_completion_tokens"],
            "endpoint": hosted["endpoint"],
            "sdk_max_retries": hosted["sdk_max_retries"],
            "store": hosted["store"],
        }
    return {
        "temperature": MODEL_SETTINGS["temperature"],
        "seed": MODEL_SETTINGS["seed"],
        "provider": MODEL_SETTINGS["provider"],
        "do_sample": MODEL_SETTINGS["do_sample"],
        "maximum_new_tokens": MODEL_SETTINGS["maximum_new_tokens"],
        "endpoint": "local_transformers_generate",
        "sdk_max_retries": 0,
        "store": False,
    }


def run_smoke_test(
    model_id: str,
    condition: str,
    api_key: str,
    *,
    case_index: int = 0,
    caller: Callable[..., NoteReviewResult] = call_note_review,
) -> Path:
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    _validate_condition_model(condition, model_id)
    runtime_settings = _condition_runtime_settings(condition)
    output_path = OUTPUT_DIR / f"{_safe_label(condition)}.json"
    if output_path.exists():
        raise RuntimeError(
            f"Smoke-test output already exists at {output_path}; preserve it rather than "
            "repeating a billable preflight"
        )
    validation = validate_development_data()
    if not validation["valid"]:
        raise RuntimeError("Training or validation data failed validation")
    case = _load_validation_case(case_index)
    try:
        result = caller(
            model_id=model_id,
            api_key=api_key,
            case_id=str(case["case_id"]),
            organ_type=str(case["organ_type"]),
            recipients=case["recipients"],
            temperature=float(runtime_settings["temperature"]),
            seed=int(runtime_settings["seed"]),
        )
    except Exception as exc:
        metadata = getattr(exc, "model_run_metadata", None)
        record = {
            "purpose": "Validation-split service and schema smoke test; not a test-set outcome",
            "status": "failure",
            "completed_at_utc": _utc_now(),
            "condition": condition,
            "case_index": case_index,
            "case_id": case["case_id"],
            "organ_type": case["organ_type"],
            "model_id_requested": model_id,
            **runtime_settings,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_model_output": getattr(exc, "raw_text", None),
            "model_run": metadata.model_dump(mode="json") if metadata is not None else None,
            "validation_file_sha256": _sha256(VALIDATION_PATH),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "script_sha256": _sha256(SCRIPT_PATH),
            "implementation_sha256": smoke_source_hashes(),
            "software_environment": {
                "python": platform.python_version(),
                "openai": importlib.metadata.version("openai"),
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
                "pydantic": importlib.metadata.version("pydantic"),
            },
        }
        _write_json(output_path, record)
        return output_path
    if result.metadata.model_id != model_id:
        raise RuntimeError("The provider-returned model identifier differs from the smoke-test request")
    ranked = rank_recipients_baseline(case["donor"], case["recipients"])
    guarded = apply_guarded_policy(
        donor_id=int(case["donor"]["donor_id"]),
        ranked=ranked,
        review=result.review,
        expected_recipient_ids=[int(item["recipient_id"]) for item in case["recipients"]],
    )
    reference_by_id = {
        int(item["recipient_id"]): (
            str(item["reference_note_state"]),
            set(item["reference_evidence_codes"]),
        )
        for item in case["recipients"]
    }
    observed_by_id = {
        int(item.recipient_id): (item.state.value, {code.value for code in item.evidence_codes})
        for item in result.review.assessments
    }
    state_matches = [
        reference_by_id[recipient_id][0] == observed_by_id[recipient_id][0]
        for recipient_id in sorted(reference_by_id)
    ]
    evidence_matches = [
        reference_by_id[recipient_id][1] == observed_by_id[recipient_id][1]
        for recipient_id in sorted(reference_by_id)
    ]
    record = {
        "purpose": "Validation-split service and schema smoke test; not a test-set outcome",
        "status": "success",
        "completed_at_utc": _utc_now(),
        "condition": condition,
        "case_index": case_index,
        "case_id": case["case_id"],
        "organ_type": case["organ_type"],
        "model_id_requested": model_id,
        **runtime_settings,
        "model_run": result.metadata.model_dump(mode="json"),
        "raw_model_output": result.raw_text,
        "parsed_review": result.review.model_dump(mode="json"),
        "guarded_decision": guarded.model_dump(mode="json"),
        "state_correct_count": sum(state_matches),
        "state_denominator": len(state_matches),
        "evidence_exact_count": sum(evidence_matches),
        "evidence_denominator": len(evidence_matches),
        "primary_reference_conforms": (
            guarded.primary_recipient_id == int(case["reference"]["primary_recipient_id"])
        ),
        "backup_reference_conforms": (
            guarded.backup_recipient_id == int(case["reference"]["backup_recipient_id"])
        ),
        "validation_file_sha256": _sha256(VALIDATION_PATH),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "script_sha256": _sha256(SCRIPT_PATH),
        "implementation_sha256": smoke_source_hashes(),
        "software_environment": {
            "python": platform.python_version(),
            "openai": importlib.metadata.version("openai"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "pydantic": importlib.metadata.version("pydantic"),
        },
    }
    _write_json(output_path, record)
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(ALLOWED_CONDITIONS))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    load_authoritative_project_env()
    if args.condition == "untuned_preflight":
        expected_model = str(MODEL_SETTINGS["base_model_snapshot"])
        adapter = False
        api_key = ""
        client: Callable[..., NoteReviewResult] = LocalNoteReviewClient(
            expected_model,
            adapter=adapter,
        )
    elif args.condition == "fine_tuned_preflight":
        training_state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
        expected_model = str(training_state["adapter"]["model_id"])
        adapter = True
        api_key = ""
        client = LocalNoteReviewClient(expected_model, adapter=adapter)
    else:
        expected_model = str(MODEL_SETTINGS["hosted_comparator"]["model_id"])
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from app/.env")
        client = call_note_review
    model_id = str(args.model_id or expected_model).strip()
    if not model_id:
        raise RuntimeError("The frozen local condition has no model identifier")
    output = run_smoke_test(
        model_id,
        args.condition,
        api_key,
        case_index=args.case_index,
        caller=client,
    )
    print(f"Saved validation smoke-test record to {output}")


if __name__ == "__main__":
    main()
