"""Verify access to the pinned hosted comparator without opening test data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI

from evaluation.project_environment import load_authoritative_project_env
from src.llm_client import OPENAI_API_BASE
from src.policy import (
    DEFAULT_PROTOCOL_PATH,
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
)


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "openai_comparator_access_check.json"
)
TEST_LOCK_PATH = APP_DIR / "pipeline-output" / "current" / "evaluation" / "test_lock.json"
SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def run() -> dict[str, Any]:
    assert_test_seed_not_retired()
    assert_protocol_frozen()
    if TEST_LOCK_PATH.exists():
        raise RuntimeError("Hosted-comparator preflight must precede the held-out test lock")
    if OUTPUT_PATH.exists():
        raise RuntimeError("The hosted-comparator access preflight is already preserved")
    load_authoritative_project_env()
    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from app/.env")

    protocol = load_protocol()
    settings = protocol["model_evaluation"]["hosted_comparator"]
    model_id = str(settings["model_id"])
    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_API_BASE,
        timeout=float(settings["timeout_seconds"]),
        max_retries=int(settings["sdk_max_retries"]),
    )
    checked_at = _utc_now()
    try:
        model = client.models.retrieve(model_id)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        details = [type(exc).__name__]
        if status is not None:
            details.append(f"status={status}")
        if request_id:
            details.append(f"request_id={request_id}")
        record = {
            "schema_version": "1.0",
            "status": "failure",
            "purpose": "Authenticated model-access check using no study case data",
            "checked_at_utc": checked_at,
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"],
            "model_id_requested": model_id,
            "model_retrievable": False,
            "error": ", ".join(details),
            "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
            "script_sha256": _sha256(SCRIPT_PATH),
            "openai_sdk_version": importlib.metadata.version("openai"),
        }
        _write_json(OUTPUT_PATH, record)
        return record

    returned = str(getattr(model, "id", "") or "")
    record = {
        "schema_version": "1.0",
        "status": "passed" if returned == model_id else "failure",
        "purpose": "Authenticated model-access check using no study case data",
        "checked_at_utc": checked_at,
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "model_id_requested": model_id,
        "model_id_returned": returned,
        "model_id_matches": returned == model_id,
        "model_retrievable": True,
        "owned_by": str(getattr(model, "owned_by", "") or "") or None,
        "test_case_data_opened": False,
        "held_out_test_lock_present": False,
        "runtime_settings": settings,
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "script_sha256": _sha256(SCRIPT_PATH),
        "openai_sdk_version": importlib.metadata.version("openai"),
    }
    _write_json(OUTPUT_PATH, record)
    return record


def main() -> None:
    record = run()
    print(
        "Hosted-comparator access preflight "
        f"{record['status']} for {record['model_id_requested']}."
    )


if __name__ == "__main__":
    main()
