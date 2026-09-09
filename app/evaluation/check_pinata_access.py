"""Perform a read-only Pinata authentication preflight without exposing credentials."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from evaluation.project_environment import load_authoritative_project_env
from src.policy import assert_protocol_frozen


APP_DIR = Path(__file__).resolve().parents[1]
TEST_AUTHENTICATION_URL = "https://api.pinata.cloud/data/testAuthentication"
OUTPUT_PATH = (
    APP_DIR
    / "pipeline-output"
    / "current"
    / "setup"
    / "preflight"
    / "pinata_access_check.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, value: dict[str, Any]) -> None:
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


def check_access(jwt: str, *, output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    if not str(jwt or "").strip():
        raise RuntimeError("PINATA_JWT is missing")
    checked_at = _utc_now()
    try:
        response = requests.get(
            TEST_AUTHENTICATION_URL,
            headers={"Authorization": f"Bearer {jwt}", "Accept": "application/json"},
            timeout=30,
        )
        status_code = int(response.status_code)
        try:
            body = response.json()
        except ValueError:
            body = None
        message = body.get("message") if isinstance(body, dict) else None
        success = status_code == 200 and isinstance(message, str) and bool(message.strip())
        error = None if success else {"error_type": "AuthenticationRejected"}
    except requests.RequestException as exc:
        status_code = None
        message = None
        success = False
        error = {"error_type": type(exc).__name__}
    record = {
        "schema_version": "1.0",
        "checked_at_utc": checked_at,
        "endpoint": TEST_AUTHENTICATION_URL,
        "method": "GET",
        "success": success,
        "http_status": status_code,
        "response_message_sha256": (
            hashlib.sha256(message.encode("utf-8")).hexdigest()
            if isinstance(message, str)
            else None
        ),
        "error": error,
        "requests_version": importlib.metadata.version("requests"),
    }
    _write_json(output_path, record)
    return record


def main() -> None:
    assert_protocol_frozen()
    load_authoritative_project_env()
    record = check_access(str(os.getenv("PINATA_JWT", "")))
    print(f"Pinata authentication success: {str(record['success']).lower()}")
    print(f"Redacted preflight record: {OUTPUT_PATH}")
    if not record["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
