"""Bind terminal model preflights into the protocol freeze before test lock."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.run_model_evaluation import (
    TEST_LOCK_PATH,
    _pretest_artifact_hashes,
    _validate_pretest_terminal_records,
)
from src.policy import DEFAULT_PROTOCOL_FREEZE_PATH, assert_protocol_frozen


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def finalize() -> dict[str, Any]:
    if TEST_LOCK_PATH.exists():
        raise RuntimeError("Cannot finalize model preflights after the held-out test lock")
    freeze = assert_protocol_frozen()
    if freeze.get("protocol_version") != "1.2.0":
        raise RuntimeError("Model-preflight finalization expects protocol version 1.2.0")
    if freeze.get("hosted_comparator_preflight_pending") is not True:
        raise RuntimeError("Model preflights have already been finalized")
    hashes = _pretest_artifact_hashes()
    missing = sorted(name for name, value in hashes.items() if not value)
    if missing:
        raise RuntimeError("Model preflight artifacts are missing: " + ", ".join(missing))
    _validate_pretest_terminal_records()

    freeze["hosted_comparator_preflight_pending"] = False
    freeze["model_pretests_finalized_at_utc"] = _utc_now()
    freeze["model_pretest_artifact_sha256"] = hashes
    _write_json(DEFAULT_PROTOCOL_FREEZE_PATH, freeze)
    assert_protocol_frozen()
    return freeze


def main() -> None:
    freeze = finalize()
    print(
        "Finalized model pretests for protocol "
        f"{freeze['protocol_id']} version {freeze['protocol_version']}."
    )


if __name__ == "__main__":
    main()
