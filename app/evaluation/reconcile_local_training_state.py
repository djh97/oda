"""Close interrupted local-training invocation records after a checkpoint resume."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.train_local_lora import STATE_PATH


SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def reconcile(path: Path = STATE_PATH) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("status") != "completed":
        raise RuntimeError("Local training must be complete before invocation reconciliation")
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise RuntimeError("Local training state has no invocation records")

    unresolved = [
        index
        for index, invocation in enumerate(invocations)
        if isinstance(invocation, dict) and not invocation.get("outcome")
    ]
    if not unresolved:
        existing = state.get("invocation_reconciliation")
        if len(invocations) > 1 and not isinstance(existing, dict):
            raise RuntimeError("Resumed training has no invocation reconciliation record")
        return state

    reconciled: list[int] = []
    for index in unresolved:
        if index + 1 >= len(invocations):
            raise RuntimeError("The final training invocation has no terminal outcome")
        current = invocations[index]
        following = invocations[index + 1]
        if not isinstance(current, dict) or not isinstance(following, dict):
            raise RuntimeError("Local training invocation record is malformed")
        if not following.get("resume_checkpoint") or not following.get("started_at_utc"):
            raise RuntimeError("An unresolved invocation was not followed by a checkpoint resume")
        current["completed_at_utc"] = following["started_at_utc"]
        current["outcome"] = "interrupted_before_checkpoint_resume"
        reconciled.append(index)

    state["invocation_reconciliation"] = {
        "reconciled_at_utc": _utc_now(),
        "script_sha256": _sha256(SCRIPT_PATH),
        "invocation_indices_zero_based": reconciled,
        "basis": "Each open invocation is immediately followed by a recorded checkpoint resume.",
        "training_hyperparameters_changed": False,
    }
    _write_json(path, state)
    return state


def main() -> None:
    state = reconcile()
    print(
        "Reconciled local training invocations "
        f"({len(state['invocations'])} invocation records)."
    )


if __name__ == "__main__":
    main()
