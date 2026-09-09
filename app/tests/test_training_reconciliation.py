from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import reconcile_local_training_state as reconciliation


def test_reconcile_closes_only_invocation_followed_by_checkpoint_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "reconcile.py"
    script.write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(reconciliation, "SCRIPT_PATH", script)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "invocations": [
                    {
                        "started_at_utc": "2026-09-07T00:00:00Z",
                        "resume_checkpoint": None,
                    },
                    {
                        "started_at_utc": "2026-09-07T01:00:00Z",
                        "resume_checkpoint": "checkpoint-200",
                        "completed_at_utc": "2026-09-07T02:00:00Z",
                        "outcome": "completed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    state = reconciliation.reconcile(state_path)
    assert state["invocations"][0]["outcome"] == "interrupted_before_checkpoint_resume"
    assert state["invocations"][0]["completed_at_utc"] == "2026-09-07T01:00:00Z"
    assert state["invocation_reconciliation"]["invocation_indices_zero_based"] == [0]
    assert reconciliation.reconcile(state_path) == state


def test_reconcile_rejects_unresolved_final_invocation(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "invocations": [
                    {"started_at_utc": "2026-09-07T00:00:00Z", "resume_checkpoint": None}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="final training invocation"):
        reconciliation.reconcile(state_path)
