from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.external_run_ledger import (
    ExternalRunLedger,
    LedgerError,
    read_ledger,
    validate_semantics,
)


def test_ledger_hash_chain_state_and_completion(tmp_path: Path) -> None:
    ledger = ExternalRunLedger.start(tmp_path)
    ledger.record(
        "pin_prepared",
        {"artifact_id": "donor", "plaintext_sha256": "a" * 64},
    )
    ledger.record("pin_accepted", {"artifact_id": "donor", "cid": "bafy-test"})
    ledger.record("pin_readback_started", {"artifact_id": "donor", "attempt": 1})
    ledger.record("pin_readback_verified", {"artifact_id": "donor", "attempt": 1})
    ledger.record("model_call_started", {"call_id": "demo"})
    ledger.record("model_call_succeeded", {"call_id": "demo", "response_id": "response"})
    ledger.record("transaction_prepared", {"transaction_id": "tx-001"})
    ledger.record("transaction_signed", {"transaction_id": "tx-001"})
    ledger.record("transaction_submitted", {"transaction_id": "tx-001"})
    ledger.record("transaction_confirmed", {"transaction_id": "tx-001"})
    ledger.complete({"transaction_count": 1})
    ledger.close()

    events = read_ledger(tmp_path / "invocations.jsonl")
    summaries = validate_semantics(events, require_all_terminal=True)
    assert summaries[0]["status"] == "completed"
    state = json.loads((tmp_path / "invocation_state.json").read_text(encoding="utf-8"))
    assert state["event_count"] == len(events)
    assert state["last_event_hash"] == events[-1]["event_hash"]
    with pytest.raises(LedgerError, match="completed canonical"):
        ExternalRunLedger.start(tmp_path)


def test_one_explicit_retry_preserves_first_invocation(tmp_path: Path) -> None:
    first = ExternalRunLedger.start(tmp_path)
    first.fail(RuntimeError("secret-bearing message is never recorded"))
    first.close()

    second = ExternalRunLedger.start(
        tmp_path,
        retry_failed=True,
        retry_reason="Provider outage after preflight",
    )
    assert second.retry_of == first.invocation_id
    second.fail(RuntimeError("second failure"))
    second.close()
    summaries = validate_semantics(read_ledger(tmp_path / "invocations.jsonl"), require_all_terminal=True)
    assert [item["status"] for item in summaries] == ["failed", "failed"]
    ledger_text = (tmp_path / "invocations.jsonl").read_text(encoding="utf-8")
    assert "secret-bearing" not in ledger_text
    with pytest.raises(LedgerError, match="one-retry limit"):
        ExternalRunLedger.start(
            tmp_path,
            retry_failed=True,
            retry_reason="Not permitted",
        )


def test_unresolved_transaction_blocks_retry(tmp_path: Path) -> None:
    first = ExternalRunLedger.start(tmp_path)
    first.record("transaction_prepared", {"transaction_id": "tx-001"})
    first.record("transaction_signed", {"transaction_id": "tx-001"})
    first.record("transaction_unresolved", {"transaction_id": "tx-001"})
    first.fail(RuntimeError("timeout"))
    first.close()

    with pytest.raises(LedgerError, match="unresolved transaction"):
        ExternalRunLedger.start(
            tmp_path,
            retry_failed=True,
            retry_reason="Must reconcile first",
        )


def test_tampering_breaks_hash_chain(tmp_path: Path) -> None:
    ledger = ExternalRunLedger.start(tmp_path)
    ledger.fail(RuntimeError("failure"))
    ledger.close()
    path = tmp_path / "invocations.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["details"]["mode"] = "tampered"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="invalid event hash"):
        read_ledger(path)
