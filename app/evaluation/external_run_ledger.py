"""Crash-aware append-only provenance ledger for the canonical external run."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0"
ZERO_HASH = "0" * 64
TERMINAL_INVOCATION_EVENTS = {
    "invocation_completed": "completed",
    "invocation_failed": "failed",
    "invocation_interrupted": "interrupted",
}
TERMINAL_TRANSACTION_EVENTS = {
    "transaction_confirmed",
    "transaction_reverted",
    "transaction_reconciled_confirmed",
    "transaction_reconciled_reverted",
}


class LedgerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event_hash(value: Mapping[str, Any]) -> str:
    material = {key: item for key, item in value.items() if key != "event_hash"}
    return hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"Ledger line {line_number} is not valid JSON") from exc
            if not isinstance(event, dict):
                raise LedgerError(f"Ledger line {line_number} is not a JSON object")
            events.append(event)
    validate_hash_chain(events)
    return events


def validate_hash_chain(events: Sequence[Mapping[str, Any]]) -> None:
    previous_hash = ZERO_HASH
    previous_time: datetime | None = None
    for expected_sequence, event in enumerate(events, start=1):
        required = {
            "schema_version",
            "invocation_id",
            "sequence",
            "timestamp_utc",
            "event_type",
            "previous_event_hash",
            "details",
            "event_hash",
        }
        if set(event) != required:
            raise LedgerError(f"Ledger event {expected_sequence} has unexpected or missing fields")
        if event["schema_version"] != SCHEMA_VERSION:
            raise LedgerError(f"Ledger event {expected_sequence} has an unsupported schema")
        if event["sequence"] != expected_sequence:
            raise LedgerError(f"Ledger event {expected_sequence} has a nonconsecutive sequence")
        if event["previous_event_hash"] != previous_hash:
            raise LedgerError(f"Ledger event {expected_sequence} breaks the hash chain")
        if event["event_hash"] != _event_hash(event):
            raise LedgerError(f"Ledger event {expected_sequence} has an invalid event hash")
        if not isinstance(event["details"], dict):
            raise LedgerError(f"Ledger event {expected_sequence} details must be an object")
        if not str(event["invocation_id"]).strip() or not str(event["event_type"]).strip():
            raise LedgerError(f"Ledger event {expected_sequence} has an empty identifier")
        try:
            timestamp = datetime.strptime(
                str(event["timestamp_utc"]),
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise LedgerError(f"Ledger event {expected_sequence} has an invalid timestamp") from exc
        if previous_time is not None and timestamp < previous_time:
            raise LedgerError(f"Ledger event {expected_sequence} moves backward in time")
        previous_time = timestamp
        previous_hash = str(event["event_hash"])


def invocation_summaries(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[str] = []
    values: dict[str, dict[str, Any]] = {}
    for event in events:
        invocation_id = str(event["invocation_id"])
        event_type = str(event["event_type"])
        if invocation_id not in values:
            if event_type != "invocation_started":
                raise LedgerError(f"Invocation {invocation_id} does not begin with invocation_started")
            ordered.append(invocation_id)
            values[invocation_id] = {
                "invocation_id": invocation_id,
                "status": "active",
                "started_sequence": event["sequence"],
                "terminal_sequence": None,
                "run_directory": event["details"].get("run_directory"),
                "retry_of": event["details"].get("retry_of"),
                "event_count": 0,
                "unresolved_transaction_ids": set(),
            }
        summary = values[invocation_id]
        summary["event_count"] += 1
        if summary["status"] != "active" and event_type not in {
            "invocation_retry_linked",
        }:
            raise LedgerError(f"Invocation {invocation_id} has events after its terminal disposition")
        if event_type in TERMINAL_INVOCATION_EVENTS:
            if summary["status"] != "active":
                raise LedgerError(f"Invocation {invocation_id} has multiple terminal dispositions")
            summary["status"] = TERMINAL_INVOCATION_EVENTS[event_type]
            summary["terminal_sequence"] = event["sequence"]
        transaction_id = event["details"].get("transaction_id")
        if event_type == "transaction_unresolved" and transaction_id:
            summary["unresolved_transaction_ids"].add(str(transaction_id))
        if event_type in TERMINAL_TRANSACTION_EVENTS and transaction_id:
            summary["unresolved_transaction_ids"].discard(str(transaction_id))

    output: list[dict[str, Any]] = []
    for invocation_id in ordered:
        item = dict(values[invocation_id])
        item["unresolved_transaction_ids"] = sorted(item["unresolved_transaction_ids"])
        output.append(item)
    return output


def validate_semantics(
    events: Sequence[Mapping[str, Any]],
    *,
    require_all_terminal: bool = False,
) -> list[dict[str, Any]]:
    validate_hash_chain(events)
    summaries = invocation_summaries(events)
    if len(summaries) > 2:
        raise LedgerError("The approved protocol permits at most two canonical invocations")
    if summaries and summaries[0]["retry_of"] is not None:
        raise LedgerError("The first canonical invocation cannot be a retry")
    if len(summaries) == 2:
        if summaries[1]["retry_of"] != summaries[0]["invocation_id"]:
            raise LedgerError("The second invocation is not linked to the first")
        if summaries[0]["status"] not in {"failed", "interrupted"}:
            raise LedgerError("A retry requires a failed or interrupted first invocation")
    if require_all_terminal and any(item["status"] == "active" for item in summaries):
        raise LedgerError("Every canonical invocation must have a terminal disposition")

    transaction_states: dict[tuple[str, str], str] = {}
    pin_states: dict[tuple[str, str], set[str]] = {}
    model_states: dict[tuple[str, str], list[str]] = {}
    for event in events:
        invocation_id = str(event["invocation_id"])
        event_type = str(event["event_type"])
        details = event["details"]
        transaction_id = details.get("transaction_id")
        if event_type.startswith("transaction_"):
            if not transaction_id:
                raise LedgerError(f"{event_type} is missing a transaction_id")
            key = (invocation_id, str(transaction_id))
            previous = transaction_states.get(key)
            expected_previous = {
                "transaction_prepared": {None},
                "transaction_signed": {"transaction_prepared"},
                "transaction_submitted": {"transaction_signed"},
                "transaction_confirmed": {"transaction_submitted"},
                "transaction_reverted": {"transaction_submitted"},
                "transaction_unresolved": {"transaction_signed", "transaction_submitted"},
                "transaction_reconciliation_started": {"transaction_unresolved"},
                "transaction_reconciled_confirmed": {"transaction_reconciliation_started"},
                "transaction_reconciled_reverted": {"transaction_reconciliation_started"},
            }.get(event_type)
            if expected_previous is None:
                raise LedgerError(f"Unsupported transaction event: {event_type}")
            if previous not in expected_previous:
                raise LedgerError(
                    f"Transaction {transaction_id} cannot move from {previous} to {event_type}"
                )
            transaction_states[key] = event_type

        artifact_id = details.get("artifact_id")
        if event_type.startswith("pin_"):
            if not artifact_id:
                raise LedgerError(f"{event_type} is missing an artifact_id")
            pin_states.setdefault((invocation_id, str(artifact_id)), set()).add(event_type)

        call_id = details.get("call_id")
        if event_type.startswith("model_call_"):
            if not call_id:
                raise LedgerError(f"{event_type} is missing a call_id")
            model_states.setdefault((invocation_id, str(call_id)), []).append(event_type)

    for summary in summaries:
        if summary["status"] != "completed":
            continue
        invocation_id = summary["invocation_id"]
        transaction_terminal = {
            "transaction_confirmed",
            "transaction_reverted",
            "transaction_reconciled_confirmed",
            "transaction_reconciled_reverted",
        }
        incomplete_transactions = [
            transaction_id
            for (owner, transaction_id), state in transaction_states.items()
            if owner == invocation_id and state not in transaction_terminal
        ]
        if incomplete_transactions:
            raise LedgerError("A completed invocation contains incomplete transactions")
        for (owner, artifact_id), states in pin_states.items():
            if owner == invocation_id and "pin_accepted" in states and "pin_readback_verified" not in states:
                raise LedgerError(
                    f"A completed invocation did not verify accepted pin {artifact_id}"
                )
        for (owner, call_id), states in model_states.items():
            if owner != invocation_id:
                continue
            if states.count("model_call_started") != 1:
                raise LedgerError(f"Model call {call_id} does not have exactly one start event")
            terminal_count = states.count("model_call_succeeded") + states.count("model_call_failed")
            if terminal_count != 1:
                raise LedgerError(f"Model call {call_id} does not have exactly one terminal event")
    return summaries


class _ProcessLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            self.handle.close()
            self.handle = None
            raise LedgerError("Another canonical workflow process holds the ledger lock") from exc

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


@dataclass
class ExternalRunLedger:
    root: Path
    invocation_id: str
    run_dir: Path
    retry_of: str | None
    _lock: _ProcessLock
    _terminal: bool = False

    @property
    def ledger_path(self) -> Path:
        return self.root / "invocations.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / "invocation_state.json"

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        mode: str = "full",
        retry_failed: bool = False,
        retry_reason: str | None = None,
    ) -> "ExternalRunLedger":
        root.mkdir(parents=True, exist_ok=True)
        process_lock = _ProcessLock(root / ".ledger.lock")
        process_lock.acquire()
        try:
            ledger_path = root / "invocations.jsonl"
            events = read_ledger(ledger_path)
            summaries = validate_semantics(events)
            if any(item["status"] == "completed" for item in summaries):
                raise LedgerError("A completed canonical invocation already exists")
            if any(item["unresolved_transaction_ids"] for item in summaries):
                raise LedgerError("An unresolved transaction blocks another canonical invocation")
            normalized_reason = str(retry_reason or "").strip()
            if not summaries:
                if retry_failed or normalized_reason:
                    raise LedgerError("The first invocation cannot use retry arguments")
                retry_of = None
            else:
                if len(summaries) >= 2:
                    raise LedgerError("The approved one-retry limit has been reached")
                if not retry_failed or not normalized_reason:
                    raise LedgerError("A retry requires --retry-failed and a nonempty reason")
                prior = summaries[-1]
                retry_of = prior["invocation_id"]
                if prior["status"] == "active":
                    _append_event(
                        ledger_path,
                        retry_of,
                        "invocation_interrupted",
                        {"reason": "A later explicit retry detected an unterminated process."},
                    )
                elif prior["status"] not in {"failed", "interrupted"}:
                    raise LedgerError("Only a failed or interrupted invocation can be retried")

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            invocation_id = f"{stamp}-{secrets.token_hex(4)}"
            run_dir = root / invocation_id
            run_dir.mkdir(parents=False, exist_ok=False)
            instance = cls(
                root=root,
                invocation_id=invocation_id,
                run_dir=run_dir,
                retry_of=retry_of,
                _lock=process_lock,
            )
            instance.record(
                "invocation_started",
                {
                    "mode": mode,
                    "run_directory": run_dir.name,
                    "retry_of": retry_of,
                    "retry_reason": normalized_reason or None,
                },
            )
            return instance
        except Exception:
            process_lock.release()
            raise

    def record(self, event_type: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._terminal:
            raise LedgerError("Cannot append to a terminal invocation")
        event = _append_event(
            self.ledger_path,
            self.invocation_id,
            event_type,
            dict(details or {}),
        )
        self._refresh_state()
        return event

    def complete(self, details: Mapping[str, Any] | None = None) -> None:
        self.validate_completion_ready(details)
        self.record("invocation_completed", details)
        self._terminal = True
        validate_semantics(read_ledger(self.ledger_path), require_all_terminal=True)
        self._refresh_state()

    def validate_completion_ready(self, details: Mapping[str, Any] | None = None) -> None:
        events = read_ledger(self.ledger_path)
        candidate: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "invocation_id": self.invocation_id,
            "sequence": len(events) + 1,
            "timestamp_utc": _utc_now(),
            "event_type": "invocation_completed",
            "previous_event_hash": events[-1]["event_hash"] if events else ZERO_HASH,
            "details": dict(details or {}),
        }
        candidate["event_hash"] = _event_hash(candidate)
        validate_semantics([*events, candidate], require_all_terminal=True)

    def fail(self, exc: BaseException) -> None:
        if self._terminal:
            return
        event_type = "invocation_interrupted" if isinstance(exc, KeyboardInterrupt) else "invocation_failed"
        self.record(event_type, {"error_type": type(exc).__name__})
        self._terminal = True
        self._refresh_state()

    def close(self) -> None:
        self._lock.release()

    def summaries(self) -> list[dict[str, Any]]:
        return validate_semantics(read_ledger(self.ledger_path))

    def _refresh_state(self) -> None:
        events = read_ledger(self.ledger_path)
        summaries = validate_semantics(events)
        _write_json_atomic(
            self.state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at_utc": _utc_now(),
                "event_count": len(events),
                "last_event_hash": events[-1]["event_hash"] if events else ZERO_HASH,
                "invocations": summaries,
            },
        )

    def __enter__(self) -> "ExternalRunLedger":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        try:
            if exc is not None:
                self.fail(exc)
            elif not self._terminal:
                self.fail(LedgerError("Invocation exited without an explicit completion"))
        finally:
            self.close()
        return False


def _append_event(
    path: Path,
    invocation_id: str,
    event_type: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    events = read_ledger(path)
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "sequence": len(events) + 1,
        "timestamp_utc": _utc_now(),
        "event_type": str(event_type),
        "previous_event_hash": events[-1]["event_hash"] if events else ZERO_HASH,
        "details": dict(details),
    }
    try:
        _canonical_bytes(event)
    except (TypeError, ValueError) as exc:
        raise LedgerError("Ledger event details are not JSON serializable") from exc
    event["event_hash"] = _event_hash(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event
