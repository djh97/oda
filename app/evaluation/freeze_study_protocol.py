"""Atomically freeze the study protocol with a fresh, unreported test seed."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.policy import RETIRED_TEST_SEEDS


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parents[1]
JOURNAL_DIR = WORKSPACE_DIR / "Frontiers_Medical_Technology_2026-09-06"
PROTOCOL_PATH = APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json"
STUDY_PROTOCOL_PATH = JOURNAL_DIR / "docs" / "STUDY_PROTOCOL.md"
MANUSCRIPT_PATH = JOURNAL_DIR / "Manuscript.tex"
FREEZE_PATH = APP_DIR / "pipeline-output" / "current" / "protocol" / "protocol_freeze.json"
FINE_TUNING_STATE_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "fine_tuning_job.json"
TEST_LOCK_PATH = APP_DIR / "pipeline-output" / "current" / "evaluation" / "test_lock.json"

EXPECTED_ACTIVE_SEED = 2026090717
MIN_REPLACEMENT_SEED = 1_000_000_000
MAX_REPLACEMENT_SEED = 2_147_483_647

OLD_STUDY_PROTOCOL_PARAGRAPH = """The development test seed `2026090703` was inspected while the note-composition
logic and validators were being designed. It was retired before any locked
classical or model evaluation and is not eligible for reporting. The later
candidate seed `2026090717` was exposed during source inspection and
is also retired. It remains in the machine-readable file only as a fail-closed
placeholder until the authors approve this protocol and a replacement is
inserted atomically. Runtime checks prevent data generation or external model
execution while either retired seed is active."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _stage_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.freeze.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _replace_exact(text: str, old: str, new: str, *, count: int, label: str) -> str:
    observed = text.count(old)
    if observed != count:
        raise RuntimeError(f"Expected {count} {label} occurrence(s), found {observed}")
    return text.replace(old, new)


def _existing_seed_values(protocol: Mapping[str, Any]) -> set[int]:
    values: set[int] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key and "seed" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)

    visit(protocol)
    return values


def _replacement_seed(protocol: Mapping[str, Any]) -> int:
    prohibited = _existing_seed_values(protocol).union(RETIRED_TEST_SEEDS)
    width = MAX_REPLACEMENT_SEED - MIN_REPLACEMENT_SEED + 1
    while True:
        candidate = MIN_REPLACEMENT_SEED + secrets.randbelow(width)
        if candidate not in prohibited:
            return candidate


def freeze_protocol(
    *,
    replacement_seed: int | None = None,
    protocol_path: Path = PROTOCOL_PATH,
    study_protocol_path: Path = STUDY_PROTOCOL_PATH,
    manuscript_path: Path = MANUSCRIPT_PATH,
    freeze_path: Path = FREEZE_PATH,
    fine_tuning_state_path: Path = FINE_TUNING_STATE_PATH,
    test_lock_path: Path = TEST_LOCK_PATH,
) -> dict[str, Any]:
    blockers = [path for path in (fine_tuning_state_path, test_lock_path) if path.exists()]
    if blockers:
        raise RuntimeError("The protocol cannot be replaced after fine-tuning or test locking has started")
    if freeze_path.exists():
        raise RuntimeError("A protocol freeze record already exists; refusing to replace it")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    active_seed = protocol["dataset"]["splits"]["test"]["seed"]
    if active_seed != EXPECTED_ACTIVE_SEED:
        raise RuntimeError("The active test seed is not the expected retired placeholder")

    candidate = _replacement_seed(protocol) if replacement_seed is None else replacement_seed
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError("The replacement test seed must be an integer")
    if not MIN_REPLACEMENT_SEED <= candidate <= MAX_REPLACEMENT_SEED:
        raise ValueError("The replacement test seed is outside the permitted range")
    if candidate in _existing_seed_values(protocol).union(RETIRED_TEST_SEEDS):
        raise ValueError("The replacement test seed is retired or already used by the protocol")

    updated_protocol = deepcopy(protocol)
    updated_protocol["dataset"]["splits"]["test"]["seed"] = candidate
    history = updated_protocol.setdefault("pre_freeze_history", [])
    recorded = {int(item["retired_test_seed"]) for item in history}
    if EXPECTED_ACTIVE_SEED not in recorded:
        history.append(
            {
                "date": "2026-09-07",
                "retired_test_seed": EXPECTED_ACTIVE_SEED,
                "reason": (
                    "The candidate split was exposed during source inspection. It was replaced "
                    "before provider upload, fine-tuning, or any locked model evaluation."
                ),
            }
        )
    recorded = {int(item["retired_test_seed"]) for item in history}
    if not set(RETIRED_TEST_SEEDS).issubset(recorded):
        raise RuntimeError("The protocol history does not contain every retired test seed")

    manuscript = manuscript_path.read_text(encoding="utf-8")
    manuscript = _replace_exact(
        manuscript,
        str(EXPECTED_ACTIVE_SEED),
        str(candidate),
        count=2,
        label="retired seed in the manuscript",
    )

    study_protocol = study_protocol_path.read_text(encoding="utf-8")
    study_protocol = _replace_exact(
        study_protocol,
        "Pending author-approved replacement",
        str(candidate),
        count=1,
        label="pending test-seed marker",
    )
    new_protocol_paragraph = f"""The development test seed `2026090703` was inspected while the note-composition
logic and validators were being designed. It was retired before any locked
classical or model evaluation and is not eligible for reporting. The later
candidate seed `2026090717` was exposed during source inspection and was also
retired. The final replacement seed `{candidate}` was generated locally from
cryptographic randomness after author delegation on 2026-09-07 and frozen
before replacement-test generation, provider upload, fine-tuning, or locked
model evaluation. Runtime checks bind subsequent execution to the frozen
machine-readable protocol."""
    study_protocol = _replace_exact(
        study_protocol,
        OLD_STUDY_PROTOCOL_PARAGRAPH,
        new_protocol_paragraph,
        count=1,
        label="test-seed history paragraph",
    )

    updated_protocol_bytes = _json_bytes(updated_protocol)
    manuscript_bytes = manuscript.encode("utf-8")
    study_protocol_bytes = study_protocol.encode("utf-8")
    freeze_record = {
        "schema_version": "1.0",
        "status": "frozen",
        "protocol_id": updated_protocol["protocol_id"],
        "protocol_version": updated_protocol["version"],
        "frozen_at_utc": _utc_now(),
        "approval_basis": (
            "Author delegated the synthetic-case design and authorized progression on 2026-09-07."
        ),
        "active_test_seed_sha256": _sha256_bytes(str(candidate).encode("ascii")),
        "retired_test_seeds": sorted(RETIRED_TEST_SEEDS),
        "source_hashes": {
            "protocol_json_sha256": _sha256_bytes(updated_protocol_bytes),
            "study_protocol_sha256": _sha256_bytes(study_protocol_bytes),
            "manuscript_sha256_at_freeze": _sha256_bytes(manuscript_bytes),
        },
    }

    staged = [
        (_stage_bytes(protocol_path, updated_protocol_bytes), protocol_path),
        (_stage_bytes(study_protocol_path, study_protocol_bytes), study_protocol_path),
        (_stage_bytes(manuscript_path, manuscript_bytes), manuscript_path),
    ]
    freeze_temporary: Path | None = None
    try:
        for temporary, destination in staged:
            os.replace(temporary, destination)
        freeze_temporary = _stage_bytes(freeze_path, _json_bytes(freeze_record))
        os.replace(freeze_temporary, freeze_path)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
        if freeze_temporary is not None and freeze_temporary.exists():
            freeze_temporary.unlink()

    if _sha256(protocol_path) != freeze_record["source_hashes"]["protocol_json_sha256"]:
        raise RuntimeError("The protocol hash differs immediately after freezing")
    return freeze_record


def main() -> None:
    freeze_protocol()
    print("Protocol freeze completed; the replacement held-out seed was not printed.")


if __name__ == "__main__":
    main()
