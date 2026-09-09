"""Create a provenance-bound active copy of the completed epoch-2 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.artifact_paths import portable_path
from evaluation.train_local_lora import (
    APP_DIR,
    CHECKPOINT_DIR,
    CHECKPOINT_INHERITANCE_PATH,
    STATE_PATH,
    _aggregate_hash,
    _directory_files,
    _runtime_memory_management,
    _sha256,
    _source_hashes,
    _training_configuration,
)
from src.policy import assert_protocol_frozen, load_protocol


SOURCE_ATTEMPT_DIR = (
    APP_DIR
    / "pipeline-output"
    / "archive"
    / "pretest_training_attempt_2026-09-08_final-eval-memory"
)
SOURCE_STATE_PATH = SOURCE_ATTEMPT_DIR / "local_lora_training.json"
SOURCE_CHECKPOINT_PATH = SOURCE_ATTEMPT_DIR / "local_lora_checkpoints" / "checkpoint-400"
SOURCE_NOTE_PATH = SOURCE_ATTEMPT_DIR / "ARCHIVE_NOTE.md"
ACTIVE_CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint-400"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read required resume artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Resume artifact is not a JSON object: {path}")
    return value


def _ensure_copy_boundaries() -> None:
    archive_root = (APP_DIR / "pipeline-output" / "archive").resolve()
    current_root = (APP_DIR / "pipeline-output" / "current" / "model").resolve()
    try:
        SOURCE_CHECKPOINT_PATH.resolve().relative_to(archive_root)
        ACTIVE_CHECKPOINT_PATH.resolve().relative_to(current_root)
    except ValueError as exc:
        raise RuntimeError("Checkpoint copy path leaves its permitted artifact root") from exc


def prepare_resume() -> dict[str, Any]:
    assert_protocol_frozen()
    protocol = load_protocol()
    configuration = _training_configuration(protocol["model_evaluation"])
    current_sources = _source_hashes()
    if STATE_PATH.exists():
        raise RuntimeError("Cannot prepare inherited checkpoint after active training starts")
    if CHECKPOINT_INHERITANCE_PATH.exists():
        raise RuntimeError("A checkpoint inheritance record already exists")
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        raise RuntimeError("The active checkpoint directory is not empty")
    if not SOURCE_NOTE_PATH.is_file():
        raise RuntimeError("The archived attempt has no explanatory note")
    _ensure_copy_boundaries()

    source_state = _load_json(SOURCE_STATE_PATH)
    if (
        source_state.get("status") != "running"
        or source_state.get("protocol_id") != protocol["protocol_id"]
        or source_state.get("protocol_version") != protocol["version"]
        or source_state.get("held_out_test_file_opened") is not False
        or source_state.get("configuration") != configuration
    ):
        raise RuntimeError("The archived training state is not eligible for continuation")
    invocations = source_state.get("invocations")
    if (
        not isinstance(invocations, list)
        or len(invocations) != 1
        or not isinstance(invocations[0], dict)
        or invocations[0].get("outcome")
    ):
        raise RuntimeError("The archived invocation history is inconsistent with interruption")

    source_sources = source_state.get("source_hashes")
    if not isinstance(source_sources, dict):
        raise RuntimeError("The archived training source hashes are missing")
    changed_sources = sorted(
        key
        for key in set(source_sources).union(current_sources)
        if source_sources.get(key) != current_sources.get(key)
    )
    if changed_sources != ["training_script"]:
        raise RuntimeError(
            "Checkpoint continuation permits only the documented training-script change; "
            f"observed changes: {changed_sources}"
        )

    trainer_state_path = SOURCE_CHECKPOINT_PATH / "trainer_state.json"
    trainer_state = _load_json(trainer_state_path)
    evaluations = [
        item
        for item in trainer_state.get("log_history", [])
        if isinstance(item, dict) and "eval_loss" in item
    ]
    expected_evaluations = [
        (200, 1.0, 0.03396410867571831),
        (400, 2.0, 0.029150526970624924),
    ]
    observed_evaluations = [
        (int(item.get("step", -1)), float(item.get("epoch", -1)), float(item["eval_loss"]))
        for item in evaluations
    ]
    if (
        int(trainer_state.get("global_step", -1)) != 400
        or float(trainer_state.get("epoch", -1)) != 2.0
        or int(trainer_state.get("max_steps", -1)) != 600
        or observed_evaluations != expected_evaluations
    ):
        raise RuntimeError("The archived checkpoint does not represent the verified epoch-2 state")

    source_files = _directory_files(SOURCE_CHECKPOINT_PATH)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_CHECKPOINT_PATH, ACTIVE_CHECKPOINT_PATH)
    active_files = _directory_files(ACTIVE_CHECKPOINT_PATH)
    if active_files != source_files:
        raise RuntimeError("The active checkpoint copy differs from the archived source")

    record = {
        "schema_version": "1.0",
        "status": "verified",
        "created_at_utc": _utc_now(),
        "purpose": "Continue the interrupted pre-test LoRA run from its latest durable checkpoint",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "held_out_test_file_opened": False,
        "scientific_configuration_changed": False,
        "configuration": configuration,
        "source_attempt": {
            "path": portable_path(SOURCE_ATTEMPT_DIR),
            "state_path": portable_path(SOURCE_STATE_PATH),
            "state_sha256": _sha256(SOURCE_STATE_PATH),
            "archive_note_path": portable_path(SOURCE_NOTE_PATH),
            "archive_note_sha256": _sha256(SOURCE_NOTE_PATH),
            "status": source_state["status"],
            "runtime_memory_management": source_state.get("runtime_memory_management"),
            "source_hashes": source_sources,
        },
        "completed_progress": {
            "global_step": 400,
            "epoch": 2.0,
            "maximum_step": 600,
            "epoch_end_validation": [
                {"step": step, "epoch": epoch, "eval_loss": loss}
                for step, epoch, loss in expected_evaluations
            ],
        },
        "source_checkpoint": {
            "path": portable_path(SOURCE_CHECKPOINT_PATH),
            "files": source_files,
            "aggregate_sha256": _aggregate_hash(source_files),
        },
        "active_checkpoint": {
            "path": portable_path(ACTIVE_CHECKPOINT_PATH),
            "files": active_files,
            "aggregate_sha256": _aggregate_hash(active_files),
        },
        "current_source_hashes": current_sources,
        "changed_source_keys": changed_sources,
        "runtime_memory_management": _runtime_memory_management(),
        "operational_changes": [
            "Retain validation loss only instead of prediction logits.",
            "Move validation losses to host memory after every prediction step.",
            "Release unreferenced CUDA cache after every validation prediction step.",
            "Use the final epoch-end validation record instead of running a duplicate validation pass.",
        ],
        "preparation_script_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
    }
    _write_json(CHECKPOINT_INHERITANCE_PATH, record)
    return record


def main() -> None:
    record = prepare_resume()
    print(
        "Prepared verified local LoRA continuation from "
        f"step {record['completed_progress']['global_step']}."
    )


if __name__ == "__main__":
    main()
