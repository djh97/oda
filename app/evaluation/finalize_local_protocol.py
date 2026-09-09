"""Bind finalized local-model development artifacts into the active freeze."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.prepare_local_model import OUTPUT_PATH as BASE_MODEL_MANIFEST_PATH
from evaluation.synthetic_dataset import DATASET_DIR
from evaluation.train_local_lora import STATE_PATH as TRAINING_STATE_PATH
from src.llm_client import SYSTEM_PROMPT
from src.policy import DEFAULT_PROTOCOL_FREEZE_PATH, DEFAULT_PROTOCOL_PATH, assert_protocol_frozen


APP_DIR = Path(__file__).resolve().parents[1]
GENERATION_MANIFEST_PATH = DATASET_DIR / "generation_manifest.json"
TRAINING_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
PREFLIGHT_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_preflight.json"
TRAINING_SCRIPT_PATH = APP_DIR / "evaluation" / "train_local_lora.py"
LOCAL_CLIENT_PATH = APP_DIR / "src" / "local_llm_client.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def finalize() -> dict[str, Any]:
    if TRAINING_STATE_PATH.exists():
        raise RuntimeError("Cannot change the development-artifact freeze after training starts")
    freeze = assert_protocol_frozen()
    protocol = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("version") != "1.1.0":
        raise RuntimeError("The local development-artifact finalizer expects protocol version 1.1.0")
    base_manifest = json.loads(BASE_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    checks = {
        "base_model_verified": base_manifest.get("status") == "verified",
        "protocol_version": generation.get("protocol_version") == protocol["version"],
        "protocol_id": generation.get("protocol_id") == protocol["protocol_id"],
        "prompt": generation.get("source_inputs", {}).get("system_prompt_sha256") == prompt_hash,
        "training_file": (
            generation.get("artifacts", {}).get("fine_tuning_training.jsonl", {}).get("sha256")
            == _sha256(TRAINING_PATH)
            == base_manifest.get("development_files", {}).get("training", {}).get("sha256")
        ),
        "validation_file": (
            generation.get("artifacts", {}).get("fine_tuning_validation.jsonl", {}).get("sha256")
            == _sha256(VALIDATION_PATH)
            == base_manifest.get("development_files", {}).get("validation", {}).get("sha256")
        ),
        "sequence_limit": (
            int(base_manifest.get("token_audit", {}).get("full_sequence_tokens", {}).get("maximum", 0))
            <= int(protocol["model_evaluation"]["maximum_sequence_tokens"])
        ),
        "assistant_loss_mask": (
            base_manifest.get("token_audit", {}).get("assistant_only_loss_mask_verified") is True
        ),
        "preflight": preflight.get("status") == "passed",
        "preflight_protocol": (
            preflight.get("source_hashes", {}).get("protocol") == _sha256(DEFAULT_PROTOCOL_PATH)
        ),
        "preflight_training_script": (
            preflight.get("source_hashes", {}).get("training_script")
            == _sha256(TRAINING_SCRIPT_PATH)
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Local development artifacts are not ready to freeze: {failed}")

    freeze["development_artifacts_finalized_at_utc"] = _utc_now()
    freeze["base_model_manifest_sha256"] = _sha256(BASE_MODEL_MANIFEST_PATH)
    freeze["system_prompt_sha256"] = prompt_hash
    freeze["development_artifacts"] = {
        "generation_manifest_sha256": _sha256(GENERATION_MANIFEST_PATH),
        "training_file_sha256": _sha256(TRAINING_PATH),
        "validation_file_sha256": _sha256(VALIDATION_PATH),
        "base_model_manifest_sha256": _sha256(BASE_MODEL_MANIFEST_PATH),
        "local_lora_preflight_sha256": _sha256(PREFLIGHT_PATH),
        "training_script_sha256": _sha256(TRAINING_SCRIPT_PATH),
        "local_llm_client_sha256": _sha256(LOCAL_CLIENT_PATH),
    }
    _write_json(DEFAULT_PROTOCOL_FREEZE_PATH, freeze)
    assert_protocol_frozen()
    return freeze


def main() -> None:
    freeze = finalize()
    print(
        "Finalized local development artifacts for protocol "
        f"{freeze['protocol_id']} version {freeze['protocol_version']}."
    )


if __name__ == "__main__":
    main()
