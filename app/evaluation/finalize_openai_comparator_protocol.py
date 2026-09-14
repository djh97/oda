"""Record and freeze the pre-test OpenAI-comparator protocol amendment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.artifact_paths import portable_path
from evaluation.publication_workspace import (
    ARTICLE_ARCHIVE_DIR,
    ARTICLE_SOURCE_DIR,
    ARTICLE_SUPPORT_DIR,
)
from src.llm_client import SYSTEM_PROMPT
from src.policy import active_test_seed, assert_protocol_frozen, load_protocol


APP_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = (
    APP_DIR
    / "pipeline-output"
    / "archive"
    / "pretest_protocol_v1.1.0_2026-09-08_openai-comparator-amendment"
)
PROTOCOL_PATH = APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json"
FREEZE_DIR = APP_DIR / "pipeline-output" / "current" / "protocol"
FREEZE_PATH = FREEZE_DIR / "protocol_freeze.json"
AMENDMENT_PATH = FREEZE_DIR / "protocol_amendment_v1.2.0.json"
LINEAGE_PATH = FREEZE_DIR / "protocol_lineage_v1.2.0.json"
TEST_LOCK_PATH = APP_DIR / "pipeline-output" / "current" / "evaluation" / "test_lock.json"
GENERATION_MANIFEST_PATH = APP_DIR.parent / "datasets" / "synthetic-v1" / "generation_manifest.json"
TRAINING_STATE_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_training.json"
STUDY_PROTOCOL_PATH = ARTICLE_SUPPORT_DIR / "STUDY_PROTOCOL.md"
MANUSCRIPT_PATH = ARTICLE_SOURCE_DIR / "Manuscript.tex"
PARENT_PROTOCOL_PATH = ARCHIVE_DIR / "oda_synth_multiorgan_v1.v1.1.0.json"
PARENT_FREEZE_PATH = ARCHIVE_DIR / "protocol_freeze.v1.1.0.json"
PARENT_GENERATOR_PATH = ARCHIVE_DIR / "synthetic_dataset.v1.1.0.py"
PARENT_LLM_CLIENT_PATH = ARCHIVE_DIR / "llm_client.v1.1.0.py"
PARENT_MANUSCRIPT_PATH = ARCHIVE_DIR / "Manuscript.v1.1.0.tex"
DATED_MANUSCRIPT_BACKUP_PATH = (
    ARTICLE_ARCHIVE_DIR
    / "manuscript_backups"
    / "Manuscript.pre-openai-comparator-2026-09-08.tex"
)
CURRENT_GENERATOR_PATH = APP_DIR / "evaluation" / "synthetic_dataset.py"
CURRENT_LLM_CLIENT_PATH = APP_DIR / "src" / "llm_client.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact(path: Path, *, protocol_version: str, protocol_sha: str) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "sha256": _sha256(path),
        "protocol_version": protocol_version,
        "protocol_source_sha256": protocol_sha,
    }


def finalize() -> dict[str, Any]:
    if TEST_LOCK_PATH.exists():
        raise RuntimeError("Cannot amend the protocol after the held-out test is locked")
    if AMENDMENT_PATH.exists() or LINEAGE_PATH.exists():
        raise RuntimeError("The OpenAI-comparator amendment is already recorded")
    required = (
        PROTOCOL_PATH,
        FREEZE_PATH,
        GENERATION_MANIFEST_PATH,
        TRAINING_STATE_PATH,
        STUDY_PROTOCOL_PATH,
        MANUSCRIPT_PATH,
        PARENT_PROTOCOL_PATH,
        PARENT_FREEZE_PATH,
        PARENT_GENERATOR_PATH,
        PARENT_LLM_CLIENT_PATH,
        PARENT_MANUSCRIPT_PATH,
        DATED_MANUSCRIPT_BACKUP_PATH,
    )
    missing = [portable_path(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing amendment evidence: " + ", ".join(missing))

    protocol = load_protocol()
    parent_protocol = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent_freeze = json.loads(PARENT_FREEZE_PATH.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    if protocol.get("version") != "1.2.0" or parent_protocol.get("version") != "1.1.0":
        raise RuntimeError("Expected protocol transition 1.1.0 to 1.2.0")
    if protocol.get("amendment_history", [])[-1].get("version") != "1.2.0":
        raise RuntimeError("The machine protocol has no matching amendment history")
    parent_protocol_sha = _sha256(PARENT_PROTOCOL_PATH)
    checks = {
        "unmodified_parent_freeze": _sha256(FREEZE_PATH) == _sha256(PARENT_FREEZE_PATH),
        "parent_freeze_protocol": (
            parent_freeze.get("protocol_version") == "1.1.0"
            and parent_freeze.get("source_hashes", {}).get("protocol_json_sha256")
            == parent_protocol_sha
        ),
        "generation_protocol": (
            generation.get("protocol_version") == "1.1.0"
            and generation.get("source_inputs", {}).get("protocol_json", {}).get("sha256")
            == parent_protocol_sha
        ),
        "generation_source": (
            generation.get("source_inputs", {}).get("generator", {}).get("sha256")
            == _sha256(PARENT_GENERATOR_PATH)
        ),
        "training_complete": (
            training.get("status") == "completed"
            and training.get("protocol_version") == "1.1.0"
            and training.get("held_out_test_file_opened") is False
        ),
        "training_protocol": training.get("source_hashes", {}).get("protocol")
        == parent_protocol_sha,
        "training_client": training.get("source_hashes", {}).get("llm_client")
        == _sha256(PARENT_LLM_CLIENT_PATH),
        "prompt_unchanged": training.get("source_hashes", {}).get("system_prompt")
        == _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "manuscript_backup": _sha256(PARENT_MANUSCRIPT_PATH)
        == _sha256(DATED_MANUSCRIPT_BACKUP_PATH),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Amendment evidence is inconsistent: {failed}")

    local_settings = protocol["model_evaluation"]
    expected_training_configuration = {
        "base_model_snapshot": local_settings["base_model_snapshot"],
        "precision": local_settings["precision"],
        "attention_implementation": local_settings["attention_implementation"],
        "maximum_sequence_tokens": local_settings["maximum_sequence_tokens"],
        **local_settings["fine_tuning"],
        "seed": local_settings["seed"],
    }
    if training.get("configuration") != expected_training_configuration:
        raise RuntimeError("Protocol 1.2.0 changed the completed local training configuration")

    amended_at = _utc_now()
    history = protocol["amendment_history"][-1]
    amendment = {
        "schema_version": "1.0",
        "status": "approved_pretest_amendment",
        "protocol_id": protocol["protocol_id"],
        "from_version": "1.1.0",
        "to_version": "1.2.0",
        "amended_at_utc": amended_at,
        "approval_basis": (
            "The author authorized implementation changes needed for a concrete and valid "
            "synthetic study and requested the documented Qwen-versus-OpenAI comparison."
        ),
        "reason": history["reason"],
        "change": history["change"],
        "unchanged": history["unchanged"],
        "held_out_test_lock_present": False,
        "held_out_model_execution_started": False,
        "selected_hosted_comparator": local_settings["hosted_comparator"],
        "official_model_and_pricing_source": (
            "https://developers.openai.com/api/docs/models/gpt-4o-mini"
        ),
        "preserved_evidence": {
            "parent_protocol_sha256": parent_protocol_sha,
            "parent_freeze_sha256": _sha256(PARENT_FREEZE_PATH),
            "generation_manifest_sha256": _sha256(GENERATION_MANIFEST_PATH),
            "completed_local_training_sha256": _sha256(TRAINING_STATE_PATH),
            "completed_local_adapter_sha256": training["adapter"]["aggregate_sha256"],
            "dated_manuscript_backup_sha256": _sha256(DATED_MANUSCRIPT_BACKUP_PATH),
        },
    }
    _write_json(AMENDMENT_PATH, amendment)

    lineage = {
        "schema_version": "1.0",
        "status": "verified_pretest_lineage",
        "protocol_id": protocol["protocol_id"],
        "created_at_utc": amended_at,
        "held_out_test_lock_present": False,
        "held_out_model_execution_started": False,
        "scientific_configuration_changed": False,
        "completed_local_training_repeated": False,
        "parent_protocol": {
            "path": portable_path(PARENT_PROTOCOL_PATH),
            "version": "1.1.0",
            "sha256": parent_protocol_sha,
        },
        "active_protocol": {
            "path": portable_path(PROTOCOL_PATH),
            "version": "1.2.0",
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "amendment": {
            "path": portable_path(AMENDMENT_PATH),
            "sha256": _sha256(AMENDMENT_PATH),
        },
        "preserved_artifacts": {
            "generation_manifest": _artifact(
                GENERATION_MANIFEST_PATH,
                protocol_version="1.1.0",
                protocol_sha=parent_protocol_sha,
            ),
            "local_lora_training": _artifact(
                TRAINING_STATE_PATH,
                protocol_version="1.1.0",
                protocol_sha=parent_protocol_sha,
            ),
        },
        "preserved_sources": {
            "synthetic_dataset_generator": {
                "path": portable_path(PARENT_GENERATOR_PATH),
                "sha256": _sha256(PARENT_GENERATOR_PATH),
            },
            "local_training_llm_client": {
                "path": portable_path(PARENT_LLM_CLIENT_PATH),
                "sha256": _sha256(PARENT_LLM_CLIENT_PATH),
            },
        },
        "post_generation_source_changes": {
            "protocol": {
                "before_sha256": parent_protocol_sha,
                "after_sha256": _sha256(PROTOCOL_PATH),
            },
            "synthetic_dataset_generator": {
                "before_sha256": _sha256(PARENT_GENERATOR_PATH),
                "after_sha256": _sha256(CURRENT_GENERATOR_PATH),
            },
        },
        "post_training_source_changes": {
            "protocol": {
                "before_sha256": training["source_hashes"]["protocol"],
                "after_sha256": _sha256(PROTOCOL_PATH),
            },
            "llm_client": {
                "before_sha256": training["source_hashes"]["llm_client"],
                "after_sha256": _sha256(CURRENT_LLM_CLIENT_PATH),
            },
        },
    }
    _write_json(LINEAGE_PATH, lineage)

    freeze = {
        "schema_version": "1.2",
        "status": "frozen",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "frozen_at_utc": amended_at,
        "approval_basis": amendment["approval_basis"],
        "active_test_seed_sha256": _sha256_bytes(
            str(active_test_seed(protocol)).encode("ascii")
        ),
        "retired_test_seeds": parent_freeze["retired_test_seeds"],
        "parent_freeze_sha256": _sha256(PARENT_FREEZE_PATH),
        "amendment_record_sha256": _sha256(AMENDMENT_PATH),
        "protocol_lineage_sha256": _sha256(LINEAGE_PATH),
        "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "source_hashes": {
            "protocol_json_sha256": _sha256(PROTOCOL_PATH),
            "study_protocol_sha256": _sha256(STUDY_PROTOCOL_PATH),
            "manuscript_sha256_at_freeze": _sha256(MANUSCRIPT_PATH),
        },
        "preserved_development_artifacts": {
            "generation_manifest_sha256": _sha256(GENERATION_MANIFEST_PATH),
            "local_lora_training_sha256": _sha256(TRAINING_STATE_PATH),
            "local_lora_adapter_sha256": training["adapter"]["aggregate_sha256"],
        },
        "hosted_comparator_preflight_pending": True,
    }
    _write_json(FREEZE_PATH, freeze)
    assert_protocol_frozen(protocol)
    return freeze


def main() -> None:
    freeze = finalize()
    print(
        "Recorded the hosted-comparator amendment and froze protocol "
        f"version {freeze['protocol_version']}."
    )


if __name__ == "__main__":
    main()
