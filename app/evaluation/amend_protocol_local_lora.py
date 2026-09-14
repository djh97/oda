"""Record the pre-test amendment from unavailable hosted tuning to local LoRA."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.prepare_local_model import (
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    OUTPUT_PATH as BASE_MODEL_MANIFEST_PATH,
)
from evaluation.publication_workspace import ARTICLE_SOURCE_DIR, ARTICLE_SUPPORT_DIR
from src.llm_client import SYSTEM_PROMPT
from src.policy import active_test_seed, assert_protocol_frozen


APP_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json"
STUDY_PROTOCOL_PATH = ARTICLE_SUPPORT_DIR / "STUDY_PROTOCOL.md"
MANUSCRIPT_PATH = ARTICLE_SOURCE_DIR / "Manuscript.tex"
FREEZE_DIR = APP_DIR / "pipeline-output" / "current" / "protocol"
FREEZE_PATH = FREEZE_DIR / "protocol_freeze.json"
ORIGINAL_FREEZE_PATH = FREEZE_DIR / "protocol_freeze_v1.0.0_openai.json"
AMENDMENT_PATH = FREEZE_DIR / "protocol_amendment_v1.1.0.json"
OPENAI_STATE_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "fine_tuning_job.json"
OPENAI_ARCHIVE_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "openai_fine_tuning_job_rejected.json"
)
LOCAL_TRAINING_STATE_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_training.json"
)
TEST_LOCK_PATH = APP_DIR / "pipeline-output" / "current" / "evaluation" / "test_lock.json"

OLD_VERSION = "1.0.0"
NEW_VERSION = "1.1.0"
MODEL_SNAPSHOT = f"{MODEL_REPOSITORY}@{MODEL_REVISION}"
OFFICIAL_OPENAI_NOTICE = (
    "https://developers.openai.com/api/docs/deprecations"
    "#update-to-openais-self-serve-fine-tuning"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=False) + "\n").encode(
        "utf-8"
    )


def _stage_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.amend.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"Expected exactly one {label} while amending the protocol")
    return value.replace(old, new, 1)


def amended_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("version") != OLD_VERSION:
        raise RuntimeError(f"Expected active protocol version {OLD_VERSION}")
    updated = deepcopy(dict(protocol))
    updated["version"] = NEW_VERSION
    updated["model_evaluation"] = {
        "provider": "local_hugging_face_transformers",
        "base_model_repository": MODEL_REPOSITORY,
        "base_model_revision": MODEL_REVISION,
        "base_model_snapshot": MODEL_SNAPSHOT,
        "base_model_license": MODEL_LICENSE,
        "precision": "float16",
        "attention_implementation": "sdpa",
        "temperature": 0,
        "do_sample": False,
        "seed": 20260907,
        "maximum_sequence_tokens": 2048,
        "maximum_new_tokens": 512,
        "attempts_per_case_per_condition": 1,
        "append_only_attempt_ledger": True,
        "interrupted_attempts_are_failures": True,
        "fine_tuning": {
            "method": "LoRA",
            "epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_train_batch_size": 8,
            "per_device_validation_batch_size": 1,
            "learning_rate": 0.0002,
            "learning_rate_scheduler": "linear",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "maximum_gradient_norm": 1.0,
            "optimizer": "adamw_torch",
            "gradient_checkpointing": True,
            "assistant_only_loss": True,
            "load_best_model_at_end": True,
            "selection_metric": "validation_loss",
            "save_and_evaluate_each_epoch": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "tf32": False,
            "single_gpu": True,
        },
        "bootstrap": deepcopy(protocol["model_evaluation"]["bootstrap"]),
        "classical_text_baseline": deepcopy(
            protocol["model_evaluation"]["classical_text_baseline"]
        ),
    }
    history = list(updated.get("amendment_history", []))
    history.append(
        {
            "version": NEW_VERSION,
            "date": "2026-09-07",
            "timing": "before any held-out test lock or held-out model execution",
            "reason": (
                "The prespecified OpenAI fine-tuning request was definitively rejected with "
                "HTTP 403 because self-serve training was unavailable to the study account. "
                "A provider reconciliation found no created job."
            ),
            "change": (
                "Replace the unavailable hosted model pair with the immutable Apache-2.0 "
                "Qwen2.5-1.5B-Instruct checkpoint and a locally trained LoRA adapter."
            ),
            "unchanged": [
                "synthetic case-generation design and frozen test seed",
                "training, validation, and test case counts",
                "task definition and note-readiness taxonomy",
                "deterministic rankers and guard",
                "outcomes and statistical analysis",
                "one-attempt held-out evaluation rule",
            ],
        }
    )
    updated["amendment_history"] = history
    return updated


def amended_study_protocol(value: str) -> str:
    value = _replace_once(
        value,
        "Protocol identifier: ODA-SYNTH-MULTIORGAN-1.0",
        "Protocol identifier: ODA-SYNTH-MULTIORGAN-1.0, version 1.1.0",
        "protocol identifier",
    )
    value = _replace_once(
        value,
        "provider upload, or external execution",
        "model adaptation, or external execution",
        "initial execution boundary",
    )
    value = _replace_once(
        value,
        "classical or API-based evaluation",
        "classical or local-model evaluation",
        "initial test-lock description",
    )
    before, marker, remainder = value.partition("## Model Conditions")
    if not marker:
        raise RuntimeError("Study protocol has no Model Conditions heading")
    _old_model, outcomes_marker, after = remainder.partition("## Outcomes")
    if not outcomes_marker:
        raise RuntimeError("Study protocol has no Outcomes heading")
    model_section = f"""## Model Conditions

The principal comparison uses the same locked test cases for four conditions.

- Deterministic baseline with no note review.
- A fixed TF-IDF word- and character-feature logistic model trained only on
  candidate notes from the training split. Its hyperparameters are fixed in
  the machine-readable protocol, and the validation split is used only for
  diagnostics rather than test-driven selection.
- The unmodified `{MODEL_SNAPSHOT}` checkpoint.
- The same checkpoint with a supervised LoRA adapter trained only on the
  training split and selected by validation loss.

Version 1.0.0 specified an OpenAI-hosted base model and supervised fine-tune.
The account could retrieve that base model, but the job-creation request was
rejected with HTTP 403 under the provider's current self-serve fine-tuning
availability policy. A metadata-filtered reconciliation confirmed that no job
was created. No held-out test lock or held-out model execution had occurred.
Version 1.1.0 therefore replaces only the unavailable model pair with the
immutable Apache-2.0 Qwen checkpoint above and a local LoRA adapter. The split,
test seed, task, prompt, taxonomy, rankers, guard, outcomes, and analysis remain
unchanged. The rejected provider record and version 1.0.0 freeze are retained.

The shared prompt states the complete evidence-code vocabulary and exact JSON
shape for both model conditions. Training uses assistant tokens only and never
computes loss on the system or user text. The maximum sequence length is 2,048
tokens, which exceeds every complete training and validation sequence after the
amended prompt is rendered. No development example is truncated. The fixed
LoRA configuration uses rank 16, alpha 32, dropout 0.05, all attention and MLP
projection modules, three epochs, per-device batch size one, gradient
accumulation over eight examples, AdamW, learning rate 2e-4, linear decay, and
a 3% warmup. The adapter with the lowest epoch-end validation loss is retained.

Both test conditions use greedy decoding with sampling disabled, a maximum of
512 new tokens, and one process-level attempt per case. Invalid output, missing
or duplicate candidates, empty responses, schema violations, and runtime
failures count as failures and cannot be silently removed. An append-only
ledger is written immediately before every model call. An interrupted
evaluation resumes at the first unattempted case and converts any unterminated
ledger entry to a retained failure. Exact model and adapter hashes, dependency
versions, GPU metadata, token counts, latency, training state, checkpoints,
loss history, raw outputs, and parsed outputs are preserved.

"""
    value = before + model_section + outcomes_marker + after
    old_source = """- OpenAI's GPT-4o mini model documentation and current self-serve fine-tuning
  deprecation schedule for model availability and the access preflight."""
    new_source = """- The Qwen2.5-1.5B-Instruct model card, immutable checkpoint, and Apache-2.0
  license for the local model definition.
- OpenAI's self-serve fine-tuning deprecation schedule for documenting the
  preserved pre-test provider rejection and protocol amendment."""
    return _replace_once(value, old_source, new_source, "model-source entry")


def amended_manuscript(value: str) -> str:
    start = "The two LLM conditions use the untuned and supervised fine-tuned forms of"
    end = "Figure~\\ref{fig:benchmark_protocol} summarizes"
    before, marker, remainder = value.partition(start)
    if not marker:
        raise RuntimeError("Manuscript model paragraph start was not found")
    _old_model, end_marker, after = remainder.partition(end)
    if not end_marker:
        raise RuntimeError("Manuscript model paragraph end was not found")
    model_paragraph = f"""The two LLM conditions use the unmodified and LoRA-adapted forms of \\texttt{{Qwen/Qwen2.5-1.5B-Instruct}} at immutable revision \\texttt{{{MODEL_REVISION}}} \\cite{{qwen25ModelCard}}. Version 1.0.0 specified an OpenAI-hosted model pair, but the provider rejected the fine-tuning request before any held-out evaluation and reconciliation confirmed that no job was created. The documented version 1.1.0 amendment changes only the unavailable model pair and preserves the same split, test seed, task, prompt, taxonomy, rankers, guard, outcomes, and analysis. LoRA training receives the 1,600 training cases and 320 validation cases in system--user--assistant format and computes loss only over assistant tokens. The fixed configuration uses three epochs, sequence length 2,048, per-device batch size one, gradient accumulation over eight examples, AdamW with learning rate $2\\times10^{{-4}}$ and linear decay, 3\\% warmup, rank 16, alpha 32, dropout 0.05, and all attention and multilayer-perceptron projection modules. The checkpoint with the lowest epoch-end validation loss is retained. Both conditions use the same explicit JSON schema and evidence-code vocabulary, greedy decoding with sampling disabled, at most 512 generated tokens, and one recorded attempt per case. An append-only ledger is written before each test call. A process interruption leaves that case as a failure on resume. Empty or malformed JSON, schema violations, incomplete candidate coverage, and runtime errors remain failures in the denominator. Each successful response preserves model and adapter identifiers and hashes, token counts, latency, software versions, and GPU metadata.

"""
    updated = before + model_paragraph + end_marker + after
    updated = updated.replace(
        "prespecified protocol ODA-SYNTH-MULTIORGAN-1.0.",
        "prespecified protocol ODA-SYNTH-MULTIORGAN-1.0, version 1.1.0.",
    )
    updated = updated.replace(
        "prespecified ODA-SYNTH-MULTIORGAN-1.0 protocol.",
        "prespecified ODA-SYNTH-MULTIORGAN-1.0 protocol, version 1.1.0.",
    )
    return updated


def amend_protocol() -> dict[str, Any]:
    if TEST_LOCK_PATH.exists():
        raise RuntimeError("Cannot amend the model protocol after the held-out test is locked")
    if LOCAL_TRAINING_STATE_PATH.exists():
        raise RuntimeError("Cannot amend the model protocol after local training has started")
    for path in (ORIGINAL_FREEZE_PATH, AMENDMENT_PATH, OPENAI_ARCHIVE_PATH):
        if path.exists():
            raise RuntimeError(f"Amendment output already exists: {path}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    original_freeze = assert_protocol_frozen(protocol)
    model_manifest = json.loads(BASE_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        model_manifest.get("status") != "verified"
        or model_manifest.get("repository") != MODEL_REPOSITORY
        or model_manifest.get("revision") != MODEL_REVISION
        or model_manifest.get("license") != MODEL_LICENSE
    ):
        raise RuntimeError("The local base-model manifest is missing or inconsistent")
    openai_state = json.loads(OPENAI_STATE_PATH.read_text(encoding="utf-8"))
    if (
        openai_state.get("status") != "job_creation_rejected"
        or openai_state.get("provider_availability", {}).get("available") is not False
        or openai_state.get("job_recovery_check", {}).get("matching_job_ids") != []
    ):
        raise RuntimeError("The preserved OpenAI state does not prove terminal unavailability")

    updated_protocol = amended_protocol(protocol)
    updated_study = amended_study_protocol(STUDY_PROTOCOL_PATH.read_text(encoding="utf-8"))
    updated_manuscript = amended_manuscript(MANUSCRIPT_PATH.read_text(encoding="utf-8"))
    updated_protocol_bytes = _json_bytes(updated_protocol)
    updated_study_bytes = updated_study.encode("utf-8")
    updated_manuscript_bytes = updated_manuscript.encode("utf-8")
    openai_state_bytes = OPENAI_STATE_PATH.read_bytes()
    original_freeze_bytes = FREEZE_PATH.read_bytes()
    amended_at = _utc_now()

    amendment = {
        "schema_version": "1.0",
        "status": "approved_pretest_amendment",
        "protocol_id": updated_protocol["protocol_id"],
        "from_version": OLD_VERSION,
        "to_version": NEW_VERSION,
        "amended_at_utc": amended_at,
        "approval_basis": (
            "Author delegated implementation and protocol decisions and authorized goal "
            "resumption on 2026-09-07."
        ),
        "reason": updated_protocol["amendment_history"][-1]["reason"],
        "held_out_test_lock_present": False,
        "held_out_model_execution_started": False,
        "unchanged": updated_protocol["amendment_history"][-1]["unchanged"],
        "official_provider_notice": OFFICIAL_OPENAI_NOTICE,
        "preserved_evidence": {
            "original_protocol_freeze_sha256": _sha256_bytes(original_freeze_bytes),
            "openai_rejection_state_sha256": _sha256_bytes(openai_state_bytes),
            "local_base_model_manifest_sha256": _sha256(BASE_MODEL_MANIFEST_PATH),
        },
    }
    freeze_record = {
        "schema_version": "1.1",
        "status": "frozen",
        "protocol_id": updated_protocol["protocol_id"],
        "protocol_version": NEW_VERSION,
        "frozen_at_utc": amended_at,
        "approval_basis": amendment["approval_basis"],
        "active_test_seed_sha256": _sha256_bytes(
            str(active_test_seed(updated_protocol)).encode("ascii")
        ),
        "retired_test_seeds": original_freeze["retired_test_seeds"],
        "amendment_record_sha256": _sha256_bytes(_json_bytes(amendment)),
        "base_model_manifest_sha256": _sha256(BASE_MODEL_MANIFEST_PATH),
        "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "source_hashes": {
            "protocol_json_sha256": _sha256_bytes(updated_protocol_bytes),
            "study_protocol_sha256": _sha256_bytes(updated_study_bytes),
            "manuscript_sha256_at_freeze": _sha256_bytes(updated_manuscript_bytes),
        },
    }

    destinations = [
        (PROTOCOL_PATH, updated_protocol_bytes),
        (STUDY_PROTOCOL_PATH, updated_study_bytes),
        (MANUSCRIPT_PATH, updated_manuscript_bytes),
        (ORIGINAL_FREEZE_PATH, original_freeze_bytes),
        (OPENAI_ARCHIVE_PATH, openai_state_bytes),
        (AMENDMENT_PATH, _json_bytes(amendment)),
        (FREEZE_PATH, _json_bytes(freeze_record)),
    ]
    staged: list[tuple[Path, Path]] = []
    try:
        staged = [(_stage_bytes(destination, content), destination) for destination, content in destinations]
        for temporary, destination in staged:
            os.replace(temporary, destination)
        OPENAI_STATE_PATH.unlink()
    finally:
        for temporary, _destination in staged:
            if temporary.exists():
                temporary.unlink()

    assert_protocol_frozen(updated_protocol)
    return freeze_record


def main() -> None:
    record = amend_protocol()
    print(
        "Recorded pre-test model-provider amendment and froze protocol "
        f"version {record['protocol_version']}."
    )
    print("The held-out test seed was retained and was not printed.")


if __name__ == "__main__":
    main()
