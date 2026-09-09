"""Deterministic local Transformers interface for synthetic note review."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .llm_client import (
    LLMClientConfigurationError,
    LLMEmptyResponseError,
    NoteReviewResult,
    SYSTEM_PROMPT,
    build_note_review_payload,
    parse_note_review,
)
from .policy import (
    DEFAULT_PROTOCOL_PATH,
    PolicyInputError,
    assert_preserved_artifact_lineage,
    load_protocol,
)
from .schemas import ModelRunMetadata


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_DIR / ".model-cache" / "Qwen2.5-1.5B-Instruct"
TRAINING_STATE_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_training.json"
)
BASE_MODEL_MANIFEST_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_base_model_manifest.json"
)
DATASET_DIR = APP_DIR.parent / "datasets" / "synthetic-v1"
TRAINING_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
GENERATION_MANIFEST_PATH = DATASET_DIR / "generation_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(files: Dict[str, Dict[str, Any]]) -> str:
    canonical = json.dumps(files, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_local_base_model(
    model_dir: Path = MODEL_DIR,
    manifest_path: Path = BASE_MODEL_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify that the local checkpoint exactly matches its retained manifest."""
    if not model_dir.is_dir() or not manifest_path.is_file():
        raise LLMClientConfigurationError("The pinned local base model or manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMClientConfigurationError("The local base-model manifest is unreadable") from exc
    settings = load_protocol()["model_evaluation"]
    if (
        manifest.get("status") != "verified"
        or manifest.get("repository") != settings["base_model_repository"]
        or manifest.get("revision") != settings["base_model_revision"]
        or manifest.get("license") != settings["base_model_license"]
    ):
        raise LLMClientConfigurationError("The local base-model manifest differs from the protocol")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise LLMClientConfigurationError("The local base-model file manifest is missing")
    observed = {
        path.relative_to(model_dir).as_posix(): path
        for path in model_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if set(observed) != set(files):
        raise LLMClientConfigurationError("The local base-model file set differs from its manifest")
    for relative_name, path in observed.items():
        record = files.get(relative_name)
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            raise LLMClientConfigurationError(
                f"The local base-model file failed verification: {relative_name}"
            )
    return manifest


def validate_local_training_state(
    path: Path = TRAINING_STATE_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise LLMClientConfigurationError("The completed local LoRA training state is missing")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMClientConfigurationError("The local LoRA training state is unreadable") from exc
    protocol = load_protocol()
    settings = protocol["model_evaluation"]
    if (
        state.get("status") != "completed"
        or state.get("protocol_id") != protocol["protocol_id"]
        or state.get("held_out_test_file_opened") is not False
    ):
        raise LLMClientConfigurationError("The local LoRA training run is not complete")
    lineage = None
    if state.get("protocol_version") != protocol["version"]:
        try:
            lineage = assert_preserved_artifact_lineage(
                "local_lora_training",
                path,
                artifact_protocol_version=state.get("protocol_version"),
                protocol_source_sha256=state.get("source_hashes", {}).get("protocol"),
            )
        except PolicyInputError as exc:
            raise LLMClientConfigurationError(
                "The completed local LoRA run has no valid protocol lineage"
            ) from exc
        if (
            lineage.get("scientific_configuration_changed") is not False
            or lineage.get("completed_local_training_repeated") is not False
        ):
            raise LLMClientConfigurationError(
                "The local LoRA protocol-lineage declaration is invalid"
            )
    expected_configuration = {
        "base_model_snapshot": settings["base_model_snapshot"],
        "precision": settings["precision"],
        "attention_implementation": settings["attention_implementation"],
        "maximum_sequence_tokens": settings["maximum_sequence_tokens"],
        **json.loads(json.dumps(settings["fine_tuning"])),
        "seed": settings["seed"],
    }
    if state.get("configuration") != expected_configuration:
        raise LLMClientConfigurationError("The local LoRA configuration differs from the protocol")
    if state.get("runtime_memory_management") != {
        "torch_empty_cache_steps": 1,
        "prediction_loss_only": True,
        "eval_accumulation_steps": 1,
        "empty_cuda_cache_after_each_prediction_step": True,
        "duplicate_post_training_evaluation": False,
        "purpose": (
            "Bound CUDA allocator growth for variable-length training and validation "
            "batches under Windows WDDM."
        ),
        "scientific_hyperparameters_changed": False,
    }:
        raise LLMClientConfigurationError("The local LoRA memory-management record is invalid")
    expected_sources = {
        "protocol": _sha256(DEFAULT_PROTOCOL_PATH),
        "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "training_file": _sha256(TRAINING_PATH),
        "validation_file": _sha256(VALIDATION_PATH),
        "generation_manifest": _sha256(GENERATION_MANIFEST_PATH),
        "base_model_manifest": _sha256(BASE_MODEL_MANIFEST_PATH),
        "training_script": _sha256(APP_DIR / "evaluation" / "train_local_lora.py"),
        "llm_client": _sha256(APP_DIR / "src" / "llm_client.py"),
        "schemas": _sha256(APP_DIR / "src" / "schemas.py"),
        "requirements_ml": _sha256(APP_DIR / "requirements-ml.txt"),
    }
    if lineage is None:
        sources_valid = state.get("source_hashes") == expected_sources
    else:
        state_sources = state.get("source_hashes")
        changed_sources = lineage.get("post_training_source_changes")
        expected_changed = {"protocol", "llm_client"}
        sources_valid = (
            isinstance(state_sources, dict)
            and isinstance(changed_sources, dict)
            and set(changed_sources) == expected_changed
            and all(
                state_sources.get(name) == expected_sources[name]
                for name in set(expected_sources).difference(expected_changed)
            )
            and all(
                isinstance(changed_sources.get(name), dict)
                and changed_sources[name].get("before_sha256") == state_sources.get(name)
                and changed_sources[name].get("after_sha256") == expected_sources[name]
                for name in expected_changed
            )
        )
    if not sources_valid:
        raise LLMClientConfigurationError("The local LoRA training inputs or sources changed")
    invocations = state.get("invocations")
    if (
        not isinstance(invocations, list)
        or not invocations
        or any(not isinstance(item, dict) or not item.get("outcome") for item in invocations)
    ):
        raise LLMClientConfigurationError("The local LoRA invocation history is incomplete")
    if invocations[0].get("resume_checkpoint"):
        inheritance = state.get("checkpoint_inheritance")
        if not isinstance(inheritance, dict):
            raise LLMClientConfigurationError(
                "The inherited local LoRA checkpoint has no provenance record"
            )
        artifact_path = inheritance.get("artifact_path")
        artifact = APP_DIR.parent.parent / str(artifact_path or "")
        if (
            not artifact.is_file()
            or inheritance.get("artifact_sha256") != _sha256(artifact)
        ):
            raise LLMClientConfigurationError(
                "The local LoRA checkpoint inheritance artifact failed verification"
            )
        try:
            inherited_record = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LLMClientConfigurationError(
                "The local LoRA checkpoint inheritance artifact is unreadable"
            ) from exc
        if (
            inheritance.get("record") != inherited_record
            or inherited_record.get("status") != "verified"
            or inherited_record.get("held_out_test_file_opened") is not False
            or inherited_record.get("scientific_configuration_changed") is not False
        ):
            raise LLMClientConfigurationError(
                "The local LoRA checkpoint inheritance record is invalid"
            )
    if len(invocations) > 1:
        reconciliation = state.get("invocation_reconciliation")
        reconciliation_path = APP_DIR / "evaluation" / "reconcile_local_training_state.py"
        if (
            not isinstance(reconciliation, dict)
            or reconciliation.get("script_sha256") != _sha256(reconciliation_path)
        ):
            raise LLMClientConfigurationError(
                "The resumed local LoRA run has no valid reconciliation record"
            )
    adapter = state.get("adapter")
    if not isinstance(adapter, dict) or not adapter.get("model_id"):
        raise LLMClientConfigurationError("The local LoRA adapter record is malformed")
    adapter_dir = APP_DIR / str(adapter.get("relative_path", ""))
    if not adapter_dir.is_dir():
        raise LLMClientConfigurationError("The local LoRA adapter directory is missing")
    files = adapter.get("files")
    if not isinstance(files, dict) or not files:
        raise LLMClientConfigurationError("The local LoRA adapter file manifest is missing")
    for relative_name, record in files.items():
        file_path = adapter_dir / relative_name
        if (
            not file_path.is_file()
            or not isinstance(record, dict)
            or record.get("sha256") != _sha256(file_path)
            or record.get("bytes") != file_path.stat().st_size
        ):
            raise LLMClientConfigurationError(
                f"The local LoRA adapter file failed verification: {relative_name}"
            )
    if adapter.get("aggregate_sha256") != _aggregate_hash(files):
        raise LLMClientConfigurationError("The local LoRA adapter aggregate hash is invalid")
    return state


class LocalNoteReviewClient:
    """Load one pinned base/adapter condition and reuse it for all calls."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter: bool,
        model_dir: Path = MODEL_DIR,
        base_model_manifest_path: Path = BASE_MODEL_MANIFEST_PATH,
        training_state_path: Path = TRAINING_STATE_PATH,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMClientConfigurationError("Local model dependencies are not installed") from exc

        settings = load_protocol()["model_evaluation"]
        validate_local_base_model(model_dir, base_model_manifest_path)
        expected_base = str(settings["base_model_snapshot"])
        training_state: dict[str, Any] | None = None
        if adapter:
            training_state = validate_local_training_state(training_state_path)
            expected_model = str(training_state["adapter"]["model_id"])
        else:
            expected_model = expected_base
        if model_id != expected_model:
            raise LLMClientConfigurationError(
                f"Requested model {model_id!r} does not match the preserved condition model"
            )
        if not torch.cuda.is_available():
            raise LLMClientConfigurationError("The prespecified local evaluation requires CUDA")
        if not model_dir.is_dir():
            raise LLMClientConfigurationError("The pinned local base-model directory is missing")

        self._torch = torch
        self.model_id = model_id
        self.model_revision = str(settings["base_model_revision"])
        self.adapter_id = None
        self.adapter_sha256 = None
        self.max_sequence_tokens = int(settings["maximum_sequence_tokens"])
        self.max_new_tokens = int(settings["maximum_new_tokens"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=str(settings["attention_implementation"]),
        )
        if adapter and training_state is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise LLMClientConfigurationError("PEFT is required for the adapted condition") from exc
            adapter_record = training_state["adapter"]
            adapter_dir = APP_DIR / str(adapter_record["relative_path"])
            self.model = PeftModel.from_pretrained(
                self.model, adapter_dir, is_trainable=False, local_files_only=True
            )
            self.adapter_id = str(adapter_record["model_id"])
            self.adapter_sha256 = str(adapter_record["aggregate_sha256"])
        self.model.to("cuda")
        self.model.eval()
        self.model.config.use_cache = True
        self.dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        self.device = torch.cuda.get_device_name(0)

    def __call__(
        self,
        *,
        model_id: str,
        api_key: str = "",
        case_id: str,
        organ_type: str,
        recipients: List[Dict[str, Any]],
        **_ignored: Any,
    ) -> NoteReviewResult:
        del api_key
        if model_id != self.model_id:
            raise LLMClientConfigurationError("Call model ID differs from the loaded local model")
        payload = build_note_review_payload(case_id, organ_type, recipients)
        expected_ids = [item["recipient_id"] for item in payload["candidates"]]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, sort_keys=True)},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_tokens = int(prompt_ids.shape[-1])
        if input_tokens + self.max_new_tokens > self.max_sequence_tokens:
            raise LLMClientConfigurationError(
                "Prompt plus generation allowance exceeds the frozen sequence limit"
            )
        prompt_ids = prompt_ids.to("cuda")
        attention_mask = self._torch.ones_like(prompt_ids)
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with self._torch.inference_mode():
                output = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            if self._torch.cuda.is_available():
                self._torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"Local model generation failed ({type(exc).__name__})") from exc
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        generated = output[0, input_tokens:]
        output_tokens = int(generated.shape[-1])
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        metadata = ModelRunMetadata(
            model_id=self.model_id,
            provider="local_hugging_face_transformers",
            model_revision=self.model_revision,
            adapter_id=self.adapter_id,
            adapter_sha256=self.adapter_sha256,
            device=self.device,
            dtype=self.dtype,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if not raw_text:
            error = LLMEmptyResponseError("Local model response contained no text")
            error.raw_text = raw_text
            error.model_run_metadata = metadata
            raise error
        try:
            review = parse_note_review(raw_text, case_id, expected_ids)
        except Exception as exc:
            exc.raw_text = raw_text
            exc.model_run_metadata = metadata
            raise
        return NoteReviewResult(review=review, metadata=metadata, raw_text=raw_text)
