"""Train and archive the prespecified local LoRA note-review adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.prepare_local_model import MODEL_DIR, OUTPUT_PATH as BASE_MODEL_MANIFEST_PATH
from evaluation.synthetic_dataset import DATASET_DIR, validate_existing
from src.llm_client import SYSTEM_PROMPT
from src.policy import DEFAULT_PROTOCOL_PATH, assert_protocol_frozen, load_protocol


APP_DIR = Path(__file__).resolve().parents[1]
TRAINING_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
GENERATION_MANIFEST_PATH = DATASET_DIR / "generation_manifest.json"
MODEL_OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "model"
STATE_PATH = MODEL_OUTPUT_DIR / "local_lora_training.json"
CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "local_lora_checkpoints"
CHECKPOINT_INHERITANCE_PATH = MODEL_OUTPUT_DIR / "local_lora_checkpoint_inheritance.json"
ADAPTER_DIR = MODEL_OUTPUT_DIR / "local_lora_adapter"
ADAPTER_MODEL_ID = "oda-qwen2.5-1.5b-instruct-lora-v1.1.0"
TORCH_EMPTY_CACHE_STEPS = 1


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _aggregate_hash(files: Mapping[str, Mapping[str, Any]]) -> str:
    canonical = json.dumps(files, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _directory_files(directory: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if not files:
        raise RuntimeError(f"Artifact directory is empty: {directory}")
    return files


def _runtime_memory_management() -> dict[str, Any]:
    return {
        "torch_empty_cache_steps": TORCH_EMPTY_CACHE_STEPS,
        "prediction_loss_only": True,
        "eval_accumulation_steps": 1,
        "empty_cuda_cache_after_each_prediction_step": True,
        "duplicate_post_training_evaluation": False,
        "purpose": (
            "Bound CUDA allocator growth for variable-length training and validation "
            "batches under Windows WDDM."
        ),
        "scientific_hyperparameters_changed": False,
    }


class EmptyCudaCacheAfterPredictionCallback(TrainerCallback):
    """Release unreferenced CUDA blocks after each validation prediction step."""

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


def _last_epoch_validation_metrics(
    log_history: Iterable[Mapping[str, Any]], expected_step: int
) -> dict[str, Any]:
    evaluations = [dict(item) for item in log_history if "eval_loss" in item]
    if not evaluations:
        raise RuntimeError("Trainer history contains no epoch-end validation metrics")
    final = evaluations[-1]
    if int(final.get("step", -1)) != expected_step:
        raise RuntimeError("Final epoch-end validation did not complete at the final update step")
    return deepcopy_json(final)


def _validate_checkpoint_inheritance(
    *,
    protocol: Mapping[str, Any],
    configuration: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not CHECKPOINT_INHERITANCE_PATH.is_file():
        raise RuntimeError("A pre-existing checkpoint has no inheritance record")
    try:
        record = json.loads(CHECKPOINT_INHERITANCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The checkpoint inheritance record is unreadable") from exc
    if (
        record.get("status") != "verified"
        or record.get("protocol_id") != protocol["protocol_id"]
        or record.get("protocol_version") != protocol["version"]
        or record.get("held_out_test_file_opened") is not False
        or record.get("scientific_configuration_changed") is not False
        or record.get("configuration") != configuration
        or record.get("current_source_hashes") != source_hashes
    ):
        raise RuntimeError("The checkpoint inheritance record differs from the frozen run")

    active = record.get("active_checkpoint")
    source = record.get("source_checkpoint")
    if not isinstance(active, dict) or not isinstance(source, dict):
        raise RuntimeError("The checkpoint inheritance manifests are missing")
    try:
        active_path = resolve_recorded_path(active.get("path"), permitted_root=CHECKPOINT_DIR)
        source_path = resolve_recorded_path(
            source.get("path"),
            permitted_root=APP_DIR / "pipeline-output" / "archive",
        )
    except ValueError as exc:
        raise RuntimeError("A checkpoint inheritance path is invalid") from exc
    for label, path, manifest in (
        ("active", active_path, active),
        ("source", source_path, source),
    ):
        if not path.is_dir():
            raise RuntimeError(f"The inherited {label} checkpoint is missing")
        files = _directory_files(path)
        if (
            manifest.get("files") != files
            or manifest.get("aggregate_sha256") != _aggregate_hash(files)
        ):
            raise RuntimeError(f"The inherited {label} checkpoint failed hash verification")
    if active.get("files") != source.get("files"):
        raise RuntimeError("The active checkpoint differs from its archived source")
    trainer_state = json.loads((active_path / "trainer_state.json").read_text(encoding="utf-8"))
    progress = record.get("completed_progress")
    if (
        not isinstance(progress, dict)
        or int(progress.get("global_step", -1)) != 400
        or float(progress.get("epoch", -1)) != 2.0
        or int(trainer_state.get("global_step", -1)) != 400
        or float(trainer_state.get("epoch", -1)) != 2.0
    ):
        raise RuntimeError("The inherited checkpoint is not the completed epoch-2 checkpoint")
    return record


class AssistantOnlyDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, path: Path, tokenizer: Any, maximum_length: int) -> None:
        self.records: list[dict[str, list[int]]] = []
        for row_number, row in enumerate(_load_jsonl(path), start=1):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise RuntimeError(f"Training row {row_number} has invalid messages")
            prompt_ids = tokenizer.apply_chat_template(
                messages[:2], tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(f"Training row {row_number} has no stable assistant boundary")
            if len(full_ids) > maximum_length:
                raise RuntimeError(
                    f"Training row {row_number} exceeds the frozen maximum sequence length"
                )
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            if all(value == -100 for value in labels):
                raise RuntimeError(f"Training row {row_number} has no assistant target")
            self.records.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _source_hashes() -> dict[str, str]:
    return {
        "protocol": _sha256(DEFAULT_PROTOCOL_PATH),
        "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "training_file": _sha256(TRAINING_PATH),
        "validation_file": _sha256(VALIDATION_PATH),
        "generation_manifest": _sha256(GENERATION_MANIFEST_PATH),
        "base_model_manifest": _sha256(BASE_MODEL_MANIFEST_PATH),
        "training_script": _sha256(Path(__file__).resolve()),
        "llm_client": _sha256(APP_DIR / "src" / "llm_client.py"),
        "schemas": _sha256(APP_DIR / "src" / "schemas.py"),
        "requirements_ml": _sha256(APP_DIR / "requirements-ml.txt"),
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "accelerate": importlib.metadata.version("accelerate"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
    }


def _training_configuration(settings: Mapping[str, Any]) -> dict[str, Any]:
    tuning = settings["fine_tuning"]
    return {
        "base_model_snapshot": settings["base_model_snapshot"],
        "precision": settings["precision"],
        "attention_implementation": settings["attention_implementation"],
        "maximum_sequence_tokens": settings["maximum_sequence_tokens"],
        **deepcopy_json(tuning),
        "seed": settings["seed"],
    }


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _validate_preconditions(settings: Mapping[str, Any]) -> dict[str, Any]:
    assert_protocol_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen local LoRA run requires a CUDA GPU")
    validation = validate_existing()
    if not validation["valid"]:
        raise RuntimeError("Synthetic development data failed validation")
    manifest = json.loads(BASE_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise RuntimeError("The local base-model manifest is not verified")
    maximum = int(manifest["token_audit"]["full_sequence_tokens"]["maximum"])
    if maximum > int(settings["maximum_sequence_tokens"]):
        raise RuntimeError("A development sequence exceeds the frozen maximum length")
    development = manifest.get("development_files", {})
    if (
        development.get("training", {}).get("sha256") != _sha256(TRAINING_PATH)
        or development.get("validation", {}).get("sha256") != _sha256(VALIDATION_PATH)
    ):
        raise RuntimeError("The base-model token audit is stale for the development files")
    return manifest


def _adapter_files() -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in ADAPTER_DIR.rglob("*") if item.is_file()):
        relative = path.relative_to(ADAPTER_DIR).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if not files:
        raise RuntimeError("The completed adapter directory is empty")
    return files


def train_local_lora() -> dict[str, Any]:
    protocol = load_protocol()
    settings = protocol["model_evaluation"]
    base_manifest = _validate_preconditions(settings)
    source_hashes = _source_hashes()
    configuration = _training_configuration(settings)
    runtime_memory_management = _runtime_memory_management()
    last_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR)) if CHECKPOINT_DIR.exists() else None
    inheritance: dict[str, Any] | None = None
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if (
            state.get("source_hashes") != source_hashes
            or state.get("configuration") != configuration
            or state.get("runtime_memory_management") != runtime_memory_management
        ):
            raise RuntimeError("Existing local training state belongs to different frozen inputs")
        if state.get("status") == "completed":
            return state
        inherited_wrapper = state.get("checkpoint_inheritance")
        if inherited_wrapper is not None:
            inheritance = _validate_checkpoint_inheritance(
                protocol=protocol,
                configuration=configuration,
                source_hashes=source_hashes,
            )
            if (
                not isinstance(inherited_wrapper, dict)
                or inherited_wrapper.get("artifact_path")
                != portable_path(CHECKPOINT_INHERITANCE_PATH)
                or inherited_wrapper.get("artifact_sha256")
                != _sha256(CHECKPOINT_INHERITANCE_PATH)
                or inherited_wrapper.get("record") != inheritance
            ):
                raise RuntimeError("The running state has an invalid checkpoint inheritance record")
    else:
        if last_checkpoint:
            inheritance = _validate_checkpoint_inheritance(
                protocol=protocol,
                configuration=configuration,
                source_hashes=source_hashes,
            )
        elif CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
            raise RuntimeError("The checkpoint directory contains no resumable Trainer checkpoint")
        state = {
            "schema_version": "1.0",
            "status": "initialized",
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"],
            "created_at_utc": _utc_now(),
            "purpose": "Supervised LoRA adaptation using training and validation splits only",
            "held_out_test_file_opened": False,
            "source_hashes": source_hashes,
            "configuration": configuration,
            "runtime_memory_management": runtime_memory_management,
            "base_model": {
                "repository": base_manifest["repository"],
                "revision": base_manifest["revision"],
                "manifest_path": portable_path(BASE_MODEL_MANIFEST_PATH),
                "parameter_count": base_manifest["parameter_count"],
            },
            "software_environment": _environment(),
            "invocations": [],
        }
        if inheritance is not None:
            state["checkpoint_inheritance"] = {
                "artifact_path": portable_path(CHECKPOINT_INHERITANCE_PATH),
                "artifact_sha256": _sha256(CHECKPOINT_INHERITANCE_PATH),
                "record": inheritance,
            }

    invocation = {
        "started_at_utc": _utc_now(),
        "resume_checkpoint": None,
        "process_id": os.getpid(),
    }
    state.setdefault("invocations", []).append(invocation)
    state["status"] = "running"
    state["last_started_at_utc"] = invocation["started_at_utc"]
    _write_json(STATE_PATH, state)

    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    maximum_length = int(settings["maximum_sequence_tokens"])
    train_dataset = AssistantOnlyDataset(TRAINING_PATH, tokenizer, maximum_length)
    validation_dataset = AssistantOnlyDataset(VALIDATION_PATH, tokenizer, maximum_length)
    state["dataset_records"] = {
        "training": len(train_dataset),
        "validation": len(validation_dataset),
    }
    _write_json(STATE_PATH, state)

    tuning = settings["fine_tuning"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation=str(settings["attention_implementation"]),
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=int(tuning["lora_rank"]),
        lora_alpha=int(tuning["lora_alpha"]),
        lora_dropout=float(tuning["lora_dropout"]),
        bias=str(tuning["lora_bias"]),
        target_modules=list(tuning["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    state["parameters"] = {
        "base": int(base_manifest["parameter_count"]),
        "model_with_adapter": total,
        "trainable": trainable,
        "trainable_proportion": trainable / total,
    }
    _write_json(STATE_PATH, state)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        overwrite_output_dir=False,
        num_train_epochs=float(tuning["epochs"]),
        per_device_train_batch_size=int(tuning["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tuning["per_device_validation_batch_size"]),
        gradient_accumulation_steps=int(tuning["gradient_accumulation_steps"]),
        learning_rate=float(tuning["learning_rate"]),
        lr_scheduler_type=str(tuning["learning_rate_scheduler"]),
        warmup_ratio=float(tuning["warmup_ratio"]),
        weight_decay=float(tuning["weight_decay"]),
        max_grad_norm=float(tuning["maximum_gradient_norm"]),
        optim=str(tuning["optimizer"]),
        fp16=True,
        gradient_checkpointing=bool(tuning["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=bool(tuning["load_best_model_at_end"]),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=[],
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        torch_empty_cache_steps=TORCH_EMPTY_CACHE_STEPS,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        callbacks=[EmptyCudaCacheAfterPredictionCallback()],
    )
    invocation["resume_checkpoint"] = portable_path(Path(last_checkpoint)) if last_checkpoint else None
    _write_json(STATE_PATH, state)

    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        final_validation = _last_epoch_validation_metrics(
            trainer.state.log_history, trainer.state.global_step
        )
        if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
            raise RuntimeError("Final adapter directory already contains files")
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
        tokenizer.save_pretrained(ADAPTER_DIR)
        files = _adapter_files()
        state["adapter"] = {
            "model_id": ADAPTER_MODEL_ID,
            "relative_path": str(ADAPTER_DIR.relative_to(APP_DIR)).replace("\\", "/"),
            "files": files,
            "aggregate_sha256": _aggregate_hash(files),
        }
        state["training_result"] = {
            "metrics": deepcopy_json(train_result.metrics),
            "final_validation_metrics": deepcopy_json(final_validation),
            "best_checkpoint": (
                portable_path(Path(trainer.state.best_model_checkpoint))
                if trainer.state.best_model_checkpoint
                else None
            ),
            "best_metric": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "log_history": deepcopy_json(trainer.state.log_history),
        }
        state["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        state["status"] = "completed"
        state["completed_at_utc"] = _utc_now()
        invocation["completed_at_utc"] = state["completed_at_utc"]
        invocation["outcome"] = "completed"
        _write_json(STATE_PATH, state)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["last_failure"] = {
            "at_utc": _utc_now(),
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
        invocation["completed_at_utc"] = state["last_failure"]["at_utc"]
        invocation["outcome"] = state["status"]
        _write_json(STATE_PATH, state)
        raise
    finally:
        del trainer
        del model
        torch.cuda.empty_cache()


def main() -> None:
    state = train_local_lora()
    print(
        f"Local LoRA training status={state['status']} "
        f"adapter={state.get('adapter', {}).get('model_id', 'pending')}"
    )


if __name__ == "__main__":
    main()
