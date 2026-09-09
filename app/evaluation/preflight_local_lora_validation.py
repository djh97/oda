"""Stress-test the resumed checkpoint on the full development validation split."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from evaluation.artifact_paths import portable_path
from evaluation.prepare_local_lora_resume import ACTIVE_CHECKPOINT_PATH
from evaluation.prepare_local_model import MODEL_DIR
from evaluation.train_local_lora import (
    APP_DIR,
    CHECKPOINT_INHERITANCE_PATH,
    CausalLMCollator,
    EmptyCudaCacheAfterPredictionCallback,
    AssistantOnlyDataset,
    VALIDATION_PATH,
)
from src.policy import assert_protocol_frozen, load_protocol


OUTPUT_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_validation_preflight.json"
)
LOSS_ABSOLUTE_TOLERANCE = 1e-4


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


class ValidationMemoryMonitor(TrainerCallback):
    def __init__(self) -> None:
        self.prediction_steps = 0
        self.maximum_allocated_bytes = 0
        self.maximum_reserved_bytes = 0

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        self.prediction_steps += 1
        self.maximum_allocated_bytes = max(
            self.maximum_allocated_bytes, torch.cuda.memory_allocated()
        )
        self.maximum_reserved_bytes = max(
            self.maximum_reserved_bytes, torch.cuda.memory_reserved()
        )
        return control


def run_validation_preflight() -> dict[str, Any]:
    assert_protocol_frozen()
    if OUTPUT_PATH.exists():
        raise RuntimeError("A validation-memory preflight record already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("The validation-memory preflight requires CUDA")
    if not ACTIVE_CHECKPOINT_PATH.is_dir() or not CHECKPOINT_INHERITANCE_PATH.is_file():
        raise RuntimeError("The verified active epoch-2 checkpoint is missing")

    settings = load_protocol()["model_evaluation"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    validation = AssistantOnlyDataset(
        VALIDATION_PATH, tokenizer, int(settings["maximum_sequence_tokens"])
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation=str(settings["attention_implementation"]),
    )
    model = PeftModel.from_pretrained(
        base_model,
        ACTIVE_CHECKPOINT_PATH,
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = False
    monitor = ValidationMemoryMonitor()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_PATH.parent / "validation-preflight-work"),
            per_device_eval_batch_size=1,
            fp16=True,
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            prediction_loss_only=True,
            eval_accumulation_steps=1,
        ),
        eval_dataset=validation,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        callbacks=[EmptyCudaCacheAfterPredictionCallback(), monitor],
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        metrics = trainer.evaluate()
        expected_loss = 0.029150526970624924
        observed_loss = float(metrics["eval_loss"])
        if monitor.prediction_steps != len(validation):
            raise RuntimeError("The validation-memory preflight did not process every record")
        loss_absolute_difference = abs(observed_loss - expected_loss)
        if not math.isclose(
            observed_loss,
            expected_loss,
            rel_tol=0.0,
            abs_tol=LOSS_ABSOLUTE_TOLERANCE,
        ):
            raise RuntimeError(
                "The inherited checkpoint validation loss differs from its archived value: "
                f"expected={expected_loss:.15f}, observed={observed_loss:.15f}, "
                f"absolute_difference={loss_absolute_difference:.3e}"
            )
        record = {
            "schema_version": "1.0",
            "status": "passed",
            "completed_at_utc": _utc_now(),
            "purpose": "Verify full-split validation stability before the epoch-3 continuation",
            "held_out_test_file_opened": False,
            "checkpoint": portable_path(ACTIVE_CHECKPOINT_PATH),
            "checkpoint_inheritance_sha256": _sha256(CHECKPOINT_INHERITANCE_PATH),
            "validation_file": portable_path(VALIDATION_PATH),
            "validation_file_sha256": _sha256(VALIDATION_PATH),
            "validation_records": len(validation),
            "prediction_steps": monitor.prediction_steps,
            "expected_eval_loss": expected_loss,
            "observed_eval_loss": observed_loss,
            "eval_loss_absolute_difference": loss_absolute_difference,
            "eval_loss_absolute_tolerance": LOSS_ABSOLUTE_TOLERANCE,
            "metrics": metrics,
            "memory": {
                "maximum_allocated_bytes": monitor.maximum_allocated_bytes,
                "maximum_reserved_bytes": monitor.maximum_reserved_bytes,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "final_allocated_bytes": torch.cuda.memory_allocated(),
                "final_reserved_bytes": torch.cuda.memory_reserved(),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "gpu_name": torch.cuda.get_device_name(0),
            },
            "settings": {
                "prediction_loss_only": True,
                "eval_accumulation_steps": 1,
                "empty_cuda_cache_after_each_prediction_step": True,
                "per_device_eval_batch_size": 1,
            },
            "source_hashes": {
                "training_script": _sha256(APP_DIR / "evaluation" / "train_local_lora.py"),
                "preflight_script": _sha256(Path(__file__).resolve()),
            },
        }
        _write_json(OUTPUT_PATH, record)
        return record
    finally:
        del trainer
        del model
        del base_model
        torch.cuda.empty_cache()


def main() -> None:
    record = run_validation_preflight()
    memory = record["memory"]
    print(
        "Validation-memory preflight passed on "
        f"{record['validation_records']} records; eval_loss={record['observed_eval_loss']:.12f}; "
        f"peak_allocated={memory['peak_allocated_bytes'] / (1024 ** 3):.2f} GiB."
    )


if __name__ == "__main__":
    main()
