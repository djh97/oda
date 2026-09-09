"""Exercise one discarded LoRA backward pass on the longest development example."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluation.artifact_paths import portable_path
from evaluation.prepare_local_model import MODEL_DIR, OUTPUT_PATH as BASE_MODEL_MANIFEST_PATH
from evaluation.train_local_lora import (
    APP_DIR,
    AssistantOnlyDataset,
    CausalLMCollator,
    TRAINING_PATH,
    VALIDATION_PATH,
)
from src.policy import DEFAULT_PROTOCOL_PATH, assert_protocol_frozen, load_protocol


OUTPUT_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_preflight.json"
)


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


def run_preflight() -> dict[str, Any]:
    assert_protocol_frozen()
    expected_source_hashes = {
        "protocol": _sha256(DEFAULT_PROTOCOL_PATH),
        "base_model_manifest": _sha256(BASE_MODEL_MANIFEST_PATH),
        "training": _sha256(TRAINING_PATH),
        "validation": _sha256(VALIDATION_PATH),
        "training_script": _sha256(APP_DIR / "evaluation" / "train_local_lora.py"),
        "preflight_script": _sha256(Path(__file__).resolve()),
    }
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "passed":
            raise RuntimeError("Existing local LoRA preflight did not pass")
        if existing.get("source_hashes") != expected_source_hashes:
            raise RuntimeError("Existing local LoRA preflight is stale for the current sources")
        return existing
    if not torch.cuda.is_available():
        raise RuntimeError("Local LoRA preflight requires CUDA")
    settings = load_protocol()["model_evaluation"]
    tuning = settings["fine_tuning"]
    seed = int(settings["seed"])
    set_seed(seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    maximum_length = int(settings["maximum_sequence_tokens"])
    training = AssistantOnlyDataset(TRAINING_PATH, tokenizer, maximum_length)
    validation = AssistantOnlyDataset(VALIDATION_PATH, tokenizer, maximum_length)
    all_records = [("training", index, value) for index, value in enumerate(training.records)]
    all_records.extend(("validation", index, value) for index, value in enumerate(validation.records))
    split, index, longest = max(all_records, key=lambda item: len(item[2]["input_ids"]))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation=str(settings["attention_implementation"]),
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=int(tuning["lora_rank"]),
            lora_alpha=int(tuning["lora_alpha"]),
            lora_dropout=float(tuning["lora_dropout"]),
            bias=str(tuning["lora_bias"]),
            target_modules=list(tuning["target_modules"]),
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    model.train()
    batch = {
        name: tensor.to("cuda")
        for name, tensor in CausalLMCollator(tokenizer.pad_token_id)([longest]).items()
    }
    try:
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise RuntimeError("Preflight loss is not finite")
        output.loss.backward()
        trainable_with_gradients = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if trainable_with_gradients < 1:
            raise RuntimeError("No trainable LoRA parameter received a gradient")
        record = {
            "schema_version": "1.0",
            "status": "passed",
            "completed_at_utc": _utc_now(),
            "purpose": (
                "Discarded implementation preflight on the longest training/validation sequence; "
                "no resulting weights were retained"
            ),
            "held_out_test_file_opened": False,
            "development_split": split,
            "development_row_index": index,
            "sequence_tokens": len(longest["input_ids"]),
            "loss_finite": True,
            "trainable_parameters_with_gradients": trainable_with_gradients,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu_name": torch.cuda.get_device_name(0),
            "source_hashes": expected_source_hashes,
            "base_model_directory": portable_path(MODEL_DIR),
        }
        _write_json(OUTPUT_PATH, record)
        return record
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        del model
        torch.cuda.empty_cache()


def main() -> None:
    record = run_preflight()
    print(
        "Local LoRA preflight passed on the longest development sequence "
        f"({record['sequence_tokens']} tokens; peak GPU memory "
        f"{record['peak_gpu_memory_bytes'] / (1024 ** 3):.2f} GiB)."
    )


if __name__ == "__main__":
    main()
