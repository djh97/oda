from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.train_local_lora import (
    AssistantOnlyDataset,
    CausalLMCollator,
    EmptyCudaCacheAfterPredictionCallback,
    _last_epoch_validation_metrics,
)
from src import local_llm_client
from src.llm_client import LLMClientConfigurationError


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        if len(messages) == 2 and add_generation_prompt:
            return [10, 11, 12]
        if len(messages) == 3 and not add_generation_prompt:
            return [10, 11, 12, 20, 21]
        raise AssertionError("unexpected template call")


def _write_row(path: Path) -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ]
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_dataset_masks_system_and_user_tokens(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    _write_row(path)
    dataset = AssistantOnlyDataset(path, FakeTokenizer(), maximum_length=5)
    assert dataset[0]["input_ids"] == [10, 11, 12, 20, 21]
    assert dataset[0]["labels"] == [-100, -100, -100, 20, 21]


def test_dataset_refuses_truncation(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    _write_row(path)
    with pytest.raises(RuntimeError, match="exceeds"):
        AssistantOnlyDataset(path, FakeTokenizer(), maximum_length=4)


def test_collator_masks_padding() -> None:
    collator = CausalLMCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]


def test_validation_callback_releases_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("evaluation.train_local_lora.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "evaluation.train_local_lora.torch.cuda.empty_cache", lambda: calls.append(True)
    )
    control = object()
    returned = EmptyCudaCacheAfterPredictionCallback().on_prediction_step(
        object(), object(), control
    )
    assert returned is control
    assert calls == [True]


def test_final_validation_uses_last_completed_epoch_entry() -> None:
    history = [
        {"loss": 0.02, "step": 590},
        {"eval_loss": 0.03, "epoch": 2.0, "step": 400},
        {"eval_loss": 0.02, "epoch": 3.0, "step": 600},
    ]
    assert _last_epoch_validation_metrics(history, 600) == history[-1]
    with pytest.raises(RuntimeError, match="final update step"):
        _last_epoch_validation_metrics(history, 601)


def test_local_base_model_files_are_verified_against_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weights.write_bytes(b"weights")

    def record(path: Path) -> dict[str, object]:
        return {
            "bytes": path.stat().st_size,
            "sha256": local_llm_client._sha256(path),
        }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "verified",
                "repository": "repository",
                "revision": "revision",
                "license": "license",
                "files": {
                    "config.json": record(config),
                    "model.safetensors": record(weights),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        local_llm_client,
        "load_protocol",
        lambda: {
            "model_evaluation": {
                "base_model_repository": "repository",
                "base_model_revision": "revision",
                "base_model_license": "license",
            }
        },
    )

    local_llm_client.validate_local_base_model(model_dir, manifest_path)
    weights.write_bytes(b"changed")
    with pytest.raises(LLMClientConfigurationError, match="failed verification"):
        local_llm_client.validate_local_base_model(model_dir, manifest_path)


def test_local_training_state_is_bound_to_protocol_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_dir = tmp_path / "app"
    dataset_dir = tmp_path / "datasets"
    adapter_dir = app_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    dataset_dir.mkdir()
    paths = {
        "protocol": app_dir / "protocol.json",
        "training_file": dataset_dir / "training.jsonl",
        "validation_file": dataset_dir / "validation.jsonl",
        "generation_manifest": dataset_dir / "manifest.json",
        "base_model_manifest": app_dir / "base.json",
        "training_script": app_dir / "evaluation" / "train_local_lora.py",
        "llm_client": app_dir / "src" / "llm_client.py",
        "schemas": app_dir / "src" / "schemas.py",
        "requirements_ml": app_dir / "requirements-ml.txt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"adapter")

    tuning = {"method": "LoRA", "epochs": 3}
    settings = {
        "base_model_snapshot": "base@revision",
        "precision": "float16",
        "attention_implementation": "sdpa",
        "maximum_sequence_tokens": 2048,
        "fine_tuning": tuning,
        "seed": 42,
    }
    protocol = {
        "protocol_id": "protocol",
        "version": "1.1.0",
        "model_evaluation": settings,
    }
    monkeypatch.setattr(local_llm_client, "APP_DIR", app_dir)
    monkeypatch.setattr(local_llm_client, "DEFAULT_PROTOCOL_PATH", paths["protocol"])
    monkeypatch.setattr(local_llm_client, "TRAINING_PATH", paths["training_file"])
    monkeypatch.setattr(local_llm_client, "VALIDATION_PATH", paths["validation_file"])
    monkeypatch.setattr(
        local_llm_client, "GENERATION_MANIFEST_PATH", paths["generation_manifest"]
    )
    monkeypatch.setattr(
        local_llm_client, "BASE_MODEL_MANIFEST_PATH", paths["base_model_manifest"]
    )
    monkeypatch.setattr(local_llm_client, "load_protocol", lambda: protocol)

    sources = {
        name: local_llm_client._sha256(path)
        for name, path in paths.items()
    }
    sources["system_prompt"] = local_llm_client.hashlib.sha256(
        local_llm_client.SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    adapter_files = {
        adapter_file.name: {
            "bytes": adapter_file.stat().st_size,
            "sha256": local_llm_client._sha256(adapter_file),
        }
    }
    state_path = app_dir / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol_id": "protocol",
                "protocol_version": "1.1.0",
                "held_out_test_file_opened": False,
                "invocations": [{"outcome": "completed"}],
                "runtime_memory_management": {
                    "torch_empty_cache_steps": 1,
                    "prediction_loss_only": True,
                    "eval_accumulation_steps": 1,
                    "empty_cuda_cache_after_each_prediction_step": True,
                    "duplicate_post_training_evaluation": False,
                    "purpose": (
                        "Bound CUDA allocator growth for variable-length training and "
                        "validation batches under Windows WDDM."
                    ),
                    "scientific_hyperparameters_changed": False,
                },
                "configuration": {
                    "base_model_snapshot": "base@revision",
                    "precision": "float16",
                    "attention_implementation": "sdpa",
                    "maximum_sequence_tokens": 2048,
                    **tuning,
                    "seed": 42,
                },
                "source_hashes": sources,
                "adapter": {
                    "model_id": "adapter-model",
                    "relative_path": "adapter",
                    "files": adapter_files,
                    "aggregate_sha256": local_llm_client._aggregate_hash(adapter_files),
                },
            }
        ),
        encoding="utf-8",
    )

    local_llm_client.validate_local_training_state(state_path)
    paths["validation_file"].write_text("changed", encoding="utf-8")
    with pytest.raises(LLMClientConfigurationError, match="inputs or sources changed"):
        local_llm_client.validate_local_training_state(state_path)
