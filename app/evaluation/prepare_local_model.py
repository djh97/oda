"""Verify and fingerprint the pinned local base model before protocol amendment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from evaluation.artifact_paths import portable_path


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
MODEL_DIR = APP_DIR / ".model-cache" / "Qwen2.5-1.5B-Instruct"
DATASET_DIR = IMPLEMENTATION_DIR / "datasets" / "synthetic-v1"
TRAINING_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
OUTPUT_PATH = (
    APP_DIR / "pipeline-output" / "current" / "model" / "local_base_model_manifest.json"
)

MODEL_REPOSITORY = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "5fee7c4ed634dc66c6e318c8ac2897b8b9154536"
MODEL_LICENSE = "Apache-2.0"
EXPECTED_ARCHITECTURE = "Qwen2ForCausalLM"
EXPECTED_PARAMETER_COUNT = 1_543_714_304
REQUIRED_FILES = {
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


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


def _percentile(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty sequence")
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _summary(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "minimum": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values),
    }


def _parameter_count(path: Path) -> int:
    count = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            shape = tensors.get_slice(name).get_shape()
            size = 1
            for dimension in shape:
                size *= int(dimension)
            count += size
    return count


def _token_audit(tokenizer: Any, paths: Iterable[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for path in paths:
        rows = _load_jsonl(path)
        split_counts[path.name] = len(rows)
        records.extend(rows)

    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    full_lengths: list[int] = []
    for row_number, row in enumerate(records, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise RuntimeError(f"Development row {row_number} does not contain three messages")
        prompt_ids = tokenizer.apply_chat_template(
            messages[:2], tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"Development row {row_number} cannot be masked at the assistant boundary"
            )
        completion_length = len(full_ids) - len(prompt_ids)
        if completion_length < 1:
            raise RuntimeError(f"Development row {row_number} has an empty assistant target")
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(completion_length)
        full_lengths.append(len(full_ids))

    return {
        "scope": "training and validation files only; the held-out test file was not opened",
        "split_record_counts": split_counts,
        "total_record_count": len(records),
        "prompt_tokens": _summary(prompt_lengths),
        "assistant_target_tokens": _summary(completion_lengths),
        "full_sequence_tokens": _summary(full_lengths),
        "assistant_only_loss_mask_verified": True,
    }


def prepare_local_model(
    *,
    model_dir: Path = MODEL_DIR,
    training_path: Path = TRAINING_PATH,
    validation_path: Path = VALIDATION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise RuntimeError(f"Pinned local model directory is missing: {model_dir}")
    model_files = sorted(
        path for path in model_dir.rglob("*") if path.is_file() and ".cache" not in path.parts
    )
    relative_names = {path.relative_to(model_dir).as_posix() for path in model_files}
    missing = sorted(REQUIRED_FILES.difference(relative_names))
    if missing:
        raise RuntimeError("Pinned local model is incomplete: " + ", ".join(missing))

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    architectures = list(getattr(config, "architectures", []) or [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeError(f"Unexpected model architecture: {architectures}")
    parameter_count = _parameter_count(model_dir / "model.safetensors")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count {parameter_count}; expected {EXPECTED_PARAMETER_COUNT}"
        )
    license_text = (model_dir / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("The retained model license is not Apache License 2.0")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The pinned tokenizer has no chat template")
    token_audit = _token_audit(tokenizer, (training_path, validation_path))

    files = {
        path.relative_to(model_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in model_files
    }
    manifest = {
        "schema_version": "1.0",
        "status": "verified",
        "verified_at_utc": _utc_now(),
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "revision_type": "immutable_commit",
        "license": MODEL_LICENSE,
        "local_directory": portable_path(model_dir),
        "architecture": EXPECTED_ARCHITECTURE,
        "parameter_count": parameter_count,
        "model_type": str(getattr(config, "model_type", "")),
        "files": files,
        "development_files": {
            "training": {
                "path": portable_path(training_path),
                "sha256": _sha256(training_path),
            },
            "validation": {
                "path": portable_path(validation_path),
                "sha256": _sha256(validation_path),
            },
        },
        "token_audit": token_audit,
        "software_environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "preparation_script_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(output_path, manifest)
    return manifest


def main() -> None:
    manifest = prepare_local_model()
    maximum = manifest["token_audit"]["full_sequence_tokens"]["maximum"]
    print(
        "Verified pinned local model and development-only token audit "
        f"({manifest['parameter_count']} parameters; maximum sequence {maximum} tokens)."
    )
    print(f"Saved manifest to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
