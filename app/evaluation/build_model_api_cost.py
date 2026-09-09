"""Compute the hosted-comparator API cost from retained token usage."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from evaluation.run_model_evaluation import validate_test_lock
from src.policy import DEFAULT_PROTOCOL_PATH, load_protocol
from src.schemas import ModelRunMetadata


APP_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
RAW_PATH = EVALUATION_DIR / "openai_raw.jsonl"
CONFIG_PATH = EVALUATION_DIR / "openai_config.json"
SUMMARY_PATH = EVALUATION_DIR / "openai_summary.json"
TEST_LOCK_PATH = EVALUATION_DIR / "test_lock.json"
OUTPUT_PATH = EVALUATION_DIR / "openai_api_cost.json"
SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _decimal_text(value: Decimal, places: str = ".00000001") -> str:
    return format(value.quantize(Decimal(places)), "f")


def compute(*, write_output: bool = True) -> dict[str, Any]:
    protocol = load_protocol()
    hosted = protocol["model_evaluation"]["hosted_comparator"]
    pricing = hosted["pricing"]
    config = _load_json(CONFIG_PATH)
    summary = _load_json(SUMMARY_PATH)
    lock = _load_json(TEST_LOCK_PATH)
    validate_test_lock(lock)
    records = _load_jsonl(RAW_PATH)
    expected_cases = int(protocol["dataset"]["splits"]["test"]["cases"])
    if (
        config.get("condition") != "openai"
        or config.get("model_id_requested") != hosted["model_id"]
        or config.get("test_lock") != lock
        or config.get("raw_file_sha256") != _sha256(RAW_PATH)
        or summary.get("condition_values") != ["openai"]
        or summary.get("records_file_sha256") != _sha256(RAW_PATH)
        or summary.get("config") != config
        or len(records) != expected_cases
    ):
        raise RuntimeError("Hosted-comparator cost inputs are incomplete or inconsistent")

    input_tokens = 0
    output_tokens = 0
    records_with_usage = 0
    missing_usage_case_ids: list[str] = []
    for record in records:
        metadata_value = record.get("model_run")
        if metadata_value is None:
            missing_usage_case_ids.append(str(record.get("case_id", "")))
            continue
        metadata = ModelRunMetadata.model_validate(metadata_value)
        if metadata.model_id != hosted["model_id"]:
            raise RuntimeError("Hosted-comparator usage names a different model")
        if metadata.input_tokens is None or metadata.output_tokens is None:
            missing_usage_case_ids.append(str(record.get("case_id", "")))
            continue
        input_tokens += int(metadata.input_tokens)
        output_tokens += int(metadata.output_tokens)
        records_with_usage += 1

    input_rate = Decimal(str(pricing["input_per_million_tokens"]))
    output_rate = Decimal(str(pricing["output_per_million_tokens"]))
    million = Decimal(1_000_000)
    input_cost = Decimal(input_tokens) * input_rate / million
    output_cost = Decimal(output_tokens) * output_rate / million
    total_cost = input_cost + output_cost
    denominator = Decimal(expected_cases)
    complete = records_with_usage == expected_cases
    record = {
        "schema_version": "1.0",
        "status": "complete_usage_cost" if complete else "observed_usage_lower_bound",
        "generated_at_utc": _utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "model_id": hosted["model_id"],
        "test_case_count": expected_cases,
        "records_with_token_usage": records_with_usage,
        "records_missing_token_usage": len(missing_usage_case_ids),
        "missing_usage_case_ids": missing_usage_case_ids,
        "token_usage": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "combined_total": input_tokens + output_tokens,
        },
        "pricing": pricing,
        "cost_usd": {
            "input": _decimal_text(input_cost),
            "output": _decimal_text(output_cost),
            "total": _decimal_text(total_cost),
            "mean_per_locked_case": _decimal_text(total_cost / denominator),
        },
        "interpretation": (
            "Complete token-accounted API cost for the locked run."
            if complete
            else "Lower bound from records with provider token usage; cases without usage are not assigned zero cost."
        ),
        "source_hashes": {
            "protocol": _sha256(DEFAULT_PROTOCOL_PATH),
            "test_lock": _sha256(TEST_LOCK_PATH),
            "raw_records": _sha256(RAW_PATH),
            "configuration": _sha256(CONFIG_PATH),
            "analysis_summary": _sha256(SUMMARY_PATH),
            "cost_script": _sha256(SCRIPT_PATH),
        },
    }
    if write_output:
        _write_json(OUTPUT_PATH, record)
    return record


def validate_cost() -> dict[str, Any]:
    recorded = _load_json(OUTPUT_PATH)
    recomputed = compute(write_output=False)
    for value in (recorded, recomputed):
        value.pop("generated_at_utc", None)
    if recorded != recomputed:
        raise RuntimeError("Hosted-comparator API cost does not reproduce")
    return _load_json(OUTPUT_PATH)


def main() -> None:
    record = compute()
    print(
        f"Recorded {record['status']} for {record['test_case_count']} hosted-comparator cases."
    )


if __name__ == "__main__":
    main()
