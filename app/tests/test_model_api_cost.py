from __future__ import annotations

import json
from pathlib import Path

from evaluation import build_model_api_cost as model_cost


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_hosted_api_cost_uses_retained_token_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = {
        "protocol_id": "protocol",
        "version": "1.2.0",
        "dataset": {"splits": {"test": {"cases": 2}}},
        "model_evaluation": {
            "hosted_comparator": {
                "model_id": "gpt-4o-mini-2024-07-18",
                "pricing": {
                    "currency": "USD",
                    "input_per_million_tokens": 0.15,
                    "output_per_million_tokens": 0.6,
                    "pricing_basis": "standard",
                    "source_url": "https://example.test/model",
                    "accessed_on": "2026-09-08",
                },
            }
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    raw_path = tmp_path / "openai_raw.jsonl"
    rows = [
        {
            "case_id": "case-1",
            "status": "success",
            "model_run": {
                "model_id": "gpt-4o-mini-2024-07-18",
                "latency_ms": 1,
                "input_tokens": 100,
                "output_tokens": 20,
            },
        },
        {
            "case_id": "case-2",
            "status": "failure",
            "model_run": {
                "model_id": "gpt-4o-mini-2024-07-18",
                "latency_ms": 1,
                "input_tokens": 200,
                "output_tokens": 30,
            },
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    lock_path = tmp_path / "test_lock.json"
    _write_json(lock_path, {"locked": True})
    config_path = tmp_path / "openai_config.json"
    config = {
        "condition": "openai",
        "model_id_requested": "gpt-4o-mini-2024-07-18",
        "test_lock": {"locked": True},
        "raw_file_sha256": model_cost._sha256(raw_path),
    }
    _write_json(config_path, config)
    summary_path = tmp_path / "openai_summary.json"
    _write_json(
        summary_path,
        {
            "condition_values": ["openai"],
            "records_file_sha256": model_cost._sha256(raw_path),
            "config": config,
        },
    )
    output_path = tmp_path / "openai_api_cost.json"

    monkeypatch.setattr(model_cost, "load_protocol", lambda: protocol)
    monkeypatch.setattr(model_cost, "DEFAULT_PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(model_cost, "RAW_PATH", raw_path)
    monkeypatch.setattr(model_cost, "CONFIG_PATH", config_path)
    monkeypatch.setattr(model_cost, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(model_cost, "TEST_LOCK_PATH", lock_path)
    monkeypatch.setattr(model_cost, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(model_cost, "validate_test_lock", lambda _value: None)

    record = model_cost.compute()

    assert record["status"] == "complete_usage_cost"
    assert record["token_usage"] == {
        "input_total": 300,
        "output_total": 50,
        "combined_total": 350,
    }
    assert record["cost_usd"]["input"] == "0.00004500"
    assert record["cost_usd"]["output"] == "0.00003000"
    assert record["cost_usd"]["total"] == "0.00007500"
    model_cost.validate_cost()


def test_missing_usage_is_reported_as_a_lower_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = {
        "protocol_id": "protocol",
        "version": "1.2.0",
        "dataset": {"splits": {"test": {"cases": 1}}},
        "model_evaluation": {
            "hosted_comparator": {
                "model_id": "gpt-4o-mini-2024-07-18",
                "pricing": {
                    "currency": "USD",
                    "input_per_million_tokens": 0.15,
                    "output_per_million_tokens": 0.6,
                },
            }
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    raw_path = tmp_path / "openai_raw.jsonl"
    raw_path.write_text(
        json.dumps({"case_id": "case-1", "status": "failure", "model_run": None}) + "\n",
        encoding="utf-8",
    )
    lock_path = tmp_path / "test_lock.json"
    _write_json(lock_path, {"locked": True})
    config = {
        "condition": "openai",
        "model_id_requested": "gpt-4o-mini-2024-07-18",
        "test_lock": {"locked": True},
        "raw_file_sha256": model_cost._sha256(raw_path),
    }
    config_path = tmp_path / "openai_config.json"
    summary_path = tmp_path / "openai_summary.json"
    _write_json(config_path, config)
    _write_json(
        summary_path,
        {
            "condition_values": ["openai"],
            "records_file_sha256": model_cost._sha256(raw_path),
            "config": config,
        },
    )

    for name, value in (
        ("load_protocol", lambda: protocol),
        ("DEFAULT_PROTOCOL_PATH", protocol_path),
        ("RAW_PATH", raw_path),
        ("CONFIG_PATH", config_path),
        ("SUMMARY_PATH", summary_path),
        ("TEST_LOCK_PATH", lock_path),
        ("OUTPUT_PATH", tmp_path / "cost.json"),
        ("validate_test_lock", lambda _value: None),
    ):
        monkeypatch.setattr(model_cost, name, value)

    record = model_cost.compute()
    assert record["status"] == "observed_usage_lower_bound"
    assert record["records_missing_token_usage"] == 1
    assert "not assigned zero cost" in record["interpretation"]
