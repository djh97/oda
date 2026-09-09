from __future__ import annotations

import base64
import csv
import hashlib
import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation import build_manuscript_tables as tables
from evaluation import build_evidence_manifest as evidence_manifest
from evaluation.build_cost_estimates import build as build_cost_estimate
from evaluation.paper_full_workflow import _expected_full_transaction_plan


VALID_CID = "b" + base64.b32encode(
    b"\x01\x70\x12\x20" + b"\0" * 32
).decode("ascii").rstrip("=").lower()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _record_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_smoke_environment_snapshot_matches_frozen_preflight_sources() -> None:
    expected_hash = _sha256(tables.PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH)
    assert expected_hash == "d206a1548f9d425094af7496bdb6b615aee9b0836bcde1962978943635daa79d"
    assert _sha256(tables.PROJECT_ENVIRONMENT_PATH) != expected_hash

    expected_sources = tables.smoke_source_hashes()
    expected_sources["project_environment"] = expected_hash
    for name in ("untuned_preflight", "fine_tuned_preflight", "openai_preflight"):
        record = json.loads(
            (tables.MODEL_DIR / "smoke_tests" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["implementation_sha256"] == expected_sources


def _metric(count: int, denominator: int) -> dict[str, object]:
    value = count / denominator
    return {
        "count": count,
        "denominator": denominator,
        "proportion": value,
        "proportion_95ci": [max(0, value - 0.02), min(1, value + 0.02)],
    }


def _analysis_summary(
    condition: str,
    model_id: str,
    raw_hash: str,
    config: dict[str, object],
    test_file_sha256: str,
) -> dict[str, object]:
    return {
        "condition_values": [condition],
        "model_id_requested_values": [model_id],
        "model_id_observed_values": [model_id],
        "test_case_count": 400,
        "raw_record_count": 400,
        "missing_record_count": 0,
        "successful_case_count": 395,
        "failure_counts": {"SyntheticFailure": 5},
        "records_file_sha256": raw_hash,
        "test_file_sha256": test_file_sha256,
        "config": config,
        "classification": {
            "candidate_count": 4000,
            "macro_f1": 0.9,
            "macro_f1_95ci": [0.88, 0.92],
            "by_class": {
                "temporary_hold": {
                    "tp": 700,
                    "support": 750,
                    "recall": 700 / 750,
                    "recall_95ci": [0.91, 0.95],
                }
            },
        },
        "case_metrics": {
            "baseline_top1_conformant": _metric(280, 400),
            "baseline_primary_unsafe": _metric(120, 400),
            "guarded_top1_conformant": _metric(350, 400),
            "unsafe_or_missing_primary": _metric(20, 400),
            "model_success": _metric(395, 400),
        },
    }


def _actor_addresses() -> dict[str, str]:
    return {
        name: f"0x{index:040x}"
        for index, name in enumerate(
            (
                "REGULATOR_PRIVATE_KEY",
                "HOSPITAL_PRIVATE_KEY",
                "ETHICS_PRIVATE_KEY",
                "MEDICAL_PRIVATE_KEY",
                "DECISION_SERVICE_PRIVATE_KEY",
                "DONOR_PRIVATE_KEY",
                *(f"RECIPIENT{i}_PRIVATE_KEY" for i in range(1, 11)),
            ),
            start=2,
        )
    }


def _transaction_rows(
    addresses: dict[str, str],
    *,
    primary_recipient_id: int = 6,
    backup_recipient_id: int = 8,
    match_id: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for expected in _expected_full_transaction_plan(primary_recipient_id):
        number = len(rows) + 1
        arguments = expected["arguments"]
        if arguments == "dynamic":
            arguments = (
                f"donor_id=1,primary={primary_recipient_id},backup={backup_recipient_id},"
                "encrypted_decision_cid=<recorded separately>"
                if expected["function"] == "createMatch"
                else f"match_id={match_id}"
            )
        rows.append({
            "category": expected["category"],
            "role": expected["role"],
            "function": expected["function"],
            "arguments": arguments,
            "sender": addresses[expected["sender_key"]],
            "tx_hash": f"0x{number:064x}",
            "status": 1,
            "block_number": number,
            "gas_used": 100,
            "effective_gas_price_wei": 1,
            "fee_wei": 100,
            "fee_native": "0.000000000000000100",
            "confirmation_seconds": 1.0,
        })
    return rows


def _canonical_transaction_summary() -> tuple[dict[str, object], list[dict[str, object]], dict[str, str]]:
    addresses = _actor_addresses()
    transactions = _transaction_rows(addresses)
    validation = tables._validate_full_transaction_records(transactions, addresses, 6, 8, 1)
    summary: dict[str, object] = {
        "deployment_tx_hash": transactions[0]["tx_hash"],
        "match_id": 1,
        "guarded_decision": {
            "primary_recipient_id": 6,
            "backup_recipient_id": 8,
        },
        "final_checks": {key: True for key in tables.EXPECTED_FINAL_CHECKS},
        "transaction_plan_validation": validation,
        "transaction_count": len(transactions),
        "gas_used_total": 4500,
        "fee_wei_total": 4500,
        "fee_native_total": "0.000000000000004500",
    }
    return summary, transactions, addresses


def test_manuscript_transaction_gate_recomputes_the_canonical_plan() -> None:
    summary, transactions, addresses = _canonical_transaction_summary()
    tables._validate_canonical_transactions(summary, transactions, addresses)

    tampered = deepcopy(transactions)
    tampered[1]["role"] = "Hospital"
    with pytest.raises(tables.ManuscriptEvidenceError, match="malformed or inconsistent"):
        tables._validate_canonical_transactions(summary, tampered, addresses)


def test_manuscript_deployment_gate_recomputes_runtime_identity(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    _write_json(artifact_path, {"deployedBytecode": {"object": "0x60016000"}})
    runtime_sha256 = hashlib.sha256(bytes.fromhex("60016000")).hexdigest()
    verification_path = tmp_path / "deployment_verification.json"
    verification = {
        "verified_at_utc": "2026-09-07T00:00:00Z",
        "contract_address": "0x0000000000000000000000000000000000000001",
        "deployment_tx_hash": "0x" + "1" * 64,
        "runtime_byte_count": 4,
        "artifact_runtime_sha256": runtime_sha256,
        "deployed_runtime_sha256": runtime_sha256,
        "matches": True,
    }
    _write_json(verification_path, verification)
    summary = {
        "contract_address": verification["contract_address"],
        "deployment_tx_hash": verification["deployment_tx_hash"],
        "deployment_verification_sha256": _sha256(verification_path),
        "deployed_runtime_sha256": runtime_sha256,
    }
    with patch.object(tables, "ARTIFACT_PATH", artifact_path):
        tables._validate_deployment_verification(tmp_path, summary)
        verification["deployed_runtime_sha256"] = "0" * 64
        _write_json(verification_path, verification)
        summary["deployment_verification_sha256"] = _sha256(verification_path)
        with pytest.raises(tables.ManuscriptEvidenceError, match="pinned runtime bytecode"):
            tables._validate_deployment_verification(tmp_path, summary)


def test_final_manuscript_fragments_require_one_consistent_evidence_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "canonical"
    evaluation_dir = tmp_path / "evaluation"
    model_dir = tmp_path / "model"
    target_dir = tmp_path / "journal"
    run_dir.mkdir()
    test_path = tmp_path / "heldout_fixture.jsonl"
    test_path.write_text(
        "".join(json.dumps({"case_id": f"case-{index}"}) + "\n" for index in range(400)),
        encoding="utf-8",
    )
    test_hash = _sha256(test_path)

    targets = {
        name: target_dir / path.name
        for name, path in tables.TARGETS.items()
    }
    target_dir.mkdir()
    for path in targets.values():
        path.write_text("% PLACEHOLDER: test\n", encoding="utf-8")

    models = {
        "tfidf_logistic": "tfidf-logistic-v1",
        "untuned": tables.PROTOCOL["model_evaluation"]["base_model_snapshot"],
        "fine_tuned": "ft:test:model",
        "openai": tables.PROTOCOL["model_evaluation"]["hosted_comparator"]["model_id"],
    }
    test_lock = {"protocol_id": "test-lock"}
    _write_json(evaluation_dir / "test_lock.json", test_lock)
    configs: dict[str, dict[str, object]] = {}
    for condition, model_id in models.items():
        raw_path = evaluation_dir / f"{condition}_raw.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            "".join(
                json.dumps({
                    "case_id": f"case-{index}",
                    "condition": condition,
                    "model_id_requested": model_id,
                }) + "\n"
                for index in range(400)
            ),
            encoding="utf-8",
        )
        config = {
            "condition": condition,
            "model_id_requested": model_id,
            "raw_file_sha256": _sha256(raw_path),
            "raw_record_count": 400,
            "test_lock": test_lock,
            "evaluation_in_progress": False,
            "software_environment": {
                "python": "3.12.10",
                "openai": "3.8.0",
                "pydantic": "2.13.5",
            },
        }
        attempt_path = evaluation_dir / f"{condition}_attempts.jsonl"
        attempt_path.write_text(
            "".join(
                json.dumps({
                    "event": "attempt_started",
                    "case_id": f"case-{index}",
                    "condition": condition,
                    "model_id_requested": model_id,
                    "started_at_utc": "2026-09-07T00:00:00Z",
                }) + "\n"
                for index in range(400)
            ),
            encoding="utf-8",
        )
        config.update({
            "attempt_log_file": str(attempt_path),
            "attempt_log_sha256": _sha256(attempt_path),
            "attempted_case_count": 400,
        })
        configs[condition] = config
        _write_json(evaluation_dir / f"{condition}_config.json", config)
        _write_json(
            evaluation_dir / f"{condition}_summary.json",
            _analysis_summary(condition, model_id, _sha256(raw_path), config, test_hash),
        )
    for left, right in tables.PAIRED_COMPARISONS:
        _write_json(
            evaluation_dir / f"{left}_vs_{right}_comparison.json",
            {
                "case_count": 400,
                "test_file_sha256": test_hash,
                "left": {
                    "label": left,
                    "sha256": _sha256(evaluation_dir / f"{left}_raw.jsonl"),
                    "config": configs[left],
                },
                "right": {
                    "label": right,
                    "sha256": _sha256(evaluation_dir / f"{right}_raw.jsonl"),
                    "config": configs[right],
                },
            },
        )
    settings = tables.PROTOCOL["model_evaluation"]
    local_run_id = "a" * 32
    result_path = model_dir / "result_files" / "file-result_metrics.csv"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("step,train_loss\n1,0.1\n", encoding="utf-8")
    job = {
        "id": "ftjob-test",
        "object": "fine_tuning.job",
        "status": "succeeded",
        "fine_tuned_model": "ft:test:model",
        "model": settings["base_model_snapshot"],
        "training_file": "file-training",
        "validation_file": "file-validation",
        "seed": settings["seed"],
        "hyperparameters": {"n_epochs": settings["fine_tuning"]["epochs"]},
        "method": {"type": "supervised"},
        "metadata": {
            "protocol": tables.PROTOCOL["protocol_id"],
            "purpose": "synthetic-note-readiness",
            "local_run_id": local_run_id,
        },
        "result_files": ["file-result"],
    }
    access_check = {
        "model_retrievable": True,
        "model_id_matches": True,
        "base_model_requested": settings["base_model_snapshot"],
    }
    access_check_path = model_dir / "fine_tuning_access_check.json"
    _write_json(access_check_path, access_check)
    fine_tuning_state = {
        "local_run_id": local_run_id,
        "request_ids": {
            "training_upload": f"oda-ft-{local_run_id}-training-upload",
            "validation_upload": f"oda-ft-{local_run_id}-validation-upload",
            "job_creation": f"oda-ft-{local_run_id}-job-creation",
        },
        "base_model_requested": settings["base_model_snapshot"],
            "protocol": {
                "id": tables.PROTOCOL["protocol_id"],
                "version": tables.PROTOCOL["version"],
                "path": str(tables.DEFAULT_PROTOCOL_PATH.resolve()),
                "sha256": _sha256(tables.DEFAULT_PROTOCOL_PATH),
            },
            "generation_manifest": {
                "path": str(tables.GENERATION_MANIFEST_PATH.resolve()),
                "sha256": _sha256(tables.GENERATION_MANIFEST_PATH),
            },
            "training_dataset": {
                "path": str(tables.TRAINING_PATH.resolve()),
                "sha256": _sha256(tables.TRAINING_PATH),
            },
            "validation_dataset": {
                "path": str(tables.VALIDATION_PATH.resolve()),
                "sha256": _sha256(tables.VALIDATION_PATH),
            },
        "training_file": {
            "id": "file-training",
            "purpose": "fine-tune",
            "filename": tables.TRAINING_PATH.name,
            "bytes": tables.TRAINING_PATH.stat().st_size,
        },
        "validation_file": {
            "id": "file-validation",
            "purpose": "fine-tune",
            "filename": tables.VALIDATION_PATH.name,
            "bytes": tables.VALIDATION_PATH.stat().st_size,
        },
        "system_prompt_sha256": hashlib.sha256(tables.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "fine_tuning_file_validation": {
            "training": {
                "row_count": tables.PROTOCOL["dataset"]["splits"]["training"]["cases"],
                "sha256": _sha256(tables.TRAINING_PATH),
            },
            "validation": {
                "row_count": tables.PROTOCOL["dataset"]["splits"]["validation"]["cases"],
                "sha256": _sha256(tables.VALIDATION_PATH),
            },
        },
        "request_without_file_ids": {
            "model": settings["base_model_snapshot"],
            "suffix": "oda-note-readiness-v1",
            "seed": settings["seed"],
            "method": {
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "n_epochs": settings["fine_tuning"]["epochs"],
                            "batch_size": settings["fine_tuning"].get("batch_size", "auto"),
                            "learning_rate_multiplier": settings["fine_tuning"].get(
                                "learning_rate_multiplier", "auto"
                            ),
                    }
                },
            },
            "metadata": {
                "protocol": tables.PROTOCOL["protocol_id"],
                "purpose": "synthetic-note-readiness",
                "local_run_id": local_run_id,
            },
        },
        "access_check": access_check,
        "fine_tuning_job_authorization": {
            "confirmed": True,
            "job_id": "ftjob-test",
        },
        "job_creation_attempts": [
            {
                "started_at_utc": "2026-09-07T00:00:00Z",
                "completed_at_utc": "2026-09-07T00:00:01Z",
                "request_id": f"oda-ft-{local_run_id}-job-creation",
                "outcome": "provider_response_received",
            }
        ],
        "job": job,
        "job_events": [{"id": "event-1", "message": "completed"}],
        "job_events_pagination": {
            "initial_page_limit": 100,
            "page_count": 1,
            "event_count": 1,
            "unique_event_count": 1,
            "final_page_has_more": False,
            "complete": True,
        },
        "archived_result_files": [
            {
                "file_id": "file-result",
                "provider_metadata": {"id": "file-result", "filename": "metrics.csv"},
                "local_path": str(result_path.resolve()),
                "bytes": result_path.stat().st_size,
                "sha256": _sha256(result_path),
            }
        ],
    }
    fine_tuning_state["provider_job_validation"] = tables.validate_provider_job(
        fine_tuning_state,
        job,
    )
    _write_json(model_dir / "fine_tuning_job.json", fine_tuning_state)
    local_training_path = model_dir / "local_lora_training.json"
    _write_json(local_training_path, fine_tuning_state)
    _write_json(model_dir / "classical_validation_diagnostics.json", {"fixture": True})
    _write_json(model_dir / "smoke_tests" / "untuned_preflight.json", {"fixture": True})
    _write_json(model_dir / "smoke_tests" / "fine_tuned_preflight.json", {"fixture": True})

    model_evidence = {"fixture_sha256": "0" * 64}
    api_cost = {
        "status": "complete_usage_cost",
        "test_case_count": 400,
        "token_usage": {"input_total": 40000, "output_total": 10000},
        "cost_usd": {"total": "0.01200000", "mean_per_locked_case": "0.00003000"},
    }
    model_api_cost_path = evaluation_dir / "openai_api_cost.json"
    _write_json(model_api_cost_path, api_cost)

    demo = tables._read_json(tables.DEMO_CASE_PATH)
    note_review_value = {
        "case_id": demo["case_id"],
        "assessments": [
            {
                "recipient_id": recipient_id,
                "state": "temporary_hold" if recipient_id == 10 else "eligible",
                "evidence_codes": [
                    "explicit_temporary_deferral" if recipient_id == 10 else "no_current_concern"
                ],
            }
            for recipient_id in range(1, 11)
        ],
    }
    note_review = tables.parse_note_review(
        note_review_value,
        str(demo["case_id"]),
        [int(item["recipient_id"]) for item in demo["recipients"]],
    )
    guarded = tables.apply_guarded_policy(
        1,
        tables.rank_recipients_baseline(demo["donor"], demo["recipients"], tables.PROTOCOL),
        note_review,
        [int(item["recipient_id"]) for item in demo["recipients"]],
    ).model_dump(mode="json")
    protocol_hash = _sha256(tables.DEFAULT_PROTOCOL_PATH)
    addresses = _actor_addresses()
    transactions = _transaction_rows(addresses)
    assert len(transactions) == 45
    _write_csv(run_dir / "transaction_manifest.csv", transactions)
    _write_json(run_dir / "actor_addresses.json", addresses)
    contract_build = {
        "offline": True,
        "command": ["forge", "build", "--force", "--offline", "--use", "solc-0.8.26.exe"],
        "forge_version": "forge test-version",
        "solc_version": "Version: 0.8.26",
        "solc_binary": {
            "filename": "solc-0.8.26.exe",
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "foundry_configuration_sha256": _sha256(tables.SMART_CONTRACTS_DIR / "foundry.toml"),
        "contract_source_sha256": _sha256(tables.SOURCE_PATH),
        "artifact_sha256": _sha256(tables.ARTIFACT_PATH),
        "artifact_compiler": {"version": f"{tables.SOLC_VERSION}+commit.test"},
    }
    _write_json(run_dir / "contract_build.json", contract_build)
    artifact_runtime = tables._artifact_bytecode(
        tables._read_json(tables.ARTIFACT_PATH),
        "deployedBytecode",
    )
    runtime_sha256 = hashlib.sha256(artifact_runtime).hexdigest()
    deployment_verification = {
        "verified_at_utc": "2026-09-07T00:00:00Z",
        "contract_address": "0x0000000000000000000000000000000000000001",
        "deployment_tx_hash": transactions[0]["tx_hash"],
        "runtime_byte_count": len(artifact_runtime),
        "artifact_runtime_sha256": runtime_sha256,
        "deployed_runtime_sha256": runtime_sha256,
        "matches": True,
    }
    _write_json(run_dir / "deployment_verification.json", deployment_verification)
    transaction_plan_validation = tables._validate_full_transaction_records(
        transactions,
        addresses,
        6,
        8,
        1,
    )
    model_metadata = tables.ModelRunMetadata(
        model_id="ft:test:model",
        provider="local_hugging_face_transformers",
        model_revision="fixture-revision",
        adapter_id="ft:test:model",
        adapter_sha256="a" * 64,
        device="fixture-gpu",
        dtype="float16",
        latency_ms=25.0,
        input_tokens=100,
        output_tokens=50,
    ).model_dump(mode="json")
    reference_comparison = {
        "reference_primary_recipient_id": int(demo["reference"]["primary_recipient_id"]),
        "reference_backup_recipient_id": int(demo["reference"]["backup_recipient_id"]),
        "observed_primary_recipient_id": int(guarded["primary_recipient_id"]),
        "observed_backup_recipient_id": int(guarded["backup_recipient_id"]),
        "primary_conforms": (
            int(guarded["primary_recipient_id"])
            == int(demo["reference"]["primary_recipient_id"])
        ),
        "backup_conforms": (
            int(guarded["backup_recipient_id"])
            == int(demo["reference"]["backup_recipient_id"])
        ),
    }
    model_and_guard_record = {
        "case_id": demo["case_id"],
        "model_id_requested": "ft:test:model",
        "model_request_config": {
            "temperature": tables.PROTOCOL["model_evaluation"]["temperature"],
            "seed": tables.PROTOCOL["model_evaluation"]["seed"],
            "provider": tables.PROTOCOL["model_evaluation"]["provider"],
            "do_sample": tables.PROTOCOL["model_evaluation"]["do_sample"],
            "maximum_new_tokens": tables.PROTOCOL["model_evaluation"]["maximum_new_tokens"],
            "system_prompt_sha256": hashlib.sha256(
                tables.SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "raw_response": json.dumps(note_review_value),
        "parsed_review": note_review.model_dump(mode="json"),
        "metadata": model_metadata,
        "guarded_decision": guarded,
        "reference_comparison": reference_comparison,
    }
    _write_json(run_dir / "model_and_guard_record.json", model_and_guard_record)
    run_summary = {
        "protocol_id": tables.PROTOCOL["protocol_id"],
        "protocol_sha256": protocol_hash,
        "network": "sepolia",
        "chain_id": 11155111,
        "contract_address": "0x0000000000000000000000000000000000000001",
        "deployment_tx_hash": transactions[0]["tx_hash"],
        "model_id_requested": "ft:test:model",
        "model": model_metadata,
        "model_evidence": model_evidence,
        "implementation_sha256": {
            "workflow": _sha256(tables.WORKFLOW_PATH),
            "llm_client": _sha256(tables.LLM_CLIENT_PATH),
            "local_llm_client": _sha256(tables.LOCAL_LLM_CLIENT_PATH),
            "policy": _sha256(tables.POLICY_PATH),
            "decision_guard": _sha256(tables.GUARD_PATH),
            "schemas": _sha256(tables.SCHEMAS_PATH),
            "secure_storage": _sha256(tables.SECURE_STORAGE_PATH),
            "ipfs_client": _sha256(tables.IPFS_CLIENT_PATH),
            "pinata_client": _sha256(tables.PINATA_CLIENT_PATH),
            "transactions": _sha256(tables.TRANSACTIONS_PATH),
                "project_environment": _sha256(
                    tables.PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH
                ),
                "artifact_paths": _sha256(tables.ARTIFACT_PATHS_PATH),
                "contract": _sha256(tables.SOURCE_PATH),
        },
        "contract_source_sha256": _sha256(tables.SOURCE_PATH),
        "foundry_artifact_sha256": _sha256(tables.ARTIFACT_PATH),
        "contract_build_sha256": _sha256(run_dir / "contract_build.json"),
        "foundry_configuration_sha256": _sha256(tables.SMART_CONTRACTS_DIR / "foundry.toml"),
        "deployment_verification_sha256": _sha256(run_dir / "deployment_verification.json"),
        "model_and_guard_record_sha256": _sha256(run_dir / "model_and_guard_record.json"),
        "deployed_runtime_sha256": runtime_sha256,
        "decision_cid": VALID_CID,
        "demo_case_id": demo["case_id"],
        "demo_case_sha256": _sha256(tables.DEMO_CASE_PATH),
        "match_id": 1,
        "guarded_decision": guarded,
        "demo_reference_comparison": reference_comparison,
        "final_checks": {key: True for key in tables.EXPECTED_FINAL_CHECKS},
        "transaction_plan_validation": transaction_plan_validation,
        "transaction_count": 45,
        "gas_used_total": 4500,
        "fee_wei_total": 4500,
        "fee_native_total": "0.000000000000004500",
    }
    _write_json(run_dir / "run_summary.json", run_summary)

    stage_counts = {
        "Deployment": 1,
        "Governance": 4,
        "Identity binding": 11,
        "Profile registration": 11,
        "Workflow eligibility": 11,
        "Match workflow": 7,
    }
    build_cost_estimate(
        run_dir,
        eth_usd=Decimal("2000"),
        price_observed_at="2026-09-07T00:00:00Z",
        price_source_url="https://example.invalid/price",
        gas_prices_gwei=(Decimal("5"), Decimal("15"), Decimal("30")),
    )
    _write_csv(
        run_dir / "transaction_stage_summary.csv",
        [
            {
                "category": stage,
                "transaction_count": count,
                "confirmation_seconds_median": 1.0,
                "confirmation_seconds_min": 0.8,
                "confirmation_seconds_max": 1.2,
            }
            for stage, count in stage_counts.items()
        ],
    )
    benchmark_dir = run_dir / "offchain_benchmark" / "run-1"
    benchmark_records: list[dict[str, object]] = []
    benchmark_events: list[dict[str, object]] = []
    for phase, count in (("warmup", 3), ("measured", 30)):
        for run_number in range(1, count + 1):
            attempt_id = f"{phase}-{run_number:03d}"
            started_at = f"2026-09-07T00:{len(benchmark_records):02d}:00Z"
            failed = phase == "measured" and run_number == 30
            record: dict[str, object] = {
                "attempt_id": attempt_id,
                "phase": phase,
                "run_number": run_number,
                "started_at_utc": started_at,
                "status": "failure" if failed else "success",
                "completed_at_utc": started_at,
            }
            if failed:
                record.update(
                    {
                        "total_seconds": 0.2,
                        "error_type": "SyntheticFailure",
                        "error": "fixture",
                    }
                )
            else:
                record.update(
                    {
                        "reference_primary_conforms": True,
                        "reference_backup_conforms": True,
                    }
                )
            benchmark_records.append(record)
            benchmark_events.extend(
                [
                    {
                        "attempt_id": attempt_id,
                        "event": "started",
                        "phase": phase,
                        "recorded_at_utc": started_at,
                        "run_number": run_number,
                    },
                    {
                        "attempt_id": attempt_id,
                        "event": "finished",
                        "phase": phase,
                        "record_sha256": _record_sha256(record),
                        "recorded_at_utc": started_at,
                        "run_number": run_number,
                        "status": record["status"],
                    },
                ]
            )
    _write_jsonl(benchmark_dir / "runs.jsonl", benchmark_records)
    _write_jsonl(benchmark_dir / "attempts.jsonl", benchmark_events)
    _write_csv(
        benchmark_dir / "timings.csv",
        [{"attempt_id": record["attempt_id"], "status": record["status"]} for record in benchmark_records],
    )
    benchmark_implementation = tables.benchmark_source_hashes()
    benchmark_implementation["project_environment"] = _sha256(
        tables.PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH
    )
    benchmark_summary = {
        "source_workflow_run": str(run_dir.resolve()),
        "source_run_summary_sha256": _sha256(run_dir / "run_summary.json"),
        "benchmark_script_sha256": _sha256(tables.BENCHMARK_PATH),
        "implementation_sha256": benchmark_implementation,
        "environment": {
            "platform": "test-platform",
            "machine": "test-machine",
            "processor": "test-processor",
            "logical_cpu_count": 1,
            "python": "3.12.10",
            "packages": {
                "openai": "3.8.0",
                "pydantic": "2.13.5",
                "cryptography": "test",
                "requests": "test",
            },
        },
        "warmup_runs_excluded": 3,
        "measured_runs_requested": 30,
        "measured_runs_attempted": 30,
        "measured_runs_successful": 29,
        "measured_runs_failed": 1,
        "failed_measured_attempts_rerun": False,
        "attempt_ledger_event_count": 66,
        "attempt_ledger_sha256": _sha256(benchmark_dir / "attempts.jsonl"),
        "raw_runs_sha256": _sha256(benchmark_dir / "runs.jsonl"),
        "replacement_attempt_count": 0,
        "interrupted_attempts_retained_as_failures": 0,
        "ipfs_fetch_retries_per_measurement": 0,
        "all_measured_primary_conform": False,
        "all_measured_backup_conform": False,
        "timings": {
            key: {
                "n": 29,
                "median_seconds": 0.2,
                "p95_seconds": 0.3,
                "min_seconds": 0.1,
                "max_seconds": 0.4,
            }
            for key in (
                "encrypted_record_retrieval",
                "protocol_ranking",
                "model_note_review",
                "deterministic_guard",
                "total",
            )
        },
    }
    _write_json(benchmark_dir / "summary.json", benchmark_summary)
    _write_json(
        benchmark_dir / "artifact_hashes.json",
        {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                benchmark_dir / "attempts.jsonl",
                benchmark_dir / "runs.jsonl",
                benchmark_dir / "timings.csv",
                benchmark_dir / "summary.json",
            )
        },
    )
    _write_json(
        run_dir / "offchain_benchmark" / "completed_run.json",
        {
            "run_directory": "run-1",
            "summary_sha256": _sha256(benchmark_dir / "summary.json"),
            "attempt_ledger_sha256": _sha256(benchmark_dir / "attempts.jsonl"),
            "raw_runs_sha256": _sha256(benchmark_dir / "runs.jsonl"),
        },
    )
    _write_json(
        run_dir / "offchain_benchmark" / "run_state.json",
        {
            "status": "completed",
            "run_directory": "run-1",
            "source_run_summary_sha256": benchmark_summary["source_run_summary_sha256"],
            "implementation_sha256": benchmark_summary["implementation_sha256"],
            "software_environment": benchmark_summary["environment"],
            "summary_sha256": _sha256(benchmark_dir / "summary.json"),
            "attempt_ledger_sha256": _sha256(benchmark_dir / "attempts.jsonl"),
            "raw_runs_sha256": _sha256(benchmark_dir / "runs.jsonl"),
        },
    )

    with (
        patch.object(tables, "TARGETS", targets),
        patch.object(tables, "EVALUATION_DIR", evaluation_dir),
        patch.object(tables, "MODEL_DIR", model_dir),
        patch.object(tables, "ACCESS_CHECK_PATH", access_check_path),
        patch.object(tables, "LOCAL_LORA_TRAINING_PATH", local_training_path),
            patch.object(tables, "TEST_PATH", test_path),
            patch.object(tables, "MODEL_API_COST_PATH", model_api_cost_path),
        patch.object(tables, "validate_test_lock"),
        patch.object(tables, "_require_reproducible_analysis"),
        patch.object(tables, "_require_reproducible_comparison"),
        patch.object(tables, "_validate_pretest_artifacts", return_value=()),
        patch.object(tables, "_validate_fine_tuning_state", return_value="ft:test:model"),
        patch.object(tables, "_fine_tuning_source_paths", return_value=()),
            patch.object(tables, "_final_model_evidence_hashes", return_value=model_evidence),
            patch.object(tables, "validate_model_api_cost", return_value=api_cost),
            patch.object(tables, "_model_summary_text", return_value="fixture model summary\n"),
        ):
        manifest = tables.build(run_dir)

    assert len(manifest["generated"]) == 7
    assert json.loads(targets["listing"].read_text(encoding="utf-8"))["case_id"] == (
        "oda-demo-kidney-001"
    )
    assert all("PLACEHOLDER" not in path.read_text(encoding="utf-8") for path in targets.values())
    assert (run_dir / "manuscript_table_manifest.json").exists()
    verified = evidence_manifest._verify_record_map(
        manifest["generated"],
        base_dir=target_dir,
        permitted_root=target_dir,
        label="test generated manifest",
    )
    assert verified == {path.resolve() for path in targets.values()}

    model_and_guard_record["raw_response"] = "{}"
    _write_json(run_dir / "model_and_guard_record.json", model_and_guard_record)
    run_summary["model_and_guard_record_sha256"] = _sha256(
        run_dir / "model_and_guard_record.json"
    )
    with pytest.raises(tables.ManuscriptEvidenceError, match="malformed or does not replay"):
        tables._validate_canonical_decision(run_dir, run_summary)
