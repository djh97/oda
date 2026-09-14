"""Build pending manuscript fragments from the frozen final evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.analyze_evaluation import analyze as recompute_analysis
from evaluation.benchmark_offchain import (
    SCRIPT_PATH as BENCHMARK_PATH,
    _resolve_run_dir,
    audit_attempt_provenance,
    benchmark_source_hashes,
)
from evaluation.build_cost_estimates import (
    TABLE7_NETWORK_SCENARIOS,
    TABLE7_OBSERVATION_DATE,
    validate_cost_artifacts,
)
from evaluation.build_model_api_cost import (
    OUTPUT_PATH as MODEL_API_COST_PATH,
    validate_cost as validate_model_api_cost,
)
from evaluation.compare_model_conditions import compare as recompute_comparison
from evaluation.manage_fine_tuning import ACCESS_CHECK_PATH, validate_provider_job
from evaluation.prepare_local_model import OUTPUT_PATH as LOCAL_BASE_MODEL_MANIFEST_PATH
from evaluation.preflight_local_lora import OUTPUT_PATH as LOCAL_LORA_PREFLIGHT_PATH
from evaluation.train_local_lora import (
    STATE_PATH as LOCAL_LORA_TRAINING_PATH,
    _training_configuration as local_training_configuration,
)
from evaluation.paper_full_workflow import (
    ARTIFACT_PATHS_PATH,
    ARTIFACT_PATH,
    DEFAULT_PROTOCOL_PATH,
    DEMO_CASE_PATH,
    GUARD_PATH,
    IPFS_CLIENT_PATH,
    LLM_CLIENT_PATH,
    LOCAL_LLM_CLIENT_PATH,
    PINATA_CLIENT_PATH,
    POLICY_PATH,
    PROJECT_ENVIRONMENT_PATH,
    SCHEMAS_PATH,
    SECURE_STORAGE_PATH,
    SMART_CONTRACTS_DIR,
    SOLC_VERSION,
    SOURCE_PATH,
    TRANSACTIONS_PATH,
    WORKFLOW_PATH,
    WorkflowError,
    _artifact_bytecode,
    _validate_full_transaction_records,
)
from evaluation.run_model_evaluation import _attempted_case_ids, validate_test_lock
from evaluation.smoke_test_model import (
    _condition_runtime_settings as smoke_runtime_settings,
    smoke_source_hashes,
)
from evaluation.synthetic_dataset import DATASET_DIR, PROTOCOL
from src.decision_guard import apply_guarded_policy
from src.ipfs_client import validate_cid
from src.llm_client import SYSTEM_PROMPT, parse_note_review
from src.local_llm_client import validate_local_training_state
from src.policy import rank_recipients_baseline
from src.schemas import ModelRunMetadata


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parents[1]
EVALUATION_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
MODEL_DIR = APP_DIR / "pipeline-output" / "current" / "model"
MANUSCRIPT_STAGING_DIR = APP_DIR / "pipeline-output" / "current" / "manuscript"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER_MARKER = "% PLACEHOLDER:"
TEST_PATH = DATASET_DIR / "test_cases.jsonl"
TRAINING_PATH = DATASET_DIR / "fine_tuning_training.jsonl"
VALIDATION_PATH = DATASET_DIR / "fine_tuning_validation.jsonl"
GENERATION_MANIFEST_PATH = DATASET_DIR / "generation_manifest.json"
TRAINING_CASES_PATH = DATASET_DIR / "training_cases.jsonl"
VALIDATION_CASES_PATH = DATASET_DIR / "validation_cases.jsonl"
CLASSICAL_BASELINE_PATH = APP_DIR / "evaluation" / "run_classical_baseline.py"
SMOKE_TEST_PATH = APP_DIR / "evaluation" / "smoke_test_model.py"
LOCAL_LORA_VALIDATION_PREFLIGHT_PATH = MODEL_DIR / "local_lora_validation_preflight.json"
LOCAL_LORA_INHERITANCE_PATH = MODEL_DIR / "local_lora_checkpoint_inheritance.json"
OPENAI_COMPARATOR_ACCESS_PATH = MODEL_DIR / "openai_comparator_access_check.json"
OPENAI_COMPARATOR_PREFLIGHT_SCRIPT_PATH = (
    APP_DIR / "evaluation" / "preflight_openai_comparator.py"
)
PROTOCOL_LINEAGE_PATH = (
    APP_DIR / "pipeline-output" / "current" / "protocol" / "protocol_lineage_v1.2.0.json"
)
PROTOCOL_AMENDMENT_PATH = (
    APP_DIR / "pipeline-output" / "current" / "protocol" / "protocol_amendment_v1.2.0.json"
)
PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH = (
    APP_DIR
    / "pipeline-output"
    / "archive"
    / "pretest_protocol_v1.1.0_2026-09-08_openai-comparator-amendment"
    / "project_environment.v1.1.0.py"
)

TARGETS = {
    "transactions": MANUSCRIPT_STAGING_DIR / "generated_tx_trace_rows.tex",
    "addresses": MANUSCRIPT_STAGING_DIR / "generated_address_rows.tex",
    "primary_results": MANUSCRIPT_STAGING_DIR / "generated_primary_result_rows.tex",
    "model_summary": MANUSCRIPT_STAGING_DIR / "generated_model_summary.tex",
    "costs": MANUSCRIPT_STAGING_DIR / "generated_cost_rows.tex",
    "latency": MANUSCRIPT_STAGING_DIR / "generated_latency_rows.tex",
    "listing": MANUSCRIPT_STAGING_DIR / "generated_note_review_example.json",
}

CONDITIONS = (
    ("tfidf_logistic", "TF--IDF logistic model + guard"),
    ("untuned", "Untuned Qwen + guard"),
    ("fine_tuned", "Fine-tuned Qwen + guard"),
    ("openai", "GPT-4o mini + guard"),
)

PAIRED_COMPARISONS = (
    ("tfidf_logistic", "untuned"),
    ("tfidf_logistic", "fine_tuned"),
    ("tfidf_logistic", "openai"),
    ("untuned", "fine_tuned"),
    ("untuned", "openai"),
    ("fine_tuned", "openai"),
)

STAGES = (
    "Deployment",
    "Governance",
    "Identity binding",
    "Profile registration",
    "Workflow eligibility",
    "Match workflow",
)

COST_FUNCTION_ORDER = {
    "Deployment": ("constructor",),
    "Governance": (
        "setHospital",
        "setEthicsCommittee",
        "setMedicalTeam",
        "setDecisionService",
    ),
    "Identity binding": ("registerDonorAuthority", "registerRecipientAddress"),
    "Profile registration": ("registerDonor", "registerRecipient"),
    "Workflow eligibility": ("setDonorEligibility", "setRecipientEligibility"),
    "Match workflow": (
        "createMatch",
        "approveMedicalTeam",
        "approveHospital",
        "approveDonorAuthority",
        "approveRecipient",
        "approveFinalTransplant",
        "finalizeMatch",
    ),
}

EXPECTED_FINAL_CHECKS = frozenset({
    "match_finalized",
    "match_not_cancelled",
    "recorded_by_decision_service",
    "onchain_primary_matches_guard",
    "onchain_backup_matches_guard",
    "onchain_active_recipient_matches_guard",
    "decision_cid_matches",
    "donor_finalized",
    "donor_has_no_open_match",
    "active_recipient_transplanted",
    "backup_recipient_not_transplanted",
    "primary_reservation_released",
    "backup_reservation_released",
    "all_transactions_succeeded",
    "transaction_plan_complete",
    "deployed_code_matches_artifact",
})


class ManuscriptEvidenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ManuscriptEvidenceError(f"Missing evidence file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManuscriptEvidenceError(f"Expected a JSON object: {path}")
    return value


def _jsonl_case_order(path: Path, *, label: str) -> list[str]:
    if not path.is_file():
        raise ManuscriptEvidenceError(f"Missing {label}: {path}")
    case_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManuscriptEvidenceError(
                    f"{label} has invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(value, Mapping) or not str(value.get("case_id", "")).strip():
                raise ManuscriptEvidenceError(
                    f"{label} has no case ID on line {line_number}"
                )
            case_ids.append(str(value["case_id"]))
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ManuscriptEvidenceError(f"{label} has missing or duplicate case IDs")
    return case_ids


def _validate_contract_build(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    build_path = run_dir / "contract_build.json"
    build = _read_json(build_path)
    if summary.get("contract_build_sha256") != _sha256(build_path):
        raise ManuscriptEvidenceError("Canonical run has a stale contract-build hash")
    if build.get("offline") is not True:
        raise ManuscriptEvidenceError("Canonical contract build was not recorded as offline")
    command = build.get("command")
    if not isinstance(command, list) or "--offline" not in command or "--use" not in command:
        raise ManuscriptEvidenceError("Canonical contract build did not record the pinned offline command")
    if (
        build.get("contract_source_sha256") != _sha256(SOURCE_PATH)
        or build.get("artifact_sha256") != summary.get("foundry_artifact_sha256")
        or not re.fullmatch(r"[0-9a-f]{64}", str(build.get("artifact_sha256", "")))
        or build.get("foundry_configuration_sha256")
        != _sha256(SMART_CONTRACTS_DIR / "foundry.toml")
        or summary.get("foundry_configuration_sha256")
        != _sha256(SMART_CONTRACTS_DIR / "foundry.toml")
    ):
        raise ManuscriptEvidenceError("Canonical contract-build inputs changed after execution")
    compiler = build.get("artifact_compiler")
    if not isinstance(compiler, Mapping) or not str(compiler.get("version", "")).startswith(
        f"{SOLC_VERSION}+"
    ):
        raise ManuscriptEvidenceError("Canonical contract artifact was not built with the pinned compiler")
    solc_binary = build.get("solc_binary")
    try:
        solc_binary_bytes = int(solc_binary.get("bytes", 0)) if isinstance(solc_binary, Mapping) else 0
    except (TypeError, ValueError) as exc:
        raise ManuscriptEvidenceError(
            "Canonical contract-build tool provenance has an invalid compiler byte count"
        ) from exc
    if (
        not isinstance(solc_binary, Mapping)
        or solc_binary_bytes < 1
        or not re.fullmatch(r"[0-9a-f]{64}", str(solc_binary.get("sha256", "")))
        or not str(build.get("forge_version", "")).strip()
        or not str(build.get("solc_version", "")).strip()
    ):
        raise ManuscriptEvidenceError("Canonical contract-build tool provenance is incomplete")
    return build_path


def _validate_deployment_verification(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    verification_path = run_dir / "deployment_verification.json"
    verification = _read_json(verification_path)
    if summary.get("deployment_verification_sha256") != _sha256(verification_path):
        raise ManuscriptEvidenceError("Canonical run has a stale deployment-verification hash")
    try:
        artifact_runtime = _artifact_bytecode(_read_json(ARTIFACT_PATH), "deployedBytecode")
    except WorkflowError as exc:
        raise ManuscriptEvidenceError("Pinned contract artifact has invalid runtime bytecode") from exc
    runtime_sha256 = hashlib.sha256(artifact_runtime).hexdigest()
    try:
        runtime_byte_count = int(verification.get("runtime_byte_count", 0))
    except (TypeError, ValueError) as exc:
        raise ManuscriptEvidenceError("Deployment verification has an invalid runtime byte count") from exc
    contract_address = str(summary.get("contract_address", "")).lower()
    deployment_tx_hash = str(summary.get("deployment_tx_hash", "")).lower()
    if (
        verification.get("matches") is not True
        or str(verification.get("contract_address", "")).lower() != contract_address
        or str(verification.get("deployment_tx_hash", "")).lower() != deployment_tx_hash
        or not re.fullmatch(r"0x[0-9a-f]{40}", contract_address)
        or not re.fullmatch(r"0x[0-9a-f]{64}", deployment_tx_hash)
        or runtime_byte_count != len(artifact_runtime)
        or verification.get("artifact_runtime_sha256") != runtime_sha256
        or verification.get("deployed_runtime_sha256") != runtime_sha256
        or summary.get("deployed_runtime_sha256") != runtime_sha256
    ):
        raise ManuscriptEvidenceError(
            "Deployment verification does not match the canonical summary and pinned runtime bytecode"
        )
    _parse_utc(verification.get("verified_at_utc"), label="deployment verification")
    return verification_path


def _validate_canonical_transactions(
    summary: Mapping[str, Any],
    transactions: Sequence[Mapping[str, str]],
    addresses: Mapping[str, Any],
) -> None:
    final_checks = summary.get("final_checks")
    if (
        not isinstance(final_checks, Mapping)
        or set(final_checks) != EXPECTED_FINAL_CHECKS
        or any(value is not True for value in final_checks.values())
    ):
        raise ManuscriptEvidenceError("Canonical run final checks are incomplete, unexpected, or failed")
    guarded = summary.get("guarded_decision")
    if not isinstance(guarded, Mapping):
        raise ManuscriptEvidenceError("Canonical run has no guarded decision")
    try:
        primary_recipient_id = int(guarded["primary_recipient_id"])
        backup_recipient_id = int(guarded["backup_recipient_id"])
        match_id = int(summary["match_id"])
        observed = _validate_full_transaction_records(
            transactions,
            addresses,
            primary_recipient_id,
            backup_recipient_id,
            match_id,
        )
        transaction_count = int(summary.get("transaction_count", -1))
        gas_used_total = int(summary.get("gas_used_total", -1))
        fee_wei_total = int(summary.get("fee_wei_total", -1))
        fee_native_total = Decimal(str(summary.get("fee_native_total", "nan")))
    except (ArithmeticError, KeyError, TypeError, ValueError, WorkflowError) as exc:
        raise ManuscriptEvidenceError("Canonical transaction evidence is malformed or inconsistent") from exc
    measured_gas = sum(int(row["gas_used"]) for row in transactions)
    measured_fee = sum(int(row["fee_wei"]) for row in transactions)
    if (
        summary.get("transaction_plan_validation") != observed
        or transaction_count != len(transactions)
        or gas_used_total != measured_gas
        or fee_wei_total != measured_fee
        or fee_native_total != Decimal(measured_fee) / Decimal(10**18)
        or str(transactions[0].get("tx_hash", "")).lower()
        != str(summary.get("deployment_tx_hash", "")).lower()
    ):
        raise ManuscriptEvidenceError(
            "Canonical transaction summary does not match the independently validated receipt ledger"
        )


def _validate_canonical_decision(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    record_path = run_dir / "model_and_guard_record.json"
    record = _read_json(record_path)
    if summary.get("model_and_guard_record_sha256") != _sha256(record_path):
        raise ManuscriptEvidenceError("Canonical run has a stale model-and-guard record hash")
    demo = _read_json(DEMO_CASE_PATH)
    expected_ids = [int(item["recipient_id"]) for item in demo["recipients"]]
    model_id = str(summary.get("model_id_requested", ""))
    expected_request = {
        "temperature": PROTOCOL["model_evaluation"]["temperature"],
        "seed": PROTOCOL["model_evaluation"]["seed"],
        "provider": PROTOCOL["model_evaluation"]["provider"],
        "do_sample": PROTOCOL["model_evaluation"]["do_sample"],
        "maximum_new_tokens": PROTOCOL["model_evaluation"]["maximum_new_tokens"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    try:
        parsed = parse_note_review(
            record.get("parsed_review", {}),
            str(demo["case_id"]),
            expected_ids,
        )
        raw = parse_note_review(
            str(record.get("raw_response", "")),
            str(demo["case_id"]),
            expected_ids,
        )
        if raw.model_dump(mode="json") != parsed.model_dump(mode="json"):
            raise ManuscriptEvidenceError(
                "Canonical raw and parsed note-review records are different"
            )
        ranked = rank_recipients_baseline(demo["donor"], demo["recipients"], PROTOCOL)
        recomputed_guard = apply_guarded_policy(
            int(demo["donor"]["donor_id"]),
            ranked,
            parsed,
            expected_ids,
        ).model_dump(mode="json")
        metadata = ModelRunMetadata.model_validate(record.get("metadata", {})).model_dump(
            mode="json"
        )
        decision_cid = validate_cid(summary.get("decision_cid"))
    except ManuscriptEvidenceError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ManuscriptEvidenceError(
            "Canonical model-and-guard evidence is malformed or does not replay"
        ) from exc
    reference = demo.get("reference")
    if not isinstance(reference, Mapping):
        raise ManuscriptEvidenceError("Demonstration case has no reference decision")
    reference_comparison = {
        "reference_primary_recipient_id": int(reference["primary_recipient_id"]),
        "reference_backup_recipient_id": int(reference["backup_recipient_id"]),
        "observed_primary_recipient_id": int(recomputed_guard["primary_recipient_id"]),
        "observed_backup_recipient_id": int(recomputed_guard["backup_recipient_id"]),
        "primary_conforms": (
            int(recomputed_guard["primary_recipient_id"])
            == int(reference["primary_recipient_id"])
        ),
        "backup_conforms": (
            int(recomputed_guard["backup_recipient_id"])
            == int(reference["backup_recipient_id"])
        ),
    }
    provider = str(metadata.get("provider") or "")
    if provider == "local_hugging_face_transformers":
        adapter_sha256 = str(metadata.get("adapter_sha256") or "")
        provider_identity_valid = (
            provider == str(expected_request["provider"])
            and metadata.get("adapter_id") == model_id
            and bool(re.fullmatch(r"[0-9a-f]{64}", adapter_sha256))
            and bool(metadata.get("model_revision"))
            and bool(metadata.get("device"))
            and bool(metadata.get("dtype"))
        )
    else:
        provider_identity_valid = (
            provider == str(expected_request["provider"])
            and bool(metadata.get("response_id"))
        )
    if (
        summary.get("demo_case_id") != demo.get("case_id")
        or summary.get("demo_case_sha256") != _sha256(DEMO_CASE_PATH)
        or record.get("case_id") != demo.get("case_id")
        or record.get("model_id_requested") != model_id
        or record.get("model_request_config") != expected_request
        or not model_id
        or metadata.get("model_id") != model_id
        or not provider_identity_valid
        or metadata.get("input_tokens") is None
        or metadata.get("output_tokens") is None
        or record.get("guarded_decision") != recomputed_guard
        or summary.get("guarded_decision") != recomputed_guard
        or record.get("reference_comparison") != reference_comparison
        or summary.get("demo_reference_comparison") != reference_comparison
        or summary.get("model") != metadata
        or summary.get("decision_cid") != decision_cid
    ):
        raise ManuscriptEvidenceError(
            "Canonical model, guard, demonstration reference, or summary evidence is inconsistent"
        )
    return record_path


def _read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        raise ManuscriptEvidenceError(f"Missing evidence file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ManuscriptEvidenceError(f"Evidence table is empty: {path}")
    return rows


def _latex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _decimal(value: Any, places: int) -> str:
    number = Decimal(str(value))
    text = f"{number:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _fixed_decimal(value: Any, places: int) -> str:
    return f"{Decimal(str(value)):.{places}f}"


def _decimal_or_na(value: Any, places: int) -> str:
    return "N/A" if value is None or value == "" else _decimal(value, places)


def _metric_cell(
    value: Any,
    interval: Sequence[Any],
    *,
    count: int | None = None,
    denominator: int | None = None,
) -> str:
    if len(interval) != 2:
        raise ManuscriptEvidenceError("A metric confidence interval must have two limits")
    first = f"{float(value):.3f}"
    second = f"[{float(interval[0]):.3f}--{float(interval[1]):.3f}]"
    if count is None or denominator is None:
        third = f"N={_latex_int(int(denominator or 0))}"
    else:
        third = f"{_latex_int(count)}/{_latex_int(denominator)}"
    return rf"\shortstack{{{first}\\{{}}{second}\\{{}}{third}}}"


def _require_recorded_file_hash(
    metadata: Mapping[str, Any],
    path: Path,
    *,
    label: str,
) -> None:
    try:
        recorded_path = resolve_recorded_path(metadata.get("path"), root=WORKSPACE_DIR)
    except ValueError as exc:
        raise ManuscriptEvidenceError(f"{label} path is malformed") from exc
    if recorded_path != path.resolve():
        raise ManuscriptEvidenceError(f"{label} path does not match the expected file")
    if str(metadata.get("sha256", "")) != _sha256(path):
        raise ManuscriptEvidenceError(f"{label} hash does not match the current file")


def _without_generation_time(value: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(value)
    normalized.pop("generated_at_utc", None)
    return normalized


def _require_reproducible_analysis(
    recorded: Mapping[str, Any],
    raw_path: Path,
) -> None:
    try:
        recomputed = recompute_analysis(raw_path, write_outputs=False)
    except Exception as exc:
        raise ManuscriptEvidenceError(
            f"Could not recompute the analysis for {raw_path.name}: {exc}"
        ) from exc
    if _without_generation_time(recorded) != _without_generation_time(recomputed):
        changed = sorted(
            key
            for key in set(recorded).union(recomputed)
            if key != "generated_at_utc" and recorded.get(key) != recomputed.get(key)
        )
        raise ManuscriptEvidenceError(
            f"{raw_path.name} analysis is not reproducible from its raw records; changed: {changed}"
        )


def _require_reproducible_comparison(
    recorded: Mapping[str, Any],
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
) -> None:
    try:
        recomputed = recompute_comparison(
            left_path,
            right_path,
            left_label,
            right_label,
            write_output=False,
        )
    except Exception as exc:
        raise ManuscriptEvidenceError(
            f"Could not recompute the paired comparison {left_label} vs {right_label}: {exc}"
        ) from exc
    if _without_generation_time(recorded) != _without_generation_time(recomputed):
        changed = sorted(
            key
            for key in set(recorded).union(recomputed)
            if key != "generated_at_utc" and recorded.get(key) != recomputed.get(key)
        )
        raise ManuscriptEvidenceError(
            f"The paired comparison {left_label} vs {right_label} is not reproducible; changed: {changed}"
        )


def _validate_fine_tuning_state(value: Mapping[str, Any]) -> str:
    if PROTOCOL["model_evaluation"].get("provider") == "local_hugging_face_transformers":
        try:
            verified = validate_local_training_state(LOCAL_LORA_TRAINING_PATH)
        except Exception as exc:
            raise ManuscriptEvidenceError(f"The local LoRA adapter failed verification: {exc}") from exc
        if dict(value) != verified:
            raise ManuscriptEvidenceError("The supplied local LoRA state differs from the retained state")
        if (
            value.get("status") != "completed"
            or value.get("protocol_id") != PROTOCOL["protocol_id"]
            or value.get("held_out_test_file_opened") is not False
        ):
            raise ManuscriptEvidenceError("The local LoRA training state is incomplete or mismatched")
        if value.get("configuration") != local_training_configuration(
            PROTOCOL["model_evaluation"]
        ):
            raise ManuscriptEvidenceError("The local LoRA configuration differs from the protocol")
        records = value.get("dataset_records")
        if not isinstance(records, Mapping) or records != {
            "training": int(PROTOCOL["dataset"]["splits"]["training"]["cases"]),
            "validation": int(PROTOCOL["dataset"]["splits"]["validation"]["cases"]),
        }:
            raise ManuscriptEvidenceError("The local LoRA training state has wrong split counts")
        training_result = value.get("training_result")
        if (
            not isinstance(training_result, Mapping)
            or int(training_result.get("global_step", 0)) < 1
            or float(training_result.get("epoch", 0)) != float(
                PROTOCOL["model_evaluation"]["fine_tuning"]["epochs"]
            )
            or not isinstance(training_result.get("log_history"), list)
        ):
            raise ManuscriptEvidenceError("The local LoRA training result is incomplete")
        return str(verified["adapter"]["model_id"])

    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning state has no protocol provenance")
    if (
        protocol.get("id") != PROTOCOL["protocol_id"]
        or protocol.get("version") != PROTOCOL["version"]
        or protocol.get("sha256") != _sha256(DEFAULT_PROTOCOL_PATH)
    ):
        raise ManuscriptEvidenceError("The fine-tuning state belongs to a different protocol")
    try:
        protocol_path = resolve_recorded_path(protocol.get("path"), root=WORKSPACE_DIR)
    except ValueError as exc:
        raise ManuscriptEvidenceError("The fine-tuning protocol path is malformed") from exc
    if protocol_path != DEFAULT_PROTOCOL_PATH.resolve():
        raise ManuscriptEvidenceError("The fine-tuning state names a different protocol file")

    generation = value.get("generation_manifest")
    training = value.get("training_dataset")
    validation = value.get("validation_dataset")
    if not all(isinstance(item, Mapping) for item in (generation, training, validation)):
        raise ManuscriptEvidenceError("The fine-tuning state has incomplete dataset provenance")
    _require_recorded_file_hash(generation, GENERATION_MANIFEST_PATH, label="Generation manifest")
    _require_recorded_file_hash(training, TRAINING_PATH, label="Fine-tuning training file")
    _require_recorded_file_hash(validation, VALIDATION_PATH, label="Fine-tuning validation file")
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if value.get("system_prompt_sha256") != prompt_hash:
        raise ManuscriptEvidenceError("The fine-tuning state used a different system prompt")

    split_settings = PROTOCOL["dataset"]["splits"]
    file_validation = value.get("fine_tuning_file_validation")
    if not isinstance(file_validation, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning files have no validation record")
    for name, expected_path, expected_rows in (
        ("training", TRAINING_PATH, int(split_settings["training"]["cases"])),
        ("validation", VALIDATION_PATH, int(split_settings["validation"]["cases"])),
    ):
        item = file_validation.get(name)
        if not isinstance(item, Mapping):
            raise ManuscriptEvidenceError(f"Missing fine-tuning {name} validation")
        if int(item.get("row_count", -1)) != expected_rows or item.get("sha256") != _sha256(expected_path):
            raise ManuscriptEvidenceError(f"Fine-tuning {name} validation does not match the frozen file")

    settings = PROTOCOL["model_evaluation"]
    request = value.get("request_without_file_ids")
    if not isinstance(request, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning state has no preserved request")
    method = request.get("method")
    metadata = request.get("metadata")
    if not isinstance(method, Mapping) or not isinstance(metadata, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning request has malformed method or metadata fields")
    local_run_id = str(value.get("local_run_id", ""))
    request_ids = value.get("request_ids")
    if not re.fullmatch(r"[0-9a-f]{32}", local_run_id):
        raise ManuscriptEvidenceError("The fine-tuning state has no valid local run ID")
    if (
        not isinstance(request_ids, Mapping)
        or set(request_ids) != {"training_upload", "validation_upload", "job_creation"}
        or len({str(item) for item in request_ids.values()}) != 3
        or any(not str(item).isascii() or not str(item) for item in request_ids.values())
    ):
        raise ManuscriptEvidenceError("The fine-tuning state has invalid provider request IDs")
    supervised = method.get("supervised")
    if not isinstance(supervised, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning request has no supervised configuration")
    hyperparameters = supervised.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning request has malformed hyperparameters")
    expected_hyperparameters = {
        "n_epochs": int(settings["fine_tuning"]["epochs"]),
        "batch_size": settings["fine_tuning"]["batch_size"],
        "learning_rate_multiplier": settings["fine_tuning"]["learning_rate_multiplier"],
    }
    if (
        request.get("model") != settings["base_model_snapshot"]
        or request.get("suffix") != "oda-note-readiness-v1"
        or int(request.get("seed", -1)) != int(settings["seed"])
        or method.get("type") != "supervised"
        or hyperparameters != expected_hyperparameters
        or dict(metadata) != {
            "protocol": PROTOCOL["protocol_id"],
            "purpose": "synthetic-note-readiness",
            "local_run_id": local_run_id,
        }
    ):
        raise ManuscriptEvidenceError("The fine-tuning request differs from the prespecified protocol")

    creation_attempts = value.get("job_creation_attempts")
    if not isinstance(creation_attempts, list) or not creation_attempts:
        raise ManuscriptEvidenceError("The fine-tuning state has no job-creation attempt ledger")
    allowed_outcomes = {
        "provider_response_received",
        "provider_job_recovered",
        "uncertain",
    }
    for attempt in creation_attempts:
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("request_id") != request_ids["job_creation"]
            or attempt.get("outcome") not in allowed_outcomes
            or not attempt.get("started_at_utc")
        ):
            raise ManuscriptEvidenceError("The fine-tuning job-creation attempt ledger is malformed")
    if creation_attempts[-1].get("outcome") not in {
        "provider_response_received",
        "provider_job_recovered",
    }:
        raise ManuscriptEvidenceError("The final fine-tuning job-creation attempt is unresolved")

    access = value.get("access_check")
    authorization = value.get("fine_tuning_job_authorization")
    if not isinstance(access, Mapping) or not (
        access.get("model_retrievable")
        and access.get("model_id_matches")
        and access.get("base_model_requested") == settings["base_model_snapshot"]
    ):
        raise ManuscriptEvidenceError("The fine-tuning state lacks a passing base-model access check")
    if _read_json(ACCESS_CHECK_PATH) != access:
        raise ManuscriptEvidenceError("The fine-tuning state and retained model-access check differ")
    if not isinstance(authorization, Mapping) or authorization.get("confirmed") is not True:
        raise ManuscriptEvidenceError("Fine-tuning job authorization was not confirmed by job creation")

    job = value.get("job")
    if not isinstance(job, Mapping) or job.get("status") != "succeeded":
        raise ManuscriptEvidenceError("The retained fine-tuning job is not successful")
    fine_tuned_model = str(job.get("fine_tuned_model", "")).strip()
    if not fine_tuned_model:
        raise ManuscriptEvidenceError("The successful job does not record a fine-tuned model ID")
    if str(authorization.get("job_id", "")) != str(job.get("id", "")):
        raise ManuscriptEvidenceError("The fine-tuning authorization and completed job IDs differ")

    provider_validation = value.get("provider_job_validation")
    if not isinstance(provider_validation, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning state has no provider-job validation")
    recomputed_validation = validate_provider_job(dict(value), dict(job))
    if (
        provider_validation.get("all_passed") is not True
        or provider_validation.get("checks") != recomputed_validation["checks"]
        or recomputed_validation["all_passed"] is not True
    ):
        raise ManuscriptEvidenceError("The provider fine-tuning job differs from the recorded request")

    events = value.get("job_events")
    pagination = value.get("job_events_pagination")
    if not isinstance(events, list) or not events or not isinstance(pagination, Mapping):
        raise ManuscriptEvidenceError("The fine-tuning state has no complete job-event archive")
    event_ids = [str(event.get("id", "")) for event in events if isinstance(event, Mapping)]
    if (
        len(event_ids) != len(events)
        or any(not event_id for event_id in event_ids)
        or len(event_ids) != len(set(event_ids))
        or pagination.get("complete") is not True
        or pagination.get("final_page_has_more") is not False
        or int(pagination.get("page_count", 0)) < 1
        or int(pagination.get("event_count", -1)) != len(events)
        or int(pagination.get("unique_event_count", -1)) != len(events)
    ):
        raise ManuscriptEvidenceError("The fine-tuning job-event archive is incomplete or inconsistent")

    result_ids = job.get("result_files")
    archived = value.get("archived_result_files")
    if not isinstance(result_ids, list) or not result_ids or not isinstance(archived, list):
        raise ManuscriptEvidenceError("The fine-tuning result files were not archived")
    archived_by_id: Dict[str, Mapping[str, Any]] = {}
    for record in archived:
        if not isinstance(record, Mapping):
            raise ManuscriptEvidenceError("The fine-tuning result-file archive is malformed")
        file_id = str(record.get("file_id", ""))
        if not file_id or file_id in archived_by_id:
            raise ManuscriptEvidenceError("The fine-tuning result-file archive has invalid IDs")
        archived_by_id[file_id] = record
    if set(archived_by_id) != {str(file_id) for file_id in result_ids}:
        raise ManuscriptEvidenceError("The archived fine-tuning result files differ from the job")
    result_root = (MODEL_DIR / "result_files").resolve()
    for file_id, record in archived_by_id.items():
        try:
            path = resolve_recorded_path(
                record.get("local_path"),
                root=WORKSPACE_DIR,
                permitted_root=result_root,
            )
        except ValueError as exc:
            raise ManuscriptEvidenceError("A fine-tuning result file leaves its archive directory") from exc
        metadata = record.get("provider_metadata")
        if (
            not path.is_file()
            or record.get("sha256") != _sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
            or not isinstance(metadata, Mapping)
            or str(metadata.get("id", "")) != file_id
        ):
            raise ManuscriptEvidenceError(f"Fine-tuning result file {file_id} failed verification")
    return fine_tuned_model


def _fine_tuning_source_paths(value: Mapping[str, Any]) -> tuple[Path, ...]:
    if PROTOCOL["model_evaluation"].get("provider") == "local_hugging_face_transformers":
        adapter = value.get("adapter")
        if not isinstance(adapter, Mapping):
            raise ManuscriptEvidenceError("The local LoRA adapter record is missing")
        adapter_root = (APP_DIR / str(adapter.get("relative_path", ""))).resolve()
        try:
            adapter_root.relative_to(MODEL_DIR.resolve())
        except ValueError as exc:
            raise ManuscriptEvidenceError("The local LoRA adapter leaves the model directory") from exc
        files = adapter.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ManuscriptEvidenceError("The local LoRA adapter manifest is empty")
        adapter_paths = tuple(adapter_root / str(name) for name in sorted(files))
        return (
            LOCAL_BASE_MODEL_MANIFEST_PATH,
            LOCAL_LORA_PREFLIGHT_PATH,
            GENERATION_MANIFEST_PATH,
            TRAINING_PATH,
            VALIDATION_PATH,
            APP_DIR / "evaluation" / "train_local_lora.py",
            APP_DIR / "requirements-ml.txt",
            *adapter_paths,
        )

    archived = value.get("archived_result_files")
    if not isinstance(archived, list):
        raise ManuscriptEvidenceError("The fine-tuning result-file archive is missing")
    try:
        result_paths = tuple(
            resolve_recorded_path(
                record["local_path"],
                root=WORKSPACE_DIR,
                permitted_root=MODEL_DIR / "result_files",
            )
            for record in archived
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManuscriptEvidenceError("The fine-tuning result-file path is malformed") from exc
    return (
        ACCESS_CHECK_PATH,
        GENERATION_MANIFEST_PATH,
        TRAINING_PATH,
        VALIDATION_PATH,
        *result_paths,
    )


def _parse_utc(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManuscriptEvidenceError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise ManuscriptEvidenceError(f"{label} timestamp has no time-zone offset")
    return parsed


def _pretest_paths() -> Dict[str, Path]:
    return {
        "generation_manifest": GENERATION_MANIFEST_PATH,
        "training_cases": TRAINING_CASES_PATH,
        "validation_cases": VALIDATION_CASES_PATH,
        "fine_tuning_training": TRAINING_PATH,
        "fine_tuning_validation": VALIDATION_PATH,
        "local_base_model_manifest": LOCAL_BASE_MODEL_MANIFEST_PATH,
        "local_lora_preflight": LOCAL_LORA_PREFLIGHT_PATH,
        "local_lora_validation_preflight": LOCAL_LORA_VALIDATION_PREFLIGHT_PATH,
        "local_lora_checkpoint_inheritance": LOCAL_LORA_INHERITANCE_PATH,
        "local_lora_training": LOCAL_LORA_TRAINING_PATH,
        "protocol_lineage": PROTOCOL_LINEAGE_PATH,
        "protocol_amendment": PROTOCOL_AMENDMENT_PATH,
        "classical_validation_diagnostics": MODEL_DIR / "classical_validation_diagnostics.json",
        "untuned_smoke_test": MODEL_DIR / "smoke_tests" / "untuned_preflight.json",
        "fine_tuned_smoke_test": MODEL_DIR / "smoke_tests" / "fine_tuned_preflight.json",
        "openai_comparator_access_check": OPENAI_COMPARATOR_ACCESS_PATH,
        "openai_smoke_test": MODEL_DIR / "smoke_tests" / "openai_preflight.json",
    }


def _final_model_evidence_paths() -> Dict[str, Path]:
    """Return every frozen artifact that the canonical workflow must bind."""
    paths = dict(_pretest_paths())
    paths["test_cases"] = TEST_PATH
    paths["test_lock"] = EVALUATION_DIR / "test_lock.json"
    for condition, _ in CONDITIONS:
        paths[f"{condition}_raw"] = EVALUATION_DIR / f"{condition}_raw.jsonl"
        paths[f"{condition}_config"] = EVALUATION_DIR / f"{condition}_config.json"
        paths[f"{condition}_summary"] = EVALUATION_DIR / f"{condition}_summary.json"
    for condition, _ in CONDITIONS:
        paths[f"{condition}_attempts"] = EVALUATION_DIR / f"{condition}_attempts.jsonl"
    for left, right in PAIRED_COMPARISONS:
        paths[f"{left}_vs_{right}_comparison"] = (
            EVALUATION_DIR / f"{left}_vs_{right}_comparison.json"
        )
    paths["openai_api_cost"] = MODEL_API_COST_PATH

    fine_tuning = _read_json(paths["local_lora_training"])
    known = {path.resolve() for path in paths.values()}
    result_paths = sorted(
        {
            path.resolve()
            for path in _fine_tuning_source_paths(fine_tuning)
            if path.resolve() not in known
        },
        key=lambda path: str(path).casefold(),
    )
    for index, path in enumerate(result_paths, start=1):
        paths[f"local_training_result_{index:03d}"] = path
    return paths


def _final_model_evidence_hashes() -> Dict[str, str]:
    return {
        f"{name}_sha256": _sha256(path)
        for name, path in _final_model_evidence_paths().items()
    }


def _validate_pretest_artifacts(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, ...]:
    paths = _pretest_paths()
    test_lock = summaries["tfidf_logistic"]["config"].get("test_lock")
    if not isinstance(test_lock, Mapping):
        raise ManuscriptEvidenceError("The test lock is unavailable for preflight validation")
    recorded_hashes = test_lock.get("pretest_artifact_sha256")
    if not isinstance(recorded_hashes, Mapping) or set(recorded_hashes) != set(paths):
        raise ManuscriptEvidenceError("The test lock does not list the exact preflight artifacts")
    for name, path in paths.items():
        if not path.is_file() or recorded_hashes.get(name) != _sha256(path):
            raise ManuscriptEvidenceError(f"Preflight artifact {name} is missing or changed")

    locked_at = _parse_utc(test_lock.get("locked_at_utc"), label="Test lock")
    access = _read_json(paths["openai_comparator_access_check"])
    hosted = PROTOCOL["model_evaluation"]["hosted_comparator"]
    if (
        access.get("status") != "passed"
        or access.get("protocol_id") != PROTOCOL["protocol_id"]
        or access.get("protocol_version") != PROTOCOL["version"]
        or access.get("model_id_requested") != hosted["model_id"]
        or access.get("model_id_returned") != hosted["model_id"]
        or access.get("model_retrievable") is not True
        or access.get("test_case_data_opened") is not False
        or access.get("held_out_test_lock_present") is not False
        or access.get("runtime_settings") != hosted
        or access.get("protocol_sha256") != _sha256(DEFAULT_PROTOCOL_PATH)
        or access.get("script_sha256") != _sha256(OPENAI_COMPARATOR_PREFLIGHT_SCRIPT_PATH)
    ):
        raise ManuscriptEvidenceError("The hosted-comparator access preflight is inconsistent")
    if _parse_utc(access.get("checked_at_utc"), label="Hosted access preflight") > locked_at:
        raise ManuscriptEvidenceError("The hosted access preflight was generated after the test lock")

    diagnostics = _read_json(paths["classical_validation_diagnostics"])
    classical_config = summaries["tfidf_logistic"]["config"]
    expected_diagnostics = {
        "protocol_id": PROTOCOL["protocol_id"],
        "script_sha256": _sha256(CLASSICAL_BASELINE_PATH),
        "training_file_sha256": _sha256(TRAINING_CASES_PATH),
        "validation_file_sha256": _sha256(VALIDATION_CASES_PATH),
        "training_candidate_count": classical_config.get("training_candidate_count"),
        "validation_candidate_count": classical_config.get("validation_candidate_count"),
        "feature_count": classical_config.get("feature_count"),
        "validation_diagnostics": classical_config.get("validation_diagnostics"),
        "hyperparameters": classical_config.get("hyperparameters"),
        "packages": classical_config.get("packages"),
    }
    for field, expected in expected_diagnostics.items():
        if diagnostics.get(field) != expected:
            raise ManuscriptEvidenceError(
                f"Classical validation diagnostics differ from the locked run for {field}"
            )
    if _parse_utc(diagnostics.get("generated_at_utc"), label="Classical diagnostics") > locked_at:
        raise ManuscriptEvidenceError("Classical validation diagnostics were generated after the test lock")

    with VALIDATION_CASES_PATH.open("r", encoding="utf-8") as handle:
        try:
            case = json.loads(next(line for line in handle if line.strip()))
        except (StopIteration, json.JSONDecodeError) as exc:
            raise ManuscriptEvidenceError("The validation split has no readable preflight case") from exc
    expected_ids = [int(item["recipient_id"]) for item in case["recipients"]]
    fine_tuning = _read_json(paths["local_lora_training"])
    expected_models = {
        "untuned": str(PROTOCOL["model_evaluation"]["base_model_snapshot"]),
        "fine_tuned": str(fine_tuning.get("adapter", {}).get("model_id", "")),
        "openai": str(PROTOCOL["model_evaluation"]["hosted_comparator"]["model_id"]),
    }
    expected_smoke_sources = smoke_source_hashes()
    expected_smoke_sources["project_environment"] = _sha256(
        PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH
    )
    for name, condition in (
        ("untuned_smoke_test", "untuned"),
        ("fine_tuned_smoke_test", "fine_tuned"),
        ("openai_smoke_test", "openai"),
    ):
        record = _read_json(paths[name])
        model_id = expected_models[condition]
        if not model_id or (
            record.get("condition") != f"{condition}_preflight"
            or record.get("case_index") != 0
            or record.get("case_id") != case["case_id"]
            or record.get("organ_type") != case["organ_type"]
            or record.get("model_id_requested") != model_id
            or record.get("validation_file_sha256") != _sha256(VALIDATION_CASES_PATH)
            or record.get("system_prompt_sha256")
            != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
            or record.get("script_sha256") != _sha256(SMOKE_TEST_PATH)
            or record.get("implementation_sha256") != expected_smoke_sources
        ):
            raise ManuscriptEvidenceError(f"The {condition} smoke-test provenance is inconsistent")
        expected_parameters = smoke_runtime_settings(f"{condition}_preflight")
        for field, expected in expected_parameters.items():
            if record.get(field) != expected:
                raise ManuscriptEvidenceError(f"The {condition} smoke test changed {field}")
        status = record.get("status")
        if status not in {"success", "failure"}:
            raise ManuscriptEvidenceError(f"The {condition} smoke test is not terminal")
        if condition in {"fine_tuned", "openai"} and status != "success":
            raise ManuscriptEvidenceError(f"The {condition} smoke test did not pass")
        metadata = ModelRunMetadata.model_validate(record.get("model_run", {}))
        if metadata.model_id != model_id:
            raise ManuscriptEvidenceError(f"The {condition} smoke test observed another model")
        raw = record.get("raw_model_output")
        if not isinstance(raw, str) or not raw.strip():
            raise ManuscriptEvidenceError(f"The {condition} smoke test has no raw output")
        if status == "success":
            parsed = parse_note_review(
                record.get("parsed_review", {}),
                case["case_id"],
                expected_ids,
            )
            raw_review = parse_note_review(raw, case["case_id"], expected_ids)
            if raw_review.model_dump(mode="json") != parsed.model_dump(mode="json"):
                raise ManuscriptEvidenceError(
                    f"The {condition} smoke-test raw and parsed outputs differ"
                )
            guarded = apply_guarded_policy(
                int(case["donor"]["donor_id"]),
                rank_recipients_baseline(case["donor"], case["recipients"], PROTOCOL),
                parsed,
                expected_ids,
            )
            if record.get("guarded_decision") != guarded.model_dump(mode="json"):
                raise ManuscriptEvidenceError(
                    f"The {condition} smoke-test guard result does not reproduce"
                )
        elif not record.get("error_type") or record.get("parsed_review") is not None:
            raise ManuscriptEvidenceError(
                f"The {condition} smoke-test failure has inconsistent evidence"
            )
        if record.get("software_environment") != summaries[condition]["config"].get(
            "software_environment"
        ):
            raise ManuscriptEvidenceError(f"The {condition} smoke-test environment changed before evaluation")
        if _parse_utc(record.get("completed_at_utc"), label=f"{condition} smoke test") > locked_at:
            raise ManuscriptEvidenceError(f"The {condition} smoke test was generated after the test lock")
    return (*paths.values(), PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH)


def _validate_analysis_summaries() -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    test_locks = []
    test_hashes = []
    test_case_order = _jsonl_case_order(TEST_PATH, label="Locked test split")
    for condition, _ in CONDITIONS:
        path = EVALUATION_DIR / f"{condition}_summary.json"
        summary = _read_json(path)
        if summary.get("condition_values") != [condition]:
            raise ManuscriptEvidenceError(f"Unexpected condition in {path.name}")
        if int(summary.get("test_case_count", 0)) != 400:
            raise ManuscriptEvidenceError(f"{path.name} is not a complete 400-case analysis")
        if int(summary.get("raw_record_count", 0)) != 400 or int(
            summary.get("missing_record_count", -1)
        ) != 0:
            raise ManuscriptEvidenceError(f"{path.name} has incomplete raw records")
        raw_path = EVALUATION_DIR / f"{condition}_raw.jsonl"
        config_path = EVALUATION_DIR / f"{condition}_config.json"
        if not raw_path.is_file() or not config_path.is_file():
            raise ManuscriptEvidenceError(f"{path.name} is missing its raw data or configuration")
        if summary.get("records_file_sha256") != _sha256(raw_path):
            raise ManuscriptEvidenceError(f"{path.name} does not match its raw records")
        if _jsonl_case_order(raw_path, label=f"{condition} raw records") != test_case_order:
            raise ManuscriptEvidenceError(
                f"{condition} raw records do not follow the locked test order"
            )
        config = summary.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("test_lock"), dict):
            raise ManuscriptEvidenceError(f"{path.name} does not contain a frozen test lock")
        current_config = _read_json(config_path)
        if config != current_config or config.get("raw_file_sha256") != _sha256(raw_path):
            raise ManuscriptEvidenceError(f"{path.name} does not match its evaluation configuration")
        if config.get("evaluation_in_progress") is True:
            raise ManuscriptEvidenceError(f"{path.name} is still marked in progress")
        attempt_path = EVALUATION_DIR / f"{condition}_attempts.jsonl"
        if not attempt_path.is_file() or config.get("attempt_log_sha256") != _sha256(attempt_path):
            raise ManuscriptEvidenceError(f"{path.name} has no valid first-attempt ledger")
        attempted, _ = _attempted_case_ids(
            attempt_path,
            condition=condition,
            model_id=str(config.get("model_id_requested", "")),
            expected_order=test_case_order,
        )
        if len(attempted) != 400 or int(config.get("attempted_case_count", -1)) != 400:
            raise ManuscriptEvidenceError(f"{path.name} first-attempt ledger is incomplete")
        failure_total = sum(int(count) for count in summary.get("failure_counts", {}).values())
        if int(summary.get("successful_case_count", -1)) + failure_total != 400:
            raise ManuscriptEvidenceError(f"{path.name} has inconsistent success and failure counts")
        _require_reproducible_analysis(summary, raw_path)
        test_locks.append(config["test_lock"])
        test_hashes.append(str(summary.get("test_file_sha256", "")))
        summaries[condition] = summary
    if any(value != test_locks[0] for value in test_locks[1:]):
        raise ManuscriptEvidenceError("Model summaries do not share one frozen test lock")
    if len(set(test_hashes)) != 1:
        raise ManuscriptEvidenceError("Model summaries do not share one test file")
    validate_test_lock(test_locks[0])
    file_lock = _read_json(EVALUATION_DIR / "test_lock.json")
    if file_lock != test_locks[0] or test_hashes[0] != _sha256(TEST_PATH):
        raise ManuscriptEvidenceError("Model summaries do not match the current frozen test lock")

    for left, right in PAIRED_COMPARISONS:
        value = _read_json(EVALUATION_DIR / f"{left}_vs_{right}_comparison.json")
        if int(value.get("case_count", 0)) != 400:
            raise ManuscriptEvidenceError(f"Incomplete paired comparison: {left} vs {right}")
        if value.get("left", {}).get("label") != left or value.get("right", {}).get("label") != right:
            raise ManuscriptEvidenceError(f"Mislabeled paired comparison: {left} vs {right}")
        if value.get("test_file_sha256") != _sha256(TEST_PATH):
            raise ManuscriptEvidenceError(f"Paired comparison uses a different test file: {left} vs {right}")
        for side, condition in (("left", left), ("right", right)):
            raw_path = EVALUATION_DIR / f"{condition}_raw.jsonl"
            side_value = value.get(side)
            if not isinstance(side_value, Mapping) or side_value.get("sha256") != _sha256(raw_path):
                raise ManuscriptEvidenceError(
                    f"Paired comparison does not match {condition} raw records"
                )
            if side_value.get("config") != summaries[condition].get("config"):
                raise ManuscriptEvidenceError(
                    f"Paired comparison does not match {condition} configuration"
                )
        _require_reproducible_comparison(
            value,
            EVALUATION_DIR / f"{left}_raw.jsonl",
            EVALUATION_DIR / f"{right}_raw.jsonl",
            left,
            right,
        )

    fine_tuning = _read_json(LOCAL_LORA_TRAINING_PATH)
    fine_tuned_model = _validate_fine_tuning_state(fine_tuning)
    requested = summaries["fine_tuned"].get("model_id_requested_values")
    if requested != [fine_tuned_model]:
        raise ManuscriptEvidenceError("The fine-tuned analysis used a different model ID")
    if summaries["untuned"].get("model_id_requested_values") != [
        PROTOCOL["model_evaluation"]["base_model_snapshot"]
    ]:
        raise ManuscriptEvidenceError("The untuned analysis used a different base-model snapshot")
    if summaries["tfidf_logistic"].get("model_id_requested_values") != [
        PROTOCOL["model_evaluation"]["classical_text_baseline"]["model_id"]
    ]:
        raise ManuscriptEvidenceError("The classical analysis used a different model identifier")
    if summaries["openai"].get("model_id_requested_values") != [
        PROTOCOL["model_evaluation"]["hosted_comparator"]["model_id"]
    ]:
        raise ManuscriptEvidenceError("The hosted analysis used a different model identifier")
    expected_models = {
        "tfidf_logistic": PROTOCOL["model_evaluation"]["classical_text_baseline"]["model_id"],
        "untuned": PROTOCOL["model_evaluation"]["base_model_snapshot"],
        "fine_tuned": fine_tuned_model,
        "openai": PROTOCOL["model_evaluation"]["hosted_comparator"]["model_id"],
    }
    for condition, expected_model in expected_models.items():
        observed = summaries[condition].get("model_id_observed_values")
        successes = int(summaries[condition].get("successful_case_count", 0))
        if successes and observed != [expected_model]:
            raise ManuscriptEvidenceError(
                f"The runtime model identifier differs for {condition}"
            )
        if not successes and observed not in ([], None):
            raise ManuscriptEvidenceError(f"Unexpected observed model identifier for {condition}")
    untuned_environment = summaries["untuned"]["config"].get("software_environment")
    fine_tuned_environment = summaries["fine_tuned"]["config"].get("software_environment")
    if not isinstance(untuned_environment, Mapping) or untuned_environment != fine_tuned_environment:
        raise ManuscriptEvidenceError("The two local LLM conditions did not use one recorded software environment")
    return summaries


def _primary_result_rows(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    baseline_summary = summaries["tfidf_logistic"]
    baseline_top1 = baseline_summary["case_metrics"]["baseline_top1_conformant"]
    baseline_unsafe = baseline_summary["case_metrics"]["baseline_primary_unsafe"]
    rows = [
        "Deterministic ranker only & N/A & N/A & "
        + _metric_cell(
            baseline_top1["proportion"],
            baseline_top1["proportion_95ci"],
            count=int(baseline_top1["count"]),
            denominator=int(baseline_top1["denominator"]),
        )
        + " & "
        + _metric_cell(
            baseline_unsafe["proportion"],
            baseline_unsafe["proportion_95ci"],
            count=int(baseline_unsafe["count"]),
            denominator=int(baseline_unsafe["denominator"]),
        )
        + " & N/A \\tabularnewline"
    ]
    for condition, label in CONDITIONS:
        summary = summaries[condition]
        classification = summary["classification"]
        hold = classification["by_class"]["temporary_hold"]
        top1 = summary["case_metrics"]["guarded_top1_conformant"]
        unsafe = summary["case_metrics"]["unsafe_or_missing_primary"]
        completion = summary["case_metrics"]["model_success"]
        cells = (
            _metric_cell(
                classification["macro_f1"],
                classification["macro_f1_95ci"],
                denominator=int(classification["candidate_count"]),
            ),
            _metric_cell(
                hold["recall"],
                hold["recall_95ci"],
                count=int(hold["tp"]),
                denominator=int(hold["support"]),
            ),
            _metric_cell(
                top1["proportion"],
                top1["proportion_95ci"],
                count=int(top1["count"]),
                denominator=int(top1["denominator"]),
            ),
            _metric_cell(
                unsafe["proportion"],
                unsafe["proportion_95ci"],
                count=int(unsafe["count"]),
                denominator=int(unsafe["denominator"]),
            ),
            _metric_cell(
                completion["proportion"],
                completion["proportion_95ci"],
                count=int(completion["count"]),
                denominator=int(completion["denominator"]),
            ),
        )
        rows.append(label + " & " + " & ".join(cells) + " \\tabularnewline")
    return "\n".join(rows) + "\n"


def _percentage(value: object) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def _delta_text(value: float, interval: Sequence[float]) -> str:
    return (
        f"{100.0 * value:+.1f} percentage points "
        f"(95\\% CI {100.0 * float(interval[0]):+.1f} to "
        f"{100.0 * float(interval[1]):+.1f})"
    )


def _model_summary_text(
    summaries: Mapping[str, Mapping[str, Any]],
    api_cost: Mapping[str, Any],
) -> str:
    untuned = summaries["untuned"]
    fine_tuned = summaries["fine_tuned"]
    hosted = summaries["openai"]
    comparison = _read_json(EVALUATION_DIR / "fine_tuned_vs_openai_comparison.json")
    hosted_minus_fine = comparison["delta_right_minus_left"]
    hosted_minus_fine_ci = comparison["delta_95ci_cluster_bootstrap"]
    fine_minus_hosted_macro = -float(hosted_minus_fine["macro_f1"])
    raw_ci = hosted_minus_fine_ci["macro_f1"]
    fine_minus_hosted_ci = [-float(raw_ci[1]), -float(raw_ci[0])]

    completion = {
        name: _percentage(value["case_metrics"]["model_success"]["proportion"])
        for name, value in (
            ("untuned", untuned),
            ("fine_tuned", fine_tuned),
            ("openai", hosted),
        )
    }
    macro = {
        name: _percentage(value["classification"]["macro_f1"])
        for name, value in (
            ("untuned", untuned),
            ("fine_tuned", fine_tuned),
            ("openai", hosted),
        )
    }
    cost = api_cost["cost_usd"]
    usage = api_cost["token_usage"]
    pricing = api_cost["pricing"]
    records_with_usage = int(api_cost["records_with_token_usage"])
    records_missing_usage = int(api_cost["records_missing_token_usage"])
    cost_qualifier = "" if api_cost["status"] == "complete_usage_cost" else "at least "
    missing_usage_text = (
        f" Provider usage metadata were unavailable for {records_missing_usage} failed cases."
        if records_missing_usage
        else ""
    )
    return (
        "Untuned Qwen, fine-tuned Qwen, and GPT-4o mini achieved macro F1 values of "
        f"{macro['untuned']}, {macro['fine_tuned']}, and {macro['openai']}. Complete guarded "
        f"decisions were produced in {completion['untuned']}, {completion['fine_tuned']}, and "
        f"{completion['openai']}, respectively. The paired macro-F1 difference for "
        "fine-tuned Qwen minus GPT-4o mini was "
        f"{_delta_text(fine_minus_hosted_macro, fine_minus_hosted_ci)}. "
        f"The hosted-model evaluation recorded {int(usage['input_total']):,} input tokens and "
        f"{int(usage['output_total']):,} output tokens for {records_with_usage} responses."
        f"{missing_usage_text} At the prespecified provider rates of USD "
        f"{_fixed_decimal(pricing['input_per_million_tokens'], 2)} per million input tokens and USD "
        f"{_fixed_decimal(pricing['output_per_million_tokens'], 2)} per million output tokens, "
        "the observed token-accounted API cost was "
        f"{cost_qualifier}USD {_fixed_decimal(cost['total'], 4)}, corresponding to "
        f"{cost_qualifier}USD {_fixed_decimal(cost['mean_per_locked_case'], 6)} per attempted "
        "test case when averaged across all "
        f"{int(api_cost['test_case_count'])} cases.\n"
    )


def _listing_json(summary: Mapping[str, Any]) -> str:
    guarded = summary.get("guarded_decision")
    if not isinstance(guarded, Mapping):
        raise ManuscriptEvidenceError("Canonical summary has no guarded decision")
    assessments = guarded.get("note_assessments")
    if not isinstance(assessments, list) or len(assessments) != 10:
        raise ManuscriptEvidenceError("Canonical note review must contain ten assessments")
    by_id: Dict[int, Mapping[str, Any]] = {}
    for assessment in assessments:
        if not isinstance(assessment, Mapping):
            raise ManuscriptEvidenceError("Canonical note assessment is not an object")
        recipient_id = int(assessment.get("recipient_id", 0))
        if recipient_id <= 0 or recipient_id in by_id:
            raise ManuscriptEvidenceError("Canonical note assessments contain an invalid recipient ID")
        by_id[recipient_id] = assessment

    preferred_ids = [
        int(guarded["baseline_primary_recipient_id"]),
        int(guarded["primary_recipient_id"]),
        int(guarded["backup_recipient_id"]),
        *(int(value) for value in guarded.get("temporary_hold_recipient_ids", [])),
        *(int(value) for value in guarded.get("review_required_recipient_ids", [])),
        *by_id,
    ]
    selected_ids: list[int] = []
    for recipient_id in preferred_ids:
        if recipient_id not in by_id:
            raise ManuscriptEvidenceError(
                f"Canonical guarded decision references missing assessment {recipient_id}"
            )
        if recipient_id not in selected_ids:
            selected_ids.append(recipient_id)
        if len(selected_ids) == 3:
            break

    value = {
        "case_id": str(summary.get("demo_case_id", "")),
        "assessments": [
            {
                "recipient_id": recipient_id,
                "state": str(by_id[recipient_id].get("state", "")),
                "evidence_codes": list(by_id[recipient_id].get("evidence_codes", [])),
            }
            for recipient_id in selected_ids
        ],
    }
    if not value["case_id"]:
        raise ManuscriptEvidenceError("Canonical summary has no demonstration case ID")
    return json.dumps(value, ensure_ascii=True, indent=2) + "\n"


def _recipient_id(row: Mapping[str, str]) -> int | None:
    match = re.search(r"(?:^|,)recipient_id=(\d+)(?:,|$)", row.get("arguments", ""))
    return int(match.group(1)) if match else None


def _one_transaction(
    rows: Sequence[Mapping[str, str]],
    function: str,
    recipient_id: int | None = None,
) -> Mapping[str, str]:
    matches = [row for row in rows if row.get("function") == function]
    if recipient_id is not None:
        matches = [row for row in matches if _recipient_id(row) == recipient_id]
    if len(matches) != 1:
        raise ManuscriptEvidenceError(
            f"Expected one {function} transaction for recipient {recipient_id}; found {len(matches)}"
        )
    return matches[0]


def _selected_recipient_roles(summary: Mapping[str, Any]) -> list[tuple[int, str]]:
    guarded = summary["guarded_decision"]
    entries = (
        (int(guarded["primary_recipient_id"]), "guarded primary"),
        (int(guarded["backup_recipient_id"]), "guarded backup"),
        (int(guarded["baseline_primary_recipient_id"]), "baseline primary"),
    )
    labels: Dict[int, list[str]] = {}
    for recipient_id, role in entries:
        labels.setdefault(recipient_id, []).append(role)
    return [(recipient_id, "; ".join(roles)) for recipient_id, roles in labels.items()]


def _transaction_rows(
    transactions: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
) -> str:
    groups: list[tuple[str, list[tuple[str, int | None, str | None]]]] = [
        ("Deployment", [("constructor", None, None)]),
        ("Governance setup", [
            ("setHospital", None, None),
            ("setMedicalTeam", None, None),
            ("setEthicsCommittee", None, None),
            ("setDecisionService", None, None),
        ]),
        ("Identity binding", [("registerDonorAuthority", None, None)]),
    ]
    recipients = _selected_recipient_roles(summary)
    groups[-1][1].extend(
        ("registerRecipientAddress", recipient_id, role)
        for recipient_id, role in recipients
    )
    groups.extend([
        ("Profile registration", [("registerDonor", None, None)] + [
            ("registerRecipient", recipient_id, role) for recipient_id, role in recipients
        ]),
        ("Workflow eligibility", [("setDonorEligibility", None, None)] + [
            ("setRecipientEligibility", recipient_id, role) for recipient_id, role in recipients
        ]),
        ("Matching", [("createMatch", None, None)]),
        ("Approvals", [
            ("approveMedicalTeam", None, None),
            ("approveHospital", None, None),
            ("approveDonorAuthority", None, None),
            ("approveRecipient", None, None),
            ("approveFinalTransplant", None, None),
        ]),
        ("Finalization", [("finalizeMatch", None, None)]),
    ])

    output: list[str] = []
    for group_index, (stage, entries) in enumerate(groups):
        if group_index:
            output.append("\\midrule")
        for row_index, (function, recipient_id, role) in enumerate(entries):
            transaction = _one_transaction(transactions, function, recipient_id)
            stage_cell = rf"\textbf{{{stage}}}" if row_index == 0 else ""
            tag = rf"\rtag{{r{recipient_id}, {role}}}" if recipient_id is not None else ""
            output.append(
                f"{stage_cell} & \\fncell{{{function}}}{tag} & "
                rf"\txhash{{{transaction['tx_hash']}}} \tabularnewline"
            )
    return "\n".join(output) + "\n"


def _address_rows(summary: Mapping[str, Any], addresses: Mapping[str, Any]) -> str:
    roles = (
        ("Smart contract", str(summary["contract_address"])),
        ("Regulator", str(addresses["REGULATOR_PRIVATE_KEY"])),
        ("Hospital", str(addresses["HOSPITAL_PRIVATE_KEY"])),
        ("Ethics committee", str(addresses["ETHICS_PRIVATE_KEY"])),
        ("Medical team", str(addresses["MEDICAL_PRIVATE_KEY"])),
        ("Decision service", str(addresses["DECISION_SERVICE_PRIVATE_KEY"])),
        ("Donor authority", str(addresses["DONOR_PRIVATE_KEY"])),
    )
    rows = [rf"{label} & \addrwrap{{{address}}} \\" for label, address in roles]
    for recipient_id, role in _selected_recipient_roles(summary):
        address = str(addresses[f"RECIPIENT{recipient_id}_PRIVATE_KEY"])
        rows.append(rf"Recipient {recipient_id} ({role}) & \addrwrap{{{address}}} \\")
    return "\n".join(rows) + "\n"


def _cost_rows(run_dir: Path, summary: Mapping[str, Any]) -> tuple[str, Sequence[Path]]:
    cost_dir = run_dir / "cost_estimate"
    function_path = cost_dir / "cost_by_function.csv"
    stage_path = cost_dir / "cost_by_stage.csv"
    assumptions_path = cost_dir / "cost_assumptions.json"
    hashes_path = cost_dir / "artifact_hashes.json"
    try:
        stage_rows, assumptions = validate_cost_artifacts(run_dir)
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise ManuscriptEvidenceError(str(exc)) from exc
    if assumptions.get("source_run_summary_sha256") != _sha256(run_dir / "run_summary.json"):
        raise ManuscriptEvidenceError("Cost estimate belongs to a different canonical run")
    if assumptions.get("gas_price_scenarios_gwei") != ["5", "15", "30"]:
        raise ManuscriptEvidenceError("Cost estimate does not use the prespecified gas scenarios")
    by_stage = {row["category"]: row for row in stage_rows}
    if set(by_stage) != set(STAGES):
        raise ManuscriptEvidenceError("Cost stages do not match the canonical manuscript table")
    function_rows = _read_csv(function_path)
    by_function = {(row["category"], row["function"]): row for row in function_rows}
    expected_functions = {
        (stage, function)
        for stage, functions in COST_FUNCTION_ORDER.items()
        for function in functions
    }
    if set(by_function) != expected_functions:
        raise ManuscriptEvidenceError("Cost functions do not match the canonical manuscript table")
    if assumptions.get("table7_observation_date") != TABLE7_OBSERVATION_DATE:
        raise ManuscriptEvidenceError("Cost estimate has a different Table 7 observation date")
    scenario_keys = [item[0] for item in TABLE7_NETWORK_SCENARIOS]

    output: list[str] = []
    for stage in STAGES:
        if output:
            output.append("\\midrule")
        for row_index, function in enumerate(COST_FUNCTION_ORDER[stage]):
            row = by_function[(stage, function)]
            stage_cell = rf"\textbf{{{stage}}}" if row_index == 0 else ""
            costs = " & ".join(
                _fixed_decimal(row[f"table7_{key}_usd"], 3) for key in scenario_keys
            )
            output.append(
                f"{stage_cell} & \\fncell{{{function}}} & "
                f"{_latex_int(int(row['transaction_count']))} & "
                f"{_latex_int(int(row['measured_gas_used']))} & "
                f"{costs} \\tabularnewline"
            )
    totals = assumptions["table7_scenario_totals_usd"]
    total_costs = " & ".join(_fixed_decimal(totals[key], 3) for key in scenario_keys)
    output.extend([
        "\\midrule",
        "\\textbf{Total} & \\textbf{Total} & "
        f"{_latex_int(int(assumptions['measured_transaction_count']))} & "
        f"{_latex_int(int(assumptions['measured_gas_used_total']))} & "
        f"{total_costs} \\tabularnewline",
    ])
    if int(assumptions["measured_transaction_count"]) != int(summary["transaction_count"]):
        raise ManuscriptEvidenceError("Cost transaction total differs from the canonical summary")
    return "\n".join(output) + "\n", (
        run_dir / "transaction_manifest.csv",
        function_path,
        stage_path,
        assumptions_path,
        hashes_path,
    )


def _benchmark_summary(run_dir: Path) -> tuple[Dict[str, Any], Path, Path]:
    root = run_dir / "offchain_benchmark"
    pointer_path = root / "completed_run.json"
    pointer = _read_json(pointer_path)
    directory = (root / str(pointer.get("run_directory", ""))).resolve()
    try:
        directory.relative_to(root.resolve())
    except ValueError as exc:
        raise ManuscriptEvidenceError("Off-chain benchmark pointer leaves its run directory") from exc
    summary_path = directory / "summary.json"
    if pointer.get("summary_sha256") != _sha256(summary_path):
        raise ManuscriptEvidenceError("Off-chain benchmark summary hash verification failed")
    summary = _read_json(summary_path)
    ledger_path = directory / "attempts.jsonl"
    raw_path = directory / "runs.jsonl"
    if pointer.get("attempt_ledger_sha256") != _sha256(ledger_path):
        raise ManuscriptEvidenceError("Off-chain benchmark ledger hash verification failed")
    if pointer.get("raw_runs_sha256") != _sha256(raw_path):
        raise ManuscriptEvidenceError("Off-chain benchmark raw-run hash verification failed")
    if summary.get("attempt_ledger_sha256") != _sha256(ledger_path):
        raise ManuscriptEvidenceError("Off-chain benchmark summary does not bind its attempt ledger")
    if summary.get("raw_runs_sha256") != _sha256(raw_path):
        raise ManuscriptEvidenceError("Off-chain benchmark summary does not bind its raw outcomes")
    if summary.get("source_run_summary_sha256") != _sha256(run_dir / "run_summary.json"):
        raise ManuscriptEvidenceError("Off-chain benchmark belongs to a different canonical run")
    try:
        source_run = resolve_recorded_path(
            summary.get("source_workflow_run"),
            root=WORKSPACE_DIR,
            permitted_root=run_dir,
        )
    except ValueError as exc:
        raise ManuscriptEvidenceError("Off-chain benchmark source-run path is malformed") from exc
    if source_run != run_dir.resolve():
        raise ManuscriptEvidenceError("Off-chain benchmark names a different canonical run")
    if summary.get("benchmark_script_sha256") != _sha256(BENCHMARK_PATH):
        raise ManuscriptEvidenceError("Off-chain benchmark script changed after measurement")
    expected_benchmark_implementation = benchmark_source_hashes()
    expected_benchmark_implementation["project_environment"] = _sha256(
        PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH
    )
    if summary.get("implementation_sha256") != expected_benchmark_implementation:
        raise ManuscriptEvidenceError("Off-chain benchmark implementation changed after measurement")
    if int(summary.get("warmup_runs_excluded", -1)) != 3:
        raise ManuscriptEvidenceError("Off-chain benchmark did not exclude three warm-up runs")
    try:
        audit = audit_attempt_provenance(
            directory,
            warmup_runs=3,
            measured_runs=30,
            require_complete=True,
        )
    except RuntimeError as exc:
        raise ManuscriptEvidenceError(str(exc)) from exc
    if int(summary.get("attempt_ledger_event_count", -1)) != len(audit["events"]):
        raise ManuscriptEvidenceError("Off-chain benchmark ledger event count is inconsistent")
    if int(summary.get("replacement_attempt_count", -1)) != 0:
        raise ManuscriptEvidenceError("Off-chain benchmark contains replacement attempts")
    requested = int(summary.get("measured_runs_requested", 0))
    attempted = int(summary.get("measured_runs_attempted", 0))
    successful = int(summary.get("measured_runs_successful", -1))
    failed = int(summary.get("measured_runs_failed", -1))
    if requested != 30 or attempted != 30 or successful < 0 or failed < 0:
        raise ManuscriptEvidenceError("Off-chain benchmark is not the complete 30-attempt study")
    if successful + failed != attempted:
        raise ManuscriptEvidenceError("Off-chain benchmark success and failure counts are inconsistent")
    measured = [record for record in audit["records"] if record["phase"] == "measured"]
    measured_successes = [record for record in measured if record["status"] == "success"]
    if attempted != len(measured) or successful != len(measured_successes):
        raise ManuscriptEvidenceError("Off-chain benchmark counts differ from its raw outcomes")
    if failed != len(measured) - len(measured_successes):
        raise ManuscriptEvidenceError("Off-chain benchmark failure count differs from its raw outcomes")
    interrupted = sum(
        1 for record in audit["records"] if record.get("error_type") == "InterruptedAttempt"
    )
    if int(summary.get("interrupted_attempts_retained_as_failures", -1)) != interrupted:
        raise ManuscriptEvidenceError("Off-chain benchmark interruption count is inconsistent")
    primary_conform = len(measured_successes) == 30 and all(
        record.get("reference_primary_conforms") is True for record in measured
    )
    backup_conform = len(measured_successes) == 30 and all(
        record.get("reference_backup_conforms") is True for record in measured
    )
    if summary.get("all_measured_primary_conform") is not primary_conform:
        raise ManuscriptEvidenceError("Off-chain benchmark primary-conformance claim is inconsistent")
    if summary.get("all_measured_backup_conform") is not backup_conform:
        raise ManuscriptEvidenceError("Off-chain benchmark backup-conformance claim is inconsistent")
    if summary.get("failed_measured_attempts_rerun") is not False:
        raise ManuscriptEvidenceError("Off-chain benchmark does not preserve first-attempt failures")
    if int(summary.get("ipfs_fetch_retries_per_measurement", -1)) != 0:
        raise ManuscriptEvidenceError("Off-chain benchmark used unprespecified IPFS fetch retries")
    hashes_path = directory / "artifact_hashes.json"
    hashes = _read_json(hashes_path)
    hashed_paths = (
        ledger_path,
        raw_path,
        directory / "timings.csv",
        summary_path,
    )
    if set(hashes) != {path.name for path in hashed_paths}:
        raise ManuscriptEvidenceError(
            "Off-chain benchmark artifact-hash map has missing or extra entries"
        )
    for path in hashed_paths:
        record = hashes.get(path.name)
        if not isinstance(record, dict):
            raise ManuscriptEvidenceError(f"Off-chain benchmark does not record {path.name}")
        if record.get("sha256") != _sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
            raise ManuscriptEvidenceError(f"Off-chain benchmark hash verification failed for {path.name}")
    state_path = root / "run_state.json"
    state = _read_json(state_path)
    if (
        state.get("status") != "completed"
        or state.get("run_directory") != directory.name
        or state.get("source_run_summary_sha256") != summary.get("source_run_summary_sha256")
        or state.get("implementation_sha256") != summary.get("implementation_sha256")
        or state.get("software_environment") != summary.get("environment")
        or state.get("summary_sha256") != _sha256(summary_path)
        or state.get("attempt_ledger_sha256") != _sha256(ledger_path)
        or state.get("raw_runs_sha256") != _sha256(raw_path)
    ):
        raise ManuscriptEvidenceError("Off-chain benchmark completion state is malformed or stale")
    return summary, pointer_path, summary_path


def _latency_rows(run_dir: Path) -> tuple[str, Sequence[Path]]:
    benchmark, pointer_path, benchmark_path = _benchmark_summary(run_dir)
    timing_specs = (
        ("Encrypted profile retrieval and authentication", "encrypted_record_retrieval"),
        ("Organ-specific ranking", "protocol_ranking"),
        ("Model note review", "model_note_review"),
        ("Deterministic guard", "deterministic_guard"),
        ("Complete repeated path", "total"),
    )
    failed = int(benchmark["measured_runs_failed"])
    failure_text = "no failures" if failed == 0 else f"{failed} failures"
    output = [
        "\\multicolumn{6}{@{}l}{\\textit{Repeated off-chain path "
        f"(30 timed runs, {failure_text})}}}} \\tabularnewline"
    ]

    def format_seconds(raw: Any) -> str:
        if raw is None or raw == "":
            return "N/A"
        if Decimal(str(raw)) < Decimal("0.001"):
            return "$<0.001$"
        return _fixed_decimal(raw, 3)

    for label, key in timing_specs:
        value = benchmark["timings"][key]
        output.append(
            f"{label} & {_latex_int(int(value['n']))} & "
            f"{format_seconds(value['median_seconds'])} & "
            f"{format_seconds(value['p95_seconds'])} & "
            f"{format_seconds(value['min_seconds'])} & "
            f"{format_seconds(value['max_seconds'])} \\tabularnewline"
        )

    stage_path = run_dir / "transaction_stage_summary.csv"
    stages = {row["category"]: row for row in _read_csv(stage_path)}
    if set(stages) != set(STAGES):
        raise ManuscriptEvidenceError("Latency stages do not match the canonical manuscript table")
    output.extend([
        "\\midrule",
        "\\multicolumn{6}{@{}l}{\\textit{Final Sepolia transaction confirmation}} \\tabularnewline",
    ])
    for stage in STAGES:
        row = stages[stage]
        output.append(
            f"{stage} & {_latex_int(int(row['transaction_count']))} & "
            f"{_decimal(row['confirmation_seconds_median'], 3)} & N/A & "
            f"{_decimal(row['confirmation_seconds_min'], 3)} & "
            f"{_decimal(row['confirmation_seconds_max'], 3)} \\tabularnewline"
        )
    benchmark_dir = benchmark_path.parent
    return "\n".join(output) + "\n", (
        pointer_path,
        benchmark_path,
        benchmark_dir / "attempts.jsonl",
        benchmark_dir / "runs.jsonl",
        benchmark_dir / "timings.csv",
        benchmark_dir / "artifact_hashes.json",
        benchmark_dir.parent / "run_state.json",
        stage_path,
    )


def _assert_targets_replaceable(replace_generated: bool) -> None:
    for path in TARGETS.values():
        if not path.exists():
            raise ManuscriptEvidenceError(f"Missing manuscript row file: {path}")
        if not replace_generated and PLACEHOLDER_MARKER not in path.read_text(encoding="utf-8"):
            raise ManuscriptEvidenceError(
                f"Refusing to replace an already generated table without --replace-generated: {path.name}"
            )


def build(run_dir: Path, *, replace_generated: bool = False) -> Dict[str, Any]:
    _assert_targets_replaceable(replace_generated)
    summary_path = run_dir / "run_summary.json"
    summary = _read_json(summary_path)
    if summary.get("protocol_id") != PROTOCOL["protocol_id"] or summary.get(
        "protocol_sha256"
    ) != _sha256(DEFAULT_PROTOCOL_PATH):
        raise ManuscriptEvidenceError("Canonical run does not match the active protocol")
    if summary.get("network") != "sepolia" or int(summary.get("chain_id", 0)) != 11155111:
        raise ManuscriptEvidenceError("Canonical run is not a Sepolia run")
    contract_build_path = _validate_contract_build(run_dir, summary)
    deployment_verification_path = _validate_deployment_verification(run_dir, summary)
    model_and_guard_path = _validate_canonical_decision(run_dir, summary)

    summaries = _validate_analysis_summaries()
    pretest_sources = _validate_pretest_artifacts(summaries)
    api_cost = validate_model_api_cost()
    fine_tuning_path = LOCAL_LORA_TRAINING_PATH
    fine_tuning_sources = _fine_tuning_source_paths(_read_json(fine_tuning_path))
    fine_tuned_model = summaries["fine_tuned"]["model_id_requested_values"][0]
    if summary.get("model_id_requested") != fine_tuned_model:
        raise ManuscriptEvidenceError("Canonical workflow did not use the evaluated fine-tuned model")
    if summary.get("model", {}).get("model_id") != fine_tuned_model:
        raise ManuscriptEvidenceError(
            "Canonical workflow runtime used a different model identifier"
        )
    expected_implementation = {
        "workflow": _sha256(WORKFLOW_PATH),
        "llm_client": _sha256(LLM_CLIENT_PATH),
        "local_llm_client": _sha256(LOCAL_LLM_CLIENT_PATH),
        "policy": _sha256(POLICY_PATH),
        "decision_guard": _sha256(GUARD_PATH),
        "schemas": _sha256(SCHEMAS_PATH),
        "secure_storage": _sha256(SECURE_STORAGE_PATH),
        "ipfs_client": _sha256(IPFS_CLIENT_PATH),
        "pinata_client": _sha256(PINATA_CLIENT_PATH),
        "transactions": _sha256(TRANSACTIONS_PATH),
        "project_environment": _sha256(PRE_ETHERSCAN_PROJECT_ENVIRONMENT_PATH),
        "artifact_paths": _sha256(ARTIFACT_PATHS_PATH),
        "contract": _sha256(SOURCE_PATH),
    }
    if summary.get("implementation_sha256") != expected_implementation:
        raise ManuscriptEvidenceError("Canonical workflow implementation changed after execution")
    if summary.get("contract_source_sha256") != _sha256(SOURCE_PATH):
        raise ManuscriptEvidenceError("Canonical workflow contract source changed")
    expected_model_evidence = _final_model_evidence_hashes()
    if summary.get("model_evidence") != expected_model_evidence:
        raise ManuscriptEvidenceError("Canonical workflow is not bound to the final model evidence")

    transaction_path = run_dir / "transaction_manifest.csv"
    address_path = run_dir / "actor_addresses.json"
    transactions = _read_csv(transaction_path)
    addresses = _read_json(address_path)
    _validate_canonical_transactions(summary, transactions, addresses)
    cost_text, cost_sources = _cost_rows(run_dir, summary)
    latency_text, latency_sources = _latency_rows(run_dir)
    contents = {
        "transactions": _transaction_rows(transactions, summary),
        "addresses": _address_rows(summary, addresses),
        "primary_results": _primary_result_rows(summaries),
        "model_summary": _model_summary_text(summaries, api_cost),
        "costs": cost_text,
        "latency": latency_text,
        "listing": _listing_json(summary),
    }
    if any("pending" in value.lower() for value in contents.values()):
        raise ManuscriptEvidenceError("Generated table content still contains a pending marker")

    for name, path in TARGETS.items():
        path.write_text(
            contents[name],
            encoding="utf-8",
            newline="\n",
        )

    sources = [
        SCRIPT_PATH,
        DEFAULT_PROTOCOL_PATH,
        summary_path,
        contract_build_path,
        deployment_verification_path,
        model_and_guard_path,
        transaction_path,
        address_path,
        *cost_sources,
        *latency_sources,
        fine_tuning_path,
        *fine_tuning_sources,
        *pretest_sources,
        EVALUATION_DIR / "test_lock.json",
        MODEL_API_COST_PATH,
        *(EVALUATION_DIR / f"{condition}_raw.jsonl" for condition, _ in CONDITIONS),
        *(EVALUATION_DIR / f"{condition}_config.json" for condition, _ in CONDITIONS),
        *(EVALUATION_DIR / f"{condition}_attempts.jsonl" for condition, _ in CONDITIONS),
        *(EVALUATION_DIR / f"{condition}_summary.json" for condition, _ in CONDITIONS),
        *(
            EVALUATION_DIR / f"{left}_vs_{right}_comparison.json"
            for left, right in PAIRED_COMPARISONS
        ),
    ]
    manifest = {
        "protocol_id": PROTOCOL["protocol_id"],
        "canonical_run": portable_path(run_dir),
        "sources": {
            portable_path(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sources
        },
        "generated": {
            portable_path(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in TARGETS.values()
        },
    }
    manifest_path = run_dir / "manuscript_table_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--replace-generated", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build(
        _resolve_run_dir(args.run_dir),
        replace_generated=args.replace_generated,
    )
    print(f"Generated {len(manifest['generated'])} manuscript table fragments")


if __name__ == "__main__":
    main()
