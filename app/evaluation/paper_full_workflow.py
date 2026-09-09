"""Run the canonical encrypted demonstration workflow on a fresh EVM deployment.

The full mode is the only workflow intended to produce the manuscript's
end-to-end blockchain evidence. It uses the model only for note-state
classification and records the recipients selected by the deterministic guard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from web3 import Web3

from evaluation.external_run_ledger import ExternalRunLedger, read_ledger
from evaluation.project_environment import load_authoritative_project_env
from evaluation.refresh_software_evidence import SOLC_VERSION, find_local_solc
from src.decision_guard import apply_guarded_policy
from src.ipfs_client import (
    fetch_encrypted_json_from_ipfs,
    fetch_json_from_ipfs,
)
from src.llm_client import SYSTEM_PROMPT, build_note_review_payload, call_note_review
from src.local_llm_client import LocalNoteReviewClient, validate_local_training_state
from src.pinata_client import pin_encrypted_json, pin_json
from src.policy import (
    DEFAULT_PROTOCOL_PATH,
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
    rank_recipients_baseline,
)
from src.schemas import BaselineCandidate, MatchResponse
from src.secure_storage import (
    canonical_json_sha256,
    decode_encryption_key,
    decrypt_json,
    encrypt_json,
)
from src.transactions import TransactionReceipt, private_key_address, send_signed_transaction, utc_now


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
SMART_CONTRACTS_DIR = IMPLEMENTATION_DIR / "smart-contracts"
ARTIFACT_PATH = SMART_CONTRACTS_DIR / "out" / "TransplantManagement.sol" / "TransplantManagement.json"
SOURCE_PATH = SMART_CONTRACTS_DIR / "src" / "TransplantManagement.sol"
DEMO_CASE_PATH = APP_DIR / "seed-data" / "demo_case.json"
CURRENT_OUTPUT_DIR = APP_DIR / "pipeline-output" / "current"
INTEGRATION_ABI_PATH = IMPLEMENTATION_DIR / "integration" / "abi" / "TransplantManagement.json"
PROTOCOL = load_protocol()
MODEL_SETTINGS = PROTOCOL["model_evaluation"]
WORKFLOW_PATH = Path(__file__).resolve()
LLM_CLIENT_PATH = APP_DIR / "src" / "llm_client.py"
LOCAL_LLM_CLIENT_PATH = APP_DIR / "src" / "local_llm_client.py"
LOCAL_TRAINING_STATE_PATH = CURRENT_OUTPUT_DIR / "model" / "local_lora_training.json"
POLICY_PATH = APP_DIR / "src" / "policy.py"
GUARD_PATH = APP_DIR / "src" / "decision_guard.py"
SCHEMAS_PATH = APP_DIR / "src" / "schemas.py"
SECURE_STORAGE_PATH = APP_DIR / "src" / "secure_storage.py"
IPFS_CLIENT_PATH = APP_DIR / "src" / "ipfs_client.py"
PINATA_CLIENT_PATH = APP_DIR / "src" / "pinata_client.py"
TRANSACTIONS_PATH = APP_DIR / "src" / "transactions.py"
PROJECT_ENVIRONMENT_PATH = APP_DIR / "evaluation" / "project_environment.py"
ARTIFACT_PATHS_PATH = APP_DIR / "evaluation" / "artifact_paths.py"
COMPLETION_FILENAME = "completion.json"

PRIVATE_KEY_NAMES = (
    "REGULATOR_PRIVATE_KEY",
    "HOSPITAL_PRIVATE_KEY",
    "ETHICS_PRIVATE_KEY",
    "MEDICAL_PRIVATE_KEY",
    "DONOR_PRIVATE_KEY",
    "DECISION_SERVICE_PRIVATE_KEY",
    *(f"RECIPIENT{i}_PRIVATE_KEY" for i in range(1, 11)),
)

MATCH_FIELDS = (
    "match_id",
    "donor_id",
    "primary_recipient_id",
    "backup_recipient_id",
    "active_recipient_id",
    "backup_promoted",
    "recorded_by",
    "decision_cid",
    "cancellation_cid",
    "medical_approved",
    "hospital_approved",
    "donor_authority_approved",
    "active_recipient_approved",
    "ethics_committee_approved",
    "finalized",
    "cancelled",
)
DONOR_FIELDS = (
    "donor_id",
    "donor_authority",
    "profile_cid",
    "registered",
    "ethically_eligible",
    "finalized",
)
RECIPIENT_FIELDS = (
    "recipient_id",
    "recipient_address",
    "profile_cid",
    "registered",
    "ethically_eligible",
    "reserved",
    "transplanted",
)


class WorkflowError(RuntimeError):
    pass


def _decode_contract_struct(
    values: Sequence[Any],
    fields: Sequence[str],
    *,
    label: str,
) -> Dict[str, Any]:
    if isinstance(values, (str, bytes)) or len(values) != len(fields):
        raise WorkflowError(
            f"{label} getter returned {len(values)} fields; expected {len(fields)}"
        )
    return dict(zip(fields, values))


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise WorkflowError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkflowError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_full_workflow_runs() -> list[Path]:
    root = CURRENT_OUTPUT_DIR / "full_workflow"
    if not root.is_dir():
        return []
    completed: list[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        completion_path = directory / COMPLETION_FILENAME
        if not completion_path.is_file():
            continue
        completion = _read_json(completion_path)
        if completion.get("status") != "completed" or completion.get("mode") not in {
            "full",
            "seed_only",
        }:
            raise WorkflowError(f"Malformed workflow completion record: {completion_path}")
        if completion["mode"] == "full":
            completed.append(directory.resolve())
    return completed


def _write_completion(run_dir: Path, *, mode: str) -> None:
    artifact_path = run_dir / "artifact_manifest.json"
    if not artifact_path.is_file():
        raise WorkflowError("Cannot complete a workflow run without its artifact manifest")
    completion: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "completed",
        "mode": mode,
        "completed_at_utc": utc_now(),
        "artifact_manifest_sha256": _sha256(artifact_path),
    }
    summary_path = run_dir / "run_summary.json"
    if mode == "full":
        if not summary_path.is_file():
            raise WorkflowError("Cannot complete a full workflow run without its run summary")
        completion["run_summary_sha256"] = _sha256(summary_path)
    _write_json(run_dir / COMPLETION_FILENAME, completion)


def _required_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    upper = value.upper()
    if not value or "YOUR_" in upper or value in {"0x...", "...", "bafk..."}:
        raise WorkflowError(f"{name} is missing or still contains a placeholder in app/.env")
    return value


def _rpc_url(network: str) -> str:
    return str(os.getenv(f"{network.upper()}_RPC_URL") or os.getenv("RPC_URL") or "").strip()


def _runtime_profile(value: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "reference_note_state",
        "reference_evidence_codes",
        "context_evidence_codes",
        "note_style",
        "note_complexity",
        "template_family_id",
        "note_instance_id",
    }
    return {key: item for key, item in value.items() if key not in excluded}


def _baseline_snapshot(ranked: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Remove internal tie-break fields from the public evidence schema."""
    return [
        BaselineCandidate(
            rank=item["rank"],
            recipient_id=item["recipient_id"],
            score=item["score"],
            priority_tier=item["priority_tier"],
            selectable=item["selectable"],
            exclusion_reasons=item["exclusion_reasons"],
            factors=item["factors"],
        ).model_dump(mode="json")
        for item in ranked
    ]


def _load_demo_case() -> Dict[str, Any]:
    case = _read_json(DEMO_CASE_PATH)
    if case.get("protocol_id") != PROTOCOL["protocol_id"]:
        raise WorkflowError("The demonstration case does not match the active protocol")
    if int(case.get("donor", {}).get("donor_id", 0)) != 1:
        raise WorkflowError("The demonstration donor ID must be 1")
    ids = [int(item.get("recipient_id", 0)) for item in case.get("recipients", [])]
    if ids != list(range(1, 11)):
        raise WorkflowError("The demonstration recipient IDs must be exactly 1 through 10")
    runtime_donor = _runtime_profile(case["donor"])
    runtime_recipients = [_runtime_profile(item) for item in case["recipients"]]
    ranking = rank_recipients_baseline(runtime_donor, runtime_recipients, PROTOCOL)
    observed = [item["recipient_id"] for item in ranking if item["selectable"]]
    if observed != case["reference"]["baseline_order"]:
        raise WorkflowError("The demonstration baseline order does not reproduce")
    return case


def _require_final_model_evidence(model_id: str) -> Dict[str, str]:
    """Require the canonical run to use the already evaluated fine-tuned model."""
    from evaluation.build_manuscript_tables import (
        _final_model_evidence_hashes,
        _validate_analysis_summaries,
        _validate_pretest_artifacts,
    )

    summaries = _validate_analysis_summaries()
    _validate_pretest_artifacts(summaries)
    fine_tuned = summaries["fine_tuned"]
    if fine_tuned.get("model_id_requested_values") != [model_id]:
        raise WorkflowError(
            "The local adapter is not the fine-tuned model used in the frozen evaluation"
        )
    return _final_model_evidence_hashes()


def _build_contract_artifact(run_dir: Path) -> tuple[list[dict[str, Any]], str, Dict[str, Any]]:
    build_env = os.environ.copy()
    for name in tuple(build_env):
        if name.upper().startswith(("FOUNDRY_", "DAPP_")):
            build_env.pop(name, None)
    build_env["FOUNDRY_OFFLINE"] = "true"
    forge = shutil.which("forge")
    if not forge:
        raise WorkflowError("Foundry forge is not available on PATH")
    try:
        solc = find_local_solc(env=build_env)
    except RuntimeError as exc:
        raise WorkflowError(str(exc)) from exc
    build_env["FOUNDRY_SOLC"] = str(solc)
    command = [forge, "build", "--force", "--offline", "--use", str(solc)]
    build = subprocess.run(
        command,
        cwd=SMART_CONTRACTS_DIR,
        env=build_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    _write_text(run_dir / "forge_build_stdout.txt", build.stdout)
    _write_text(run_dir / "forge_build_stderr.txt", build.stderr)
    if build.returncode != 0:
        raise WorkflowError("forge build failed; inspect the saved build logs")

    artifact = _read_json(ARTIFACT_PATH)
    abi = artifact.get("abi")
    bytecode_field = artifact.get("bytecode", {})
    bytecode = bytecode_field.get("object") if isinstance(bytecode_field, dict) else bytecode_field
    if not isinstance(abi, list) or not bytecode:
        raise WorkflowError("Foundry artifact does not contain an ABI and deployment bytecode")
    if not str(bytecode).startswith("0x"):
        bytecode = "0x" + str(bytecode)
    compiler = _compiler_metadata(artifact).get("compiler", {})
    compiler_version = str(compiler.get("version", "")) if isinstance(compiler, dict) else ""
    if not compiler_version.startswith(f"{SOLC_VERSION}+"):
        raise WorkflowError(
            f"Foundry artifact compiler {compiler_version or '<missing>'} is not Solidity {SOLC_VERSION}"
        )
    forge_version = subprocess.run(
        [forge, "--version"],
        cwd=SMART_CONTRACTS_DIR,
        env=build_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    solc_version = subprocess.run(
        [str(solc), "--version"],
        cwd=SMART_CONTRACTS_DIR,
        env=build_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if forge_version.returncode != 0 or solc_version.returncode != 0:
        raise WorkflowError("Unable to record the pinned contract-build tool versions")
    _write_json(
        run_dir / "contract_build.json",
        {
            "offline": True,
            "command": ["forge", "build", "--force", "--offline", "--use", solc.name],
            "forge_version": forge_version.stdout.strip() or forge_version.stderr.strip(),
            "solc_version": solc_version.stdout.strip() or solc_version.stderr.strip(),
            "solc_binary": {
                "filename": solc.name,
                "bytes": solc.stat().st_size,
                "sha256": _sha256(solc),
            },
            "foundry_configuration_sha256": _sha256(SMART_CONTRACTS_DIR / "foundry.toml"),
            "contract_source_sha256": _sha256(SOURCE_PATH),
            "artifact_sha256": _sha256(ARTIFACT_PATH),
            "artifact_compiler": compiler,
        },
    )
    return abi, str(bytecode), artifact


def _check_signers(w3: Web3, keys: Mapping[str, str]) -> Dict[str, str]:
    addresses = {
        name: Web3.to_checksum_address(private_key_address(w3, key, role=name))
        for name, key in keys.items()
    }
    duplicates: Dict[str, list[str]] = defaultdict(list)
    for name, address in addresses.items():
        duplicates[address.lower()].append(name)
    collisions = [names for names in duplicates.values() if len(names) > 1]
    if collisions:
        raise WorkflowError(f"Each synthetic actor must use a distinct key; repeated roles: {collisions}")
    return addresses


def _require_funded_signers(
    w3: Web3,
    addresses: Mapping[str, str],
    required_names: Iterable[str],
) -> None:
    names = list(dict.fromkeys(required_names))
    unfunded = [name for name in names if int(w3.eth.get_balance(addresses[name])) == 0]
    if unfunded:
        raise WorkflowError(f"The following required synthetic signers have zero native-token balance: {unfunded}")


def _fetch_profile_with_retry(gateway: str, cid: str, key: str, aad: str) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            return fetch_encrypted_json_from_ipfs(gateway, cid, key, expected_aad=aad)
        except Exception as exc:
            last_error = exc
            if attempt < 6:
                time.sleep(float(attempt * 2))
    raise WorkflowError(f"Unable to retrieve and authenticate newly pinned profile {cid}: {last_error}")


def _sanitized_error(exc: BaseException) -> Dict[str, Any]:
    details: Dict[str, Any] = {"error_type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(status_code, int):
        details["http_status"] = status_code
    if request_id:
        details["provider_request_id"] = str(request_id)
    return details


def _recoverable_pin(
    ledger: ExternalRunLedger,
    *,
    artifact_id: str,
    plaintext_sha256: str,
) -> tuple[Dict[str, Any], str | None] | None:
    if ledger.retry_of is None:
        return None
    prior_events = [
        event
        for event in read_ledger(ledger.ledger_path)
        if event["invocation_id"] == ledger.retry_of
        and event["details"].get("artifact_id") == artifact_id
    ]
    prepared = next(
        (
            event
            for event in reversed(prior_events)
            if event["event_type"] == "pin_prepared"
            and event["details"].get("plaintext_sha256") == plaintext_sha256
        ),
        None,
    )
    if prepared is None:
        return None
    relative_path = Path(str(prepared["details"].get("envelope_path", "")))
    prior_run = (ledger.root / ledger.retry_of).resolve()
    source_path = (prior_run / relative_path).resolve()
    try:
        source_path.relative_to(prior_run)
    except ValueError as exc:
        raise WorkflowError("Prior encrypted-envelope path escapes its invocation directory") from exc
    envelope = _read_json(source_path)
    if canonical_json_sha256(envelope) != prepared["details"].get(
        "ciphertext_envelope_sha256"
    ):
        raise WorkflowError("Prior encrypted envelope no longer matches its ledger digest")
    accepted = next(
        (
            event
            for event in reversed(prior_events)
            if event["event_type"] == "pin_accepted"
            and event["details"].get("plaintext_sha256") == plaintext_sha256
        ),
        None,
    )
    return envelope, str(accepted["details"]["cid"]) if accepted is not None else None


def _pin_and_verify_encrypted_artifact(
    run_dir: Path,
    ledger: ExternalRunLedger,
    *,
    artifact_id: str,
    value: Mapping[str, Any],
    pinata_jwt: str,
    gateway: str,
    encryption_key: str,
    aad: str,
    name: str,
    readback_attempts: int = 6,
) -> str:
    if readback_attempts < 1:
        raise ValueError("readback_attempts must be positive")
    plaintext_sha256 = canonical_json_sha256(value)
    recovered = _recoverable_pin(
        ledger,
        artifact_id=artifact_id,
        plaintext_sha256=plaintext_sha256,
    )
    if recovered is None:
        envelope = encrypt_json(value, encryption_key, aad=aad)
        recovered_cid = None
    else:
        envelope, recovered_cid = recovered
    envelope_sha256 = canonical_json_sha256(envelope)
    envelope_path = run_dir / "encrypted_envelopes" / f"{artifact_id}.json"
    if envelope_path.exists():
        raise WorkflowError(f"Encrypted envelope already exists for {artifact_id}")
    _write_json(envelope_path, envelope)
    common = {
        "artifact_id": artifact_id,
        "aad": aad,
        "name": name,
        "envelope_path": str(envelope_path.relative_to(run_dir)).replace("\\", "/"),
        "plaintext_sha256": plaintext_sha256,
        "ciphertext_envelope_sha256": envelope_sha256,
        "reused_envelope_from_invocation": ledger.retry_of if recovered is not None else None,
    }
    ledger.record("pin_prepared", common)
    if recovered_cid is not None:
        cid = recovered_cid
        ledger.record(
            "pin_accepted",
            {
                **common,
                "cid": cid,
                "reused_existing_pin": True,
                "source_invocation_id": ledger.retry_of,
            },
        )
    else:
        if recovered is not None:
            ledger.record(
                "pin_upload_reconciliation_started",
                {"artifact_id": artifact_id, "source_invocation_id": ledger.retry_of},
            )
        try:
            cid = pin_json(pinata_jwt, envelope, name=name)
        except Exception as exc:
            ledger.record("pin_upload_unresolved", {**common, **_sanitized_error(exc)})
            raise
        ledger.record(
            "pin_accepted",
            {**common, "cid": cid, "reused_existing_pin": False},
        )

    for attempt in range(1, readback_attempts + 1):
        ledger.record(
            "pin_readback_started",
            {"artifact_id": artifact_id, "cid": cid, "attempt": attempt},
        )
        try:
            fetched_envelope = fetch_json_from_ipfs(
                gateway,
                cid,
                timeout=20,
                retries=0,
            )
            if canonical_json_sha256(fetched_envelope) != envelope_sha256:
                raise WorkflowError("Gateway envelope differs from the retained ciphertext")
            fetched_value = decrypt_json(
                fetched_envelope,
                encryption_key,
                expected_aad=aad,
            )
            if canonical_json_sha256(fetched_value) != plaintext_sha256:
                raise WorkflowError("Decrypted readback differs from the retained source object")
        except Exception as exc:
            ledger.record(
                "pin_readback_failed",
                {
                    "artifact_id": artifact_id,
                    "cid": cid,
                    "attempt": attempt,
                    **_sanitized_error(exc),
                },
            )
            if attempt < readback_attempts:
                time.sleep(float(attempt * 2))
                continue
            raise WorkflowError(
                f"Unable to retrieve and authenticate newly pinned artifact {artifact_id}"
            ) from exc
        ledger.record(
            "pin_readback_verified",
            {
                "artifact_id": artifact_id,
                "cid": cid,
                "attempt": attempt,
                "plaintext_sha256": plaintext_sha256,
                "ciphertext_envelope_sha256": envelope_sha256,
            },
        )
        return cid
    raise AssertionError("Unreachable Pinata readback state")


def _pin_profiles(
    case: Mapping[str, Any],
    pinata_jwt: str,
    gateway: str,
    encryption_key: str,
    *,
    run_dir: Path | None = None,
    ledger: ExternalRunLedger | None = None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], str, Dict[int, str]]:
    donor = _runtime_profile(case["donor"])
    recipients = [_runtime_profile(item) for item in case["recipients"]]
    if ledger is not None:
        if run_dir is None:
            raise WorkflowError("A run directory is required for journaled Pinata uploads")
        donor_cid = _pin_and_verify_encrypted_artifact(
            run_dir,
            ledger,
            artifact_id="donor-1",
            value=donor,
            pinata_jwt=pinata_jwt,
            gateway=gateway,
            encryption_key=encryption_key,
            aad="donor:1",
            name=f"oda-demo-{case['case_id']}-donor-1",
        )
        recipient_cids: Dict[int, str] = {}
        for recipient in recipients:
            recipient_id = int(recipient["recipient_id"])
            recipient_cids[recipient_id] = _pin_and_verify_encrypted_artifact(
                run_dir,
                ledger,
                artifact_id=f"recipient-{recipient_id}",
                value=recipient,
                pinata_jwt=pinata_jwt,
                gateway=gateway,
                encryption_key=encryption_key,
                aad=f"recipient:{recipient_id}",
                name=f"oda-demo-{case['case_id']}-recipient-{recipient_id}",
            )
        return donor, recipients, donor_cid, recipient_cids

    donor_cid = pin_encrypted_json(
        pinata_jwt,
        donor,
        encryption_key,
        aad="donor:1",
        name=f"oda-demo-{case['case_id']}-donor-1",
    )
    recipient_cids: Dict[int, str] = {}
    for recipient in recipients:
        recipient_id = int(recipient["recipient_id"])
        recipient_cids[recipient_id] = pin_encrypted_json(
            pinata_jwt,
            recipient,
            encryption_key,
            aad=f"recipient:{recipient_id}",
            name=f"oda-demo-{case['case_id']}-recipient-{recipient_id}",
        )

    fetched_donor = _fetch_profile_with_retry(gateway, donor_cid, encryption_key, "donor:1")
    if canonical_json_sha256(fetched_donor) != canonical_json_sha256(donor):
        raise WorkflowError("Retrieved donor profile differs from the encrypted source object")
    for recipient in recipients:
        recipient_id = int(recipient["recipient_id"])
        fetched = _fetch_profile_with_retry(
            gateway,
            recipient_cids[recipient_id],
            encryption_key,
            f"recipient:{recipient_id}",
        )
        if canonical_json_sha256(fetched) != canonical_json_sha256(recipient):
            raise WorkflowError(f"Retrieved recipient profile {recipient_id} differs from its source object")
    return donor, recipients, donor_cid, recipient_cids


def _receipt_row(
    receipt: TransactionReceipt,
    *,
    category: str,
    role: str,
    function: str,
    arguments: str,
) -> Dict[str, Any]:
    fee_wei = int(receipt.gas_used) * int(receipt.effective_gas_price_wei)
    return {
        "category": category,
        "role": role,
        "function": function,
        "arguments": arguments,
        "sender": receipt.sender,
        "tx_hash": receipt.tx_hash,
        "status": receipt.status,
        "block_number": receipt.block_number,
        "gas_used": receipt.gas_used,
        "effective_gas_price_wei": receipt.effective_gas_price_wei,
        "fee_wei": fee_wei,
        "fee_native": format(Decimal(fee_wei) / Decimal(10**18), ".18f"),
        "submitted_at_utc": receipt.submitted_at_utc,
        "confirmed_at_utc": receipt.confirmed_at_utc,
        "confirmation_seconds": receipt.confirmation_seconds,
    }


def _expected_full_transaction_plan(primary_recipient_id: int) -> list[Dict[str, str]]:
    plan: list[Dict[str, str]] = [
        {
            "category": "Deployment",
            "role": "Regulator",
            "function": "constructor",
            "arguments": "initial_regulator=regulator",
            "sender_key": "REGULATOR_PRIVATE_KEY",
        }
    ]
    for function_name, label in (
        ("setHospital", "hospital"),
        ("setMedicalTeam", "medical_team"),
        ("setEthicsCommittee", "ethics_committee"),
        ("setDecisionService", "decision_service"),
    ):
        plan.append({
            "category": "Governance",
            "role": "Regulator",
            "function": function_name,
            "arguments": f"account={label},enabled=true",
            "sender_key": "REGULATOR_PRIVATE_KEY",
        })
    plan.append({
        "category": "Identity binding",
        "role": "Regulator",
        "function": "registerDonorAuthority",
        "arguments": "donor_id=1",
        "sender_key": "REGULATOR_PRIVATE_KEY",
    })
    for recipient_id in range(1, 11):
        plan.append({
            "category": "Identity binding",
            "role": "Regulator",
            "function": "registerRecipientAddress",
            "arguments": f"recipient_id={recipient_id}",
            "sender_key": "REGULATOR_PRIVATE_KEY",
        })
    plan.append({
        "category": "Profile registration",
        "role": "Hospital",
        "function": "registerDonor",
        "arguments": "donor_id=1,encrypted_profile_cid=<recorded separately>",
        "sender_key": "HOSPITAL_PRIVATE_KEY",
    })
    for recipient_id in range(1, 11):
        plan.append({
            "category": "Profile registration",
            "role": "Hospital",
            "function": "registerRecipient",
            "arguments": (
                f"recipient_id={recipient_id},encrypted_profile_cid=<recorded separately>"
            ),
            "sender_key": "HOSPITAL_PRIVATE_KEY",
        })
    plan.append({
        "category": "Workflow eligibility",
        "role": "SyntheticEthicsCommittee",
        "function": "setDonorEligibility",
        "arguments": "donor_id=1,eligible=true",
        "sender_key": "ETHICS_PRIVATE_KEY",
    })
    for recipient_id in range(1, 11):
        plan.append({
            "category": "Workflow eligibility",
            "role": "SyntheticEthicsCommittee",
            "function": "setRecipientEligibility",
            "arguments": f"recipient_id={recipient_id},eligible=true",
            "sender_key": "ETHICS_PRIVATE_KEY",
        })
    plan.append({
        "category": "Match workflow",
        "role": "DecisionService",
        "function": "createMatch",
        "arguments": "dynamic",
        "sender_key": "DECISION_SERVICE_PRIVATE_KEY",
    })
    for role, function_name, sender_key in (
        ("SyntheticMedicalTeam", "approveMedicalTeam", "MEDICAL_PRIVATE_KEY"),
        ("Hospital", "approveHospital", "HOSPITAL_PRIVATE_KEY"),
        ("DonorAuthority", "approveDonorAuthority", "DONOR_PRIVATE_KEY"),
        ("ActiveRecipient", "approveRecipient", f"RECIPIENT{primary_recipient_id}_PRIVATE_KEY"),
        ("SyntheticEthicsCommittee", "approveFinalTransplant", "ETHICS_PRIVATE_KEY"),
        ("SyntheticMedicalTeam", "finalizeMatch", "MEDICAL_PRIVATE_KEY"),
    ):
        plan.append({
            "category": "Match workflow",
            "role": role,
            "function": function_name,
            "arguments": "dynamic",
            "sender_key": sender_key,
        })
    return plan


def _validate_full_transaction_records(
    records: Sequence[Mapping[str, Any]],
    addresses: Mapping[str, str],
    primary_recipient_id: int,
    backup_recipient_id: int,
    match_id: int,
) -> Dict[str, Any]:
    plan = _expected_full_transaction_plan(primary_recipient_id)
    if len(plan) != 45 or len(records) != len(plan):
        raise WorkflowError(
            f"Canonical workflow must contain exactly 45 transactions; found {len(records)}"
        )
    seen_hashes: set[str] = set()
    for index, (record, expected) in enumerate(zip(records, plan), start=1):
        for field in ("category", "role", "function"):
            if str(record.get(field, "")) != expected[field]:
                raise WorkflowError(f"Transaction {index} differs from the prescribed {field}")
        if expected["arguments"] != "dynamic" and record.get("arguments") != expected["arguments"]:
            raise WorkflowError(f"Transaction {index} differs from the prescribed arguments")
        if expected["arguments"] == "dynamic":
            arguments = str(record.get("arguments", ""))
            if expected["function"] == "createMatch":
                valid_arguments = arguments == (
                    f"donor_id=1,primary={primary_recipient_id},backup={backup_recipient_id},"
                    "encrypted_decision_cid=<recorded separately>"
                )
            else:
                valid_arguments = arguments == f"match_id={match_id}"
            if not valid_arguments:
                raise WorkflowError(f"Transaction {index} has malformed dynamic arguments")
        expected_sender = Web3.to_checksum_address(addresses[expected["sender_key"]])
        try:
            observed_sender = Web3.to_checksum_address(str(record.get("sender", "")))
        except Exception as exc:
            raise WorkflowError(f"Transaction {index} has a malformed sender") from exc
        if observed_sender != expected_sender:
            raise WorkflowError(f"Transaction {index} was signed by the wrong synthetic role")
        tx_hash = str(record.get("tx_hash", "")).lower()
        if not re.fullmatch(r"0x[0-9a-f]{64}", tx_hash) or tx_hash in seen_hashes:
            raise WorkflowError(f"Transaction {index} has a malformed or duplicate hash")
        seen_hashes.add(tx_hash)
        try:
            status = int(record["status"])
            block_number = int(record["block_number"])
            gas_used = int(record["gas_used"])
            gas_price = int(record["effective_gas_price_wei"])
            fee_wei = int(record["fee_wei"])
            fee_native = Decimal(str(record["fee_native"]))
            confirmation_seconds = Decimal(str(record["confirmation_seconds"]))
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"Transaction {index} has malformed receipt measurements") from exc
        if (
            status != 1
            or block_number < 0
            or gas_used <= 0
            or gas_price <= 0
            or fee_wei != gas_used * gas_price
            or fee_native != Decimal(fee_wei) / Decimal(10**18)
            or not confirmation_seconds.is_finite()
            or confirmation_seconds < 0
        ):
            raise WorkflowError(f"Transaction {index} has inconsistent receipt measurements")
    return {
        "complete": True,
        "expected_transaction_count": len(plan),
        "observed_transaction_count": len(records),
        "unique_transaction_hash_count": len(seen_hashes),
    }


def _send(
    records: list[Dict[str, Any]],
    w3: Web3,
    function: Any,
    key: str,
    *,
    checkpoint_path: Path,
    category: str,
    role: str,
    function_name: str,
    arguments: str,
    ledger: ExternalRunLedger | None = None,
) -> TransactionReceipt:
    transaction_id = f"tx-{len(records) + 1:03d}"
    receipt = send_signed_transaction(
        w3,
        function,
        key,
        role=f"{role}.{function_name}",
        timeout_seconds=float(os.getenv("TX_RECEIPT_TIMEOUT_S", "600")),
        poll_latency_seconds=float(os.getenv("TX_POLL_LATENCY_S", "2")),
        priority_fee_gwei=float(os.getenv("TX_PRIORITY_FEE_GWEI", "2")),
        event_recorder=ledger.record if ledger is not None else None,
        event_context={
            "transaction_id": transaction_id,
            "workflow_stage": category,
            "role": role,
            "contract_function": function_name,
            "argument_sha256": hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
        },
    )
    records.append(
        _receipt_row(
            receipt,
            category=category,
            role=role,
            function=function_name,
            arguments=arguments,
        )
    )
    _write_csv(checkpoint_path, records, list(records[0]))
    return receipt


def _aggregate_transactions(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["category"]), str(row["function"]))].append(row)
    output: list[Dict[str, Any]] = []
    for (category, function), rows in groups.items():
        gas = sum(int(row["gas_used"]) for row in rows)
        fee_wei = sum(int(row["fee_wei"]) for row in rows)
        times = [float(row["confirmation_seconds"]) for row in rows]
        output.append({
            "category": category,
            "function": function,
            "transaction_count": len(rows),
            "gas_used_total": gas,
            "fee_wei_total": fee_wei,
            "fee_native_total": format(Decimal(fee_wei) / Decimal(10**18), ".18f"),
            "confirmation_seconds_median": round(statistics.median(times), 3),
            "confirmation_seconds_min": round(min(times), 3),
            "confirmation_seconds_max": round(max(times), 3),
        })
    return output


def _aggregate_transaction_stages(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["category"]), []).append(row)
    output: list[Dict[str, Any]] = []
    for category, rows in groups.items():
        gas = sum(int(row["gas_used"]) for row in rows)
        fee_wei = sum(int(row["fee_wei"]) for row in rows)
        times = [float(row["confirmation_seconds"]) for row in rows]
        output.append({
            "category": category,
            "transaction_count": len(rows),
            "gas_used_total": gas,
            "fee_wei_total": fee_wei,
            "fee_native_total": format(Decimal(fee_wei) / Decimal(10**18), ".18f"),
            "confirmation_seconds_median": round(statistics.median(times), 3),
            "confirmation_seconds_min": round(min(times), 3),
            "confirmation_seconds_max": round(max(times), 3),
        })
    return output


def _sync_integration_files(
    network: str,
    chain_id: int,
    abi: list[dict[str, Any]],
    contract_address: str,
    deployment: TransactionReceipt,
) -> None:
    deployed_runtime_sha256 = hashlib.sha256(
        _artifact_bytecode(_read_json(ARTIFACT_PATH), "deployedBytecode")
    ).hexdigest()
    artifact_sha256 = _sha256(ARTIFACT_PATH)
    source_sha256 = _sha256(SOURCE_PATH)
    abi_value = {
        "contract": "TransplantManagement",
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
        "abi": abi,
    }
    _write_json(INTEGRATION_ABI_PATH, abi_value)
    _write_json(
        IMPLEMENTATION_DIR / "integration" / "addresses" / f"{network}.json",
        {
            "address": contract_address,
            "chain_id": chain_id,
            "deployed_at_utc": deployment.confirmed_at_utc,
            "deployment_tx_hash": deployment.tx_hash,
            "network": network,
            "protocol_id": PROTOCOL["protocol_id"],
            "source_sha256": source_sha256,
            "artifact_sha256": artifact_sha256,
            "deployed_runtime_sha256": deployed_runtime_sha256,
        },
    )


def _artifact_manifest(run_dir: Path) -> Dict[str, Any]:
    excluded = {"artifact_manifest.json", COMPLETION_FILENAME}
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name not in excluded)
    return {
        "generated_at_utc": utc_now(),
        "artifacts": {
            str(path.relative_to(run_dir)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }


def _compiler_metadata(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = artifact.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {"raw_metadata_sha256": hashlib.sha256(metadata.encode("utf-8")).hexdigest()}
    if not isinstance(metadata, dict):
        return {}
    return {
        "compiler": metadata.get("compiler", {}),
        "evm_version": metadata.get("settings", {}).get("evmVersion"),
        "optimizer": metadata.get("settings", {}).get("optimizer", {}),
        "via_ir": metadata.get("settings", {}).get("viaIR"),
    }


def _artifact_bytecode(artifact: Mapping[str, Any], field: str) -> bytes:
    value = artifact.get(field, {})
    encoded = value.get("object") if isinstance(value, Mapping) else value
    normalized = str(encoded or "").removeprefix("0x")
    if not normalized or len(normalized) % 2:
        raise WorkflowError(f"Foundry artifact has malformed {field}")
    try:
        return bytes.fromhex(normalized)
    except ValueError as exc:
        raise WorkflowError(f"Foundry artifact has non-hexadecimal {field}") from exc


def _verify_deployment(
    run_dir: Path,
    w3: Web3,
    artifact: Mapping[str, Any],
    deployment: TransactionReceipt,
) -> Dict[str, Any]:
    if not deployment.contract_address:
        raise WorkflowError("Deployment receipt did not contain a contract address")
    expected = _artifact_bytecode(artifact, "deployedBytecode")
    try:
        observed = bytes(w3.eth.get_code(deployment.contract_address))
    except Exception as exc:
        raise WorkflowError("Unable to retrieve deployed contract bytecode") from exc
    if not observed or observed != expected:
        raise WorkflowError("Deployed contract bytecode differs from the pinned Foundry artifact")
    record = {
        "verified_at_utc": utc_now(),
        "contract_address": Web3.to_checksum_address(deployment.contract_address),
        "deployment_tx_hash": deployment.tx_hash,
        "runtime_byte_count": len(observed),
        "artifact_runtime_sha256": hashlib.sha256(expected).hexdigest(),
        "deployed_runtime_sha256": hashlib.sha256(observed).hexdigest(),
        "matches": True,
    }
    _write_json(run_dir / "deployment_verification.json", record)
    return record


def _prepare_full_decision(
    run_dir: Path,
    case: Mapping[str, Any],
    donor: Dict[str, Any],
    recipients: list[Dict[str, Any]],
    donor_cid: str,
    recipient_cids: Mapping[int, str],
    *,
    model_id: str,
    pinata_jwt: str,
    pinata_gateway: str,
    encryption_key: str,
    ledger: ExternalRunLedger | None = None,
    model_caller: Any | None = None,
) -> tuple[list[Dict[str, Any]], Any, Any, Dict[str, Any], str]:
    """Complete all fallible off-chain decision steps before deployment."""
    ranked = rank_recipients_baseline(donor, recipients, PROTOCOL)
    model_call_id = "canonical-demonstration-note-review"
    if ledger is not None:
        model_payload = build_note_review_payload(
            str(case["case_id"]),
            str(case["organ_type"]),
            recipients,
        )
        ledger.record(
            "model_call_started",
            {
                "call_id": model_call_id,
                "requested_model_id": model_id,
                "case_id": str(case["case_id"]),
                "system_prompt_sha256": hashlib.sha256(
                    SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "payload_sha256": canonical_json_sha256(model_payload),
                "temperature": MODEL_SETTINGS["temperature"],
                "seed": MODEL_SETTINGS["seed"],
                "provider": MODEL_SETTINGS["provider"],
                "do_sample": MODEL_SETTINGS["do_sample"],
                "maximum_new_tokens": MODEL_SETTINGS["maximum_new_tokens"],
            },
        )
    try:
        selected_caller = model_caller or call_note_review
        model_result = selected_caller(
            model_id=model_id,
            api_key="",
            case_id=str(case["case_id"]),
            organ_type=str(case["organ_type"]),
            recipients=recipients,
        )
    except Exception as exc:
        if ledger is not None:
            ledger.record(
                "model_call_failed",
                {"call_id": model_call_id, **_sanitized_error(exc)},
            )
        raise
    if ledger is not None:
        ledger.record(
            "model_call_succeeded",
            {
                "call_id": model_call_id,
                "response_id": model_result.metadata.response_id,
                "observed_model_id": model_result.metadata.model_id,
                "provider": model_result.metadata.provider,
                "model_revision": model_result.metadata.model_revision,
                "adapter_id": model_result.metadata.adapter_id,
                "adapter_sha256": model_result.metadata.adapter_sha256,
                "device": model_result.metadata.device,
                "dtype": model_result.metadata.dtype,
                "raw_response_sha256": hashlib.sha256(
                    model_result.raw_text.encode("utf-8")
                ).hexdigest(),
                "parsed_response_sha256": canonical_json_sha256(
                    model_result.review.model_dump(mode="json")
                ),
                "input_tokens": model_result.metadata.input_tokens,
                "output_tokens": model_result.metadata.output_tokens,
                "latency_ms": model_result.metadata.latency_ms,
                "system_fingerprint": model_result.metadata.system_fingerprint,
                "service_tier": model_result.metadata.service_tier,
            },
        )
    if model_result.metadata.model_id != model_id:
        raise WorkflowError(
            "The runtime model identifier differs from the evaluated model"
        )
    guarded = apply_guarded_policy(
        1,
        ranked,
        model_result.review,
        [int(item["recipient_id"]) for item in recipients],
    )
    reference = case["reference"]
    demo_comparison = {
        "reference_primary_recipient_id": int(reference["primary_recipient_id"]),
        "reference_backup_recipient_id": int(reference["backup_recipient_id"]),
        "observed_primary_recipient_id": guarded.primary_recipient_id,
        "observed_backup_recipient_id": guarded.backup_recipient_id,
        "primary_conforms": guarded.primary_recipient_id == int(reference["primary_recipient_id"]),
        "backup_conforms": guarded.backup_recipient_id == int(reference["backup_recipient_id"]),
    }
    _write_json(
        run_dir / "model_and_guard_record.json",
        {
            "case_id": case["case_id"],
            "model_id_requested": model_id,
            "model_request_config": {
                "temperature": MODEL_SETTINGS["temperature"],
                "seed": MODEL_SETTINGS["seed"],
                "provider": MODEL_SETTINGS["provider"],
                "do_sample": MODEL_SETTINGS["do_sample"],
                "maximum_new_tokens": MODEL_SETTINGS["maximum_new_tokens"],
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            },
            "raw_response": model_result.raw_text,
            "parsed_review": model_result.review.model_dump(mode="json"),
            "metadata": model_result.metadata.model_dump(mode="json"),
            "guarded_decision": guarded.model_dump(mode="json"),
            "reference_comparison": demo_comparison,
        },
    )
    decision_artifact = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "case_id": case["case_id"],
        "donor_id": 1,
        "profile_cids": {
            "donor": donor_cid,
            "recipients": {str(key): value for key, value in recipient_cids.items()},
        },
        "baseline_ranking": ranked,
        "note_review": model_result.review.model_dump(mode="json"),
        "model_run": model_result.metadata.model_dump(mode="json"),
        "model_request_config": {
            "model_id_requested": model_id,
            "temperature": MODEL_SETTINGS["temperature"],
            "seed": MODEL_SETTINGS["seed"],
            "provider": MODEL_SETTINGS["provider"],
            "do_sample": MODEL_SETTINGS["do_sample"],
            "maximum_new_tokens": MODEL_SETTINGS["maximum_new_tokens"],
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "guarded_decision": guarded.model_dump(mode="json"),
    }
    if ledger is not None:
        decision_cid = _pin_and_verify_encrypted_artifact(
            run_dir,
            ledger,
            artifact_id="guarded-decision",
            value=decision_artifact,
            pinata_jwt=pinata_jwt,
            gateway=pinata_gateway,
            encryption_key=encryption_key,
            aad="decision:donor:1",
            name=f"oda-demo-{case['case_id']}-guarded-decision",
        )
    else:
        decision_cid = pin_encrypted_json(
            pinata_jwt,
            decision_artifact,
            encryption_key,
            aad="decision:donor:1",
            name=f"oda-demo-{case['case_id']}-guarded-decision",
        )
        retrieved_decision = _fetch_profile_with_retry(
            pinata_gateway,
            decision_cid,
            encryption_key,
            "decision:donor:1",
        )
        if canonical_json_sha256(retrieved_decision) != canonical_json_sha256(decision_artifact):
            raise WorkflowError("Retrieved decision artifact differs from the encrypted source object")
    return ranked, model_result, guarded, demo_comparison, decision_cid


def _finalize_seed_run(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
    network: str,
    chain_id: int,
    abi: list[dict[str, Any]],
    deployment: TransactionReceipt,
) -> None:
    _write_csv(run_dir / "transaction_manifest.csv", records, list(records[0]))
    _write_csv(
        run_dir / "transaction_summary.csv",
        _aggregate_transactions(records),
        [
            "category",
            "function",
            "transaction_count",
            "gas_used_total",
            "fee_wei_total",
            "fee_native_total",
            "confirmation_seconds_median",
            "confirmation_seconds_min",
            "confirmation_seconds_max",
        ],
    )
    _sync_integration_files(network, chain_id, abi, str(deployment.contract_address), deployment)
    _write_json(run_dir / "artifact_manifest.json", _artifact_manifest(run_dir))


def _execute_run(ledger: ExternalRunLedger, *, stop_after_seed: bool = False) -> Path:
    assert_test_seed_not_retired(PROTOCOL)
    assert_protocol_frozen(PROTOCOL)
    load_authoritative_project_env()
    network = str(os.getenv("NETWORK", "sepolia")).strip().lower()
    if network != "sepolia":
        raise WorkflowError("The canonical manuscript workflow must use NETWORK=sepolia")
    rpc_url = _rpc_url(network)
    if not rpc_url:
        raise WorkflowError("SEPOLIA_RPC_URL is missing in app/.env")

    case = _load_demo_case()
    keys = {name: _required_env(name) for name in PRIVATE_KEY_NAMES}
    encryption_key = _required_env("OFFCHAIN_ENCRYPTION_KEY")
    decode_encryption_key(encryption_key)
    pinata_jwt = _required_env("PINATA_JWT")
    pinata_gateway = _required_env("PINATA_GATEWAY")
    model_id = None
    model_evidence = None
    model_client = None
    if not stop_after_seed:
        training_state = validate_local_training_state(LOCAL_TRAINING_STATE_PATH)
        model_id = str(training_state["adapter"]["model_id"])
        model_evidence = _require_final_model_evidence(model_id)
        model_client = LocalNoteReviewClient(model_id, adapter=True)

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise WorkflowError("Unable to connect to the configured Sepolia RPC endpoint")
    if int(w3.eth.chain_id) != 11155111:
        raise WorkflowError(f"Expected Sepolia chain ID 11155111, found {w3.eth.chain_id}")
    addresses = _check_signers(w3, keys)
    initially_required_signers = [
        "REGULATOR_PRIVATE_KEY",
        "HOSPITAL_PRIVATE_KEY",
        "ETHICS_PRIVATE_KEY",
    ]
    if not stop_after_seed:
        initially_required_signers.extend([
            "MEDICAL_PRIVATE_KEY",
            "DONOR_PRIVATE_KEY",
            "DECISION_SERVICE_PRIVATE_KEY",
        ])
    _require_funded_signers(w3, addresses, initially_required_signers)

    run_dir = ledger.run_dir
    command_record = {
        "argv": sys.argv,
        "mode": "seed_only" if stop_after_seed else "full",
        "started_at_utc": utc_now(),
        "invocation_id": ledger.invocation_id,
        "retry_of": ledger.retry_of,
    }
    _write_json(run_dir / "command.json", command_record)
    _write_json(run_dir / "actor_addresses.json", addresses)

    build_started = time.perf_counter()
    abi, bytecode, artifact = _build_contract_artifact(run_dir)
    build_seconds = time.perf_counter() - build_started
    profile_started = time.perf_counter()
    donor, recipients, donor_cid, recipient_cids = _pin_profiles(
        case,
        pinata_jwt,
        pinata_gateway,
        encryption_key,
        run_dir=run_dir,
        ledger=ledger,
    )
    profile_preparation_seconds = time.perf_counter() - profile_started
    _write_json(
        run_dir / "encrypted_profile_cids.json",
        {"donor": donor_cid, "recipients": {str(key): value for key, value in recipient_cids.items()}},
    )
    prepared_decision = None
    decision_preparation_seconds = None
    if not stop_after_seed:
        if model_client is None or model_id is None:
            raise WorkflowError("Full workflow model configuration was not initialized")
        decision_started = time.perf_counter()
        prepared_decision = _prepare_full_decision(
            run_dir,
            case,
            donor,
            recipients,
            donor_cid,
            recipient_cids,
            model_id=model_id,
            pinata_jwt=pinata_jwt,
            pinata_gateway=pinata_gateway,
            encryption_key=encryption_key,
            ledger=ledger,
            model_caller=model_client,
        )
        decision_preparation_seconds = time.perf_counter() - decision_started
        selected_recipient_id = int(prepared_decision[2].primary_recipient_id)
        _require_funded_signers(
            w3,
            addresses,
            [f"RECIPIENT{selected_recipient_id}_PRIVATE_KEY"],
        )

    contract_factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    records: list[Dict[str, Any]] = []
    deployment = _send(
        records,
        w3,
        contract_factory.constructor(addresses["REGULATOR_PRIVATE_KEY"]),
        keys["REGULATOR_PRIVATE_KEY"],
        checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
        category="Deployment",
        role="Regulator",
        function_name="constructor",
        arguments="initial_regulator=regulator",
        ledger=ledger,
    )
    if not deployment.contract_address:
        raise WorkflowError("Deployment receipt did not contain a contract address")
    deployment_verification = _verify_deployment(run_dir, w3, artifact, deployment)
    contract = w3.eth.contract(address=deployment.contract_address, abi=abi)

    role_setup = (
        ("setHospital", addresses["HOSPITAL_PRIVATE_KEY"], "hospital"),
        ("setMedicalTeam", addresses["MEDICAL_PRIVATE_KEY"], "medical_team"),
        ("setEthicsCommittee", addresses["ETHICS_PRIVATE_KEY"], "ethics_committee"),
        ("setDecisionService", addresses["DECISION_SERVICE_PRIVATE_KEY"], "decision_service"),
    )
    for function_name, address, label in role_setup:
        _send(
            records,
            w3,
            getattr(contract.functions, function_name)(address, True),
            keys["REGULATOR_PRIVATE_KEY"],
            checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
            category="Governance",
            role="Regulator",
            function_name=function_name,
            arguments=f"account={label},enabled=true",
            ledger=ledger,
        )

    _send(
        records,
        w3,
        contract.functions.registerDonorAuthority(addresses["DONOR_PRIVATE_KEY"]),
        keys["REGULATOR_PRIVATE_KEY"],
        checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
        category="Identity binding",
        role="Regulator",
        function_name="registerDonorAuthority",
        arguments="donor_id=1",
        ledger=ledger,
    )
    for recipient_id in range(1, 11):
        _send(
            records,
            w3,
            contract.functions.registerRecipientAddress(addresses[f"RECIPIENT{recipient_id}_PRIVATE_KEY"]),
            keys["REGULATOR_PRIVATE_KEY"],
            checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
            category="Identity binding",
            role="Regulator",
            function_name="registerRecipientAddress",
            arguments=f"recipient_id={recipient_id}",
            ledger=ledger,
        )
    if int(contract.functions.donorCounter().call()) != 1 or int(contract.functions.recipientCounter().call()) != 10:
        raise WorkflowError("Fresh contract identity counters do not match the demonstration case")

    _send(
        records,
        w3,
        contract.functions.registerDonor(addresses["DONOR_PRIVATE_KEY"], donor_cid),
        keys["HOSPITAL_PRIVATE_KEY"],
        checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
        category="Profile registration",
        role="Hospital",
        function_name="registerDonor",
        arguments="donor_id=1,encrypted_profile_cid=<recorded separately>",
        ledger=ledger,
    )
    for recipient_id in range(1, 11):
        _send(
            records,
            w3,
            contract.functions.registerRecipient(
                addresses[f"RECIPIENT{recipient_id}_PRIVATE_KEY"],
                recipient_cids[recipient_id],
            ),
            keys["HOSPITAL_PRIVATE_KEY"],
            checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
            category="Profile registration",
            role="Hospital",
            function_name="registerRecipient",
            arguments=f"recipient_id={recipient_id},encrypted_profile_cid=<recorded separately>",
            ledger=ledger,
        )

    _send(
        records,
        w3,
        contract.functions.setDonorEligibility(1, True),
        keys["ETHICS_PRIVATE_KEY"],
        checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
        category="Workflow eligibility",
        role="SyntheticEthicsCommittee",
        function_name="setDonorEligibility",
        arguments="donor_id=1,eligible=true",
        ledger=ledger,
    )
    for recipient_id in range(1, 11):
        _send(
            records,
            w3,
            contract.functions.setRecipientEligibility(recipient_id, True),
            keys["ETHICS_PRIVATE_KEY"],
            checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
            category="Workflow eligibility",
            role="SyntheticEthicsCommittee",
            function_name="setRecipientEligibility",
            arguments=f"recipient_id={recipient_id},eligible=true",
            ledger=ledger,
        )

    seed_summary = {
        "case_id": case["case_id"],
        "contract_address": deployment.contract_address,
        "donor_cid": donor_cid,
        "recipient_cids": {str(key): value for key, value in recipient_cids.items()},
        "transactions": len(records),
    }
    _write_json(run_dir / "seed_summary.json", seed_summary)

    if stop_after_seed:
        _finalize_seed_run(run_dir, records, network, int(w3.eth.chain_id), abi, deployment)
        _write_completion(run_dir, mode="seed_only")
        return run_dir

    if prepared_decision is None:
        raise WorkflowError("Full workflow decision preparation did not complete")
    ranked, model_result, guarded, demo_comparison, decision_cid = prepared_decision

    create_receipt = _send(
        records,
        w3,
        contract.functions.createMatch(
            1,
            guarded.primary_recipient_id,
            guarded.backup_recipient_id,
            decision_cid,
        ),
        keys["DECISION_SERVICE_PRIVATE_KEY"],
        checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
        category="Match workflow",
        role="DecisionService",
        function_name="createMatch",
        arguments=(
            f"donor_id=1,primary={guarded.primary_recipient_id},"
            f"backup={guarded.backup_recipient_id},encrypted_decision_cid=<recorded separately>"
        ),
        ledger=ledger,
    )
    create_chain_receipt = w3.eth.get_transaction_receipt(create_receipt.tx_hash)
    events = contract.events.MatchCreated().process_receipt(create_chain_receipt)
    if len(events) != 1:
        raise WorkflowError("Expected exactly one MatchCreated event")
    try:
        event_args = events[0]["args"]
        match_id = int(event_args["matchId"])
        match_event_valid = (
            match_id > 0
            and int(event_args["donorId"]) == 1
            and int(event_args["primaryRecipientId"]) == guarded.primary_recipient_id
            and int(event_args["backupRecipientId"]) == guarded.backup_recipient_id
            and str(event_args["decisionCID"]) == decision_cid
            and Web3.to_checksum_address(event_args["recordedBy"])
            == addresses["DECISION_SERVICE_PRIVATE_KEY"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError("MatchCreated event fields are malformed") from exc
    if not match_event_valid:
        raise WorkflowError("MatchCreated event differs from the guarded decision")

    approval_steps = (
        (
            contract.functions.approveMedicalTeam(match_id),
            keys["MEDICAL_PRIVATE_KEY"],
            "SyntheticMedicalTeam",
            "approveMedicalTeam",
        ),
        (
            contract.functions.approveHospital(match_id),
            keys["HOSPITAL_PRIVATE_KEY"],
            "Hospital",
            "approveHospital",
        ),
        (
            contract.functions.approveDonorAuthority(match_id),
            keys["DONOR_PRIVATE_KEY"],
            "DonorAuthority",
            "approveDonorAuthority",
        ),
        (
            contract.functions.approveRecipient(match_id),
            keys[f"RECIPIENT{guarded.primary_recipient_id}_PRIVATE_KEY"],
            "ActiveRecipient",
            "approveRecipient",
        ),
        (
            contract.functions.approveFinalTransplant(match_id),
            keys["ETHICS_PRIVATE_KEY"],
            "SyntheticEthicsCommittee",
            "approveFinalTransplant",
        ),
        (
            contract.functions.finalizeMatch(match_id),
            keys["MEDICAL_PRIVATE_KEY"],
            "SyntheticMedicalTeam",
            "finalizeMatch",
        ),
    )
    for function, key, role, function_name in approval_steps:
        _send(
            records,
            w3,
            function,
            key,
            checkpoint_path=run_dir / "transaction_manifest.checkpoint.csv",
            category="Match workflow",
            role=role,
            function_name=function_name,
            arguments=f"match_id={match_id}",
            ledger=ledger,
        )

    match = _decode_contract_struct(
        contract.functions.matches(match_id).call(),
        MATCH_FIELDS,
        label="Match",
    )
    donor_chain = _decode_contract_struct(
        contract.functions.donors(1).call(),
        DONOR_FIELDS,
        label="Donor",
    )
    primary_chain = _decode_contract_struct(
        contract.functions.recipients(guarded.primary_recipient_id).call(),
        RECIPIENT_FIELDS,
        label="Primary recipient",
    )
    backup_chain = _decode_contract_struct(
        contract.functions.recipients(guarded.backup_recipient_id).call(),
        RECIPIENT_FIELDS,
        label="Backup recipient",
    )
    transaction_plan = _validate_full_transaction_records(
        records,
        addresses,
        guarded.primary_recipient_id,
        guarded.backup_recipient_id,
        match_id,
    )
    final_checks = {
        "match_finalized": bool(match["finalized"]),
        "match_not_cancelled": not bool(match["cancelled"]),
        "recorded_by_decision_service": (
            str(match["recorded_by"]).lower()
            == addresses["DECISION_SERVICE_PRIVATE_KEY"].lower()
        ),
        "onchain_primary_matches_guard": (
            int(match["primary_recipient_id"]) == guarded.primary_recipient_id
        ),
        "onchain_backup_matches_guard": (
            int(match["backup_recipient_id"]) == guarded.backup_recipient_id
        ),
        "onchain_active_recipient_matches_guard": (
            int(match["active_recipient_id"]) == guarded.primary_recipient_id
        ),
        "decision_cid_matches": str(match["decision_cid"]) == decision_cid,
        "donor_finalized": bool(donor_chain["finalized"]),
        "donor_has_no_open_match": not bool(contract.functions.donorHasOpenMatch(1).call()),
        "active_recipient_transplanted": bool(primary_chain["transplanted"]),
        "backup_recipient_not_transplanted": not bool(backup_chain["transplanted"]),
        "primary_reservation_released": not bool(primary_chain["reserved"]),
        "backup_reservation_released": not bool(backup_chain["reserved"]),
        "all_transactions_succeeded": all(int(row["status"]) == 1 for row in records),
        "transaction_plan_complete": transaction_plan["complete"],
        "deployed_code_matches_artifact": deployment_verification["matches"],
    }
    if not all(final_checks.values()):
        raise WorkflowError(f"Final on-chain invariants failed: {final_checks}")

    _write_csv(run_dir / "transaction_manifest.csv", records, list(records[0]))
    aggregate = _aggregate_transactions(records)
    _write_csv(
        run_dir / "transaction_summary.csv",
        aggregate,
        [
            "category",
            "function",
            "transaction_count",
            "gas_used_total",
            "fee_wei_total",
            "fee_native_total",
            "confirmation_seconds_median",
            "confirmation_seconds_min",
            "confirmation_seconds_max",
        ],
    )
    stage_aggregate = _aggregate_transaction_stages(records)
    _write_csv(
        run_dir / "transaction_stage_summary.csv",
        stage_aggregate,
        [
            "category",
            "transaction_count",
            "gas_used_total",
            "fee_wei_total",
            "fee_native_total",
            "confirmation_seconds_median",
            "confirmation_seconds_min",
            "confirmation_seconds_max",
        ],
    )
    run_summary = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "demo_case_id": case["case_id"],
        "demo_case_sha256": _sha256(DEMO_CASE_PATH),
        "network": network,
        "chain_id": int(w3.eth.chain_id),
        "contract_address": deployment.contract_address,
        "deployment_tx_hash": deployment.tx_hash,
        "contract_source_sha256": _sha256(SOURCE_PATH),
        "foundry_artifact_sha256": _sha256(ARTIFACT_PATH),
        "contract_build_sha256": _sha256(run_dir / "contract_build.json"),
        "foundry_configuration_sha256": _sha256(SMART_CONTRACTS_DIR / "foundry.toml"),
        "deployment_verification_sha256": _sha256(run_dir / "deployment_verification.json"),
        "model_and_guard_record_sha256": _sha256(run_dir / "model_and_guard_record.json"),
        "deployed_runtime_sha256": deployment_verification["deployed_runtime_sha256"],
        "decision_cid": decision_cid,
        "match_id": match_id,
        "model_id_requested": model_id,
        "model": model_result.metadata.model_dump(mode="json"),
        "model_evidence": model_evidence,
        "guarded_decision": guarded.model_dump(mode="json"),
        "demo_reference_comparison": demo_comparison,
        "predeployment_timings_seconds": {
            "contract_build": round(build_seconds, 6),
            "encrypted_profile_upload_and_readback": round(profile_preparation_seconds, 6),
            "decision_preparation_and_persistence": (
                round(decision_preparation_seconds, 6)
                if decision_preparation_seconds is not None
                else None
            ),
        },
        "final_checks": final_checks,
        "transaction_plan_validation": transaction_plan,
        "transaction_count": len(records),
        "gas_used_total": sum(int(row["gas_used"]) for row in records),
        "fee_wei_total": sum(int(row["fee_wei"]) for row in records),
        "fee_native_total": format(
            Decimal(sum(int(row["fee_wei"]) for row in records)) / Decimal(10**18),
            ".18f",
        ),
        "completed_at_utc": utc_now(),
        "implementation_sha256": {
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
            "project_environment": _sha256(PROJECT_ENVIRONMENT_PATH),
            "artifact_paths": _sha256(ARTIFACT_PATHS_PATH),
            "contract": _sha256(SOURCE_PATH),
        },
    }
    _write_json(run_dir / "run_summary.json", run_summary)
    ui_evidence = MatchResponse.model_validate({
        "donor_id": 1,
        "baseline_top": _baseline_snapshot(ranked),
        "guarded_decision": guarded.model_dump(mode="json"),
        "model_run": model_result.metadata.model_dump(mode="json"),
        "match_cid": decision_cid,
        "onchain": {
            "tx_hash": create_receipt.tx_hash,
            "match_id": match_id,
            "gas_used": create_receipt.gas_used,
            "contract_address": deployment.contract_address,
        },
    }).model_dump(mode="json")
    _write_json(run_dir / "ui_evidence_snapshot.json", ui_evidence)
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "openai",
                "pydantic",
                "web3",
                "cryptography",
                "requests",
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            )
        },
        "chain_id": int(w3.eth.chain_id),
        "network": network,
        "contract_build": {
            **_compiler_metadata(artifact),
            "provenance": _read_json(run_dir / "contract_build.json"),
        },
    }
    _write_json(run_dir / "environment_manifest.json", environment)
    _sync_integration_files(network, int(w3.eth.chain_id), abi, deployment.contract_address, deployment)
    _write_json(run_dir / "artifact_manifest.json", _artifact_manifest(run_dir))
    return run_dir


def run(
    *,
    stop_after_seed: bool = False,
    retry_failed: bool = False,
    retry_reason: str | None = None,
) -> Path:
    assert_test_seed_not_retired(PROTOCOL)
    assert_protocol_frozen(PROTOCOL)
    if stop_after_seed:
        raise WorkflowError(
            "The approved canonical Sepolia protocol disables seed-only execution"
        )

    ledger = ExternalRunLedger.start(
        CURRENT_OUTPUT_DIR / "full_workflow",
        mode="full",
        retry_failed=retry_failed,
        retry_reason=retry_reason,
    )
    try:
        run_dir = _execute_run(ledger, stop_after_seed=False)
        run_summary = _read_json(run_dir / "run_summary.json")
        completion_details = {
            "run_directory": run_dir.name,
            "transaction_count": int(run_summary["transaction_count"]),
            "run_summary_sha256": _sha256(run_dir / "run_summary.json"),
        }
        ledger.validate_completion_ready(completion_details)
        _write_completion(run_dir, mode="full")
        ledger.complete(completion_details)
        summaries = ledger.summaries()
        _write_json(
            CURRENT_OUTPUT_DIR / "latest_full_workflow.json",
            {
                "completed_at_utc": run_summary["completed_at_utc"],
                "canonical_invocation_id": ledger.invocation_id,
                "run_directory": str(run_dir.relative_to(APP_DIR)).replace("\\", "/"),
                "run_summary_sha256": _sha256(run_dir / "run_summary.json"),
                "ui_evidence_snapshot_sha256": _sha256(
                    run_dir / "ui_evidence_snapshot.json"
                ),
                "invocation_ledger_path": str(
                    ledger.ledger_path.relative_to(APP_DIR)
                ).replace("\\", "/"),
                "invocation_ledger_sha256": _sha256(ledger.ledger_path),
                "invocation_count": len(summaries),
                "invocations": [
                    {
                        "invocation_id": item["invocation_id"],
                        "status": item["status"],
                        "retry_of": item["retry_of"],
                        "event_count": item["event_count"],
                    }
                    for item in summaries
                ],
            },
        )
        return run_dir
    except BaseException as exc:
        ledger.fail(exc)
        raise
    finally:
        ledger.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop-after-seed",
        action="store_true",
        help="Retained for compatibility; prohibited by the approved canonical protocol.",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-reason", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = run(
        stop_after_seed=args.stop_after_seed,
        retry_failed=args.retry_failed,
        retry_reason=args.retry_reason,
    )
    print(f"Workflow artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
