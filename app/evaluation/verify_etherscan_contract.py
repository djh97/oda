"""Publish and verify the canonical contract source on Etherscan."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from evaluation.paper_full_workflow import _sha256
from evaluation.project_environment import (
    APP_DIR,
    CONTROLLED_ENV_NAMES,
    load_authoritative_project_env,
)


SMART_CONTRACTS_DIR = APP_DIR.parent / "smart-contracts"
SOURCE_PATH = SMART_CONTRACTS_DIR / "src" / "TransplantManagement.sol"
FOUNDRY_PATH = SMART_CONTRACTS_DIR / "foundry.toml"
POINTER_PATH = APP_DIR / "pipeline-output" / "current" / "latest_full_workflow.json"
CONTRACT_ID = "src/TransplantManagement.sol:TransplantManagement"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_run() -> Path:
    pointer = _read_json(POINTER_PATH)
    raw_path = pointer.get("run_directory") or pointer.get("run_dir")
    if not raw_path:
        raise RuntimeError("The canonical workflow pointer has no run directory")
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = APP_DIR / candidate
    run_dir = candidate.resolve()
    root = (APP_DIR / "pipeline-output" / "current" / "full_workflow").resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("The canonical workflow pointer leaves its evidence root") from exc
    if not run_dir.is_dir():
        raise RuntimeError(f"Canonical run directory not found: {run_dir}")
    return run_dir


def _constructor_args(address: str) -> str:
    normalized = address.strip().lower()
    if not normalized.startswith("0x") or len(normalized) != 42:
        raise RuntimeError("The canonical regulator address is invalid")
    try:
        int(normalized[2:], 16)
    except ValueError as exc:
        raise RuntimeError("The canonical regulator address is invalid") from exc
    return "0x" + normalized[2:].rjust(64, "0")


def _api_key() -> str:
    load_authoritative_project_env()
    value = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not value or value == "YOUR_ETHERSCAN_API_KEY":
        raise RuntimeError(
            "ETHERSCAN_API_KEY is missing from app/.env"
        )
    return value


def _verification_environment(api_key: str) -> dict[str, str]:
    process_env = os.environ.copy()
    for name in CONTROLLED_ENV_NAMES:
        process_env.pop(name, None)
    process_env["ETHERSCAN_API_KEY"] = api_key
    return process_env


def verify() -> Path:
    api_key = _api_key()

    run_dir = _canonical_run()
    summary = _read_json(run_dir / "run_summary.json")
    build = _read_json(run_dir / "contract_build.json")
    deployment = _read_json(run_dir / "deployment_verification.json")
    addresses = _read_json(run_dir / "actor_addresses.json")

    if _sha256(SOURCE_PATH) != build["contract_source_sha256"]:
        raise RuntimeError("Contract source differs from the canonical deployed build")
    if _sha256(FOUNDRY_PATH) != build["foundry_configuration_sha256"]:
        raise RuntimeError("Foundry configuration differs from the canonical deployed build")
    if deployment.get("matches") is not True:
        raise RuntimeError("Canonical runtime-bytecode verification did not pass")
    if deployment["contract_address"].lower() != summary["contract_address"].lower():
        raise RuntimeError("Canonical contract addresses do not match")

    compiler = build["artifact_compiler"]["version"]
    command = [
        "forge",
        "verify-contract",
        "--watch",
        "--chain",
        "sepolia",
        "--verifier",
        "etherscan",
        "--compiler-version",
        compiler,
        "--num-of-optimizations",
        "200",
        "--via-ir",
        "--constructor-args",
        _constructor_args(addresses["REGULATOR_PRIVATE_KEY"]),
        summary["contract_address"],
        CONTRACT_ID,
    ]

    result = subprocess.run(
        command,
        cwd=SMART_CONTRACTS_DIR,
        env=_verification_environment(api_key),
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = run_dir / "external_verification" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    record = {
        "schema_version": "1.0",
        "attempted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract_address": summary["contract_address"],
        "deployment_tx_hash": deployment["deployment_tx_hash"],
        "contract_id": CONTRACT_ID,
        "chain": "sepolia",
        "compiler_version": compiler,
        "optimizer": True,
        "optimizer_runs": 200,
        "via_ir": True,
        "constructor_regulator_address": addresses["REGULATOR_PRIVATE_KEY"],
        "source_sha256": _sha256(SOURCE_PATH),
        "foundry_configuration_sha256": _sha256(FOUNDRY_PATH),
        "deployed_runtime_sha256": deployment["deployed_runtime_sha256"],
        "return_code": result.returncode,
        "api_key_recorded": False,
    }
    record_path = output_dir / "verification_attempt.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Etherscan verification failed: {detail}")
    return record_path


def check_status(guid: str) -> Path:
    if re.fullmatch(r"[a-zA-Z0-9]+", guid) is None:
        raise ValueError("Etherscan verification GUID contains invalid characters")
    api_key = _api_key()
    run_dir = _canonical_run()
    summary = _read_json(run_dir / "run_summary.json")
    deployment = _read_json(run_dir / "deployment_verification.json")
    result = subprocess.run(
        [
            "forge",
            "verify-check",
            "--chain",
            "sepolia",
            "--verifier",
            "etherscan",
            "--retries",
            "1",
            "--delay",
            "1",
            guid,
        ],
        cwd=SMART_CONTRACTS_DIR,
        env=_verification_environment(api_key),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    verified = result.returncode == 0 and "Pass - Verified" in result.stdout
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = run_dir / "external_verification" / f"status_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    record_path = output_dir / "verification_status.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checked_at_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "chain": "sepolia",
                "contract_address": summary["contract_address"],
                "deployment_tx_hash": deployment["deployment_tx_hash"],
                "guid": guid,
                "status": "verified" if verified else "not_verified",
                "return_code": result.returncode,
                "api_key_recorded": False,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not verified:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Etherscan verification is not complete: {detail}")
    return record_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-guid", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check_guid:
        print(f"Etherscan verification status written to {check_status(args.check_guid)}")
    else:
        print(f"Etherscan verification record written to {verify()}")


if __name__ == "__main__":
    main()
