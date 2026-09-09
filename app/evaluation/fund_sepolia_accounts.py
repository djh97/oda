"""Plan or execute auditable Sepolia funding for synthetic workflow EOAs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values
from web3 import Web3

from evaluation.check_environment import ACTOR_KEYS


APP_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = APP_DIR / ".env"
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "setup" / "funding"
SEPOLIA_CHAIN_ID = 11_155_111
TRANSFER_GAS = 21_000
MINIMUM_PLANNING_FEE_WEI = 5_000_000_000
TARGET_ROUNDING_WEI = 100_000_000_000_000  # 0.0001 ETH

# Maximum observed Foundry gas for each canonical sender's expected calls.
# Targets apply an additional 50% headroom at the planning fee below.
EXPECTED_GAS_BY_KEY = {
    "REGULATOR_PRIVATE_KEY": 4_229_200,
    "HOSPITAL_PRIVATE_KEY": 1_382_200,
    "ETHICS_PRIVATE_KEY": 403_820,
    "MEDICAL_PRIVATE_KEY": 136_087,
    "DONOR_PRIVATE_KEY": 60_188,
    "DECISION_SERVICE_PRIVATE_KEY": 264_394,
    **{f"RECIPIENT{index}_PRIVATE_KEY": 60_339 for index in range(1, 11)},
}


class FundingError(RuntimeError):
    """Raised before or during a guarded funding operation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_up(value: int, unit: int = TARGET_ROUNDING_WEI) -> int:
    return ((value + unit - 1) // unit) * unit


def _eth(wei: int) -> str:
    return format(Decimal(wei) / Decimal(10**18), "f")


def _tx_hash(value: Any) -> str:
    rendered = Web3.to_hex(value).lower()
    if len(rendered) != 66 or not rendered.startswith("0x"):
        raise FundingError("A funding transaction hash is malformed")
    int(rendered[2:], 16)
    return rendered


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_configuration(path: Path = ENV_PATH) -> tuple[str, dict[str, str], int]:
    if not path.is_file():
        raise FundingError("The project environment file is missing")
    values = {key: str(value or "").strip() for key, value in dotenv_values(path, interpolate=False).items()}
    if values.get("NETWORK", "").lower() != "sepolia":
        raise FundingError("NETWORK must be sepolia")
    rpc_url = values.get("SEPOLIA_RPC_URL", "")
    parsed = urlsplit(rpc_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise FundingError("SEPOLIA_RPC_URL is not a complete HTTP or HTTPS URL")

    keys: dict[str, str] = {}
    addresses: dict[str, str] = {}
    for name in ACTOR_KEYS:
        key = values.get(name, "")
        if not key:
            raise FundingError(f"{name} is missing")
        try:
            address = Web3().eth.account.from_key(key).address.lower()
        except Exception as exc:
            raise FundingError(f"{name} is not a valid EVM private key") from exc
        if address in addresses:
            raise FundingError(f"Synthetic EOA keys are not distinct: {addresses[address]} and {name}")
        addresses[address] = name
        keys[name] = key

    try:
        priority_gwei = Decimal(values.get("TX_PRIORITY_FEE_GWEI", "2"))
    except InvalidOperation as exc:
        raise FundingError("TX_PRIORITY_FEE_GWEI must be numeric") from exc
    if not priority_gwei.is_finite() or priority_gwei < 0:
        raise FundingError("TX_PRIORITY_FEE_GWEI must be finite and nonnegative")
    priority_fee_wei = int(priority_gwei * Decimal(10**9))
    return rpc_url, keys, priority_fee_wei


def calculate_targets(planning_fee_wei: int) -> dict[str, int]:
    if not isinstance(planning_fee_wei, int) or isinstance(planning_fee_wei, bool) or planning_fee_wei <= 0:
        raise FundingError("Planning fee must be a positive integer")
    return {
        name: _round_up(math.ceil(gas * planning_fee_wei * 1.5))
        for name, gas in EXPECTED_GAS_BY_KEY.items()
    }


def required_regulator_balance(
    regulator_target_wei: int,
    deficits_wei: Iterable[int],
    funding_fee_cap_wei: int,
) -> int:
    deficits = [int(value) for value in deficits_wei if int(value) > 0]
    return regulator_target_wei + sum(deficits) + len(deficits) * TRANSFER_GAS * funding_fee_cap_wei


def _connect(rpc_url: str) -> Web3:
    provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30})
    w3 = Web3(provider)
    try:
        connected = w3.is_connected()
    except Exception:
        raise FundingError("Sepolia RPC connectivity check failed") from None
    if not connected:
        raise FundingError("Sepolia RPC did not connect")
    try:
        chain_id = int(w3.eth.chain_id)
    except Exception:
        raise FundingError("Sepolia RPC chain-ID check failed") from None
    if chain_id != SEPOLIA_CHAIN_ID:
        raise FundingError(f"Refusing funding on chain ID {chain_id}; expected Sepolia {SEPOLIA_CHAIN_ID}")
    return w3


def _build_plan(w3: Web3, keys: Mapping[str, str], priority_fee_wei: int) -> dict[str, Any]:
    try:
        latest = w3.eth.get_block("latest")
        base_fee_wei = int(latest["baseFeePerGas"])
    except Exception as exc:
        raise FundingError(f"Could not obtain Sepolia EIP-1559 fee data ({type(exc).__name__})") from exc
    if base_fee_wei < 0:
        raise FundingError("Sepolia returned a negative base fee")
    funding_fee_cap_wei = base_fee_wei * 2 + priority_fee_wei
    planning_fee_wei = max(funding_fee_cap_wei, MINIMUM_PLANNING_FEE_WEI)
    targets = calculate_targets(planning_fee_wei)

    accounts: list[dict[str, Any]] = []
    for name in ACTOR_KEYS:
        address = Web3.to_checksum_address(w3.eth.account.from_key(keys[name]).address)
        try:
            balance_wei = int(w3.eth.get_balance(address, "latest"))
        except Exception as exc:
            raise FundingError(f"Could not query the balance for {name} ({type(exc).__name__})") from exc
        target_wei = targets[name]
        deficit_wei = max(0, target_wei - balance_wei) if name != "REGULATOR_PRIVATE_KEY" else 0
        accounts.append(
            {
                "key_name": name,
                "address": address,
                "expected_gas": EXPECTED_GAS_BY_KEY[name],
                "balance_wei": balance_wei,
                "balance_eth": _eth(balance_wei),
                "target_wei": target_wei,
                "target_eth": _eth(target_wei),
                "deficit_wei": deficit_wei,
                "deficit_eth": _eth(deficit_wei),
            }
        )

    regulator = next(item for item in accounts if item["key_name"] == "REGULATOR_PRIVATE_KEY")
    deficits = [item["deficit_wei"] for item in accounts if item["key_name"] != "REGULATOR_PRIVATE_KEY"]
    required_wei = required_regulator_balance(
        int(regulator["target_wei"]),
        deficits,
        funding_fee_cap_wei,
    )
    sufficient = int(regulator["balance_wei"]) >= required_wei
    return {
        "schema_version": "1.0",
        "artifact_type": "sepolia_synthetic_eoa_funding",
        "created_at_utc": _utc_now(),
        "status": "planned" if sufficient else "insufficient_regulator_balance",
        "chain_id": SEPOLIA_CHAIN_ID,
        "rpc_origin": f"{urlsplit(w3.provider.endpoint_uri).scheme}://{urlsplit(w3.provider.endpoint_uri).netloc}",
        "fee_assumptions": {
            "base_fee_wei": base_fee_wei,
            "priority_fee_wei": priority_fee_wei,
            "funding_transaction_max_fee_wei": funding_fee_cap_wei,
            "planning_fee_wei": planning_fee_wei,
            "planning_fee_floor_gwei": 5,
            "role_gas_headroom_multiplier": "1.5",
            "native_transfer_gas": TRANSFER_GAS,
        },
        "regulator_required_before_funding_wei": required_wei,
        "regulator_required_before_funding_eth": _eth(required_wei),
        "regulator_balance_sufficient": sufficient,
        "accounts": accounts,
        "transactions": [],
        "script_sha256": _script_sha256(),
        "secret_values_retained": False,
    }


def _report_path(execute: bool) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "execution" if execute else "plan"
    return OUTPUT_DIR / f"sepolia_funding_{suffix}_{timestamp}.json"


def _execute_plan(w3: Web3, keys: Mapping[str, str], plan: dict[str, Any], path: Path) -> None:
    if not plan["regulator_balance_sufficient"]:
        raise FundingError(
            "The regulator balance cannot cover all deficits, funding fees, and its retained workflow reserve"
        )
    regulator_key = keys["REGULATOR_PRIVATE_KEY"]
    regulator_address = Web3.to_checksum_address(w3.eth.account.from_key(regulator_key).address)
    fee_cap = int(plan["fee_assumptions"]["funding_transaction_max_fee_wei"])
    priority_fee = int(plan["fee_assumptions"]["priority_fee_wei"])
    timeout_seconds = 600
    poll_latency_seconds = 2
    plan["status"] = "executing"
    plan["execution_started_at_utc"] = _utc_now()
    _write_json(path, plan)

    for account in plan["accounts"]:
        deficit = int(account["deficit_wei"])
        if account["key_name"] == "REGULATOR_PRIVATE_KEY" or deficit == 0:
            continue
        try:
            nonce = int(w3.eth.get_transaction_count(regulator_address, "pending"))
            tx = {
                "chainId": SEPOLIA_CHAIN_ID,
                "from": regulator_address,
                "to": Web3.to_checksum_address(account["address"]),
                "value": deficit,
                "nonce": nonce,
                "gas": TRANSFER_GAS,
                "maxPriorityFeePerGas": priority_fee,
                "maxFeePerGas": fee_cap,
                "type": 2,
            }
            record: dict[str, Any] = {
                "key_name": account["key_name"],
                "recipient": account["address"],
                "value_wei": deficit,
                "value_eth": _eth(deficit),
                "nonce": nonce,
                "status": "prepared",
                "prepared_at_utc": _utc_now(),
            }
            plan["transactions"].append(record)
            _write_json(path, plan)

            signed = w3.eth.account.sign_transaction(tx, regulator_key)
            signed_hash = _tx_hash(signed.hash)
            record.update({"status": "signed", "signed_hash": signed_hash, "signed_at_utc": _utc_now()})
            _write_json(path, plan)

            submitted_at = _utc_now()
            started = time.perf_counter()
            submitted_hash_value = w3.eth.send_raw_transaction(signed.raw_transaction)
            submitted_hash = _tx_hash(submitted_hash_value)
            if submitted_hash != signed_hash:
                raise FundingError("The RPC returned a hash different from the signed funding transaction")
            record.update(
                {
                    "status": "submitted",
                    "submitted_hash": submitted_hash,
                    "submitted_at_utc": submitted_at,
                }
            )
            _write_json(path, plan)

            receipt = w3.eth.wait_for_transaction_receipt(
                submitted_hash_value,
                timeout=timeout_seconds,
                poll_latency=poll_latency_seconds,
            )
            if int(receipt["status"]) != 1:
                raise FundingError(f"Funding transaction reverted: {submitted_hash}")
            if _tx_hash(receipt["transactionHash"]) != submitted_hash:
                raise FundingError("Funding receipt hash differs from the submitted hash")
            receipt_sender = Web3.to_checksum_address(receipt["from"])
            receipt_recipient = Web3.to_checksum_address(receipt["to"])
            if receipt_sender != regulator_address or receipt_recipient != account["address"]:
                raise FundingError("Funding receipt sender or recipient differs from the signed transaction")
            chain_tx = w3.eth.get_transaction(submitted_hash)
            if int(chain_tx["value"]) != deficit:
                raise FundingError("Confirmed funding transaction value differs from the planned deficit")
            record.update(
                {
                    "status": "confirmed",
                    "confirmed_at_utc": _utc_now(),
                    "confirmation_seconds": round(time.perf_counter() - started, 3),
                    "block_number": int(receipt["blockNumber"]),
                    "gas_used": int(receipt["gasUsed"]),
                    "effective_gas_price_wei": int(receipt["effectiveGasPrice"]),
                    "fee_wei": int(receipt["gasUsed"]) * int(receipt["effectiveGasPrice"]),
                }
            )
            _write_json(path, plan)
        except Exception as exc:
            if plan["transactions"]:
                plan["transactions"][-1]["status"] = "unresolved"
                plan["transactions"][-1]["error_type"] = type(exc).__name__
            plan["status"] = "unresolved"
            plan["execution_stopped_at_utc"] = _utc_now()
            _write_json(path, plan)
            raise FundingError(
                "Funding stopped with an unresolved transaction; inspect the redacted funding record before retrying"
            ) from None

    refreshed_accounts = []
    for account in plan["accounts"]:
        final_balance = int(w3.eth.get_balance(account["address"], "latest"))
        refreshed_accounts.append(
            {
                "key_name": account["key_name"],
                "address": account["address"],
                "final_balance_wei": final_balance,
                "final_balance_eth": _eth(final_balance),
                "target_wei": account["target_wei"],
                "meets_target": final_balance >= int(account["target_wei"]),
            }
        )
    plan["final_accounts"] = refreshed_accounts
    plan["status"] = "complete" if all(item["meets_target"] for item in refreshed_accounts) else "incomplete"
    plan["execution_completed_at_utc"] = _utc_now()
    _write_json(path, plan)
    if plan["status"] != "complete":
        raise FundingError("At least one synthetic EOA remains below its funding target")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send the planned Sepolia transfers")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        rpc_url, keys, priority_fee_wei = _load_configuration()
        w3 = _connect(rpc_url)
        plan = _build_plan(w3, keys, priority_fee_wei)
        path = _report_path(args.execute)
        _write_json(path, plan)

        regulator = next(item for item in plan["accounts"] if item["key_name"] == "REGULATOR_PRIVATE_KEY")
        transfer_count = sum(
            int(item["deficit_wei"]) > 0
            for item in plan["accounts"]
            if item["key_name"] != "REGULATOR_PRIVATE_KEY"
        )
        print(f"Sepolia chain ID verified: {plan['chain_id']}")
        print(f"Regulator balance: {regulator['balance_eth']} ETH")
        print(f"Required before funding: {plan['regulator_required_before_funding_eth']} ETH")
        print(f"Accounts requiring transfers: {transfer_count}")
        print(f"Redacted funding record: {path}")
        if not plan["regulator_balance_sufficient"]:
            raise FundingError("Regulator balance is insufficient; no transaction was sent")
        if args.execute:
            _execute_plan(w3, keys, plan, path)
            print("All synthetic Sepolia EOAs meet their role-based funding targets.")
        else:
            print("Plan only; no transaction was sent.")
    except FundingError as exc:
        parser.exit(1, f"Funding setup failed: {exc}\n")


if __name__ == "__main__":
    main()
