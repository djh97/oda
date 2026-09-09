"""Shared, fail-closed helpers for signed EVM transactions."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from web3 import Web3


class TransactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransactionReceipt:
    tx_hash: str
    sender: str
    status: int
    gas_used: int
    effective_gas_price_wei: int
    block_number: int
    submitted_at_utc: str
    confirmed_at_utc: str
    confirmation_seconds: float
    contract_address: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def private_key_address(w3: Web3, private_key: str, *, role: str = "signer") -> str:
    value = str(private_key or "").strip()
    if not value:
        raise TransactionError(f"Private key is missing for {role}")
    try:
        return w3.eth.account.from_key(value).address
    except Exception as exc:
        raise TransactionError(f"Invalid private key for {role}") from exc


def _receipt_field(receipt: Any, name: str) -> Any:
    if isinstance(receipt, Mapping):
        return receipt.get(name)
    return getattr(receipt, name, None)


def _transaction_hash(value: Any, *, label: str) -> str:
    try:
        normalized = Web3.to_hex(value).lower()
    except Exception as exc:
        raise TransactionError(f"{label} is malformed") from exc
    if len(normalized) != 66 or not normalized.startswith("0x"):
        raise TransactionError(f"{label} is not a 32-byte hash")
    try:
        int(normalized[2:], 16)
    except ValueError as exc:
        raise TransactionError(f"{label} is not hexadecimal") from exc
    return normalized


def _safe_error_details(exc: BaseException) -> dict[str, Any]:
    details: dict[str, Any] = {"error_type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(status_code, int):
        details["http_status"] = status_code
    if request_id:
        details["provider_request_id"] = str(request_id)
    return details


def _validate_success_receipt(
    receipt: Any,
    *,
    tx_hash: str,
    sender: str,
    gas_limit: int,
    fee_cap: int,
    role: str,
) -> tuple[int, int, int, str | None]:
    receipt_hash = _transaction_hash(
        _receipt_field(receipt, "transactionHash"),
        label="Receipt transaction hash",
    )
    if receipt_hash != tx_hash:
        raise TransactionError(f"Receipt transaction hash differs from submission for {role}")
    receipt_sender = _receipt_field(receipt, "from")
    try:
        sender_matches = Web3.to_checksum_address(receipt_sender) == Web3.to_checksum_address(sender)
    except Exception as exc:
        raise TransactionError(f"Receipt sender is malformed for {role}") from exc
    if not sender_matches:
        raise TransactionError(f"Receipt sender differs from signer for {role}")
    try:
        gas_used = int(_receipt_field(receipt, "gasUsed"))
        effective_gas_price = int(_receipt_field(receipt, "effectiveGasPrice"))
        block_number = int(_receipt_field(receipt, "blockNumber"))
    except (TypeError, ValueError) as exc:
        raise TransactionError(f"Receipt measurements are malformed for {role}") from exc
    if gas_used <= 0 or gas_used > gas_limit:
        raise TransactionError(f"Receipt gas use is outside the submitted limit for {role}")
    if effective_gas_price <= 0:
        raise TransactionError(f"Receipt omitted the effective gas price for {role}")
    if effective_gas_price > fee_cap:
        raise TransactionError(f"Receipt effective gas price exceeds the submitted cap for {role}")
    if block_number < 0:
        raise TransactionError(f"Receipt block number is invalid for {role}")
    contract_address_value = _receipt_field(receipt, "contractAddress")
    try:
        contract_address = (
            Web3.to_checksum_address(contract_address_value)
            if contract_address_value
            else None
        )
    except Exception as exc:
        raise TransactionError(f"Receipt contract address is malformed for {role}") from exc
    if contract_address == Web3.to_checksum_address("0x" + "0" * 40):
        raise TransactionError(f"Receipt contract address is the zero address for {role}")
    return gas_used, effective_gas_price, block_number, contract_address


def send_signed_transaction(
    w3: Web3,
    function: Any,
    private_key: str,
    *,
    role: str,
    timeout_seconds: float = 600.0,
    poll_latency_seconds: float = 2.0,
    priority_fee_gwei: float = 2.0,
    gas_multiplier: float = 1.20,
    gas_padding: int = 10_000,
    event_recorder: Callable[[str, Mapping[str, Any]], Any] | None = None,
    event_context: Mapping[str, Any] | None = None,
) -> TransactionReceipt:
    numeric_parameters = {
        "timeout_seconds": timeout_seconds,
        "poll_latency_seconds": poll_latency_seconds,
        "priority_fee_gwei": priority_fee_gwei,
        "gas_multiplier": gas_multiplier,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_parameters.values()
    ):
        raise TransactionError("Transaction timing and fee parameters must be finite numbers")
    if timeout_seconds <= 0 or poll_latency_seconds <= 0:
        raise TransactionError("Transaction timeout and polling interval must be positive")
    if priority_fee_gwei < 0 or gas_multiplier < 1:
        raise TransactionError("Priority fee cannot be negative and gas multiplier cannot be below one")
    if not isinstance(gas_padding, int) or isinstance(gas_padding, bool) or gas_padding < 0:
        raise TransactionError("Gas padding must be a nonnegative integer")

    sender = private_key_address(w3, private_key, role=role)
    try:
        estimated_gas = int(function.estimate_gas({"from": sender}))
    except Exception as exc:
        raise TransactionError(
            f"Gas estimation failed for {role} ({type(exc).__name__})"
        ) from exc
    if estimated_gas <= 0:
        raise TransactionError(f"Gas estimation returned a nonpositive value for {role}")

    try:
        nonce = int(w3.eth.get_transaction_count(sender, "pending"))
        chain_id = int(w3.eth.chain_id)
        latest = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas")
    except Exception as exc:
        raise TransactionError(
            f"Transaction metadata lookup failed for {role} ({type(exc).__name__})"
        ) from exc
    if nonce < 0 or chain_id <= 0:
        raise TransactionError(f"Transaction metadata is invalid for {role}")

    gas_limit = math.ceil(estimated_gas * float(gas_multiplier)) + gas_padding
    tx_fields: dict[str, Any] = {
        "from": sender,
        "nonce": nonce,
        "gas": gas_limit,
        "chainId": chain_id,
    }
    if base_fee is None:
        try:
            gas_price = int(w3.eth.gas_price)
        except Exception as exc:
            raise TransactionError(
                f"Legacy gas-price lookup failed for {role} ({type(exc).__name__})"
            ) from exc
        if gas_price <= 0:
            raise TransactionError(f"Legacy gas price is nonpositive for {role}")
        tx_fields["gasPrice"] = gas_price
    else:
        try:
            base_fee_wei = int(base_fee)
            priority_fee = int(w3.to_wei(priority_fee_gwei, "gwei"))
        except Exception as exc:
            raise TransactionError(f"EIP-1559 fee data is invalid for {role}") from exc
        if base_fee_wei < 0 or priority_fee < 0:
            raise TransactionError(f"EIP-1559 fee data is negative for {role}")
        tx_fields["maxPriorityFeePerGas"] = priority_fee
        tx_fields["maxFeePerGas"] = base_fee_wei * 2 + priority_fee

    context = dict(event_context or {})
    if event_recorder is not None and not str(context.get("transaction_id", "")).strip():
        raise TransactionError("Journaled transactions require a transaction_id")
    try:
        transaction = function.build_transaction(tx_fields)
        if not isinstance(transaction, Mapping):
            raise TransactionError("Built transaction is not a mapping")
        numeric_fields = tuple(field for field in tx_fields if field != "from")
        if any(int(transaction.get(field, -1)) != int(tx_fields[field]) for field in numeric_fields):
            raise TransactionError("Built transaction changed a critical numeric field")
        if Web3.to_checksum_address(transaction.get("from")) != Web3.to_checksum_address(sender):
            raise TransactionError("Built transaction changed the sender")
    except TransactionError:
        raise
    except Exception as exc:
        raise TransactionError(f"Transaction build failed for {role} ({type(exc).__name__})") from exc

    prepared_details = {
        **context,
        "sender": Web3.to_checksum_address(sender),
        "chain_id": chain_id,
        "nonce": nonce,
        "estimated_gas": estimated_gas,
        "gas_limit": gas_limit,
        "fee_fields": {
            key: int(tx_fields[key])
            for key in ("gasPrice", "maxPriorityFeePerGas", "maxFeePerGas")
            if key in tx_fields
        },
    }
    if event_recorder is not None:
        event_recorder("transaction_prepared", prepared_details)

    try:
        signed = w3.eth.account.sign_transaction(transaction, private_key=private_key)
        raw_transaction = getattr(signed, "raw_transaction", None)
        if raw_transaction is None:
            raw_transaction = signed.rawTransaction
        raw_bytes = bytes(raw_transaction)
        signed_hash_value = getattr(signed, "hash", None)
        if signed_hash_value is None:
            signed_hash_value = Web3.keccak(raw_bytes)
        signed_hash = _transaction_hash(signed_hash_value, label="Signed transaction hash")
    except TransactionError:
        raise
    except Exception as exc:
        raise TransactionError(f"Transaction signing failed for {role} ({type(exc).__name__})") from exc

    signed_details = {
        **context,
        "sender": Web3.to_checksum_address(sender),
        "nonce": nonce,
        "signed_transaction_hash": signed_hash,
        "raw_signed_transaction_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    if event_recorder is not None:
        event_recorder("transaction_signed", signed_details)

    submitted_at = utc_now()
    started = time.perf_counter()
    try:
        tx_hash_value = w3.eth.send_raw_transaction(raw_transaction)
        tx_hash = _transaction_hash(tx_hash_value, label="Submitted transaction hash")
    except Exception as exc:
        if event_recorder is not None:
            event_recorder(
                "transaction_unresolved",
                {**signed_details, "phase": "submission", **_safe_error_details(exc)},
            )
        raise TransactionError(
            f"Transaction submission is unresolved for {role} ({type(exc).__name__})"
        ) from exc
    if signed_hash != tx_hash:
        if event_recorder is not None:
            event_recorder(
                "transaction_unresolved",
                {
                    **signed_details,
                    "phase": "submission_hash_mismatch",
                    "rpc_returned_hash": tx_hash,
                },
            )
        raise TransactionError("The node returned a hash that differs from the signed transaction")

    submitted_details = {
        **context,
        "sender": Web3.to_checksum_address(sender),
        "nonce": nonce,
        "signed_transaction_hash": signed_hash,
        "rpc_returned_hash": tx_hash,
        "submitted_at_utc": submitted_at,
    }
    if event_recorder is not None:
        event_recorder("transaction_submitted", submitted_details)
    try:
        receipt = w3.eth.wait_for_transaction_receipt(
            tx_hash_value,
            timeout=timeout_seconds,
            poll_latency=poll_latency_seconds,
        )
        elapsed = time.perf_counter() - started
    except Exception as exc:
        if event_recorder is not None:
            event_recorder(
                "transaction_unresolved",
                {**submitted_details, "phase": "receipt_polling", **_safe_error_details(exc)},
            )
        raise TransactionError(
            f"Transaction receipt is unresolved for {role} ({type(exc).__name__})"
        ) from exc

    try:
        status = int(_receipt_field(receipt, "status"))
    except (TypeError, ValueError) as exc:
        if event_recorder is not None:
            event_recorder(
                "transaction_unresolved",
                {**submitted_details, "phase": "receipt_validation", **_safe_error_details(exc)},
            )
        raise TransactionError(f"Receipt status is malformed for {role}") from exc
    if status != 1:
        if event_recorder is not None:
            event_recorder(
                "transaction_reverted",
                {
                    **submitted_details,
                    "status": status,
                    "block_number": _receipt_field(receipt, "blockNumber"),
                    "gas_used": _receipt_field(receipt, "gasUsed"),
                    "effective_gas_price_wei": _receipt_field(receipt, "effectiveGasPrice"),
                },
            )
        raise TransactionError(f"Transaction reverted for {role}: {tx_hash}")
    fee_cap = int(tx_fields.get("maxFeePerGas", tx_fields.get("gasPrice", 0)))
    try:
        gas_used, effective_gas_price, block_number, contract_address = _validate_success_receipt(
            receipt,
            tx_hash=tx_hash,
            sender=sender,
            gas_limit=gas_limit,
            fee_cap=fee_cap,
            role=role,
        )
    except TransactionError as exc:
        if event_recorder is not None:
            event_recorder(
                "transaction_unresolved",
                {**submitted_details, "phase": "receipt_validation", **_safe_error_details(exc)},
            )
        raise
    confirmed_at = utc_now()
    if event_recorder is not None:
        event_recorder(
            "transaction_confirmed",
            {
                **submitted_details,
                "status": status,
                "block_number": block_number,
                "gas_used": gas_used,
                "effective_gas_price_wei": effective_gas_price,
                "contract_address": contract_address,
                "confirmed_at_utc": confirmed_at,
                "confirmation_seconds": round(elapsed, 3),
            },
        )
    return TransactionReceipt(
        tx_hash=tx_hash,
        sender=Web3.to_checksum_address(sender),
        status=status,
        gas_used=gas_used,
        effective_gas_price_wei=effective_gas_price,
        block_number=block_number,
        submitted_at_utc=submitted_at,
        confirmed_at_utc=confirmed_at,
        confirmation_seconds=round(elapsed, 3),
        contract_address=contract_address,
    )
