from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List

from web3 import Web3

from .web3_client import Web3Context


class ChainDataError(RuntimeError):
    pass


MAX_RECIPIENT_RECORDS = 10_000


def _call(function: Any, *, label: str) -> Any:
    try:
        return function.call()
    except Exception as exc:
        raise ChainDataError(f"Unable to read {label} from the configured chain") from exc


def _record(value: Any, *, fields: int, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ChainDataError(f"{label} has an invalid contract response type")
    if len(value) != fields:
        raise ChainDataError(f"{label} has {len(value)} fields; expected {fields}")
    return value


def _flag(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ChainDataError(f"{label} is not a Boolean contract value")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChainDataError(f"{label} is not a valid integer")
    return value


def _address(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not Web3.is_address(value):
        raise ChainDataError(f"{label} is not a valid EVM address")
    return value


def get_donor(ctx: Web3Context, donor_id: int) -> Dict[str, Any]:
    if isinstance(donor_id, bool) or not isinstance(donor_id, int) or donor_id < 1:
        raise ChainDataError("Donor ID must be a positive integer")
    value = _record(
        _call(ctx.contract.functions.donors(donor_id), label=f"donor {donor_id}"),
        fields=6,
        label=f"Donor {donor_id}",
    )
    returned_id = _integer(value[0], label="Donor ID")
    if returned_id not in {0, donor_id}:
        raise ChainDataError("Donor response does not match the requested ID")
    if not isinstance(value[2], str):
        raise ChainDataError("Donor profile CID is not a string")
    return {
        "donorId": returned_id,
        "donorAuthority": _address(value[1], label="Donor authority"),
        "profileCID": value[2],
        "registered": _flag(value[3], label="Donor registered flag"),
        "ethicallyEligible": _flag(value[4], label="Donor eligibility flag"),
        "finalized": _flag(value[5], label="Donor finalized flag"),
        "hasOpenMatch": _flag(
            _call(
                ctx.contract.functions.donorHasOpenMatch(donor_id),
                label=f"donor {donor_id} open-match flag",
            ),
            label="Donor open-match flag",
        ),
    }


def get_all_recipients(ctx: Web3Context) -> List[Dict[str, Any]]:
    raw_count = _call(ctx.contract.functions.recipientCounter(), label="recipient counter")
    count = _integer(raw_count, label="Recipient counter")
    if count > MAX_RECIPIENT_RECORDS:
        raise ChainDataError("Recipient counter is outside the supported range")
    recipients: List[Dict[str, Any]] = []
    for recipient_id in range(1, count + 1):
        value = _record(
            _call(
                ctx.contract.functions.recipients(recipient_id),
                label=f"recipient {recipient_id}",
            ),
            fields=7,
            label=f"Recipient {recipient_id}",
        )
        returned_id = _integer(value[0], label="Recipient ID")
        if returned_id != recipient_id:
            raise ChainDataError("Recipient response does not match the requested ID")
        if not isinstance(value[2], str):
            raise ChainDataError("Recipient profile CID is not a string")
        recipient = {
            "recipientId": returned_id,
            "recipientAddress": _address(value[1], label="Recipient address"),
            "profileCID": value[2],
            "registered": _flag(value[3], label="Recipient registered flag"),
            "ethicallyEligible": _flag(value[4], label="Recipient eligibility flag"),
            "reserved": _flag(value[5], label="Recipient reserved flag"),
            "transplanted": _flag(value[6], label="Recipient transplanted flag"),
        }
        recipients.append(recipient)
    return recipients


def require_eligible_donor(donor: Dict[str, Any]) -> None:
    if not donor["registered"]:
        raise ChainDataError("Donor profile is not registered on-chain")
    if not donor["ethicallyEligible"]:
        raise ChainDataError("Donor is not eligible under the workflow gate")
    if donor["finalized"]:
        raise ChainDataError("Donor workflow is already finalized")
    if donor.get("hasOpenMatch"):
        raise ChainDataError("Donor already has an open match")
    if not donor["profileCID"]:
        raise ChainDataError("Donor has no encrypted profile CID")


def filter_eligible_recipients(recipients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        recipient
        for recipient in recipients
        if recipient["registered"]
        and recipient["ethicallyEligible"]
        and recipient["profileCID"]
        and not recipient["reserved"]
        and not recipient["transplanted"]
    ]
