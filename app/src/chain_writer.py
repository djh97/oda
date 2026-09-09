from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .ipfs_client import IPFSError, validate_cid
from .transactions import TransactionError, private_key_address, send_signed_transaction


class ChainWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class TxResult:
    tx_hash: str
    gas_used: int
    match_id: Optional[int]


def send_create_match(
    ctx: Any,
    decision_service_private_key: str,
    donor_id: int,
    primary_id: int,
    backup_id: int,
    decision_cid: str,
) -> TxResult:
    w3 = ctx.w3
    contract = ctx.contract
    if not decision_service_private_key:
        raise ChainWriteError("DECISION_SERVICE_PRIVATE_KEY is missing")
    identifiers = {
        "donor ID": donor_id,
        "primary recipient ID": primary_id,
        "backup recipient ID": backup_id,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in identifiers.values()
    ):
        raise ChainWriteError("Donor and recipient IDs must be positive integers")
    if primary_id == backup_id:
        raise ChainWriteError("Primary and backup recipient IDs must be different")
    try:
        decision_cid = validate_cid(decision_cid)
    except IPFSError as exc:
        raise ChainWriteError("The encrypted decision CID is invalid") from exc

    try:
        sender = private_key_address(w3, decision_service_private_key, role="decision service")
    except TransactionError as exc:
        raise ChainWriteError(str(exc)) from exc
    try:
        authorized = contract.functions.authorizedDecisionServices(sender).call()
    except Exception as exc:
        raise ChainWriteError("Unable to verify the decision-service role on-chain") from exc
    if not isinstance(authorized, bool):
        raise ChainWriteError("Decision-service authorization returned a malformed value")
    if not authorized:
        raise ChainWriteError(f"Decision-service address is not authorized on-chain: {sender}")

    function = contract.functions.createMatch(
        donor_id,
        primary_id,
        backup_id,
        str(decision_cid),
    )
    try:
        receipt = send_signed_transaction(
            w3,
            function,
            decision_service_private_key,
            role="decision service createMatch",
        )
    except TransactionError as exc:
        raise ChainWriteError(str(exc)) from exc

    try:
        chain_receipt = w3.eth.get_transaction_receipt(receipt.tx_hash)
        events = contract.events.MatchCreated().process_receipt(chain_receipt)
    except Exception as exc:
        raise ChainWriteError("Unable to decode the MatchCreated transaction event") from exc
    if len(events) != 1:
        raise ChainWriteError("createMatch did not emit exactly one MatchCreated event")
    args = events[0]["args"]
    try:
        match_id = int(args["matchId"])
        event_matches = (
            match_id > 0
            and int(args["donorId"]) == donor_id
            and int(args["primaryRecipientId"]) == primary_id
            and int(args["backupRecipientId"]) == backup_id
            and str(args["decisionCID"]) == str(decision_cid)
            and str(args["recordedBy"]).lower() == str(sender).lower()
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainWriteError("MatchCreated event fields are malformed") from exc
    if not event_matches:
        raise ChainWriteError("MatchCreated event does not match the submitted decision")
    return TxResult(tx_hash=receipt.tx_hash, gas_used=receipt.gas_used, match_id=match_id)
