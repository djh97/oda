from dataclasses import dataclass
from typing import Optional, Tuple, Any

from web3 import Web3

class ChainWriteError(RuntimeError):
    pass

@dataclass
class TxResult:
    tx_hash: str
    gas_used: int
    match_id: Optional[int]

def _pk_to_addr(w3: Web3, private_key: str) -> str:
    return w3.eth.account.from_key(private_key).address

def send_create_match(ctx: Any, llm_private_key: str, donor_id: int, primary_id: int, backup_id: int, match_cid: str) -> TxResult:
    """
    Sends createMatch(donorId, primaryRecipientId, backupRecipientId, matchCID) from LLM EOA.
    Returns tx hash, gas used, and match id (best-effort).
    """
    w3: Web3 = ctx.w3
    contract = ctx.contract

    if not llm_private_key:
        raise ChainWriteError("LLM_PRIVATE_KEY missing")
    if not match_cid or match_cid == "NA":
        raise ChainWriteError("matchCID missing/invalid")

    sender = _pk_to_addr(w3, llm_private_key)

    # Ensure sender is authorized LLM on-chain
    try:
        ok = contract.functions.authorizedLLMs(sender).call()
        if not ok:
            raise ChainWriteError(f"LLM address is not authorized on-chain: {sender}")
    except Exception:
        # if contract doesn't expose authorizedLLMs, skip check
        pass

    # Snapshot matchCounter before (helps derive match id if event parsing fails)
    before = None
    try:
        before = int(contract.functions.matchCounter().call())
    except Exception:
        before = None

    nonce = w3.eth.get_transaction_count(sender)

    fn = contract.functions.createMatch(int(donor_id), int(primary_id), int(backup_id), str(match_cid))
    try:
        gas_est = fn.estimate_gas({"from": sender})
    except Exception:
        gas_est = 350000

    # EIP-1559 values (safe defaults)
    try:
        latest = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", None)
    except Exception:
        base_fee = None

    max_priority = w3.to_wei(2, "gwei")
    if base_fee is not None:
        max_fee = int(base_fee) + int(max_priority) * 2
    else:
        max_fee = w3.to_wei(30, "gwei")

    tx = fn.build_transaction({
        "from": sender,
        "nonce": nonce,
        "gas": int(gas_est) + 30000,
        "maxFeePerGas": int(max_fee),
        "maxPriorityFeePerGas": int(max_priority),
        "chainId": w3.eth.chain_id,
    })

    signed = w3.eth.account.sign_transaction(tx, private_key=llm_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    txh = tx_hash.hex()
    gas_used = int(receipt.gasUsed)

    match_id = None

    # Try parse MatchCreated event if it exists
    try:
        ev = contract.events.MatchCreated().process_receipt(receipt)
        if ev and len(ev) > 0:
            # take first event
            args = ev[0]["args"]
            match_id = int(args.get("matchId"))
    except Exception:
        match_id = None

    # Fallback: matchCounter after
    if match_id is None:
        try:
            after = int(contract.functions.matchCounter().call())
            if before is not None and after == before + 1:
                match_id = after
            else:
                match_id = after
        except Exception:
            match_id = None

    return TxResult(tx_hash=txh, gas_used=gas_used, match_id=match_id)