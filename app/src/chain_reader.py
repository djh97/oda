from typing import Any, Dict, List, Tuple

from .web3_client import Web3Context

class ChainDataError(RuntimeError):
    pass

def get_donor(ctx: Web3Context, donor_id: int) -> Dict[str, Any]:
    d = ctx.contract.functions.donors(donor_id).call()
    # Donor struct:
    # (donorId, donorAddress, bloodType, hlaTyping, organType, ipfsHash, registered, ethicalApproved)
    return {
        "donorId": int(d[0]),
        "donorAddress": d[1],
        "bloodType": d[2],
        "hlaTyping": d[3],
        "organType": d[4],
        "ipfsHash": d[5],
        "registered": bool(d[6]),
        "ethicalApproved": bool(d[7]),
    }

def get_all_recipients(ctx: Web3Context) -> List[Dict[str, Any]]:
    count = int(ctx.contract.functions.recipientCounter().call())
    out: List[Dict[str, Any]] = []
    for rid in range(1, count + 1):
        r = ctx.contract.functions.recipients(rid).call()
        # Recipient struct:
        # (recipientId, recipientAddress, bloodType, hlaTyping, organType, ipfsHash, registered, matched, ethicalApproved)
        rec = {
            "recipientId": int(r[0]),
            "recipientAddress": r[1],
            "bloodType": r[2],
            "hlaTyping": r[3],
            "organType": r[4],
            "ipfsHash": r[5],
            "registered": bool(r[6]),
            "matched": bool(r[7]),
            "ethicalApproved": bool(r[8]),
        }
        # Skip empty/uninitialized slots (defensive)
        if rec["recipientId"] == 0:
            continue
        out.append(rec)
    return out

def require_eligible_donor(d: Dict[str, Any]) -> None:
    if not d["registered"]:
        raise ChainDataError("Donor is not registered on-chain.")
    if not d["ethicalApproved"]:
        raise ChainDataError("Donor is not ethically approved yet.")
    if not d["ipfsHash"]:
        raise ChainDataError("Donor has no IPFS CID on-chain.")

def filter_eligible_recipients(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eligible = []
    for r in recs:
        if r["registered"] and r["ethicalApproved"] and r["ipfsHash"] and not r["matched"]:
            eligible.append(r)
    return eligible
