import os
import csv
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from web3 import Web3

from src.config import get_settings
from src.web3_client import load_abi, load_contract_address
from src.tx_logger import append_tx

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output" / "approvals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address


def send_tx_with_timing(w3: Web3, fn, sender_addr: str, private_key: str, gas: int) -> Tuple[str, Any, float]:
    """
    Builds, signs, sends, waits receipt. Returns (tx_hash_hex, receipt, receipt_ms).
    """
    tx = fn.build_transaction({
        "from": sender_addr,
        "gas": gas,
        "nonce": w3.eth.get_transaction_count(sender_addr),
        "chainId": w3.eth.chain_id,
        "maxFeePerGas": w3.to_wei(30, "gwei"),
        "maxPriorityFeePerGas": w3.to_wei(2, "gwei"),
    })

    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)

    t0 = time.perf_counter()
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    t1 = time.perf_counter()

    return tx_hash.hex(), receipt, (t1 - t0) * 1000.0


def log_tx(network: str, role: str, fn_name: str, tx_hash: str, receipt, notes: str = ""):
    gas_used = None
    try:
        gas_used = int(receipt.gasUsed) if receipt is not None else None
    except Exception:
        gas_used = None

    append_tx(
        network=network,
        role=role,
        function=fn_name,
        tx_hash=tx_hash,
        gas_used=gas_used,
        notes=notes
    )


def write_run_csv(path: Path, rows: list):
    fieldnames = [
        "timestamp_utc",
        "match_id",
        "step",
        "role",
        "function",
        "tx_hash",
        "gas_used",
        "receipt_ms",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", type=int, required=True)
    args = parser.parse_args()

    load_dotenv(APP_DIR / ".env")

    settings = get_settings()
    if not settings.sepolia_rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL missing in .env")

    w3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    assert w3.is_connected(), "Not connected to Sepolia"

    abi = load_abi(settings.abi_path)
    addr = load_contract_address(settings.address_path)
    contract = w3.eth.contract(address=addr, abi=abi)

    # Keys
    regulator_pk = os.getenv("REGULATOR_PRIVATE_KEY", "").strip()
    hospital_pk = os.getenv("HOSPITAL_PRIVATE_KEY", "").strip()
    ethics_pk = os.getenv("ETHICS_PRIVATE_KEY", "").strip()
    medical_pk = os.getenv("MEDICAL_PRIVATE_KEY", "").strip()
    donor_pk = os.getenv("DONOR_PRIVATE_KEY", "").strip()

    if not regulator_pk or not hospital_pk or not ethics_pk or not medical_pk or not donor_pk:
        raise RuntimeError("Missing one or more required keys: REGULATOR/HOSPITAL/ETHICS/MEDICAL/DONOR private keys")

    regulator_addr = pk_to_addr(w3, regulator_pk)
    hospital_addr = pk_to_addr(w3, hospital_pk)
    ethics_addr = pk_to_addr(w3, ethics_pk)
    medical_addr = pk_to_addr(w3, medical_pk)
    donor_addr = pk_to_addr(w3, donor_pk)

    match_id = int(args.match_id)

    # Read match to get active recipient id
    m = contract.functions.matches(match_id).call()
    # Expected layout (from our V21):
    # [matchId, donorId, primaryRecipientId, backupRecipientId, activeRecipientId, backupPromoted, matchedByLLM, matchCID,
    #  medicalApproved, hospitalApproved, donorApproved, activeRecipientApproved, ethicalCommitteeApproved, finalized]
    # So: activeRecipientId index = 4
    active_recipient_id = int(m[4])

    # For recipient approval we need that recipient’s private key
    rpk = os.getenv(f"RECIPIENT{active_recipient_id}_PRIVATE_KEY", "").strip()
    if not rpk:
        raise RuntimeError(f"Missing RECIPIENT{active_recipient_id}_PRIVATE_KEY in .env (needed for approveRecipient)")
    recipient_addr = pk_to_addr(w3, rpk)

    print("Connected:", w3.is_connected())
    print("Network:", settings.network)
    print("Contract:", addr)
    print("matchId:", match_id)
    print("activeRecipientId:", active_recipient_id)
    print("Regulator:", regulator_addr)
    print("Hospital:", hospital_addr)
    print("MedicalTeam:", medical_addr)
    print("Ethics:", ethics_addr)
    print("Donor:", donor_addr)
    print(f"Recipient{active_recipient_id}:", recipient_addr)

    rows = []
    ts = now_utc()

    # Helper to skip already-approved flags
    def refresh_match():
        return contract.functions.matches(match_id).call()

    # 1) approveMedicalTeam
    m = refresh_match()
    medicalApproved = bool(m[8])
    if not medicalApproved:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.approveMedicalTeam(match_id),
            medical_addr,
            medical_pk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id}"
        log_tx(settings.network, "MedicalTeam", "approveMedicalTeam", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 1,
            "role": "MedicalTeam",
            "function": "approveMedicalTeam",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("approveMedicalTeam tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("approveMedicalTeam: already approved")

    # 2) approveHospital
    m = refresh_match()
    hospitalApproved = bool(m[9])
    if not hospitalApproved:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.approveHospital(match_id),
            hospital_addr,
            hospital_pk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id}"
        log_tx(settings.network, "Hospital", "approveHospital", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 2,
            "role": "Hospital",
            "function": "approveHospital",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("approveHospital tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("approveHospital: already approved")

    # 3) approveDonor
    m = refresh_match()
    donorApproved = bool(m[10])
    if not donorApproved:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.approveDonor(match_id),
            donor_addr,
            donor_pk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id}"
        log_tx(settings.network, "Donor", "approveDonor", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 3,
            "role": "Donor",
            "function": "approveDonor",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("approveDonor tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("approveDonor: already approved")

    # 4) approveRecipient (active recipient)
    m = refresh_match()
    activeRecipientApproved = bool(m[11])
    if not activeRecipientApproved:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.approveRecipient(match_id),
            recipient_addr,
            rpk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id},recipient={active_recipient_id}"
        log_tx(settings.network, f"Recipient{active_recipient_id}", "approveRecipient", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 4,
            "role": f"Recipient{active_recipient_id}",
            "function": "approveRecipient",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("approveRecipient tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("approveRecipient: already approved")

    # 5) approveFinalTransplant (ethics committee)
    m = refresh_match()
    ethicalApproved = bool(m[12])
    if not ethicalApproved:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.approveFinalTransplant(match_id),
            ethics_addr,
            ethics_pk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id}"
        log_tx(settings.network, "EthicsCommittee", "approveFinalTransplant", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 5,
            "role": "EthicsCommittee",
            "function": "approveFinalTransplant",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("approveFinalTransplant tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("approveFinalTransplant: already approved")

    # Check approval status
    approved = contract.functions.isTransplantApproved(match_id).call()
    print("\nisTransplantApproved:", approved)

    # 6) finalizeMatch (we'll use Regulator to be consistent)
    m = refresh_match()
    finalized = bool(m[13])
    if not finalized:
        txh, rcpt, ms = send_tx_with_timing(
            w3,
            contract.functions.finalizeMatch(match_id),
            regulator_addr,
            regulator_pk,
            gas=250000
        )
        gas_used = int(rcpt.gasUsed)
        notes = f"match={match_id},approved={approved}"
        log_tx(settings.network, "Regulator", "finalizeMatch", txh, rcpt, notes=notes)
        rows.append({
            "timestamp_utc": ts,
            "match_id": match_id,
            "step": 6,
            "role": "Regulator",
            "function": "finalizeMatch",
            "tx_hash": txh,
            "gas_used": gas_used,
            "receipt_ms": round(ms, 2),
            "notes": notes,
        })
        print("finalizeMatch tx:", txh, "| gas:", gas_used, "| ms:", round(ms, 2))
    else:
        print("finalizeMatch: already finalized")

    # Write run file
    out_csv = OUT_DIR / f"approval_chain_match{match_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    write_run_csv(out_csv, rows)
    print("\nSaved approval chain log:")
    print(" ", out_csv)


if __name__ == "__main__":
    main()
