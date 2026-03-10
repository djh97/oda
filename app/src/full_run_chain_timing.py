# app/src/full_run_chain_timing.py
#
# Full end-to-end on-chain workflow timing runner:
# - Deploys a fresh contract
# - Runs governance setup, identity binding, profile registration, eligibility approvals
# - Records a match (createMatch), runs approvals, finalizes
# - Measures confirmation time for every tx (submit -> mined receipt returned)
# - Writes an audit CSV with tx hashes and categories
#
# Fixes included:
# - ABI+bytecode loaded from the same Foundry artifact (prevents ABI mismatch)
# - Nonce handling uses pending nonce + local nonce cache (prevents nonce-too-low)
# - Receipt waiting uses configurable timeout/poll latency + graceful timeout logging
#
# Run:
#   python -m src.full_run_chain_timing
#
# Required .env (app/.env):
#   SEPOLIA_RPC_URL=...
#   FOUNDRY_ARTIFACT_PATH=... (recommended)
#   REGULATOR_PRIVATE_KEY=...
#   HOSPITAL_PRIVATE_KEY=...
#   ETHICS_PRIVATE_KEY=...
#   LLM_PRIVATE_KEY=...
#   MEDICAL_PRIVATE_KEY=...
#   DONOR_PRIVATE_KEY=...
#   RECIPIENT1_PRIVATE_KEY=... ... RECIPIENT10_PRIVATE_KEY=...
#   SEED_DONOR_CID=...
#   SEED_RECIPIENT1_CID=... ... SEED_RECIPIENT10_CID=...
#
# Optional .env:
#   TX_MAX_FEE_GWEI=60
#   TX_PRIORITY_FEE_GWEI=3
#   TX_RECEIPT_TIMEOUT_S=600
#   TX_POLL_LATENCY_S=2
#   MATCH_RATIONALE_CID=cid-placeholder

import os
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import TimeExhausted


# ----------------------------
# Paths
# ----------------------------
APP_DIR = Path(__file__).resolve().parents[1]          # .../app
REPO_ROOT = APP_DIR.parents[0]                         # repo root
OUT_DIR = APP_DIR / "pipeline-output" / "timing_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIR = APP_DIR / "seed-data"


# ----------------------------
# Helpers
# ----------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_0x(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x
    return x if x.startswith("0x") else "0x" + x


def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address


def load_seed_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def hla_list_to_string(hla) -> str:
    if isinstance(hla, list):
        return ",".join([str(x).strip() for x in hla if str(x).strip()])
    if isinstance(hla, str):
        return ",".join([x.strip() for x in hla.split(",") if x.strip()])
    return ""


def norm_bt(bt: str) -> str:
    bt = (bt or "").strip().upper()
    if bt.startswith("AB"):
        return "AB"
    if bt.startswith("A"):
        return "A"
    if bt.startswith("B"):
        return "B"
    if bt.startswith("O"):
        return "O"
    return bt


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestamp_utc",
        "category",
        "role",
        "function",
        "tx_hash",
        "gas_used",
        "confirmation_s",
        "status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_foundry_artifact() -> Tuple[List[dict], str]:
    """
    Loads (abi, bytecode) from a Foundry artifact JSON.
    Override with FOUNDRY_ARTIFACT_PATH in app/.env.
    """
    override = os.getenv("FOUNDRY_ARTIFACT_PATH", "").strip()
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))

    # Common Foundry output location
    candidates.append(REPO_ROOT / "smart-contracts" / "out" / "TransplantManagement.sol" / "TransplantManagement.json")

    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                art = json.load(f)

            abi = art.get("abi")
            bytecode_obj = art.get("bytecode", {})
            if isinstance(bytecode_obj, dict):
                bytecode = bytecode_obj.get("object", "")
            else:
                bytecode = bytecode_obj

            if not abi or not bytecode:
                raise RuntimeError(f"Artifact missing abi/bytecode: {p}")

            return abi, normalize_0x(bytecode)

    raise RuntimeError("Could not locate Foundry artifact. Set FOUNDRY_ARTIFACT_PATH in app/.env.")


def fn_exists(contract, fn_name: str) -> bool:
    return hasattr(contract.functions, fn_name)


def build_fn(contract, fn_name: str, args: list):
    if not fn_exists(contract, fn_name):
        raise AttributeError(f"Contract ABI does not contain function: {fn_name}")
    return getattr(contract.functions, fn_name)(*args)


def first_input_type(contract, fn_name: str) -> Optional[str]:
    for item in contract.abi:
        if item.get("type") == "function" and item.get("name") == fn_name:
            inputs = item.get("inputs", [])
            if inputs:
                return inputs[0].get("type")
            return None
    return None


def send_tx_with_confirmation_time(
    w3: Web3,
    fn,
    sender_addr: str,
    private_key: str,
    gas: int,
    max_fee_gwei: float,
    max_priority_gwei: float,
    nonce: int,
    receipt_timeout_s: int,
    poll_latency_s: float,
) -> Tuple[str, Optional[dict], float, str]:
    """
    Returns:
      (tx_hash_hex, receipt_or_none, confirmation_seconds, status)

    status:
      - "mined" if receipt obtained within timeout
      - "timeout_pending" if not mined within timeout
    """
    tx = fn.build_transaction({
        "from": sender_addr,
        "gas": gas,
        "nonce": nonce,
        "chainId": w3.eth.chain_id,
        "maxFeePerGas": w3.to_wei(max_fee_gwei, "gwei"),
        "maxPriorityFeePerGas": w3.to_wei(max_priority_gwei, "gwei"),
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)

    t0 = time.perf_counter()
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    try:
        receipt = w3.eth.wait_for_transaction_receipt(
            tx_hash,
            timeout=receipt_timeout_s,
            poll_latency=poll_latency_s,
        )
        t1 = time.perf_counter()
        return tx_hash.hex(), receipt, (t1 - t0), "mined"
    except TimeExhausted:
        t1 = time.perf_counter()
        # Not mined within timeout; keep tx hash for auditability
        return tx_hash.hex(), None, (t1 - t0), "timeout_pending"


def main():
    load_dotenv(APP_DIR / ".env")

    rpc = (os.getenv("SEPOLIA_RPC_URL") or "").strip()
    if not rpc:
        raise RuntimeError("Missing SEPOLIA_RPC_URL in app/.env")

    # Fee caps for tx submission (not used for paper cost accounting)
    max_fee_gwei = float(os.getenv("TX_MAX_FEE_GWEI", "60"))
    max_priority_gwei = float(os.getenv("TX_PRIORITY_FEE_GWEI", "3"))

    # Receipt waiting behavior
    receipt_timeout_s = int(os.getenv("TX_RECEIPT_TIMEOUT_S", "600"))  # 10 min
    poll_latency_s = float(os.getenv("TX_POLL_LATENCY_S", "2"))

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

    # Keys
    regulator_pk = (os.getenv("REGULATOR_PRIVATE_KEY") or "").strip()
    hospital_pk = (os.getenv("HOSPITAL_PRIVATE_KEY") or "").strip()
    ethics_pk = (os.getenv("ETHICS_PRIVATE_KEY") or "").strip()
    llm_pk = (os.getenv("LLM_PRIVATE_KEY") or "").strip()
    medical_pk = (os.getenv("MEDICAL_PRIVATE_KEY") or "").strip()
    donor_pk = (os.getenv("DONOR_PRIVATE_KEY") or "").strip()

    if not all([regulator_pk, hospital_pk, ethics_pk, llm_pk, medical_pk, donor_pk]):
        raise RuntimeError("Missing one or more required private keys in app/.env")

    recipient_pks: List[str] = []
    for i in range(1, 11):
        pk = (os.getenv(f"RECIPIENT{i}_PRIVATE_KEY") or "").strip()
        if not pk:
            raise RuntimeError(f"Missing RECIPIENT{i}_PRIVATE_KEY in app/.env")
        recipient_pks.append(pk)

    regulator_addr = pk_to_addr(w3, regulator_pk)
    hospital_addr = pk_to_addr(w3, hospital_pk)
    ethics_addr = pk_to_addr(w3, ethics_pk)
    llm_addr = pk_to_addr(w3, llm_pk)
    medical_addr = pk_to_addr(w3, medical_pk)
    donor_addr = pk_to_addr(w3, donor_pk)
    recipient_addrs = [pk_to_addr(w3, pk) for pk in recipient_pks]

    # Nonce cache (pending) to avoid nonce-too-low
    nonce_cache: Dict[str, int] = {}

    def next_nonce(addr: str) -> int:
        if addr not in nonce_cache:
            nonce_cache[addr] = w3.eth.get_transaction_count(addr, "pending")
        n = nonce_cache[addr]
        nonce_cache[addr] += 1
        return n

    # Seed CIDs
    donor_cid = (os.getenv("SEED_DONOR_CID") or "").strip()
    if not donor_cid:
        raise RuntimeError("Missing SEED_DONOR_CID in app/.env")

    recipient_cids: Dict[int, str] = {}
    for i in range(1, 11):
        cid = (os.getenv(f"SEED_RECIPIENT{i}_CID") or "").strip()
        if not cid:
            raise RuntimeError(f"Missing SEED_RECIPIENT{i}_CID in app/.env")
        recipient_cids[i] = cid

    abi, bytecode = load_foundry_artifact()

    rows: List[Dict[str, Any]] = []

    def log_row(category: str, role: str, fn_name: str, txh: str, receipt: Optional[dict], confirm_s: float, status: str, notes: str = ""):
        gas_used = int(receipt["gasUsed"]) if receipt is not None else ""
        rows.append({
            "timestamp_utc": now_utc(),
            "category": category,
            "role": role,
            "function": fn_name,
            "tx_hash": txh,
            "gas_used": gas_used,
            "confirmation_s": round(confirm_s, 3),
            "status": status,
            "notes": notes,
        })

    # 0) Deploy fresh contract
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    deploy_fn = Contract.constructor(regulator_addr)

    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, deploy_fn, regulator_addr, regulator_pk, gas=3_000_000,
        max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(regulator_addr),
        receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
    )

    if rcpt is None:
        out_path = OUT_DIR / f"full_chain_timing_{run_id()}.csv"
        log_row("Deployment", "Regulator", "deployContract", txh, None, confirm_s, status, "deployment_not_mined_in_time")
        write_csv(out_path, rows)
        raise RuntimeError(f"Deployment tx not mined within timeout. Wrote partial CSV: {out_path}")

    contract_addr = rcpt.get("contractAddress")
    if not contract_addr:
        raise RuntimeError("Deployment receipt missing contractAddress")

    log_row("Deployment", "Regulator", "deployContract", txh, rcpt, confirm_s, status, f"contract={contract_addr}")

    contract = w3.eth.contract(address=contract_addr, abi=abi)

    # Load seed profiles for registration fields
    donor_seed = load_seed_json(SEED_DIR / "donor_1.json")
    donor_bt = norm_bt(donor_seed.get("blood_type", ""))
    donor_hla = hla_list_to_string(donor_seed.get("hla_typing", []))
    donor_organ = str(donor_seed.get("organ_type", "Kidney")).strip()

    recipient_seed = {i: load_seed_json(SEED_DIR / f"recipient_{i}.json") for i in range(1, 11)}

    # 1) Governance setup
    for fn_name, fn_args in [
        ("registerHospital", [hospital_addr]),
        ("registerEthicalCommittee", [ethics_addr]),
        ("registerMedicalTeam", [medical_addr]),
        ("registerLLM", [llm_addr]),
    ]:
        fn = build_fn(contract, fn_name, fn_args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, regulator_addr, regulator_pk, gas=300_000,
            max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
            nonce=next_nonce(regulator_addr),
            receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
        )
        log_row("Governance setup", "Regulator", fn_name, txh, rcpt, confirm_s, status)

    # 2) Identity binding (if supported by ABI)
    donor_id: Optional[int] = None
    recipient_ids: Dict[int, int] = {}

    if fn_exists(contract, "registerDonorAddress") and fn_exists(contract, "registerRecipientAddress"):
        fn = build_fn(contract, "registerDonorAddress", [donor_addr])
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, regulator_addr, regulator_pk, gas=300_000,
            max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
            nonce=next_nonce(regulator_addr),
            receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
        )
        log_row("Identity binding", "Regulator", "registerDonorAddress", txh, rcpt, confirm_s, status)

        if fn_exists(contract, "registeredDonorAddresses"):
            donor_id = int(contract.functions.registeredDonorAddresses(donor_addr).call())

        for i, raddr in enumerate(recipient_addrs, start=1):
            fn = build_fn(contract, "registerRecipientAddress", [raddr])
            txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
                w3, fn, regulator_addr, regulator_pk, gas=300_000,
                max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
                nonce=next_nonce(regulator_addr),
                receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
            )
            log_row("Identity binding", "Regulator", "registerRecipientAddress", txh, rcpt, confirm_s, status, f"i={i}")

            if fn_exists(contract, "registeredRecipientAddresses"):
                recipient_ids[i] = int(contract.functions.registeredRecipientAddresses(raddr).call())

    # 3) Profile registration (supports id-based or address-based ABI)
    reg_donor_first = first_input_type(contract, "registerDonor")
    if reg_donor_first == "uint256":
        if donor_id is None:
            raise RuntimeError("registerDonor expects donorId (uint256) but donorId is not available")
        donor_reg_args = [donor_id, donor_bt, donor_hla, donor_organ, donor_cid]
    elif reg_donor_first == "address":
        donor_reg_args = [donor_addr, donor_bt, donor_hla, donor_organ, donor_cid]
    else:
        raise RuntimeError(f"Unexpected registerDonor first input type: {reg_donor_first}")

    fn = build_fn(contract, "registerDonor", donor_reg_args)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, hospital_addr, hospital_pk, gas=550_000,
        max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(hospital_addr),
        receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
    )
    log_row("Profile registration", "Hospital", "registerDonor", txh, rcpt, confirm_s, status)

    reg_rec_first = first_input_type(contract, "registerRecipient")
    for i in range(1, 11):
        r = recipient_seed[i]
        bt = norm_bt(r.get("blood_type", ""))
        hla = hla_list_to_string(r.get("hla_typing", []))
        organ = str(r.get("organ_type", "Kidney")).strip()
        cid = recipient_cids[i]

        if reg_rec_first == "uint256":
            if i not in recipient_ids:
                raise RuntimeError(f"registerRecipient expects recipientId but recipientId is not available for i={i}")
            args = [recipient_ids[i], bt, hla, organ, cid]
        elif reg_rec_first == "address":
            args = [recipient_addrs[i - 1], bt, hla, organ, cid]
        else:
            raise RuntimeError(f"Unexpected registerRecipient first input type: {reg_rec_first}")

        fn = build_fn(contract, "registerRecipient", args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, hospital_addr, hospital_pk, gas=550_000,
            max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
            nonce=next_nonce(hospital_addr),
            receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
        )
        log_row("Profile registration", "Hospital", "registerRecipient", txh, rcpt, confirm_s, status, f"i={i}")

    # 4) Eligibility approvals
    donor_appr_first = first_input_type(contract, "approveDonorEthicalCommittee")
    if donor_appr_first == "uint256":
        if donor_id is None:
            raise RuntimeError("approveDonorEthicalCommittee expects donorId but donorId is not available")
        args = [donor_id]
    elif donor_appr_first == "address":
        args = [donor_addr]
    else:
        raise RuntimeError(f"Unexpected approveDonorEthicalCommittee first input type: {donor_appr_first}")

    fn = build_fn(contract, "approveDonorEthicalCommittee", args)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, ethics_addr, ethics_pk, gas=300_000,
        max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(ethics_addr),
        receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
    )
    log_row("Eligibility approvals", "Ethics committee", "approveDonorEthicalCommittee", txh, rcpt, confirm_s, status)

    rec_appr_first = first_input_type(contract, "approveRecipientEthicalCommittee")
    for i in range(1, 11):
        if rec_appr_first == "uint256":
            if i not in recipient_ids:
                raise RuntimeError(f"approveRecipientEthicalCommittee expects recipientId but recipientId is not available for i={i}")
            args = [recipient_ids[i]]
        elif rec_appr_first == "address":
            args = [recipient_addrs[i - 1]]
        else:
            raise RuntimeError(f"Unexpected approveRecipientEthicalCommittee first input type: {rec_appr_first}")

        fn = build_fn(contract, "approveRecipientEthicalCommittee", args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, ethics_addr, ethics_pk, gas=300_000,
            max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
            nonce=next_nonce(ethics_addr),
            receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
        )
        log_row("Eligibility approvals", "Ethics committee", "approveRecipientEthicalCommittee", txh, rcpt, confirm_s, status, f"i={i}")

    # Optional CID passed to createMatch
    rationale_cid = (os.getenv("MATCH_RATIONALE_CID") or "").strip() or "cid-placeholder"

    # 5) Matching
    create_first = first_input_type(contract, "createMatch")
    if create_first == "uint256":
        if donor_id is None or not recipient_ids:
            raise RuntimeError("createMatch expects ids but donorId/recipientIds are not available")
        primary = recipient_ids.get(1)
        backup = recipient_ids.get(7)
        if primary is None or backup is None:
            raise RuntimeError("Missing recipient ids for primary/backup")
        create_args = [donor_id, primary, backup, rationale_cid]
    elif create_first == "address":
        primary = recipient_addrs[0]
        backup = recipient_addrs[6]
        create_args = [donor_addr, primary, backup, rationale_cid]
    else:
        raise RuntimeError(f"Unexpected createMatch first input type: {create_first}")

    fn = build_fn(contract, "createMatch", create_args)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, llm_addr, llm_pk, gas=600_000,
        max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(llm_addr),
        receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
    )
    log_row("Matching", "Authorized LLM", "createMatch", txh, rcpt, confirm_s, status)

    # Determine matchId
    match_id: Optional[int] = None
    if fn_exists(contract, "matchCounter"):
        match_id = int(contract.functions.matchCounter().call())
    if match_id is None:
        raise RuntimeError("Could not resolve matchId after createMatch.")

    # 6) Approvals
    for cat, role, sender_addr, sender_pk, fn_name in [
        ("Approvals", "Medical team", medical_addr, medical_pk, "approveMedicalTeam"),
        ("Approvals", "Hospital", hospital_addr, hospital_pk, "approveHospital"),
        ("Approvals", "Donor", donor_addr, donor_pk, "approveDonor"),
        ("Approvals", "Recipient", recipient_addrs[0], recipient_pks[0], "approveRecipient"),
        ("Approvals", "Ethics committee", ethics_addr, ethics_pk, "approveFinalTransplant"),
    ]:
        fn = build_fn(contract, fn_name, [match_id])
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, sender_addr, sender_pk, gas=300_000,
            max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
            nonce=next_nonce(sender_addr),
            receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
        )
        log_row(cat, role, fn_name, txh, rcpt, confirm_s, status, f"matchId={match_id}")

    # 7) Finalization
    fn = build_fn(contract, "finalizeMatch", [match_id])
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, regulator_addr, regulator_pk, gas=400_000,
        max_fee_gwei=max_fee_gwei, max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(regulator_addr),
        receipt_timeout_s=receipt_timeout_s, poll_latency_s=poll_latency_s,
    )
    log_row("Finalization", "Regulator", "finalizeMatch", txh, rcpt, confirm_s, status, f"matchId={match_id}")

    out_path = OUT_DIR / f"full_chain_timing_{run_id()}.csv"
    write_csv(out_path, rows)

    print("Contract:", contract_addr)
    print("Match ID:", match_id)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()