import os
import csv
import json
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import ContractLogicError

from src.config import get_settings
from src.web3_client import load_abi, load_contract_address

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ts_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# 09-Mar-2026 assumptions provided by you
# NOTE: For L2s this estimates "execution gas component only" (excludes L1 data posting fees).
CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon":  {"token": "MATIC", "token_usd": 0.177,   "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}

def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    cost_token = gas_used * base_fee_gwei * 1e-9
    return cost_token * token_usd

def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address

def dummy_addr(i: int) -> str:
    # deterministic non-zero dummy addresses (not private-key owned; fine for estimateGas)
    # 0x000... is invalid; we make 0x000...00(i+1)
    return Web3.to_checksum_address("0x" + ("00" * 19) + f"{i+1:02x}")

def est_safe(fn, sender: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        g = int(fn.estimate_gas({"from": sender}))
        return g, None
    except ContractLogicError as e:
        return None, str(e)
    except Exception as e:
        return None, f"estimate_gas failed: {e}"

def main():
    load_dotenv(APP_DIR / ".env")
    settings = get_settings()
    if not settings.sepolia_rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL missing")

    w3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

    abi = load_abi(settings.abi_path)
    addr = load_contract_address(settings.address_path)
    contract = w3.eth.contract(address=addr, abi=abi)

    regulator_pk = os.getenv("REGULATOR_PRIVATE_KEY")
    hospital_pk = os.getenv("HOSPITAL_PRIVATE_KEY")
    ethics_pk = os.getenv("ETHICS_PRIVATE_KEY")
    llm_pk = os.getenv("LLM_PRIVATE_KEY")
    medical_pk = os.getenv("MEDICAL_PRIVATE_KEY")
    donor_pk = os.getenv("DONOR_PRIVATE_KEY")
    r1_pk = os.getenv("RECIPIENT1_PRIVATE_KEY")

    if not all([regulator_pk, hospital_pk, ethics_pk, llm_pk, medical_pk, donor_pk, r1_pk]):
        raise RuntimeError("Missing one or more required private keys in .env")

    regulator = pk_to_addr(w3, regulator_pk)
    hospital = pk_to_addr(w3, hospital_pk)
    ethics = pk_to_addr(w3, ethics_pk)
    llm = pk_to_addr(w3, llm_pk)
    medical = pk_to_addr(w3, medical_pk)
    donor = pk_to_addr(w3, donor_pk)
    recipient1 = pk_to_addr(w3, r1_pk)

    # Use existing IDs (these exist in your seeded contract)
    donor_id = int(os.getenv("BENCH_DONOR_ID", "1"))
    recipient_id_1 = 1
    recipient_id_2 = 2

    # Existing matchId for approval estimates: pick latest
    try:
        match_id = int(contract.functions.matchCounter().call())
        if match_id < 1:
            match_id = 1
    except Exception:
        match_id = 1

    # Seed strings (only calldata)
    donor_cid = os.getenv("SEED_DONOR_CID", "bafyDONORCID")
    r1_cid = os.getenv("SEED_RECIPIENT1_CID", "bafyRECIP1CID")
    bt = "O"
    hla = "A1,A2,B7,DR15"
    organ = "Kidney"
    match_cid = "bafyMATCHCID"

    # Dummy addresses for estimating one-time “registration” paths without revert
    dummy_hospital = dummy_addr(10)
    dummy_ethics = dummy_addr(11)
    dummy_medical = dummy_addr(12)
    dummy_llm = dummy_addr(13)
    dummy_donor = dummy_addr(20)
    dummy_recipient = dummy_addr(21)

    rows: List[Dict[str, Any]] = []

    def add_row(label: str, sender_role: str, gas_val: Optional[int], err: Optional[str]):
        row = {
            "function": label,
            "sender_role": sender_role,
            "estimated_gas": "" if gas_val is None else gas_val,
            "error": "" if err is None else err[:180],
        }
        for chain, p in CHAIN_PARAMS.items():
            if gas_val is None:
                row[f"{chain}_usd_at_basefee"] = ""
            else:
                row[f"{chain}_usd_at_basefee"] = round(gas_to_usd(gas_val, p["base_fee_gwei"], p["token_usd"]), 8)
        rows.append(row)

    # ---- Estimates ----
    # Use DUMMY recipients/entities to avoid “already registered” reverts
    g, e = est_safe(contract.functions.registerHospital(dummy_hospital), regulator)
    add_row("registerHospital", "Regulator", g, e)

    g, e = est_safe(contract.functions.registerEthicalCommittee(dummy_ethics), regulator)
    add_row("registerEthicalCommittee", "Regulator", g, e)

    g, e = est_safe(contract.functions.registerMedicalTeam(dummy_medical), regulator)
    add_row("registerMedicalTeam", "Regulator", g, e)

    g, e = est_safe(contract.functions.registerLLM(dummy_llm), regulator)
    add_row("registerLLM", "Regulator", g, e)

    g, e = est_safe(contract.functions.registerDonorAddress(dummy_donor), regulator)
    add_row("registerDonorAddress", "Regulator", g, e)

    g, e = est_safe(contract.functions.registerRecipientAddress(dummy_recipient), regulator)
    add_row("registerRecipientAddress", "Regulator", g, e)

    # Hospital registration estimates (use dummy donor/recipient pre-registered?)
    # NOTE: registerDonor/registerRecipient require pre-registration. estimate may revert if dummy not pre-registered.
    # So we estimate using your real addresses but accept if it reverts (since already registered). This is still informative.
    g, e = est_safe(contract.functions.registerDonor(donor, bt, hla, organ, donor_cid), hospital)
    add_row("registerDonor", "Hospital", g, e)

    g, e = est_safe(contract.functions.registerRecipient(recipient1, bt, hla, organ, r1_cid), hospital)
    add_row("registerRecipient", "Hospital", g, e)

    # Ethical approvals (use existing donor/recipient IDs; will revert if already approved -> captured)
    g, e = est_safe(contract.functions.approveDonorEthicalCommittee(donor_id), ethics)
    add_row("approveDonorEthicalCommittee", "EthicsCommittee", g, e)

    g, e = est_safe(contract.functions.approveRecipientEthicalCommittee(recipient_id_1), ethics)
    add_row("approveRecipientEthicalCommittee", "EthicsCommittee", g, e)

    # createMatch estimate (may revert if donor/recipients not approved; your state is approved, so should pass)
    g, e = est_safe(contract.functions.createMatch(donor_id, recipient_id_1, recipient_id_2, match_cid), llm)
    add_row("createMatch", "LLM", g, e)

    # Approvals chain (use existing matchId; may revert if already approved; captured)
    g, e = est_safe(contract.functions.approveMedicalTeam(match_id), medical)
    add_row("approveMedicalTeam", "MedicalTeam", g, e)

    g, e = est_safe(contract.functions.approveHospital(match_id), hospital)
    add_row("approveHospital", "Hospital", g, e)

    g, e = est_safe(contract.functions.approveDonor(match_id), donor)
    add_row("approveDonor", "Donor", g, e)

    g, e = est_safe(contract.functions.approveRecipient(match_id), recipient1)
    add_row("approveRecipient", "Recipient", g, e)

    g, e = est_safe(contract.functions.approveFinalTransplant(match_id), ethics)
    add_row("approveFinalTransplant", "EthicsCommittee", g, e)

    stamp = ts_utc()
    out_csv = OUT_DIR / f"gas_estimates_{stamp}.csv"
    out_json = OUT_DIR / f"gas_estimates_{stamp}.json"

    fields = ["function", "sender_role", "estimated_gas", "error"] + [f"{c}_usd_at_basefee" for c in CHAIN_PARAMS.keys()]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    obj = {
        "assumptions_date": "09-Mar-2026",
        "note": "USD estimates use base_fee_gwei * gas (execution gas only). L2 estimates exclude L1 data posting fees; treat as lower-bound. Estimates that revert are captured in the error column.",
        "contract_address": addr,
        "match_id_used_for_approval_estimates": match_id,
        "chain_params": CHAIN_PARAMS,
        "rows": rows,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_json)

if __name__ == "__main__":
    main()