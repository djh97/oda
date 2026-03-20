import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

from src.config import get_settings
from src.web3_client import load_abi, load_contract_address
from src.tx_logger import append_tx


APP_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = APP_DIR / "seed-data"
OUT_DIR = APP_DIR / "pipeline-output"
SEED_RUNS_DIR = OUT_DIR / "seed_runs"


def run_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address


def load_seed_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing seed file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def hla_list_to_string(hla):
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


def send_tx(w3: Web3, tx: dict, private_key: str):
    acct = w3.eth.account.from_key(private_key)
    tx.setdefault("chainId", w3.eth.chain_id)
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash.hex(), receipt


def log_and_append(network: str, role: str, function: str, tx_hash: str, receipt, notes: str = ""):
    gas_used = int(receipt.gasUsed) if receipt is not None else None
    append_tx(network=network, role=role, function=function, tx_hash=tx_hash, gas_used=gas_used, notes=notes)


def main():
    load_dotenv(APP_DIR / ".env")
    settings = get_settings()
    if not settings.sepolia_rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL missing in .env")

    w3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

    abi = load_abi(settings.abi_path)
    addr = load_contract_address(settings.address_path)
    contract = w3.eth.contract(address=addr, abi=abi)

    regulator_pk = os.getenv("REGULATOR_PRIVATE_KEY", "").strip()
    hospital_pk = os.getenv("HOSPITAL_PRIVATE_KEY", "").strip()
    ethics_pk = os.getenv("ETHICS_PRIVATE_KEY", "").strip()

    if not regulator_pk or not hospital_pk or not ethics_pk:
        raise RuntimeError("Missing REGULATOR_PRIVATE_KEY, HOSPITAL_PRIVATE_KEY, or ETHICS_PRIVATE_KEY in .env")

    regulator_addr = pk_to_addr(w3, regulator_pk)
    hospital_addr = pk_to_addr(w3, hospital_pk)
    ethics_addr = pk_to_addr(w3, ethics_pk)
    medical_addr = pk_to_addr(w3, os.getenv("MEDICAL_PRIVATE_KEY", "").strip())
    llm_addr = pk_to_addr(w3, os.getenv("LLM_PRIVATE_KEY", "").strip())
    donor_addr = pk_to_addr(w3, os.getenv("DONOR_PRIVATE_KEY", "").strip())
    recipient_addrs = [pk_to_addr(w3, os.getenv(f"RECIPIENT{i}_PRIVATE_KEY", "").strip()) for i in range(1, 11)]

    donor_seed = load_seed_json(SEED_DIR / "donor_1.json")
    recipient_seed = {i: load_seed_json(SEED_DIR / f"recipient_{i}.json") for i in range(1, 11)}

    donor_cid = (os.getenv("SEED_DONOR_CID") or "").strip()
    recipient_cids = {i: (os.getenv(f"SEED_RECIPIENT{i}_CID") or "").strip() for i in range(1, 11)}

    rows = []
    max_fee_gwei = float(os.getenv("TX_MAX_FEE_GWEI", "60"))
    max_priority_gwei = float(os.getenv("TX_PRIORITY_FEE_GWEI", "3"))
    nonce_cache = {}

    def next_nonce(addr: str) -> int:
        if addr not in nonce_cache:
            nonce_cache[addr] = w3.eth.get_transaction_count(addr, "pending")
        nonce = nonce_cache[addr]
        nonce_cache[addr] += 1
        return nonce

    def build_tx(fn, sender_addr: str, gas: int):
        return fn.build_transaction(
            {
                "from": sender_addr,
                "gas": gas,
                "nonce": next_nonce(sender_addr),
                "chainId": w3.eth.chain_id,
                "maxFeePerGas": w3.to_wei(max_fee_gwei, "gwei"),
                "maxPriorityFeePerGas": w3.to_wei(max_priority_gwei, "gwei"),
            }
        )

    def record(role: str, function: str, tx_hash: str, receipt, notes: str = ""):
        gas_used = int(receipt.gasUsed) if receipt is not None else ""
        log_and_append(settings.network, role, function, tx_hash, receipt, notes)
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "role": role,
                "function": function,
                "tx_hash": tx_hash,
                "gas_used": gas_used,
                "notes": notes,
            }
        )

    for fn_name, fn_args in [
        ("registerHospital", [hospital_addr]),
        ("registerEthicalCommittee", [ethics_addr]),
        ("registerMedicalTeam", [medical_addr]),
        ("registerLLM", [llm_addr]),
    ]:
        fn = getattr(contract.functions, fn_name)(*fn_args)
        txh, rcpt = send_tx(w3, build_tx(fn, regulator_addr, 300000), regulator_pk)
        record("Regulator", fn_name, txh, rcpt)

    fn = contract.functions.registerDonorAddress(donor_addr)
    txh, rcpt = send_tx(w3, build_tx(fn, regulator_addr, 300000), regulator_pk)
    donor_id = int(contract.functions.registeredDonorAddresses(donor_addr).call())
    record("Regulator", "registerDonorAddress", txh, rcpt, f"donorId={donor_id}")

    recipient_ids = {}
    for i, raddr in enumerate(recipient_addrs, start=1):
        fn = contract.functions.registerRecipientAddress(raddr)
        txh, rcpt = send_tx(w3, build_tx(fn, regulator_addr, 300000), regulator_pk)
        recipient_ids[i] = int(contract.functions.registeredRecipientAddresses(raddr).call())
        record("Regulator", "registerRecipientAddress", txh, rcpt, f"recipientId={recipient_ids[i]}")

    donor_bt = norm_bt(donor_seed.get("blood_type", ""))
    donor_hla = hla_list_to_string(donor_seed.get("hla_typing", []))
    donor_organ = str(donor_seed.get("organ_type", "Kidney")).strip()
    fn = contract.functions.registerDonor(donor_addr, donor_bt, donor_hla, donor_organ, donor_cid)
    txh, rcpt = send_tx(w3, build_tx(fn, hospital_addr, 550000), hospital_pk)
    record("Hospital", "registerDonor", txh, rcpt, f"donorId={donor_id}")

    for i in range(1, 11):
        r = recipient_seed[i]
        bt = norm_bt(r.get("blood_type", ""))
        hla = hla_list_to_string(r.get("hla_typing", []))
        organ = str(r.get("organ_type", "Kidney")).strip()
        fn = contract.functions.registerRecipient(recipient_addrs[i - 1], bt, hla, organ, recipient_cids[i])
        txh, rcpt = send_tx(w3, build_tx(fn, hospital_addr, 550000), hospital_pk)
        record("Hospital", "registerRecipient", txh, rcpt, f"recipientId={recipient_ids[i]}")

    fn = contract.functions.approveDonorEthicalCommittee(donor_id)
    txh, rcpt = send_tx(w3, build_tx(fn, ethics_addr, 250000), ethics_pk)
    record("EthicsCommittee", "approveDonorEthicalCommittee", txh, rcpt, f"id={donor_id}")

    for i in range(1, 11):
        fn = contract.functions.approveRecipientEthicalCommittee(recipient_ids[i])
        txh, rcpt = send_tx(w3, build_tx(fn, ethics_addr, 250000), ethics_pk)
        record("EthicsCommittee", "approveRecipientEthicalCommittee", txh, rcpt, f"id={recipient_ids[i]}")

    SEED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SEED_RUNS_DIR / f"seed_only_{run_ts()}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "role", "function", "tx_hash", "gas_used", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Contract:", addr)
    print("Donor ID:", donor_id)
    print("Recipient IDs:", [recipient_ids[i] for i in range(1, 11)])
    print("Seed log:", out_csv)


if __name__ == "__main__":
    main()
