import os
import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from web3 import Web3

from src.config import get_settings
from src.web3_client import load_abi, load_contract_address
from src.tx_logger import append_tx

APP_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = APP_DIR / "seed-data"
OUT_DIR = APP_DIR / "pipeline-output"
SEED_RUNS_DIR = OUT_DIR / "seed_runs"


def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address


def send_tx(w3: Web3, tx: dict, private_key: str):
    acct = w3.eth.account.from_key(private_key)
    tx["nonce"] = w3.eth.get_transaction_count(acct.address)
    tx.setdefault("maxFeePerGas", w3.to_wei(30, "gwei"))
    tx.setdefault("maxPriorityFeePerGas", w3.to_wei(2, "gwei"))
    tx.setdefault("chainId", w3.eth.chain_id)
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash.hex(), receipt


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


def run_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main():
    settings = get_settings()
    if not settings.sepolia_rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL missing in .env")

    w3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    assert w3.is_connected(), "Not connected to Sepolia"

    abi = load_abi(settings.abi_path)
    addr = load_contract_address(settings.address_path)
    contract = w3.eth.contract(address=addr, abi=abi)

    regulator_pk = os.getenv("REGULATOR_PRIVATE_KEY")
    hospital_pk = os.getenv("HOSPITAL_PRIVATE_KEY")
    ethics_pk = os.getenv("ETHICS_PRIVATE_KEY")
    llm_pk = os.getenv("LLM_PRIVATE_KEY")
    medical_pk = os.getenv("MEDICAL_PRIVATE_KEY")

    if not regulator_pk or not hospital_pk or not ethics_pk:
        raise RuntimeError("Missing REGULATOR_PRIVATE_KEY / HOSPITAL_PRIVATE_KEY / ETHICS_PRIVATE_KEY in .env")
    if not llm_pk or not medical_pk:
        raise RuntimeError("Missing LLM_PRIVATE_KEY and/or MEDICAL_PRIVATE_KEY in .env")

    regulator_addr = pk_to_addr(w3, regulator_pk)
    hospital_addr = pk_to_addr(w3, hospital_pk)
    ethics_addr = pk_to_addr(w3, ethics_pk)
    llm_addr = pk_to_addr(w3, llm_pk)
    medical_addr = pk_to_addr(w3, medical_pk)

    donor_pk = os.getenv("DONOR_PRIVATE_KEY")
    if not donor_pk:
        raise RuntimeError("Missing DONOR_PRIVATE_KEY in .env")
    donor_addr = pk_to_addr(w3, donor_pk)

    recipient_addrs = []
    for i in range(1, 11):
        pk = os.getenv(f"RECIPIENT{i}_PRIVATE_KEY")
        if not pk:
            raise RuntimeError(f"Missing RECIPIENT{i}_PRIVATE_KEY in .env")
        recipient_addrs.append(pk_to_addr(w3, pk))

    donor_cid = os.getenv("SEED_DONOR_CID", "").strip()
    if not donor_cid:
        raise RuntimeError("Missing SEED_DONOR_CID in .env")

    recipient_cids = {}
    for i in range(1, 11):
        cid = os.getenv(f"SEED_RECIPIENT{i}_CID", "").strip()
        if not cid:
            raise RuntimeError(f"Missing SEED_RECIPIENT{i}_CID in .env")
        recipient_cids[i] = cid

    donor_seed = load_seed_json(SEED_DIR / "donor_1.json")
    donor_bt = norm_bt(donor_seed.get("blood_type", ""))
    donor_hla_str = hla_list_to_string(donor_seed.get("hla_typing", []))
    donor_organ = str(donor_seed.get("organ_type", "Kidney")).strip()

    recipient_seed = {i: load_seed_json(SEED_DIR / f"recipient_{i}.json") for i in range(1, 11)}

    # Run-specific log file for reproducibility
    SEED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run_ts()
    run_log_path = SEED_RUNS_DIR / f"seed_sepolia_{run_id}.csv"

    def log(role: str, fn_name: str, txh: str, receipt, notes: str = ""):
        gas_used = int(receipt.gasUsed) if receipt is not None else None
        # append to global log
        append_tx(network=settings.network, role=role, function=fn_name, tx_hash=txh, gas_used=gas_used, notes=notes)
        # append to run-specific log
        append_tx(network=settings.network, role=role, function=fn_name, tx_hash=txh, gas_used=gas_used, notes=notes, path=run_log_path)

    def do_tx(from_addr: str, pk: str, role: str, fn, fn_name: str, gas: int, notes: str):
        tx = fn.build_transaction({"from": from_addr, "gas": gas})
        txh, rcpt = send_tx(w3, tx, pk)
        log(role, fn_name, txh, rcpt, notes=notes)
        return txh, rcpt

    print("Connected:", w3.is_connected())
    print("Contract:", addr)
    print("Regulator:", regulator_addr)
    print("Hospital:", hospital_addr)
    print("Ethics:", ethics_addr)
    print("LLM:", llm_addr)
    print("MedicalTeam:", medical_addr)
    print("Donor:", donor_addr)
    for i, a in enumerate(recipient_addrs, start=1):
        print(f"Recipient{i}:", a)

    # 1) Register entities
    print("\n[1] Registering entities (regulator)...")
    if not contract.functions.registeredHospitals(hospital_addr).call():
        txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                       contract.functions.registerHospital(hospital_addr), "registerHospital", 300000, "seed")
        print(" registerHospital tx:", txh)
    else:
        print(" registerHospital: already registered")

    if not contract.functions.registeredEthicalCommittee(ethics_addr).call():
        txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                       contract.functions.registerEthicalCommittee(ethics_addr), "registerEthicalCommittee", 300000, "seed")
        print(" registerEthicalCommittee tx:", txh)
    else:
        print(" registerEthicalCommittee: already registered")

    if not contract.functions.registeredMedicalTeams(medical_addr).call():
        txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                       contract.functions.registerMedicalTeam(medical_addr), "registerMedicalTeam", 300000, "seed")
        print(" registerMedicalTeam tx:", txh)
    else:
        print(" registerMedicalTeam: already registered")

    if not contract.functions.authorizedLLMs(llm_addr).call():
        txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                       contract.functions.registerLLM(llm_addr), "registerLLM", 300000, "seed")
        print(" registerLLM tx:", txh)
    else:
        print(" registerLLM: already registered")

    # 2) Pre-register addresses
    print("\n[2] Pre-registering donor/recipients addresses (regulator)...")
    if int(contract.functions.registeredDonorAddresses(donor_addr).call()) == 0:
        txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                       contract.functions.registerDonorAddress(donor_addr), "registerDonorAddress", 300000, "seed")
        print(" registerDonorAddress tx:", txh)
    else:
        print(" registerDonorAddress: already pre-registered")

    for i, raddr in enumerate(recipient_addrs, start=1):
        if int(contract.functions.registeredRecipientAddresses(raddr).call()) == 0:
            txh, _ = do_tx(regulator_addr, regulator_pk, "Regulator",
                           contract.functions.registerRecipientAddress(raddr), "registerRecipientAddress", 300000, f"seed r{i}")
            print(f" registerRecipientAddress r{i} tx:", txh)
        else:
            print(f" registerRecipientAddress r{i}: already pre-registered")

    donor_id = int(contract.functions.registeredDonorAddresses(donor_addr).call())
    recipient_ids = [int(contract.functions.registeredRecipientAddresses(a).call()) for a in recipient_addrs]

    # 3) Register donor / recipients
    print("\n[3] Registering donor/recipients (hospital)...")
    donor_struct = contract.functions.donors(donor_id).call()
    donor_registered = bool(donor_struct[6])
    if not donor_registered:
        txh, _ = do_tx(hospital_addr, hospital_pk, "Hospital",
                       contract.functions.registerDonor(donor_addr, donor_bt, donor_hla_str, donor_organ, donor_cid),
                       "registerDonor", 650000, "seed donor_1")
        print(" registerDonor tx:", txh)
    else:
        print(" registerDonor: already registered")

    for i, (rid, raddr) in enumerate(zip(recipient_ids, recipient_addrs), start=1):
        seed = recipient_seed[i]
        r_bt = norm_bt(seed.get("blood_type", ""))
        # enforce your convention (if you want it): "seed_recipient_i"
        r_hla_str = f"seed_recipient_{i}"
        r_organ = str(seed.get("organ_type", "Kidney")).strip()

        r_struct = contract.functions.recipients(rid).call()
        r_registered = bool(r_struct[6])
        if not r_registered:
            txh, _ = do_tx(hospital_addr, hospital_pk, "Hospital",
                           contract.functions.registerRecipient(raddr, r_bt, r_hla_str, r_organ, recipient_cids[i]),
                           "registerRecipient", 650000, f"seed recipient_{i}")
            print(f" registerRecipient r{i} tx:", txh)
        else:
            print(f" registerRecipient r{i}: already registered")

    # 4) Ethical approvals
    print("\n[4] Ethical approvals (ethics committee)...")
    donor_struct = contract.functions.donors(donor_id).call()
    donor_eth_ok = bool(donor_struct[7])
    if not donor_eth_ok:
        txh, _ = do_tx(ethics_addr, ethics_pk, "EthicsCommittee",
                       contract.functions.approveDonorEthicalCommittee(donor_id),
                       "approveDonorEthicalCommittee", 250000, "seed donor_1")
        print(" approveDonorEthicalCommittee tx:", txh)
    else:
        print(" approveDonorEthicalCommittee: already approved")

    for i, rid in enumerate(recipient_ids, start=1):
        r_struct = contract.functions.recipients(rid).call()
        r_eth_ok = bool(r_struct[8])
        if not r_eth_ok:
            txh, _ = do_tx(ethics_addr, ethics_pk, "EthicsCommittee",
                           contract.functions.approveRecipientEthicalCommittee(rid),
                           "approveRecipientEthicalCommittee", 250000, f"seed recipient_{i}")
            print(f" approveRecipientEthicalCommittee r{i} tx:", txh)
        else:
            print(f" approveRecipientEthicalCommittee r{i}: already approved")

    print("\nDone.")
    print("Donor ID:", donor_id, "Recipient IDs:", recipient_ids)
    print("Run log:", run_log_path)


if __name__ == "__main__":
    load_dotenv(APP_DIR / ".env")
    main()