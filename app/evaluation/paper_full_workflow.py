import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import TimeExhausted

from src.tx_logger import append_tx


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[0]
PIPELINE_DIR = APP_DIR / "pipeline-output"
TIMING_DIR = PIPELINE_DIR / "timing_runs"
SEED_DIR = APP_DIR / "seed-data"
SMART_CONTRACTS_DIR = REPO_ROOT / "smart-contracts"
INTEGRATION_DIR = REPO_ROOT / "integration"
ABI_PATH = INTEGRATION_DIR / "abi" / "TransplantManagement.json"
ADDRESS_PATH = INTEGRATION_DIR / "addresses" / "sepolia.json"
STANDARD_INPUT_PATH = SMART_CONTRACTS_DIR / "standard-input.json"

TIMING_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

TX_MANIFEST = PIPELINE_DIR / "tx_manifest.csv"
FINAL_COST_CSV = PIPELINE_DIR / "final_cost_table.csv"
FINAL_COST_JSON = PIPELINE_DIR / "final_cost_table.json"
FINAL_COST_DETAILED = PIPELINE_DIR / "final_cost_table_detailed.csv"

CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon": {"token": "MATIC", "token_usd": 0.177, "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}


@dataclass
class TxRecord:
    timestamp_utc: str
    category: str
    role: str
    function: str
    tx_hash: str
    gas_used: int
    confirmation_s: float
    status: str
    notes: str
    label: str = ""
    effective_gas_price_gwei: float = 0.0
    fee_eth: float = 0.0


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_0x(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return value
    return value if value.startswith("0x") else "0x" + value


def pk_to_addr(w3: Web3, pk: str) -> str:
    return w3.eth.account.from_key(pk).address


def load_seed_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def hla_list_to_string(hla: Any) -> str:
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


def load_foundry_artifact() -> Tuple[List[dict], str]:
    override = os.getenv("FOUNDRY_ARTIFACT_PATH", "").strip()
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.append(REPO_ROOT / "smart-contracts" / "out" / "TransplantManagement.sol" / "TransplantManagement.json")

    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            artifact = json.load(f)
        abi = artifact.get("abi")
        bytecode_obj = artifact.get("bytecode", {})
        bytecode = bytecode_obj.get("object", "") if isinstance(bytecode_obj, dict) else bytecode_obj
        if abi and bytecode:
            return abi, normalize_0x(bytecode)

    raise RuntimeError("Could not locate Foundry artifact. Set FOUNDRY_ARTIFACT_PATH in app/.env.")


def fn_exists(contract: Any, fn_name: str) -> bool:
    return hasattr(contract.functions, fn_name)


def build_fn(contract: Any, fn_name: str, args: list):
    if not fn_exists(contract, fn_name):
        raise AttributeError(f"Contract ABI does not contain function: {fn_name}")
    return getattr(contract.functions, fn_name)(*args)


def first_input_type(contract: Any, fn_name: str) -> Optional[str]:
    for item in contract.abi:
        if item.get("type") == "function" and item.get("name") == fn_name:
            inputs = item.get("inputs", [])
            if inputs:
                return inputs[0].get("type")
            return None
    return None


def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    return gas_used * float(base_fee_gwei) * 1e-9 * float(token_usd)


def send_tx_with_confirmation_time(
    w3: Web3,
    fn: Any,
    sender_addr: str,
    private_key: str,
    gas: int,
    max_fee_gwei: float,
    max_priority_gwei: float,
    nonce: int,
    receipt_timeout_s: int,
    poll_latency_s: float,
) -> Tuple[str, Optional[dict], float, str]:
    tx = fn.build_transaction(
        {
            "from": sender_addr,
            "gas": gas,
            "nonce": nonce,
            "chainId": w3.eth.chain_id,
            "maxFeePerGas": w3.to_wei(max_fee_gwei, "gwei"),
            "maxPriorityFeePerGas": w3.to_wei(max_priority_gwei, "gwei"),
        }
    )
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
        return tx_hash.hex(), None, (t1 - t0), "timeout_pending"


def write_timing_csv(path: Path, rows: List[TxRecord]) -> None:
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
        "label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in fieldnames})


def write_manifest(path: Path, rows: List[TxRecord]) -> None:
    fieldnames = ["category", "function", "tx_hash", "label"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": row.category,
                    "function": row.function,
                    "tx_hash": normalize_0x(row.tx_hash),
                    "label": row.label,
                }
            )


def write_cost_outputs(rows: List[TxRecord]) -> None:
    detailed_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "category": "",
            "function": "",
            "count": 0,
            "gas_used_total": 0,
            "tx_hashes": [],
        }
    )

    for row in rows:
        detailed: Dict[str, object] = {
            "category": row.category,
            "function": row.function,
            "tx_hash": row.tx_hash,
            "gas_used": row.gas_used,
            "effective_gas_price_gwei": row.effective_gas_price_gwei,
            "fee_eth": row.fee_eth,
            "source": "observed_receipt",
            "note": row.label or row.notes,
        }
        for chain, params in CHAIN_PARAMS.items():
            detailed[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(row.gas_used, params["base_fee_gwei"], params["token_usd"]), 8
            )
        detailed_rows.append(detailed)

        key = (row.category, row.function)
        agg = grouped[key]
        agg["category"] = row.category
        agg["function"] = row.function
        agg["count"] += 1
        agg["gas_used_total"] += row.gas_used
        agg["tx_hashes"].append(row.tx_hash)

    agg_rows: List[Dict[str, object]] = []
    for _, agg in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out = dict(agg)
        for chain, params in CHAIN_PARAMS.items():
            out[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(int(out["gas_used_total"]), params["base_fee_gwei"], params["token_usd"]), 8
            )
        agg_rows.append(out)

    total_gas = sum(int(r["gas_used_total"]) for r in agg_rows)
    totals = {
        "category": "TOTAL",
        "function": "all",
        "count": sum(int(r["count"]) for r in agg_rows),
        "gas_used_total": total_gas,
        "tx_hashes": [],
    }
    for chain, params in CHAIN_PARAMS.items():
        totals[f"{chain}_usd_at_basefee"] = round(
            gas_to_usd(total_gas, params["base_fee_gwei"], params["token_usd"]), 8
        )
    agg_rows.append(totals)

    detailed_fields = list(detailed_rows[0].keys()) if detailed_rows else [
        "category",
        "function",
        "tx_hash",
        "gas_used",
        "effective_gas_price_gwei",
        "fee_eth",
        "source",
        "note",
    ] + [f"{chain}_usd_at_basefee" for chain in CHAIN_PARAMS]
    with FINAL_COST_DETAILED.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fields)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    agg_fields = [
        "category",
        "function",
        "count",
        "gas_used_total",
        *[f"{chain}_usd_at_basefee" for chain in CHAIN_PARAMS],
        "tx_hashes",
    ]
    with FINAL_COST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for row in agg_rows:
            out = dict(row)
            out["tx_hashes"] = ",".join(out.get("tx_hashes", []))
            writer.writerow(out)

    payload = {
        "generated_utc": now_utc(),
        "rows": agg_rows,
        "detailed_csv": str(FINAL_COST_DETAILED),
        "aggregated_csv": str(FINAL_COST_CSV),
    }
    with FINAL_COST_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sync_integration_files(contract_addr: str, abi: List[dict]) -> None:
    ABI_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADDRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ABI_PATH.open("w", encoding="utf-8", newline="") as f:
        json.dump(abi, f, indent=2)

    address_payload = {
        "network": "sepolia",
        "chainId": 11155111,
        "contractName": "TransplantManagement",
        "address": contract_addr,
    }
    with ADDRESS_PATH.open("w", encoding="utf-8", newline="") as f:
        json.dump(address_payload, f, indent=2)


def generate_standard_input(contract_addr: str) -> None:
    cmd = [
        "forge",
        "verify-contract",
        contract_addr,
        "src/TransplantManagement.sol:TransplantManagement",
        "--chain",
        "sepolia",
        "--show-standard-json-input",
    ]
    result = subprocess.run(
        cmd,
        cwd=SMART_CONTRACTS_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    STANDARD_INPUT_PATH.write_text(result.stdout, encoding="utf-8")


def main() -> None:
    load_dotenv(APP_DIR / ".env")

    rpc = (os.getenv("SEPOLIA_RPC_URL") or "").strip()
    if not rpc:
        raise RuntimeError("Missing SEPOLIA_RPC_URL in app/.env")

    max_fee_gwei = float(os.getenv("TX_MAX_FEE_GWEI", "60"))
    max_priority_gwei = float(os.getenv("TX_PRIORITY_FEE_GWEI", "3"))
    receipt_timeout_s = int(os.getenv("TX_RECEIPT_TIMEOUT_S", "600"))
    poll_latency_s = float(os.getenv("TX_POLL_LATENCY_S", "2"))

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

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

    donor_cid = (os.getenv("SEED_DONOR_CID") or "").strip()
    if not donor_cid:
        raise RuntimeError("Missing SEED_DONOR_CID in app/.env")
    recipient_cids: Dict[int, str] = {}
    for i in range(1, 11):
        cid = (os.getenv(f"SEED_RECIPIENT{i}_CID") or "").strip()
        if not cid:
            raise RuntimeError(f"Missing SEED_RECIPIENT{i}_CID in app/.env")
        recipient_cids[i] = cid

    regulator_addr = pk_to_addr(w3, regulator_pk)
    hospital_addr = pk_to_addr(w3, hospital_pk)
    ethics_addr = pk_to_addr(w3, ethics_pk)
    llm_addr = pk_to_addr(w3, llm_pk)
    medical_addr = pk_to_addr(w3, medical_pk)
    donor_addr = pk_to_addr(w3, donor_pk)
    recipient_addrs = [pk_to_addr(w3, pk) for pk in recipient_pks]

    abi, bytecode = load_foundry_artifact()
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    nonce_cache: Dict[str, int] = {}

    def next_nonce(addr: str) -> int:
        if addr not in nonce_cache:
            nonce_cache[addr] = w3.eth.get_transaction_count(addr, "pending")
        nonce = nonce_cache[addr]
        nonce_cache[addr] += 1
        return nonce

    records: List[TxRecord] = []

    def add_record(
        category: str,
        role: str,
        function: str,
        tx_hash: str,
        receipt: Optional[dict],
        confirmation_s: float,
        status: str,
        notes: str = "",
        label: str = "",
    ) -> None:
        gas_used = int(receipt["gasUsed"]) if receipt is not None else 0
        eff_price_wei = int(receipt.get("effectiveGasPrice", 0)) if receipt is not None else 0
        fee_eth = (gas_used * eff_price_wei) / 1e18 if eff_price_wei else 0.0
        record = TxRecord(
            timestamp_utc=now_utc(),
            category=category,
            role=role,
            function=function,
            tx_hash=normalize_0x(tx_hash),
            gas_used=gas_used,
            confirmation_s=round(confirmation_s, 3),
            status=status,
            notes=notes,
            label=label,
            effective_gas_price_gwei=(eff_price_wei / 1e9) if eff_price_wei else 0.0,
            fee_eth=fee_eth,
        )
        records.append(record)
        append_tx(
            network="sepolia",
            role=role,
            function=function,
            tx_hash=normalize_0x(tx_hash),
            gas_used=gas_used if gas_used else None,
            notes=notes,
        )

    deploy_fn = Contract.constructor(regulator_addr)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3,
        deploy_fn,
        regulator_addr,
        regulator_pk,
        gas=3_000_000,
        max_fee_gwei=max_fee_gwei,
        max_priority_gwei=max_priority_gwei,
        nonce=next_nonce(regulator_addr),
        receipt_timeout_s=receipt_timeout_s,
        poll_latency_s=poll_latency_s,
    )
    if rcpt is None or not rcpt.get("contractAddress"):
        raise RuntimeError("Deployment did not complete successfully.")
    contract_addr = rcpt["contractAddress"]
    add_record("Deployment", "Regulator", "deployContract", txh, rcpt, confirm_s, status, notes=f"contract={contract_addr}", label=f"contract={contract_addr}")
    sync_integration_files(contract_addr, abi)
    generate_standard_input(contract_addr)

    contract = w3.eth.contract(address=contract_addr, abi=abi)

    donor_seed = load_seed_json(SEED_DIR / "donor_1.json")
    donor_bt = norm_bt(donor_seed.get("blood_type", ""))
    donor_hla = hla_list_to_string(donor_seed.get("hla_typing", []))
    donor_organ = str(donor_seed.get("organ_type", "Kidney")).strip()
    recipient_seed = {i: load_seed_json(SEED_DIR / f"recipient_{i}.json") for i in range(1, 11)}

    for fn_name, fn_args, label in [
        ("registerHospital", [hospital_addr], f"hospital={hospital_addr}"),
        ("registerMedicalTeam", [medical_addr], f"medicalTeam={medical_addr}"),
        ("registerEthicalCommittee", [ethics_addr], f"member={ethics_addr}"),
        ("registerLLM", [llm_addr], f"llm={llm_addr}"),
    ]:
        fn = build_fn(contract, fn_name, fn_args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, regulator_addr, regulator_pk, 300_000, max_fee_gwei, max_priority_gwei,
            next_nonce(regulator_addr), receipt_timeout_s, poll_latency_s,
        )
        add_record("Governance setup", "Regulator", fn_name, txh, rcpt, confirm_s, status, label=label)

    donor_id: Optional[int] = None
    recipient_ids: Dict[int, int] = {}

    fn = build_fn(contract, "registerDonorAddress", [donor_addr])
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, regulator_addr, regulator_pk, 300_000, max_fee_gwei, max_priority_gwei,
        next_nonce(regulator_addr), receipt_timeout_s, poll_latency_s,
    )
    if fn_exists(contract, "registeredDonorAddresses"):
        donor_id = int(contract.functions.registeredDonorAddresses(donor_addr).call())
    add_record("Identity binding", "Regulator", "registerDonorAddress", txh, rcpt, confirm_s, status, label=f"donorId={donor_id},addr={donor_addr}")

    for i, raddr in enumerate(recipient_addrs, start=1):
        fn = build_fn(contract, "registerRecipientAddress", [raddr])
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, regulator_addr, regulator_pk, 300_000, max_fee_gwei, max_priority_gwei,
            next_nonce(regulator_addr), receipt_timeout_s, poll_latency_s,
        )
        if fn_exists(contract, "registeredRecipientAddresses"):
            recipient_ids[i] = int(contract.functions.registeredRecipientAddresses(raddr).call())
        add_record(
            "Identity binding",
            "Regulator",
            "registerRecipientAddress",
            txh,
            rcpt,
            confirm_s,
            status,
            label=f"recipientId={recipient_ids.get(i, i)},addr={raddr}",
        )

    reg_donor_first = first_input_type(contract, "registerDonor")
    if reg_donor_first == "uint256":
        donor_reg_args = [donor_id, donor_bt, donor_hla, donor_organ, donor_cid]
    else:
        donor_reg_args = [donor_addr, donor_bt, donor_hla, donor_organ, donor_cid]
    fn = build_fn(contract, "registerDonor", donor_reg_args)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, hospital_addr, hospital_pk, 550_000, max_fee_gwei, max_priority_gwei,
        next_nonce(hospital_addr), receipt_timeout_s, poll_latency_s,
    )
    add_record("Profile registration", "Hospital", "registerDonor", txh, rcpt, confirm_s, status, label=f"donorId={donor_id},organ={donor_organ}")

    reg_rec_first = first_input_type(contract, "registerRecipient")
    for i in range(1, 11):
        r = recipient_seed[i]
        bt = norm_bt(r.get("blood_type", ""))
        hla = hla_list_to_string(r.get("hla_typing", []))
        organ = str(r.get("organ_type", "Kidney")).strip()
        cid = recipient_cids[i]
        args = [recipient_ids[i], bt, hla, organ, cid] if reg_rec_first == "uint256" else [recipient_addrs[i - 1], bt, hla, organ, cid]
        fn = build_fn(contract, "registerRecipient", args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, hospital_addr, hospital_pk, 550_000, max_fee_gwei, max_priority_gwei,
            next_nonce(hospital_addr), receipt_timeout_s, poll_latency_s,
        )
        add_record("Profile registration", "Hospital", "registerRecipient", txh, rcpt, confirm_s, status, label=f"recipientId={recipient_ids[i]},organ={organ}")

    donor_appr_first = first_input_type(contract, "approveDonorEthicalCommittee")
    fn = build_fn(contract, "approveDonorEthicalCommittee", [donor_id] if donor_appr_first == "uint256" else [donor_addr])
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, ethics_addr, ethics_pk, 300_000, max_fee_gwei, max_priority_gwei,
        next_nonce(ethics_addr), receipt_timeout_s, poll_latency_s,
    )
    add_record("Eligibility approvals", "EthicsCommittee", "approveDonorEthicalCommittee", txh, rcpt, confirm_s, status, label=f"id={donor_id}")

    rec_appr_first = first_input_type(contract, "approveRecipientEthicalCommittee")
    for i in range(1, 11):
        args = [recipient_ids[i]] if rec_appr_first == "uint256" else [recipient_addrs[i - 1]]
        fn = build_fn(contract, "approveRecipientEthicalCommittee", args)
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, ethics_addr, ethics_pk, 300_000, max_fee_gwei, max_priority_gwei,
            next_nonce(ethics_addr), receipt_timeout_s, poll_latency_s,
        )
        add_record("Eligibility approvals", "EthicsCommittee", "approveRecipientEthicalCommittee", txh, rcpt, confirm_s, status, label=f"id={recipient_ids[i]}")

    rationale_cid = (os.getenv("MATCH_RATIONALE_CID") or "").strip() or "cid-placeholder"
    create_args = [donor_id, recipient_ids[1], recipient_ids[7], rationale_cid]
    fn = build_fn(contract, "createMatch", create_args)
    txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
        w3, fn, llm_addr, llm_pk, 600_000, max_fee_gwei, max_priority_gwei,
        next_nonce(llm_addr), receipt_timeout_s, poll_latency_s,
    )
    add_record("Match workflow", "AuthorizedLLM", "createMatch", txh, rcpt, confirm_s, status, label="primary=1,backup=7")

    match_id = int(contract.functions.matchCounter().call())

    for role, sender_addr, sender_pk, fn_name, label in [
        ("MedicalTeam", medical_addr, medical_pk, "approveMedicalTeam", f"matchId={match_id}"),
        ("Hospital", hospital_addr, hospital_pk, "approveHospital", f"matchId={match_id}"),
        ("Donor", donor_addr, donor_pk, "approveDonor", f"matchId={match_id}"),
        ("Recipient1", recipient_addrs[0], recipient_pks[0], "approveRecipient", f"matchId={match_id},recipient={recipient_ids[1]}"),
        ("EthicsCommittee", ethics_addr, ethics_pk, "approveFinalTransplant", f"matchId={match_id}"),
        ("Finalizer", regulator_addr, regulator_pk, "finalizeMatch", f"matchId={match_id}"),
    ]:
        fn = build_fn(contract, fn_name, [match_id])
        txh, rcpt, confirm_s, status = send_tx_with_confirmation_time(
            w3, fn, sender_addr, sender_pk, 400_000, max_fee_gwei, max_priority_gwei,
            next_nonce(sender_addr), receipt_timeout_s, poll_latency_s,
        )
        add_record("Match workflow", role, fn_name, txh, rcpt, confirm_s, status, label=label)

    timing_path = TIMING_DIR / f"paper_full_workflow_{run_id()}.csv"
    write_timing_csv(timing_path, records)
    write_manifest(TX_MANIFEST, records)
    write_cost_outputs(records)

    print("Contract:", contract_addr)
    print("Match ID:", match_id)
    print("Timing CSV:", timing_path)
    print("Manifest:", TX_MANIFEST)
    print("Cost CSV:", FINAL_COST_CSV)
    print("Cost JSON:", FINAL_COST_JSON)
    print("Cost Detailed CSV:", FINAL_COST_DETAILED)
    print("ABI:", ABI_PATH)
    print("Address:", ADDRESS_PATH)
    print("Standard Input:", STANDARD_INPUT_PATH)


if __name__ == "__main__":
    main()
