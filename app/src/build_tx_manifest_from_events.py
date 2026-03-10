#!/usr/bin/env python3
"""
Build app/pipeline-output/tx_manifest.csv by scanning contract events on Sepolia.

- Uses ONLY on-chain logs (no Etherscan API).
- Captures tx hashes for: deployment, governance setup, identity binding,
  profile registration, ethical approvals, match recording, approvals, finalization.
- This will include ALL recipients that were actually registered/approved on-chain.

Then rerun your cost script; it will auto-use tx_manifest.csv.
"""
import time
from requests.exceptions import HTTPError
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from web3 import Web3

from pathlib import Path

# Script location: repo_root/app/scripts/this_file.py
REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"

OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUT_DIR / "tx_manifest.csv"

DEFAULT_DEPLOYMENT_TX = "0x4b876f900c2553d37ad7a78aaf2a93a57c2045e54e611a907b556ffbc695814d"

# Try common ABI locations in your repo
ABI_CANDIDATES = [
    REPO_ROOT / "integration" / "abi" / "TransplantManagement.json",
    APP_DIR / "integration" / "abi" / "TransplantManagement.json",  # keep as fallback if present
    APP_DIR / "abi" / "TransplantManagement.json",
]

CHUNK_SIZE = 5000  # blocks per log query


def get_logs_with_backoff(w3: Web3, params: dict, max_retries: int = 6) -> List[dict]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return w3.eth.get_logs(params)
        except HTTPError as e:
            # Infura commonly returns 429 rate limit
            if getattr(e.response, "status_code", None) == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception as e:
            # Some providers return JSON-RPC errors (also retryable)
            msg = str(e).lower()
            if ("too many requests" in msg or "rate" in msg) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []

def normalize_tx(tx: str) -> str:
    tx = (tx or "").strip()
    return tx if tx.startswith("0x") else "0x" + tx


def load_abi() -> List[dict]:
    for p in ABI_CANDIDATES:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise RuntimeError(
        "Could not find TransplantManagement ABI. Tried:\n"
        + "\n".join(str(p) for p in ABI_CANDIDATES)
    )


def collect_logs(
    w3: Web3,
    contract_addr: str,
    topic0: str,
    start_block: int,
    end_block: int,
) -> List[dict]:
    logs: List[dict] = []
    b = start_block

    # ✅ ensure topic is 0x-prefixed
    topic0 = topic0 if topic0.startswith("0x") else "0x" + topic0

    while b <= end_block:
        to_b = min(b + CHUNK_SIZE - 1, end_block)
        part = get_logs_with_backoff(
            w3,
            {
                "fromBlock": b,
                "toBlock": to_b,
                "address": contract_addr,
                "topics": [topic0],
            },
        )
        logs.extend(part)

        # Small throttle to avoid 429
        time.sleep(0.25)
        logs.extend(part)
        b = to_b + 1
    return logs


def main() -> None:
    load_dotenv(APP_DIR / ".env")
    rpc = (os.getenv("SEPOLIA_RPC_URL") or "").strip()
    if not rpc:
        raise RuntimeError("Missing SEPOLIA_RPC_URL in app/.env")

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

    deploy_tx = normalize_tx(os.getenv("DEPLOYMENT_TX_HASH", DEFAULT_DEPLOYMENT_TX))
    deploy_receipt = w3.eth.get_transaction_receipt(deploy_tx)

    contract_addr = deploy_receipt.get("contractAddress")
    if not contract_addr:
        raise RuntimeError("Deployment receipt missing contractAddress")

    start_block = int(deploy_receipt["blockNumber"])
    end_block = int(w3.eth.block_number)
    start_override = os.getenv("LOGS_FROM_BLOCK")
    end_override = os.getenv("LOGS_TO_BLOCK")

    if start_override:
        start_block = int(start_override)
    if end_override:
        end_block = int(end_override)

    print(f"Scanning logs blocks: {start_block}..{end_block}")

    abi = load_abi()
    contract = w3.eth.contract(address=contract_addr, abi=abi)

    # Map (event -> (category,function,label_builder))
    # label_builder receives event args and returns a short label
    def _lbl(*parts):  # small helper
        return ",".join([p for p in parts if p])

    EVENT_SPECS: List[Tuple[object, str, str, object]] = [
        (contract.events.HospitalRegistered, "Governance setup", "registerHospital",
         lambda a: _lbl(f"hospital={a.get('hospital','')}")),
        (contract.events.MedicalTeamRegistered, "Governance setup", "registerMedicalTeam",
         lambda a: _lbl(f"medicalTeam={a.get('medicalTeam','')}")),
        (contract.events.EthicalCommitteeMemberRegistered, "Governance setup", "registerEthicalCommittee",
         lambda a: _lbl(f"member={a.get('committeeMember','')}")),
        (contract.events.LLMRegistered, "Governance setup", "registerLLM",
         lambda a: _lbl(f"llm={a.get('llmAddress','')}")),

        (contract.events.DonorAddressRegistered, "Identity binding", "registerDonorAddress",
         lambda a: _lbl(f"donorId={a.get('donorId','')}", f"addr={a.get('donorAddress','')}")),
        (contract.events.RecipientAddressRegistered, "Identity binding", "registerRecipientAddress",
         lambda a: _lbl(f"recipientId={a.get('recipientId','')}", f"addr={a.get('recipientAddress','')}")),

        (contract.events.DonorRegistered, "Profile registration", "registerDonor",
         lambda a: _lbl(f"donorId={a.get('donorId','')}", f"organ={a.get('organType','')}")),
        (contract.events.RecipientRegistered, "Profile registration", "registerRecipient",
         lambda a: _lbl(f"recipientId={a.get('recipientId','')}", f"organ={a.get('organType','')}")),

        (contract.events.MatchCreated, "Match workflow", "createMatch",
         lambda a: _lbl(f"matchId={a.get('matchId','')}", f"donorId={a.get('donorId','')}",
                        f"primary={a.get('primaryRecipientId','')}", f"backup={a.get('backupRecipientId','')}")),

        (contract.events.MatchFinalized, "Match workflow", "finalizeMatch",
         lambda a: _lbl(f"matchId={a.get('matchId','')}", f"approved={a.get('approved','')}")),
    ]

    # EthicalApprovalGranted needs routing by entityType
    def add_ethical(txh: str, args: Dict, out_rows: List[Dict]) -> None:
        et = str(args.get("entityType", ""))
        if et == "Donor":
            out_rows.append({
                "category": "Eligibility approvals",
                "function": "approveDonorEthicalCommittee",
                "tx_hash": txh,
                "label": f"id={args.get('id','')}",
            })
        elif et == "Recipient":
            out_rows.append({
                "category": "Eligibility approvals",
                "function": "approveRecipientEthicalCommittee",
                "tx_hash": txh,
                "label": f"id={args.get('id','')}",
            })

    # ApprovalGranted needs routing by approvedBy string
    def add_approval(txh: str, args: Dict, out_rows: List[Dict]) -> None:
        who = str(args.get("approvedBy", ""))
        fn_map = {
            "Medical Team": "approveMedicalTeam",
            "Hospital": "approveHospital",
            "Donor": "approveDonor",
            "Recipient": "approveRecipient",
            "Ethical Committee": "approveFinalTransplant",
        }
        fn = fn_map.get(who)
        if fn:
            out_rows.append({
                "category": "Match workflow",
                "function": fn,
                "tx_hash": txh,
                "label": f"matchId={args.get('matchId','')}",
            })

    # Collect rows
    rows: List[Dict[str, str]] = []

    # Always include deployment as the first row (for cost tables)
    rows.append({
        "category": "Deployment",
        "function": "deployContract",
        "tx_hash": deploy_tx,
        "label": f"contract={contract_addr}",
    })

    # 1) Fixed-spec events
    for evt_cls, cat, fn, label_fn in EVENT_SPECS:
        sig = evt_cls().abi["name"] + "(" + ",".join([i["type"] for i in evt_cls().abi["inputs"]]) + ")"
        topic0 = w3.keccak(text=sig).hex()
        logs = collect_logs(w3, contract_addr, topic0, start_block, end_block)
        for lg in logs:
            ev = evt_cls().process_log(lg)
            txh = lg["transactionHash"].hex()
            rows.append({
                "category": cat,
                "function": fn,
                "tx_hash": "0x" + txh,
                "label": label_fn(dict(ev["args"])),
            })

    # 2) Ethical approvals (route by entityType)
    ethical_sig = contract.events.EthicalApprovalGranted().abi["name"] + "(" + ",".join(
        [i["type"] for i in contract.events.EthicalApprovalGranted().abi["inputs"]]
    ) + ")"
    ethical_topic0 = w3.keccak(text=ethical_sig).hex()
    ethical_logs = collect_logs(w3, contract_addr, ethical_topic0, start_block, end_block)
    for lg in ethical_logs:
        ev = contract.events.EthicalApprovalGranted().process_log(lg)
        txh = "0x" + lg["transactionHash"].hex()
        add_ethical(txh, dict(ev["args"]), rows)

    # 3) Approvals (route by approvedBy)
    appr_sig = contract.events.ApprovalGranted().abi["name"] + "(" + ",".join(
        [i["type"] for i in contract.events.ApprovalGranted().abi["inputs"]]
    ) + ")"
    appr_topic0 = w3.keccak(text=appr_sig).hex()
    appr_logs = collect_logs(w3, contract_addr, appr_topic0, start_block, end_block)
    for lg in appr_logs:
        ev = contract.events.ApprovalGranted().process_log(lg)
        txh = "0x" + lg["transactionHash"].hex()
        add_approval(txh, dict(ev["args"]), rows)

    # De-duplicate while preserving order (same tx can emit multiple events)
    seen = set()
    deduped: List[Dict[str, str]] = []
    for r in rows:
        key = (r["category"], r["function"], r["tx_hash"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # Write manifest CSV
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "function", "tx_hash", "label"])
        w.writeheader()
        for r in deduped:
            w.writerow(r)

    print(f"Wrote manifest: {MANIFEST_CSV}")
    print(f"Contract: {contract_addr}")
    print(f"Blocks: {start_block}..{end_block}")
    print(f"Rows: {len(deduped)}")


if __name__ == "__main__":
    main()