import os
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from web3 import Web3

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "final_cost_table_observed_from_chain.csv"
OUT_JSON = OUT_DIR / "final_cost_table_observed_from_chain.json"

# As of 09-Mar-2026 (your provided values)
CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon": {"token": "MATIC", "token_usd": 0.177, "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}

def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    return gas_used * float(base_fee_gwei) * 1e-9 * float(token_usd)

# Put your seeding + approval tx hashes here (no 0x prefix required, both ok)
TX_MAP: List[Tuple[str, str]] = [
    ("registerHospital", "0148d6b3cce2198b1ccfa3ca146c3453e096c96ed667292f961e8f94fd0b4dc8"),
    ("registerEthicalCommittee", "2a20062d79c76ef3b0ec0650e1ba3c7743a4206621cc75c3e71e0dd354bc2f14"),
    ("registerMedicalTeam", "37b71bfe5e15fa94db17302e158069c66fa16366fbf5c4ef082c65042ebc1789"),
    ("registerLLM", "e38ca66e01e0f009d7406374ef1429255b9730b8feb9648682cb692836d5bd53"),

    ("registerDonorAddress", "a768228bea4f30a15e95a6a479ea16316208550bb969c9cf2c41c53c67e194e7"),
    ("registerRecipientAddress", "0c462cf6e5f1648e22516fa11afa0760a89edb80d402298fac55df874ae1997b"),
    ("registerRecipientAddress", "1db484de9b35b94292ea735a5029c7eb76d574d20127b79b2eecd5eb2ac86ccb"),

    ("registerDonor", "431161bdb3425b9707b47201eb057e9c9bb88c9b2c43aa535ac5de245abd5bab"),
    ("registerRecipient", "79a4a192fb2aeeaf3bec83a57d267acacd54e5d8c7133462f7491a9a228aa2a2"),
    ("registerRecipient", "357d8a97ae2d789d2ecb2f26a785f684b56f7518f12a2498b80bb10aab67fefd"),

    ("approveDonorEthicalCommittee", "da07fa2645cd1ad544dccc072343807a4f687e4312bb864ac78c0eede1269951"),
    ("approveRecipientEthicalCommittee", "09b118f30a4948fce79f900622489760926ada4b79821ee8236015d2b24bcdd8"),
    ("approveRecipientEthicalCommittee", "56a521df3ed19577f880c93a99f7a37f0e4b0a2a3e9c549a62c65e455b97a658"),
]

def normalize_tx(tx: str) -> str:
    tx = tx.strip()
    if not tx.startswith("0x"):
        tx = "0x" + tx
    return tx

def main():
    load_dotenv(APP_DIR / ".env")

    rpc = os.getenv("SEPOLIA_RPC_URL", "").strip()
    if not rpc:
        raise RuntimeError("Missing SEPOLIA_RPC_URL in app/.env")

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia")

    rows = []
    for fn, txh in TX_MAP:
        txh = normalize_tx(txh)
        receipt = w3.eth.get_transaction_receipt(txh)
        gas_used = int(receipt["gasUsed"])

        row = {
            "function": fn,
            "gas_used": gas_used,
            "source": "observed_receipt",
            "note": f"tx={txh}",
        }
        for chain, params in CHAIN_PARAMS.items():
            row[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(gas_used, params["base_fee_gwei"], params["token_usd"]), 8
            )
        rows.append(row)

    # Also include latest createMatch from tx_log if present
    tx_log = OUT_DIR / "tx_log.csv"
    if tx_log.exists():
        with tx_log.open("r", encoding="utf-8") as f:
            tx_rows = list(csv.DictReader(f))
        # pick latest createMatch entry
        create = [r for r in tx_rows if (r.get("function") == "createMatch" and r.get("tx_hash"))]
        if create:
            latest = sorted(create, key=lambda r: r.get("timestamp_utc", ""))[-1]
            txh = normalize_tx(latest["tx_hash"])
            receipt = w3.eth.get_transaction_receipt(txh)
            gas_used = int(receipt["gasUsed"])
            row = {
                "function": "createMatch",
                "gas_used": gas_used,
                "source": "observed_receipt",
                "note": f"tx={txh}",
            }
            for chain, params in CHAIN_PARAMS.items():
                row[f"{chain}_usd_at_basefee"] = round(
                    gas_to_usd(gas_used, params["base_fee_gwei"], params["token_usd"]), 8
                )
            rows.append(row)

    # Write outputs
    fieldnames = ["function", "gas_used", "source", "note"] + [
        f"{chain}_usd_at_basefee" for chain in CHAIN_PARAMS.keys()
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "network_receipts_source": "sepolia",
        "chains": CHAIN_PARAMS,
        "rows": rows,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print(f"Rows: {len(rows)}")

if __name__ == "__main__":
    main()