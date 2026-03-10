#!/usr/bin/env python3
"""
Prepare receipt-observed gas + cost tables for the Sepolia validation run.

Outputs (written to app/pipeline-output/):
  - final_cost_table.csv                 (aggregated by category+function + subtotals + total)
  - final_cost_table.json                (same data + full tx hash lists)
  - final_cost_table_detailed.csv        (one row per transaction receipt)

How it works:
  - Fetches receipts from Sepolia for a manifest of tx hashes (includes deployment).
  - Uses ONLY receipt-observed gasUsed.
  - Computes USD-at-basefee for multiple chains using fixed assumptions (same as your existing files).
  - Aggregates repeated function calls (e.g., multiple registerRecipient calls) within a category.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from web3 import Web3

# -----------------------
# Paths
# -----------------------
APP_DIR = Path(__file__).resolve().parents[1]  # script lives under app/...
OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_AGG_CSV = OUT_DIR / "final_cost_table.csv"
OUT_AGG_JSON = OUT_DIR / "final_cost_table.json"
OUT_DETAILED_CSV = OUT_DIR / "final_cost_table_detailed.csv"

# Optional: if you create this file, the script will use it instead of DEFAULT_TX_ITEMS
# Format:
#   category,function,tx_hash,label(optional)
MANIFEST_CSV = OUT_DIR / "tx_manifest.csv"

# -----------------------
# Cost assumptions (your provided values)
# -----------------------
CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon": {"token": "MATIC", "token_usd": 0.177, "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}


def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    # USD = gas * (gwei * 1e-9 ETH/gas) * USD/ETH
    return gas_used * float(base_fee_gwei) * 1e-9 * float(token_usd)


def normalize_tx(tx: str) -> str:
    tx = (tx or "").strip()
    if not tx:
        return tx
    return tx if tx.startswith("0x") else "0x" + tx


@dataclass(frozen=True)
class TxItem:
    category: str
    function: str
    tx_hash: str
    label: str = ""  # optional, for human readability


# -----------------------
# Default tx manifest
# -----------------------
# IMPORTANT: This is the subset you listed in your paper tables + approvals + deployment.
# If you want to include more recipients (r3, r5, ...), add them here or create tx_manifest.csv.
DEFAULT_TX_ITEMS: List[TxItem] = [
    # Deployment (provided by you)
    TxItem("Deployment", "deployContract", "0x4b876f900c2553d37ad7a78aaf2a93a57c2045e54e611a907b556ffbc695814d", "Contract deployment"),

    # Governance setup
    TxItem("Governance setup", "registerHospital", "0x0148d6b3cce2198b1ccfa3ca146c3453e096c96ed667292f961e8f94fd0b4dc8", "Hospital"),
    TxItem("Governance setup", "registerEthicalCommittee", "0x2a20062d79c76ef3b0ec0650e1ba3c7743a4206621cc75c3e71e0dd354bc2f14", "Ethics"),
    TxItem("Governance setup", "registerMedicalTeam", "0x37b71bfe5e15fa94db17302e158069c66fa16366fbf5c4ef082c65042ebc1789", "Medical team"),
    TxItem("Governance setup", "registerLLM", "0xe38ca66e01e0f009d7406374ef1429255b9730b8feb9648682cb692836d5bd53", "Authorized LLM"),

    # Identity binding (subset used for traceability in the paper tables)
    TxItem("Identity binding", "registerDonorAddress", "0xa768228bea4f30a15e95a6a479ea16316208550bb969c9cf2c41c53c67e194e7", "donor"),
    TxItem("Identity binding", "registerRecipientAddress", "0x0c462cf6e5f1648e22516fa11afa0760a89edb80d402298fac55df874ae1997b", "r1"),
    TxItem("Identity binding", "registerRecipientAddress", "0x1db484de9b35b94292ea735a5029c7eb76d574d20127b79b2eecd5eb2ac86ccb", "r2"),
    TxItem("Identity binding", "registerRecipientAddress", "0x9a12db2bb2d74b64d4dc009b47597c8ae77405df6c680f3b91d4b2469c341c6e", "r4"),
    TxItem("Identity binding", "registerRecipientAddress", "0x58cb17149761df23e21288e8c6412b8c83da197ae070b482b0dc1c49fbbcf722", "r7"),

    # Profile registration (subset)
    TxItem("Profile registration", "registerDonor", "0x431161bdb3425b9707b47201eb057e9c9bb88c9b2c43aa535ac5de245abd5bab", "donor"),
    TxItem("Profile registration", "registerRecipient", "0x79a4a192fb2aeeaf3bec83a57d267acacd54e5d8c7133462f7491a9a228aa2a2", "r1"),
    TxItem("Profile registration", "registerRecipient", "0x357d8a97ae2d789d2ecb2f26a785f684b56f7518f12a2498b80bb10aab67fefd", "r2"),
    TxItem("Profile registration", "registerRecipient", "0x01083eace6f59dd3df2f683db3eb212885be623181d09eadf1bf1b06c9eaaf26", "r4"),
    TxItem("Profile registration", "registerRecipient", "0xe25761b1bf2dd284d53c2daa5c71f187fe810f59edf9c22d2591c66a7f2c8321", "r7"),

    # Eligibility approvals (subset)
    TxItem("Eligibility approvals", "approveDonorEthicalCommittee", "0xda07fa2645cd1ad544dccc072343807a4f687e4312bb864ac78c0eede1269951", "donor"),
    TxItem("Eligibility approvals", "approveRecipientEthicalCommittee", "0x09b118f30a4948fce79f900622489760926ada4b79821ee8236015d2b24bcdd8", "r1"),
    TxItem("Eligibility approvals", "approveRecipientEthicalCommittee", "0x56a521df3ed19577f880c93a99f7a37f0e4b0a2a3e9c549a62c65e455b97a658", "r2"),
    TxItem("Eligibility approvals", "approveRecipientEthicalCommittee", "0x39dbea3a82aaeff8aee5c50c4f25482cab3b84b011a9d07f8f5043c6ae9e6b9d", "r4"),
    TxItem("Eligibility approvals", "approveRecipientEthicalCommittee", "0xaa390833e6a7cc7f4540e7645dbbc2ca964fe74838ca26a487f6412a32ab3d20", "r7"),

    # Match workflow
    TxItem("Match workflow", "createMatch", "0x9352830f8447ee88b9c3b52f99b5aa312b00303ea1a98ecd112aee06180f8260", "match record"),
    TxItem("Match workflow", "approveMedicalTeam", "0x3b804a4b134595800c779200b3356c73c40fb42c6d88a0a45c7c06832a828d1a", ""),
    TxItem("Match workflow", "approveHospital", "0x45b4adb0f40cc8c53aa6203d4943192c0fca8246d29aad9078a9d1a5bbb2b200", ""),
    TxItem("Match workflow", "approveDonor", "0x960f241ce46ba0528bb0c478502e836ff24fc819458970675b59b1f992918cce", ""),
    TxItem("Match workflow", "approveRecipient", "0x100d2ef061662e0b0790445d88bbd74972d2817e973f089045293e4702761cae", ""),
    TxItem("Match workflow", "approveFinalTransplant", "0xab1e5167bd1002d12100633f5cf8577510271ba57999f54f4f5e08229227abf3", ""),
    TxItem("Match workflow", "finalizeMatch", "0x588425e76a56a997e847c3909ca9a1a80ab08c85269fe58458995a42a480938f", ""),
]


def load_manifest() -> List[TxItem]:
    if not MANIFEST_CSV.exists():
        return DEFAULT_TX_ITEMS

    items: List[TxItem] = []
    with MANIFEST_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"category", "function", "tx_hash"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"{MANIFEST_CSV} must have columns: category,function,tx_hash[,label]"
            )
        for r in reader:
            items.append(
                TxItem(
                    category=(r.get("category") or "").strip(),
                    function=(r.get("function") or "").strip(),
                    tx_hash=normalize_tx(r.get("tx_hash") or ""),
                    label=(r.get("label") or "").strip(),
                )
            )
    return items


def short_hash(txh: str, n: int = 10) -> str:
    txh = normalize_tx(txh)
    if len(txh) <= 2 + n:
        return txh
    return txh[: 2 + n] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rpc-env",
        default="SEPOLIA_RPC_URL",
        help="Env var containing the Sepolia RPC URL (default: SEPOLIA_RPC_URL)",
    )
    args = parser.parse_args()

    load_dotenv(APP_DIR / ".env")

    rpc = (os.getenv(args.rpc_env) or "").strip()
    if not rpc:
        raise RuntimeError(f"Missing {args.rpc_env} in {APP_DIR / '.env'}")

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Not connected to Sepolia via RPC")

    tx_items = load_manifest()

    detailed_rows: List[Dict[str, object]] = []
    for it in tx_items:
        if not it.tx_hash:
            continue

        txh = normalize_tx(it.tx_hash)
        receipt = w3.eth.get_transaction_receipt(txh)

        gas_used = int(receipt["gasUsed"])
        eff_price_wei = int(receipt.get("effectiveGasPrice", 0))  # may be missing on some RPCs
        fee_eth = (gas_used * eff_price_wei) / 1e18 if eff_price_wei else None

        contract_addr = receipt.get("contractAddress")
        note_bits = []
        if it.label:
            note_bits.append(f"label={it.label}")
        note_bits.append(f"tx={txh}")
        if contract_addr:
            note_bits.append(f"contract={contract_addr}")

        row: Dict[str, object] = {
            "category": it.category,
            "function": it.function,
            "tx_hash": txh,
            "gas_used": gas_used,
            "effective_gas_price_gwei": (eff_price_wei / 1e9) if eff_price_wei else "",
            "fee_eth": fee_eth if fee_eth is not None else "",
            "source": "observed_receipt",
            "note": "; ".join(note_bits),
        }

        for chain, params in CHAIN_PARAMS.items():
            row[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(gas_used, params["base_fee_gwei"], params["token_usd"]), 8
            )

        detailed_rows.append(row)

    # -------- Aggregate by (category, function) --------
    agg: Dict[Tuple[str, str], Dict[str, object]] = {}
    tx_hashes_by_key: Dict[Tuple[str, str], List[str]] = {}

    for r in detailed_rows:
        key = (str(r["category"]), str(r["function"]))
        tx_hashes_by_key.setdefault(key, []).append(str(r["tx_hash"]))
        if key not in agg:
            agg[key] = {
                "category": key[0],
                "function": key[1],
                "tx_count": 0,
                "gas_used": 0,
                "source": "observed_receipt",
                "note": "",
            }
            for chain in CHAIN_PARAMS.keys():
                agg[key][f"{chain}_usd_at_basefee"] = 0.0

        agg[key]["tx_count"] = int(agg[key]["tx_count"]) + 1
        agg[key]["gas_used"] = int(agg[key]["gas_used"]) + int(r["gas_used"])
        for chain, params in CHAIN_PARAMS.items():
            agg[key][f"{chain}_usd_at_basefee"] = round(
                float(agg[key][f"{chain}_usd_at_basefee"])
                + gas_to_usd(int(r["gas_used"]), params["base_fee_gwei"], params["token_usd"]),
                8,
            )

    # Add readable notes with hashes (shortened) and keep full list in JSON
    agg_rows: List[Dict[str, object]] = []
    for (cat, fn), row in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        txs = tx_hashes_by_key.get((cat, fn), [])
        if len(txs) <= 2:
            note = "txs=" + ",".join(short_hash(t) for t in txs)
        else:
            note = "txs=" + ",".join(short_hash(t) for t in txs[:2]) + f",+{len(txs)-2} more"
        row["note"] = note
        agg_rows.append(row)

    # -------- Subtotals per category + grand total --------
    def make_total_row(category: str, label: str, rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
        total_gas = sum(int(r["gas_used"]) for r in rows)
        total_count = sum(int(r.get("tx_count", 0)) for r in rows)
        out: Dict[str, object] = {
            "category": category,
            "function": label,
            "tx_count": total_count,
            "gas_used": total_gas,
            "source": "observed_receipt",
            "note": "",
        }
        for chain, params in CHAIN_PARAMS.items():
            out[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(total_gas, params["base_fee_gwei"], params["token_usd"]), 8
            )
        return out

    # group agg_rows by category
    by_cat: Dict[str, List[Dict[str, object]]] = {}
    for r in agg_rows:
        by_cat.setdefault(str(r["category"]), []).append(r)

    final_rows: List[Dict[str, object]] = []
    for cat in sorted(by_cat.keys()):
        final_rows.extend(by_cat[cat])
        final_rows.append(make_total_row(cat, "SUBTOTAL", by_cat[cat]))

    final_rows.append(make_total_row("TOTAL", "TOTAL", agg_rows))

    # -------- Write CSVs --------
    usd_cols = [f"{c}_usd_at_basefee" for c in CHAIN_PARAMS.keys()]

    agg_fieldnames = ["category", "function", "tx_count", "gas_used", "source", "note"] + usd_cols
    with OUT_AGG_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fieldnames)
        w.writeheader()
        for r in final_rows:
            w.writerow({k: r.get(k, "") for k in agg_fieldnames})

    det_fieldnames = [
        "category",
        "function",
        "tx_hash",
        "gas_used",
        "effective_gas_price_gwei",
        "fee_eth",
        "source",
        "note",
    ] + usd_cols
    with OUT_DETAILED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=det_fieldnames)
        w.writeheader()
        for r in detailed_rows:
            w.writerow({k: r.get(k, "") for k in det_fieldnames})

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "network_receipts_source": "sepolia",
        "assumptions": {"chains": CHAIN_PARAMS},
        "manifest_source": str(MANIFEST_CSV) if MANIFEST_CSV.exists() else "DEFAULT_TX_ITEMS",
        "detailed_rows": detailed_rows,
        "aggregated_rows": final_rows,
        "tx_hashes_by_category_function": {
            f"{cat}::{fn}": txs for (cat, fn), txs in tx_hashes_by_key.items()
        },
    }
    with OUT_AGG_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:")
    print(f"  {OUT_AGG_CSV}")
    print(f"  {OUT_DETAILED_CSV}")
    print(f"  {OUT_AGG_JSON}")
    print(f"Detailed txs: {len(detailed_rows)} | Aggregated rows (incl. subtotals): {len(final_rows)}")


if __name__ == "__main__":
    main()