import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

APP_DIR = Path(__file__).resolve().parents[1]
PIPE_DIR = APP_DIR / "pipeline-output"

TX_LOG = PIPE_DIR / "tx_log.csv"

# pick your latest estimate file explicitly or change this to auto-find newest
GAS_EST_FILE = PIPE_DIR / "gas_estimates_20260308T232701Z.csv"

def ts_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# 09-Mar-2026 assumptions (same as cost_summary)
CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon":  {"token": "MATIC", "token_usd": 0.177,   "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}

def sanitize_note(note: str) -> str:
    """Shrink noisy revert payloads to readable messages."""
    if not note:
        return ""
    # Common revert strings from web3 / solidity
    prefixes = [
        "('execution reverted: ",
        "execution reverted: ",
    ]
    for p in prefixes:
        if note.startswith(p):
            note = note[len(p):]
            break

    # If it looks like "Message', '0x...."
    # Keep only the first quoted part if present
    if "', '0x" in note:
        note = note.split("', '0x", 1)[0]

    # Trim trailing quotes/parentheses/commas
    return note.strip(" '),\"")

def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    return gas_used * base_fee_gwei * 1e-9 * token_usd

def load_latest_observed_create_match() -> Optional[Dict[str, Any]]:
    if not TX_LOG.exists():
        return None
    rows = []
    with TX_LOG.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("function") or "") == "createMatch":
                g = row.get("gas_used", "").strip()
                if not g:
                    continue
                try:
                    gas_used = int(float(g))
                except Exception:
                    continue
                rows.append({
                    "tx_hash": row.get("tx_hash", ""),
                    "gas_used": gas_used,
                    "notes": row.get("notes", ""),
                    "timestamp_utc": row.get("timestamp_utc", ""),
                })
    return rows[-1] if rows else None

def load_estimates() -> Dict[str, Dict[str, Any]]:
    if not GAS_EST_FILE.exists():
        raise RuntimeError(f"Missing {GAS_EST_FILE}")
    out = {}
    with GAS_EST_FILE.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row["function"]
            gas = row.get("estimated_gas", "").strip()
            err = row.get("error", "").strip()
            out[fn] = {"estimated_gas": int(gas) if gas else None, "error": err}
    return out

def main():
    estimates = load_estimates()
    observed = load_latest_observed_create_match()

    stamp = ts_utc()
    out_csv = PIPE_DIR / f"final_cost_table_{stamp}.csv"
    out_json = PIPE_DIR / f"final_cost_table_{stamp}.json"

    # choose createMatch gas: observed if available else estimate
    create_match_gas = None
    create_match_source = "estimate"
    create_match_tx = ""
    if observed:
        create_match_gas = observed["gas_used"]
        create_match_source = "observed_receipt"
        create_match_tx = observed["tx_hash"]
    else:
        create_match_gas = estimates.get("createMatch", {}).get("estimated_gas")
        create_match_source = "estimate"

    # Build final rows
    functions_order = [
        "registerHospital",
        "registerEthicalCommittee",
        "registerMedicalTeam",
        "registerLLM",
        "registerDonorAddress",
        "registerRecipientAddress",
        "registerDonor",
        "registerRecipient",
        "approveDonorEthicalCommittee",
        "approveRecipientEthicalCommittee",
        "createMatch",
        "approveMedicalTeam",
        "approveHospital",
        "approveDonor",
        "approveRecipient",
        "approveFinalTransplant",
    ]

    rows_out: List[Dict[str, Any]] = []

    def add(fn: str, gas: Optional[int], source: str, note: str):
        row = {
            "function": fn,
            "gas_used": "" if gas is None else gas,
            "source": source,
            "note": sanitize_note(note),
        }
        for chain, p in CHAIN_PARAMS.items():
            row[f"{chain}_usd_at_basefee"] = "" if gas is None else round(gas_to_usd(gas, p["base_fee_gwei"], p["token_usd"]), 8)
        rows_out.append(row)

    for fn in functions_order:
        if fn == "createMatch":
            add("createMatch", create_match_gas, create_match_source, f"tx={create_match_tx}" if create_match_tx else "")
            continue

        est_g = estimates.get(fn, {}).get("estimated_gas")
        err = estimates.get(fn, {}).get("error", "")
        if est_g is None:
            # not estimatable on current state
            add(fn, None, "n/a", err or "not estimated")
        else:
            add(fn, est_g, "estimate", "")

    # write CSV
    fields = ["function","gas_used","source","note"] + [f"{c}_usd_at_basefee" for c in CHAIN_PARAMS.keys()]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fields})

    obj = {
        "assumptions_date": "09-Mar-2026",
        "note": "USD estimates use base_fee_gwei * gas (execution gas only). L2 estimates exclude L1 data posting fees; treat as lower-bound.",
        "inputs": {
            "tx_log": str(TX_LOG),
            "gas_estimates": str(GAS_EST_FILE),
        },
        "createMatch": {
            "gas_used": create_match_gas,
            "source": create_match_source,
            "tx_hash": create_match_tx,
            "observed_row": observed,
        },
        "rows": rows_out,
        "chain_params": CHAIN_PARAMS,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_json)

if __name__ == "__main__":
    main()