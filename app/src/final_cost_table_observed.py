import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
TX_LOG = OUT_DIR / "tx_log.csv"
MATCH_RUNS = OUT_DIR / "match_runs.csv"

OUT_CSV = OUT_DIR / "final_cost_table.csv"
OUT_JSON = OUT_DIR / "final_cost_table.json"

# As of 09-Mar-2026 (your provided values)
CHAIN_PARAMS = {
    "ethereum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.04},
    "polygon": {"token": "MATIC", "token_usd": 0.177, "base_fee_gwei": 146.00},
    "arbitrum": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.02},
    "optimism": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.01},
    "zksync_era": {"token": "ETH", "token_usd": 1944.53, "base_fee_gwei": 0.05},
}

# If you want to force “one row per function”, keep only latest tx per function
KEEP_LATEST_PER_FUNCTION = True


def gas_to_usd(gas_used: int, base_fee_gwei: float, token_usd: float) -> float:
    # cost = gas * base_fee(gwei) * 1e-9 ETH * token_usd
    return gas_used * float(base_fee_gwei) * 1e-9 * float(token_usd)


def load_tx_log_rows() -> List[Dict]:
    if not TX_LOG.exists():
        raise RuntimeError(f"Missing {TX_LOG}")
    with TX_LOG.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_match_runs_rows() -> List[Dict]:
    if not MATCH_RUNS.exists():
        return []
    with MATCH_RUNS.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_latest_by_timestamp(rows: List[Dict], ts_field: str = "timestamp_utc") -> Dict[str, Dict]:
    """
    Returns dict function -> row (latest by timestamp).
    """
    best: Dict[str, Tuple[str, Dict]] = {}
    for r in rows:
        fn = (r.get("function") or "").strip()
        ts = (r.get(ts_field) or "").strip()
        if not fn or not ts:
            continue
        if fn not in best or ts > best[fn][0]:
            best[fn] = (ts, r)
    return {fn: item[1] for fn, item in best.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tx_rows = load_tx_log_rows()
    # Normalize + filter only rows that have gas_used and tx_hash
    cleaned = []
    for r in tx_rows:
        gas = r.get("gas_used")
        txh = (r.get("tx_hash") or "").strip()
        fn = (r.get("function") or "").strip()
        if not fn or not txh:
            continue
        try:
            gas_i = int(gas) if gas not in (None, "", "None") else None
        except Exception:
            gas_i = None
        if gas_i is None:
            continue
        cleaned.append({**r, "gas_used": gas_i, "tx_hash": txh, "function": fn})

    # Also include createMatch from match_runs.csv if present and not already in tx_log
    match_rows = load_match_runs_rows()
    for mr in match_rows:
        fn = "createMatch"
        txh = (mr.get("tx_hash") or "").strip()
        gas = mr.get("gas_used")
        if not txh:
            continue
        try:
            gas_i = int(gas) if gas not in (None, "", "None") else None
        except Exception:
            gas_i = None
        if gas_i is None:
            continue
        cleaned.append({
            "timestamp_utc": mr.get("timestamp_utc", ""),
            "network": "sepolia",
            "role": "LLM",
            "function": fn,
            "tx_hash": txh,
            "gas_used": gas_i,
            "notes": f"match_id={mr.get('match_id','')},cid={mr.get('match_cid','')}",
        })

    if not cleaned:
        raise RuntimeError("No observed transactions found in tx_log.csv or match_runs.csv")

    # Optionally keep only the latest tx per function
    if KEEP_LATEST_PER_FUNCTION:
        latest_map = pick_latest_by_timestamp(cleaned, ts_field="timestamp_utc")
        observed = list(latest_map.values())
    else:
        observed = cleaned

    observed.sort(key=lambda r: r.get("function", ""))

    # Build output rows
    out_rows = []
    for r in observed:
        fn = r["function"]
        gas_used = int(r["gas_used"])
        txh = r["tx_hash"]

        row = {
            "function": fn,
            "gas_used": gas_used,
            "source": "observed_receipt",
            "note": f"tx={txh}",
        }
        for chain, params in CHAIN_PARAMS.items():
            row[f"{chain}_usd_at_basefee"] = round(
                gas_to_usd(gas_used, params["base_fee_gwei"], params["token_usd"]),
                8
            )
        out_rows.append(row)

    # Write CSV (overwrite)
    fieldnames = ["function", "gas_used", "source", "note"] + [
        f"{chain}_usd_at_basefee" for chain in CHAIN_PARAMS.keys()
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    # Write JSON (overwrite)
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "observed_only",
        "keep_latest_per_function": KEEP_LATEST_PER_FUNCTION,
        "chains": CHAIN_PARAMS,
        "rows": out_rows,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print(f"Functions included: {len(out_rows)}")


if __name__ == "__main__":
    main()