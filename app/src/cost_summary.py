import csv
import json
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import Dict, Any, List, Tuple

APP_DIR = Path(__file__).resolve().parents[1]
PIPE_DIR = APP_DIR / "pipeline-output"
IN_TX_LOG = PIPE_DIR / "tx_log.csv"
OUT_DIR = PIPE_DIR
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
    # cost_token = gas * baseFee(gwei) * 1e-9
    cost_token = gas_used * base_fee_gwei * 1e-9
    return cost_token * token_usd

def load_tx_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing tx log: {path}. Run the pipeline at least once to generate tx_log.csv.")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # Skip rows without gas_used
            g = row.get("gas_used", "").strip()
            if not g:
                continue
            try:
                gas_used = int(float(g))
            except Exception:
                continue
            rows.append({
                "timestamp_utc": row.get("timestamp_utc", ""),
                "network": row.get("network", ""),
                "role": row.get("role", ""),
                "function": row.get("function", ""),
                "tx_hash": row.get("tx_hash", ""),
                "gas_used": gas_used,
                "notes": row.get("notes", ""),
            })
    return rows

def group_by_function(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for r in rows:
        fn = (r.get("function") or "").strip() or "unknown"
        groups.setdefault(fn, []).append(int(r["gas_used"]))
    return groups

def summarize_gas(gases: List[int]) -> Dict[str, Any]:
    return {
        "count": len(gases),
        "min_gas": min(gases),
        "avg_gas": int(round(mean(gases))),
        "max_gas": max(gases),
    }

def main():
    rows = load_tx_log(IN_TX_LOG)
    groups = group_by_function(rows)

    stamp = ts_utc()
    out_csv = OUT_DIR / f"cost_summary_{stamp}.csv"
    out_json = OUT_DIR / f"cost_summary_{stamp}.json"

    # CSV columns
    fields = [
        "function",
        "count",
        "min_gas",
        "avg_gas",
        "max_gas",
    ]
    # Add per-chain USD columns (avg gas)
    for chain in CHAIN_PARAMS.keys():
        fields.append(f"{chain}_usd_at_basefee")

    summary_rows = []
    json_obj: Dict[str, Any] = {
        "assumptions_date": "09-Mar-2026",
        "note": "USD estimates use gas_used * base_fee_gwei (execution gas only). For L2s (Arbitrum/Optimism/zkSync), this excludes L1 data posting fees; treat as lower-bound.",
        "chain_params": CHAIN_PARAMS,
        "input_tx_log": str(IN_TX_LOG),
        "functions": {},
    }

    for fn, gases in sorted(groups.items()):
        s = summarize_gas(gases)
        row_out = {
            "function": fn,
            **s,
        }
        # USD estimates using avg gas
        for chain, p in CHAIN_PARAMS.items():
            usd = gas_to_usd(s["avg_gas"], p["base_fee_gwei"], p["token_usd"])
            row_out[f"{chain}_usd_at_basefee"] = round(usd, 8)  # keep small values visible
        summary_rows.append(row_out)

        # JSON function detail
        json_obj["functions"][fn] = {
            **s,
            "usd_estimates_avg_gas": {
                chain: round(gas_to_usd(s["avg_gas"], p["base_fee_gwei"], p["token_usd"]), 8)
                for chain, p in CHAIN_PARAMS.items()
            }
        }

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # Write JSON
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_json)

if __name__ == "__main__":
    main()