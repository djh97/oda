import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "tx_log.csv"

FIELDS = [
    "timestamp_utc",
    "network",
    "role",
    "function",
    "tx_hash",
    "gas_used",
    "notes",
]

def append_tx(network: str, role: str, function: str, tx_hash: str, gas_used: Optional[int], notes: str = "") -> str:
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "network": network,
            "role": role,
            "function": function,
            "tx_hash": tx_hash,
            "gas_used": "" if gas_used is None else int(gas_used),
            "notes": notes,
        })
    return str(CSV_PATH)