import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
TX_LOG = OUT_DIR / "tx_log.csv"

FIELDNAMES = ["timestamp_utc", "network", "role", "function", "tx_hash", "gas_used", "notes"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()


def append_tx(network: str, role: str, function: str, tx_hash: str, gas_used: Optional[int], notes: str = "", path: Path = TX_LOG):
    _ensure_file(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writerow(
            {
                "timestamp_utc": _now_utc(),
                "network": network,
                "role": role,
                "function": function,
                "tx_hash": tx_hash,
                "gas_used": gas_used if gas_used is not None else "",
                "notes": notes,
            }
        )