import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "match_runs.csv"

FIELDS = [
    "timestamp_utc",
    "donor_id",
    "baseline_top1",
    "llm_primary",
    "llm_backup",
    "overrode_baseline",
    "override_reason",
    "match_cid",
    "tx_hash",
    "match_id",
    "gas_used",
]

def append_match_row(row: Dict[str, Any]) -> str:
    """
    Appends one row to pipeline-output/match_runs.csv
    Creates file with header if missing.
    Returns path as string.
    """
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return str(CSV_PATH)

def make_row(
    donor_id: int,
    baseline_top1: int,
    llm_primary: int,
    llm_backup: int,
    overrode_baseline: bool,
    override_reason: Optional[str],
    match_cid: str,
    tx_hash: str,
    match_id: Optional[int],
    gas_used: Optional[int],
) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "donor_id": donor_id,
        "baseline_top1": baseline_top1,
        "llm_primary": llm_primary,
        "llm_backup": llm_backup,
        "overrode_baseline": bool(overrode_baseline),
        "override_reason": override_reason or "",
        "match_cid": match_cid,
        "tx_hash": tx_hash,
        "match_id": "" if match_id is None else int(match_id),
        "gas_used": "" if gas_used is None else int(gas_used),
    }