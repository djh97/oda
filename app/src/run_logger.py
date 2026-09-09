from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = APP_DIR / "pipeline-output" / "current" / "ui" / "match_runs.csv"
FIELDS = [
    "timestamp_utc",
    "donor_id",
    "baseline_primary",
    "guarded_primary",
    "guarded_backup",
    "guard_changed_primary",
    "temporary_hold_count",
    "review_required_count",
    "model_id",
    "model_latency_ms",
    "decision_cid",
    "tx_hash",
    "match_id",
    "gas_used",
]


def append_match_row(row: Dict[str, Any]) -> str:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_PATH.exists()
    with OUTPUT_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
    return str(OUTPUT_PATH)


def make_row(**values: Any) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **values,
    }
