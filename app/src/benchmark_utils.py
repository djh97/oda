import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

APP_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = APP_DIR / "pipeline-output" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ts_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def write_csv(filename: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return str(path)

def write_json(filename: str, obj: Dict[str, Any]) -> str:
    path = OUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return str(path)