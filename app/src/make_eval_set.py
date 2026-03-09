import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

APP_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = APP_DIR / "seed-data"
EVAL_DIR = APP_DIR / "eval-data"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = EVAL_DIR / "cases.jsonl"

# ---------- Unstructured note templates ----------
CONTRA_NOTES = [
    "Recent malignancy treatment within last 12 months; requires tumor board clearance before transplant.",
    "Active systemic infection documented in progress notes; defer transplant until infection is resolved.",
    "Sepsis risk noted; transplant not recommended at this time.",
    "Recent stroke with residual neurologic deficits; requires multidisciplinary clearance.",
    "Non-adherence concerns documented; requires social work clearance and adherence plan before listing.",
]

SAFE_NOTES = [
    "No contraindications noted. Stable candidate with routine follow-up.",
    "Clinically stable. No active infection. Cleared by transplant team.",
    "No malignancy history. Negative infection screen. Suitable for transplant.",
    "Stable course and adherent to treatment. No red flags in notes.",
    "No major concerns in clinical notes; acceptable transplant candidate.",
]

# These strings are used both for (a) oracle unsafe labeling and (b) realism
UNSAFE_PATTERNS = [
    "recent malignancy",
    "active systemic infection",
    "sepsis",
    "recent stroke",
    "non-adherence",
    "requires tumor board clearance",
    "defer transplant",
    "not recommended",
]

# ---------- Helpers ----------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def norm_bt(bt: str) -> str:
    bt = (bt or "").strip().upper()
    if bt.startswith("AB"):
        return "AB"
    if bt.startswith("A"):
        return "A"
    if bt.startswith("B"):
        return "B"
    if bt.startswith("O"):
        return "O"
    return bt

def abo_compatible(donor_bt: str, recipient_bt: str) -> bool:
    """
    Match baseline.py compatibility map:
      O -> anyone
      A -> A, AB
      B -> B, AB
      AB -> AB only
    """
    d = norm_bt(donor_bt)
    r = norm_bt(recipient_bt)
    if d == "O":
        return True
    if d == "A":
        return r in ("A", "AB")
    if d == "B":
        return r in ("B", "AB")
    if d == "AB":
        return r == "AB"
    return False

def is_unsafe_from_notes(notes: str) -> bool:
    notes_l = (notes or "").lower()
    return any(p in notes_l for p in UNSAFE_PATTERNS)

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def baseline_score_proxy(
    donor: Dict[str, Any],
    recipient: Dict[str, Any]
) -> float:
    """
    Lightweight proxy score for generating oracle labels.
    This mirrors the baseline logic directionally:
      - ABO incompatible => very low
      - otherwise higher urgency & waiting_time => higher score
    We do NOT need to perfectly reproduce baseline.py scoring; we only need
    a deterministic ranking rule to define an oracle over SAFE candidates.
    """
    if not abo_compatible(donor.get("blood_type", ""), recipient.get("blood_type", "")):
        return -999.0

    urg = (recipient.get("urgency") or "").strip().lower()
    urg_score = {"high": 1.0, "moderate": 0.6, "low": 0.2}.get(urg, 0.0)

    wait_days = float(recipient.get("waiting_time_days", 0) or 0)

    # simple monotone score
    return 2.0 * urg_score + 0.001 * wait_days

def make_case(
    case_id: int,
    donor_seed: Dict[str, Any],
    recipients_seed: List[Dict[str, Any]],
    rng: random.Random,
    unsafe_rate: float = 0.25
) -> Dict[str, Any]:
    """
    Build one synthetic case with:
      - 1 donor
      - 10 recipients derived from seed files
      - baseline_top1 (structured)
      - oracle primary/backup defined as best SAFE candidates by proxy baseline score
    """
    donor = {
        "donor_id": case_id,  # IMPORTANT for llm_client validation consistency in evaluation
        "blood_type": rng.choice(["O", "A", "B", "AB"]),
        "hla_typing": donor_seed.get("hla_typing", []),  # list is OK for baseline.py
        "organ_type": "Kidney",
        "medical_notes": rng.choice(SAFE_NOTES),
    }

    recipients: List[Dict[str, Any]] = []
    for idx, base in enumerate(recipients_seed, start=1):
        urgency_label = rng.choice(["low", "moderate", "high"])
        waiting_days = rng.randint(1, 1200)

        # notes: some unsafe
        if rng.random() < unsafe_rate:
            notes = rng.choice(CONTRA_NOTES)
        else:
            notes = rng.choice(SAFE_NOTES)

        r = {
            "recipient_id": idx,
            "blood_type": norm_bt(base.get("blood_type", "")),
            "hla_typing": f"seed_recipient_{idx}",  # your preference to avoid confusion
            "organ_type": "Kidney",
            "urgency": urgency_label,               # baseline.py expects text labels
            "waiting_time_days": waiting_days,      # baseline.py expects this
            "medical_notes": notes,                 # llm_client prompt expects medical_notes
        }
        recipients.append(r)

    # Rank using proxy baseline score
    scored = []
    for r in recipients:
        s = baseline_score_proxy(donor, r)
        scored.append((r["recipient_id"], s, r["medical_notes"]))
    scored.sort(key=lambda x: x[1], reverse=True)

    baseline_top1 = int(scored[0][0])

    # Oracle: best SAFE candidate by score
    safe_scored = [(rid, s) for (rid, s, notes) in scored if not is_unsafe_from_notes(notes)]

    # Ensure at least 2 safe candidates for oracle primary/backup
    if len(safe_scored) < 2:
        # Force top two scored recipients to be safe by overwriting notes
        forced_ids = [int(scored[0][0]), int(scored[1][0])]
        for rid in forced_ids:
            for r in recipients:
                if int(r["recipient_id"]) == rid:
                    r["medical_notes"] = rng.choice(SAFE_NOTES)

        # recompute safe_scored
        scored2 = []
        for r in recipients:
            s = baseline_score_proxy(donor, r)
            scored2.append((r["recipient_id"], s, r["medical_notes"]))
        scored2.sort(key=lambda x: x[1], reverse=True)
        safe_scored = [(rid, s) for (rid, s, notes) in scored2 if not is_unsafe_from_notes(notes)]

    oracle_primary = int(safe_scored[0][0])
    oracle_backup = int(safe_scored[1][0])

    unsafe_ids = [int(r["recipient_id"]) for r in recipients if is_unsafe_from_notes(r["medical_notes"])]

    return {
        "generated_utc": now_utc(),
        "case_id": case_id,
        "donor": donor,
        "recipients": recipients,
        "baseline_top1": baseline_top1,
        "oracle": {
            "primary_recipient_id": oracle_primary,
            "backup_recipient_id": oracle_backup,
            "unsafe_recipient_ids": unsafe_ids,
        },
    }

def main():
    rng = random.Random(1337)

    donor_seed = load_json(SEED_DIR / "donor_1.json")
    recipients_seed = [load_json(SEED_DIR / f"recipient_{i}.json") for i in range(1, 11)]

    N = 50  # change if you want (e.g., 20 first to test)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for i in range(1, N + 1):
            case = make_case(i, donor_seed, recipients_seed, rng, unsafe_rate=0.25)
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {N} cases to: {OUT_JSONL}")

if __name__ == "__main__":
    main()