from typing import Any, Dict, List, Tuple

# Basic ABO compatibility (donor -> recipient)
# For simplicity:
# - O donors can donate to anyone
# - A donors to A, AB
# - B donors to B, AB
# - AB donors to AB only
ABO_COMPAT = {
    "O": {"O", "A", "B", "AB"},
    "A": {"A", "AB"},
    "B": {"B", "AB"},
    "AB": {"AB"},
}

URGENCY_SCORE = {
    "high": 1.0,
    "moderate": 0.6,
    "low": 0.2,
}

def _norm_blood(bt: str) -> str:
    # Accept "O", "O+", "O-" etc -> just take ABO portion
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
    d = _norm_blood(donor_bt)
    r = _norm_blood(recipient_bt)
    return r in ABO_COMPAT.get(d, set())

def hla_overlap(donor_hla: List[str], recipient_hla: List[str]) -> int:
    dset = {x.strip().upper() for x in donor_hla or [] if str(x).strip()}
    rset = {x.strip().upper() for x in recipient_hla or [] if str(x).strip()}
    return len(dset.intersection(rset))

def get_urgency(recipient: Dict[str, Any]) -> float:
    urg = str(recipient.get("urgency", "")).strip().lower()
    return URGENCY_SCORE.get(urg, 0.0)

def get_waiting_days(recipient: Dict[str, Any]) -> float:
    try:
        return float(recipient.get("waiting_time_days", 0) or 0)
    except Exception:
        return 0.0

def rank_recipients_baseline(
    donor: Dict[str, Any],
    recipients: List[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of candidates with baseline score and factors.
    weights keys: abo, hla, urgency, waiting
    """
    w = weights or {"abo": 3.0, "hla": 1.0, "urgency": 2.0, "waiting": 0.5}

    donor_bt = donor.get("blood_type") or donor.get("bloodType") or ""
    donor_hla = donor.get("hla_typing") or donor.get("hlaTyping") or []
    if isinstance(donor_hla, str):
        donor_hla = [x.strip() for x in donor_hla.split(",") if x.strip()]

    # Normalize waiting time for recipients (0..1)
    waiting_vals = [get_waiting_days(r) for r in recipients]
    max_wait = max(waiting_vals) if waiting_vals else 0.0

    scored = []
    for r in recipients:
        r_bt = r.get("blood_type") or r.get("bloodType") or ""
        r_hla = r.get("hla_typing") or r.get("hlaTyping") or []
        if isinstance(r_hla, str):
            r_hla = [x.strip() for x in r_hla.split(",") if x.strip()]

        abo_ok = abo_compatible(donor_bt, r_bt)
        hla = hla_overlap(donor_hla, r_hla)
        urg = get_urgency(r)
        wait_days = get_waiting_days(r)
        wait_norm = (wait_days / max_wait) if max_wait > 0 else 0.0

        # Hard gate: if ABO incompatible, force very low score
        if not abo_ok:
            score = -999.0
        else:
            score = (
                w["abo"] * 1.0 +
                w["hla"] * float(hla) +
                w["urgency"] * float(urg) +
                w["waiting"] * float(wait_norm)
            )

        scored.append({
            "recipient_id": int(r.get("recipient_id") or r.get("recipientId") or 0),
            "score": float(score),
            "factors": {
                "abo_compatible": abo_ok,
                "hla_overlap": hla,
                "urgency": r.get("urgency", "unknown"),
                "waiting_time_days": wait_days,
                "waiting_norm": round(wait_norm, 4),
            }
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    # Add rank
    for i, item in enumerate(scored, start=1):
        item["rank"] = i
    return scored