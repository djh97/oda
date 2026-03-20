import json
from typing import Any, Dict, List, Tuple

class LLMError(RuntimeError):
    pass

def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMError("LLM returned empty output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON. Output was: {text[:400]}") from e

def _validate_llm_output(data: Dict[str, Any], donor_id: int, recipient_ids: List[int]) -> Dict[str, Any]:
    required = ["donor_id", "primary_recipient_id", "backup_recipient_id", "risk_flags"]
    for k in required:
        if k not in data:
            raise LLMError(f"LLM JSON missing key: {k}")

    # Defaults
    data.setdefault("overrode_baseline", False)
    data.setdefault("override_reason", None)
    data.setdefault("explanation", "")

    if int(data["donor_id"]) != int(donor_id):
        raise LLMError(f"LLM donor_id mismatch. Expected {donor_id}, got {data['donor_id']}")

    primary = int(data["primary_recipient_id"])
    backup = int(data["backup_recipient_id"])
    if primary not in recipient_ids:
        raise LLMError(f"LLM primary_recipient_id not in candidates: {primary}")
    if backup not in recipient_ids:
        raise LLMError(f"LLM backup_recipient_id not in candidates: {backup}")
    if primary == backup:
        raise LLMError("LLM chose the same recipient for primary and backup")

    rf = data.get("risk_flags", [])
    if not isinstance(rf, list):
        raise LLMError("LLM risk_flags must be a list")
    for item in rf:
        if not isinstance(item, dict):
            raise LLMError("Each risk_flags entry must be an object")
        if "recipient_id" not in item or "risk_level" not in item or "flags" not in item:
            raise LLMError("Each risk_flags entry must have recipient_id, risk_level, flags")

    return data

def _risk_map(risk_flags: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in risk_flags or []:
        try:
            rid = int(r.get("recipient_id", 0))
        except Exception:
            continue
        out[rid] = r
    return out

def _is_contraindicated(r: Dict[str, Any]) -> bool:
    """
    Conservative rule: treat 'high' risk as contraindicated if flags indicate hard stop.
    You can extend this list later.
    """
    if not r:
        return False
    level = str(r.get("risk_level", "")).lower().strip()
    flags = [str(x).lower() for x in (r.get("flags") or [])]

    hard_flags = [
        "active infection",
        "sepsis",
        "bacteremia",
        "active malignancy",
        "recent malignancy treatment",
        "requires tumor board clearance",
        "unstable",
        "defer transplant",
    ]

    if level == "high":
        for hf in hard_flags:
            if any(hf in f for f in flags):
                return True
    return False

def _post_process_llm(
    llm_out: Dict[str, Any],
    baseline_top1: int,
    baseline_top2: int,
    baseline_order: List[int]
) -> Dict[str, Any]:
    """
    Enforce logical consistency:
    1) overrode_baseline must match whether primary != baseline_top1
    2) If primary is contraindicated, demote it and pick the best non-contra candidate from baseline order.
    """
    primary = int(llm_out["primary_recipient_id"])
    backup = int(llm_out["backup_recipient_id"])

    rm = _risk_map(llm_out.get("risk_flags", []))
    primary_contra = _is_contraindicated(rm.get(primary))

    def first_safe(exclude: List[int] | None = None) -> int | None:
        excluded = set(exclude or [])
        for rid in baseline_order:
            rid = int(rid)
            if rid in excluded:
                continue
            if not _is_contraindicated(rm.get(rid)):
                return rid
        return None

    if primary_contra:
        # choose first non-contra from baseline order
        new_primary = first_safe(exclude=[primary])
        if new_primary is None:
            # fallback: baseline_top2 if exists
            new_primary = baseline_top2 if baseline_top2 != primary else baseline_top1

        # Set backup: prefer old primary if not equal; else baseline_top2
        new_backup = primary if new_primary != primary else baseline_top2
        if new_backup == new_primary:
            # final fallback
            for rid in baseline_order:
                if rid != new_primary:
                    new_backup = rid
                    break

        llm_out["primary_recipient_id"] = int(new_primary)
        llm_out["backup_recipient_id"] = int(new_backup)

        # Ensure override_reason explains the change
        reason = llm_out.get("override_reason") or ""
        add = "Primary candidate flagged as contraindicated by risk_flags; promoted next best suitable candidate."
        llm_out["override_reason"] = (reason + " " + add).strip() if reason else add
        if not llm_out.get("explanation"):
            llm_out["explanation"] = llm_out["override_reason"]

    # Now enforce overrode_baseline consistency
    primary = int(llm_out["primary_recipient_id"])
    llm_out["overrode_baseline"] = (primary != int(baseline_top1))
    if llm_out["overrode_baseline"] and not llm_out.get("override_reason"):
        llm_out["override_reason"] = "LLM selected a different primary than baseline_top1."

    if not llm_out.get("explanation"):
        llm_out["explanation"] = llm_out.get("override_reason") or ""
        
    # --- Baseline-first enforcement ---
    # If baseline_top1 is NOT contraindicated, force primary to baseline_top1.
    rm = _risk_map(llm_out.get("risk_flags", []))
    baseline1_contra = _is_contraindicated(rm.get(int(baseline_top1)))

    if not baseline1_contra:
        # Force primary to baseline_top1
        forced_primary = int(baseline_top1)

        # Choose backup: prefer baseline_top2 unless equal; otherwise next in baseline_order
        forced_backup = int(baseline_top2)
        if forced_backup == forced_primary:
            for rid in baseline_order:
                if int(rid) != forced_primary:
                    forced_backup = int(rid)
                    break

        llm_out["primary_recipient_id"] = forced_primary
        llm_out["backup_recipient_id"] = forced_backup
        llm_out["overrode_baseline"] = False
        llm_out["override_reason"] = None

        # Ensure risk_flags include low entries for primary/backup if missing
        rm2 = _risk_map(llm_out.get("risk_flags", []))
        def ensure_low(rid: int):
            if rid not in rm2:
                llm_out.setdefault("risk_flags", []).append({
                    "recipient_id": rid,
                    "risk_level": "low",
                    "flags": []
                })

        ensure_low(forced_primary)
        ensure_low(forced_backup)

        if not llm_out.get("explanation"):
            llm_out["explanation"] = "Baseline top-1 is not contraindicated; following baseline ranking."
    else:
        # Baseline top-1 is contraindicated, so force the highest-ranked safe fallback.
        safe_primary = first_safe(exclude=[int(baseline_top1)])
        if safe_primary is None:
            safe_primary = int(baseline_top2) if int(baseline_top2) != int(baseline_top1) else int(baseline_top1)

        safe_backup = first_safe(exclude=[int(baseline_top1), int(safe_primary)])
        if safe_backup is None:
            safe_backup = int(baseline_top1) if int(baseline_top1) != int(safe_primary) else int(baseline_top2)
            if safe_backup == int(safe_primary):
                for rid in baseline_order:
                    rid = int(rid)
                    if rid != int(safe_primary):
                        safe_backup = rid
                        break

        llm_out["primary_recipient_id"] = int(safe_primary)
        llm_out["backup_recipient_id"] = int(safe_backup)
        llm_out["overrode_baseline"] = True

        reason = llm_out.get("override_reason") or ""
        prefix = f"baseline top-1 candidate (recipient {baseline_top1}) contraindicated"
        if prefix not in reason.lower():
            llm_out["override_reason"] = (
                (reason + " ").strip() + f"Selected next highest-ranked suitable candidate (recipient {safe_primary})."
            ).strip()
        if not llm_out.get("explanation") or "clinically unstable" in str(llm_out.get("explanation", "")).lower():
            llm_out["explanation"] = (
                f"Baseline top-1 candidate (recipient {baseline_top1}) was flagged as contraindicated; "
                f"promoted the highest-ranked suitable fallback (recipient {safe_primary})."
            )

        rm2 = _risk_map(llm_out.get("risk_flags", []))
        for rid in [int(safe_primary), int(safe_backup)]:
            if rid not in rm2:
                llm_out.setdefault("risk_flags", []).append(
                    {"recipient_id": rid, "risk_level": "low", "flags": []}
                )

    # Recompute overrode_baseline consistency (final)
    llm_out["overrode_baseline"] = (int(llm_out["primary_recipient_id"]) != int(baseline_top1))
    if llm_out["overrode_baseline"] and not llm_out.get("override_reason"):
        llm_out["override_reason"] = "baseline top-1 candidate contraindicated; selected next suitable candidate."
    if not llm_out.get("explanation"):
        llm_out["explanation"] = llm_out.get("override_reason") or ""

    return llm_out

def call_llm_decision_support(
    model_id: str,
    api_key: str,
    donor_id: int,
    donor_json: Dict[str, Any],
    recipients_json: List[Dict[str, Any]],
    baseline_ranked: List[Dict[str, Any]],
) -> Dict[str, Any]:
    recipient_ids = [int(r.get("recipient_id") or r.get("recipientId") or 0) for r in recipients_json]
    recipient_ids = [x for x in recipient_ids if x != 0]

    baseline_top1 = int(baseline_ranked[0]["recipient_id"])
    baseline_top2 = int(baseline_ranked[1]["recipient_id"])
    baseline_order = [int(x["recipient_id"]) for x in baseline_ranked]

    payload = {
        "donor": donor_json,
        "recipients": recipients_json,
        "baseline_ranking": baseline_ranked,
        "baseline_top1": baseline_top1,
        "baseline_top2": baseline_top2,
    }

    system = (
        "You are a clinical decision-support assistant for organ allocation. "
        "Return ONLY valid JSON (no markdown, no extra text). "
        "Your default behavior is to FOLLOW the baseline ranking unless there is a clear contraindication. "
        "Use BOTH structured fields and unstructured medical_notes. "
        "RULES:\n"
        "1) If baseline_top1 has NO clear contraindication in medical_notes, you MUST set primary_recipient_id = baseline_top1.\n"
        "2) You may override baseline_top1 ONLY when medical_notes provide a clear contraindication / hard stop for transplant now "
        "(e.g., active infection/sepsis, recent malignancy treatment requiring clearance, 'defer transplant', 'not recommended').\n"
        "3) backup_recipient_id should normally be baseline_top2, unless baseline_top2 is contraindicated, then choose the next suitable.\n"
        "4) Do NOT choose a contraindicated candidate as PRIMARY.\n"
        "5) IMPORTANT wording: when overriding, explicitly reference the rejected baseline candidate, e.g., "
        "'baseline top-1 candidate (recipient 4) flagged: ...'. Do NOT call the rejected candidate 'primary recipient'.\n"
        "6) risk_flags MUST include entries for: (a) selected primary, (b) selected backup, and (c) any candidate you flag as high risk/contraindicated. "
        "If primary/backup have no concerns, set risk_level='low' and flags=[].\n"
        "Schema:\n"
        "{"
        "\"donor_id\": <int>, "
        "\"primary_recipient_id\": <int>, "
        "\"backup_recipient_id\": <int>, "
        "\"overrode_baseline\": <true|false>, "
        "\"override_reason\": <string|null>, "
        "\"risk_flags\": ["
        "{\"recipient_id\": <int>, \"risk_level\": \"low|medium|high\", \"flags\": [\"...\"]}"
        "], "
        "\"explanation\": <string>"
        "}"
    )
    user = json.dumps(payload, ensure_ascii=False)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        data = _safe_json_loads(text)
        data = _validate_llm_output(data, donor_id, recipient_ids)
        data = _post_process_llm(data, baseline_top1, baseline_top2, baseline_order)
        return data
    except Exception as e:
        raise LLMError(f"LLM call failed: {e}")
