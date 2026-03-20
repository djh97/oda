import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Set

from dotenv import load_dotenv

from src.baseline import rank_recipients_baseline
from src.llm_client import call_llm_decision_support

APP_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = APP_DIR / "eval-data"
OUT_DIR = APP_DIR / "pipeline-output" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_JSONL = EVAL_DIR / "cases.jsonl"
OUT_CSV = OUT_DIR / "baseline_vs_llm.csv"
OUT_SUMMARY_JSON = OUT_DIR / "baseline_vs_llm_summary.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rate(x: int, total: int) -> float:
    return round(x / max(1, total), 4)


def prf_from_tp_sum(tp_sum: int, total_cases: int) -> Dict[str, float]:
    """
    True Precision/Recall/F1 for a Top-2 set task:
      reference set size = 2 (oracle_primary + oracle_backup)
      predicted set size = 2 (baseline_top2 OR {llm_primary, llm_backup})

    For each case: TP in {0,1,2}
    Micro-averaged across cases:
      precision = TP_total / (2 * N)
      recall    = TP_total / (2 * N)
      F1        = 2PR/(P+R)
    """
    denom = 2 * max(1, total_cases)
    precision = tp_sum / denom
    recall = tp_sum / denom
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    load_dotenv(APP_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_id = os.getenv("OPENAI_MODEL_ID", "").strip()
    if not api_key or not model_id:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_MODEL_ID in .env")

    if not IN_JSONL.exists():
        raise RuntimeError(f"Missing {IN_JSONL}. Run: python -m evaluation.make_eval_set")

    rows: List[Dict[str, Any]] = []
    total = 0

    # Metrics
    baseline_correct = 0
    llm_correct = 0
    baseline_unsafe = 0
    llm_unsafe = 0
    llm_override = 0
    baseline_top2_hit = 0
    llm_top2_hit = 0
    llm_failures = 0
    baseline_unsafe_but_llm_safe = 0
    baseline_safe_but_llm_unsafe = 0

    # True PRF counters (Top-2 set match)
    baseline_tp_sum = 0
    llm_tp_sum = 0

    with IN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            case = json.loads(line)

            donor = case["donor"]
            recs = case["recipients"]

            # We'll call this "reference label" in the paper
            oracle_primary = int(case["oracle"]["primary_recipient_id"])
            oracle_backup = int(case["oracle"]["backup_recipient_id"])
            ref_set: Set[int] = {oracle_primary, oracle_backup}

            unsafe_ids = set(int(x) for x in case["oracle"]["unsafe_recipient_ids"])

            # ---- Baseline ----
            baseline_ranked = rank_recipients_baseline(donor, recs)
            baseline_top1 = int(baseline_ranked[0]["recipient_id"])
            baseline_top2 = [int(baseline_ranked[0]["recipient_id"]), int(baseline_ranked[1]["recipient_id"])]
            baseline_set = set(baseline_top2)

            baseline_correct += int(baseline_top1 == oracle_primary)
            baseline_top2_hit += int(oracle_primary in baseline_top2)
            baseline_unsafe += int(baseline_top1 in unsafe_ids)

            baseline_tp = len(ref_set.intersection(baseline_set))  # 0..2
            baseline_tp_sum += baseline_tp

            # ---- LLM ----
            try:
                llm_out = call_llm_decision_support(
                    model_id=model_id,
                    api_key=api_key,
                    donor_id=int(case["case_id"]),  # synthetic donor_id for eval; validated inside function
                    donor_json=donor,
                    recipients_json=recs,
                    baseline_ranked=baseline_ranked,
                )
                llm_primary = int(llm_out["primary_recipient_id"])
                llm_backup = int(llm_out["backup_recipient_id"])
                # Cross-safety comparisons
                if (baseline_top1 in unsafe_ids) and (llm_primary not in unsafe_ids):
                    baseline_unsafe_but_llm_safe += 1
                if (baseline_top1 not in unsafe_ids) and (llm_primary in unsafe_ids):
                    baseline_safe_but_llm_unsafe += 1
                llm_set = {llm_primary, llm_backup}

                llm_correct += int(llm_primary == oracle_primary)
                llm_top2_hit += int(oracle_primary in [llm_primary, llm_backup])
                llm_unsafe += int(llm_primary in unsafe_ids)
                llm_override += int(bool(llm_out.get("overrode_baseline", False)))

                llm_tp = len(ref_set.intersection(llm_set))  # 0..2
                llm_tp_sum += llm_tp

                rows.append({
                    "case_id": case["case_id"],
                    "oracle_primary": oracle_primary,
                    "oracle_backup": oracle_backup,
                    "baseline_top1": baseline_top1,
                    "baseline_top2_1": baseline_top2[0],
                    "baseline_top2_2": baseline_top2[1],
                    "baseline_unsafe": baseline_top1 in unsafe_ids,
                    "llm_primary": llm_primary,
                    "llm_backup": llm_backup,
                    "llm_unsafe": llm_primary in unsafe_ids,
                    "overrode_baseline": bool(llm_out.get("overrode_baseline", False)),
                    "baseline_tp": baseline_tp,
                    "llm_tp": llm_tp,
                })

            except Exception as e:
                llm_failures += 1
                rows.append({
                    "case_id": case["case_id"],
                    "oracle_primary": oracle_primary,
                    "oracle_backup": oracle_backup,
                    "baseline_top1": baseline_top1,
                    "baseline_top2_1": baseline_top2[0],
                    "baseline_top2_2": baseline_top2[1],
                    "baseline_unsafe": baseline_top1 in unsafe_ids,
                    "llm_primary": "",
                    "llm_backup": "",
                    "llm_unsafe": "",
                    "overrode_baseline": "",
                    "baseline_tp": baseline_tp,
                    "llm_tp": "",
                    "llm_error": str(e)[:200],
                })

    # PRF for top-2 set match
    baseline_prf = prf_from_tp_sum(baseline_tp_sum, total)
    llm_prf = prf_from_tp_sum(llm_tp_sum, total)

    summary = {
        "generated_utc": now_utc(),
        "cases": total,
        "llm_failures": llm_failures,

        "baseline_top1_accuracy": rate(baseline_correct, total),
        "llm_top1_accuracy": rate(llm_correct, total),

        "baseline_top2_hit": rate(baseline_top2_hit, total),
        "llm_top2_hit": rate(llm_top2_hit, total),

        "baseline_unsafe_rate": rate(baseline_unsafe, total),
        "llm_unsafe_rate": rate(llm_unsafe, total),

        "llm_override_rate": rate(llm_override, total),

        # True Precision/Recall/F1 (Top-2 set overlap)
        "baseline_top2_precision": baseline_prf["precision"],
        "baseline_top2_recall": baseline_prf["recall"],
        "baseline_top2_f1": baseline_prf["f1"],

        "llm_top2_precision": llm_prf["precision"],
        "llm_top2_recall": llm_prf["recall"],
        "llm_top2_f1": llm_prf["f1"],
        
        "baseline_unsafe_count": baseline_unsafe,
        "llm_unsafe_count": llm_unsafe,
        "llm_override_count": llm_override,
        "unsafe_avoided_count": baseline_unsafe_but_llm_safe,
        "unsafe_introduced_count": baseline_safe_but_llm_unsafe,
    }

    # Write CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write summary JSON
    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Saved:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_SUMMARY_JSON}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
