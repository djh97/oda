import os
import time
import statistics as stats
from typing import Dict, Any, List

from dotenv import load_dotenv

from src.config import get_settings
from src.web3_client import make_web3_context
from src.chain_reader import get_donor, get_all_recipients, require_eligible_donor, filter_eligible_recipients
from src.ipfs_client import fetch_json_from_ipfs
from src.baseline import rank_recipients_baseline
from src.llm_client import call_llm_decision_support
from evaluation.benchmark_utils import ts_utc, write_csv, write_json

def _ms(x: float) -> float:
    return round(x * 1000.0, 2)

def summarize(values: List[float]) -> Dict[str, Any]:
    return {
        "avg_ms": _ms(stats.mean(values)),
        "min_ms": _ms(min(values)),
        "max_ms": _ms(max(values)),
        "runs": len(values),
    }

def main():
    load_dotenv()
    settings = get_settings()
    ctx = make_web3_context(settings)

    donor_id = int(os.getenv("BENCH_DONOR_ID", "1"))
    runs = int(os.getenv("BENCH_RUNS", "10"))

    if not settings.openai_api_key or not settings.openai_model_id:
        raise RuntimeError("OPENAI_API_KEY / OPENAI_MODEL_ID missing in .env")

    donor_struct = get_donor(ctx, donor_id)
    require_eligible_donor(donor_struct)

    recipients_structs = get_all_recipients(ctx)
    eligible = filter_eligible_recipients(recipients_structs)
    if len(eligible) < 2:
        raise RuntimeError("Need at least 2 eligible recipients")

    ipfs_times = []
    baseline_times = []
    llm_times = []

    rows = []
    print(f"Benchmarking donor_id={donor_id} for runs={runs} (no pinata, no on-chain tx)...")

    for i in range(runs):
        # IPFS fetch
        t0 = time.perf_counter()
        donor_json = fetch_json_from_ipfs(settings.pinata_gateway, donor_struct["ipfsHash"])
        recipients_json = [fetch_json_from_ipfs(settings.pinata_gateway, r["ipfsHash"]) for r in eligible]
        t1 = time.perf_counter()
        ipfs_s = t1 - t0
        ipfs_times.append(ipfs_s)

        # baseline
        t2 = time.perf_counter()
        ranked = rank_recipients_baseline(donor_json, recipients_json)
        ranked_top = ranked[:10]
        t3 = time.perf_counter()
        baseline_s = t3 - t2
        baseline_times.append(baseline_s)

        # LLM
        t4 = time.perf_counter()
        _ = call_llm_decision_support(
            model_id=settings.openai_model_id,
            api_key=settings.openai_api_key,
            donor_id=donor_id,
            donor_json=donor_json,
            recipients_json=recipients_json,
            baseline_ranked=ranked_top,
        )
        t5 = time.perf_counter()
        llm_s = t5 - t4
        llm_times.append(llm_s)

        print(f" Run {i+1}/{runs}: IPFS={_ms(ipfs_s)}ms | baseline={_ms(baseline_s)}ms | LLM={_ms(llm_s)}ms")

        rows.append({
            "run": i + 1,
            "donor_id": donor_id,
            "recipient_count": len(eligible),
            "ipfs_fetch_total_ms": _ms(ipfs_s),
            "baseline_ms": _ms(baseline_s),
            "llm_ms": _ms(llm_s),
        })

    summary = {
        "network": settings.network,
        "donor_id": donor_id,
        "runs": runs,
        "recipient_count": len(eligible),
        "model_id": settings.openai_model_id,
        "pinata_gateway": settings.pinata_gateway,
        "ipfs_fetch_total": summarize(ipfs_times),
        "baseline_scoring": summarize(baseline_times),
        "llm_inference": summarize(llm_times),
    }

    stamp = ts_utc()
    csv_path = write_csv(
        filename=f"benchmark_offchain_{stamp}.csv",
        rows=rows,
        fieldnames=["run","donor_id","recipient_count","ipfs_fetch_total_ms","baseline_ms","llm_ms"],
    )
    json_path = write_json(
        filename=f"benchmark_offchain_{stamp}.json",
        obj=summary,
    )

    print("\nSummary (ms):")
    print(" IPFS_fetch_total:", summary["ipfs_fetch_total"])
    print(" Baseline_scoring:", summary["baseline_scoring"])
    print(" LLM_inference:", summary["llm_inference"])
    print(f"\nSaved:\n {csv_path}\n {json_path}")

if __name__ == "__main__":
    main()
