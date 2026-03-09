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
from src.pinata_client import pin_json
from src.chain_writer import send_create_match
from src.tx_logger import append_tx
from src.benchmark_utils import ts_utc, write_csv, write_json

def _ms(x: float) -> float:
    return round(x * 1000.0, 2)

def summarize_ms(values_ms: List[float]) -> Dict[str, Any]:
    return {
        "avg_ms": round(stats.mean(values_ms), 2),
        "min_ms": round(min(values_ms), 2),
        "max_ms": round(max(values_ms), 2),
        "runs": len(values_ms),
    }

def main():
    load_dotenv()
    settings = get_settings()
    ctx = make_web3_context(settings)

    donor_id = int(os.getenv("BENCH_DONOR_ID", "1"))
    runs = int(os.getenv("BENCH_CHAIN_RUNS", "3"))

    if not settings.openai_api_key or not settings.openai_model_id:
        raise RuntimeError("OPENAI_API_KEY / OPENAI_MODEL_ID missing in .env")
    if not settings.pinata_jwt:
        raise RuntimeError("PINATA_JWT missing in .env")
    if not settings.llm_private_key:
        raise RuntimeError("LLM_PRIVATE_KEY missing in .env")

    donor_struct = get_donor(ctx, donor_id)
    require_eligible_donor(donor_struct)

    recipients_structs = get_all_recipients(ctx)
    eligible = filter_eligible_recipients(recipients_structs)
    if len(eligible) < 2:
        raise RuntimeError("Need at least 2 eligible recipients")

    pinata_times_ms = []
    chain_receipt_times_ms = []
    gas_used_list = []

    rows = []
    print(f"Chain benchmark donor_id={donor_id} runs={runs} (includes Pinata + on-chain createMatch)...")

    # Fetch donor/recipients once (so chain benchmark isolates pinata+chain)
    donor_json = fetch_json_from_ipfs(settings.pinata_gateway, donor_struct["ipfsHash"])
    recipients_json = [fetch_json_from_ipfs(settings.pinata_gateway, r["ipfsHash"]) for r in eligible]
    ranked = rank_recipients_baseline(donor_json, recipients_json)
    ranked_top = ranked[:10]

    for i in range(runs):
        # LLM once per run (keeps rationale realistic; still mostly isolates chain+pinata)
        llm_out = call_llm_decision_support(
            model_id=settings.openai_model_id,
            api_key=settings.openai_api_key,
            donor_id=donor_id,
            donor_json=donor_json,
            recipients_json=recipients_json,
            baseline_ranked=ranked_top,
        )

        rationale_obj = {
            "donor_id": donor_id,
            "baseline_top1": int(ranked_top[0]["recipient_id"]),
            "baseline_top2": int(ranked_top[1]["recipient_id"]),
            "baseline_top10": ranked_top,
            "llm_decision": llm_out,
        }

        # Pinata upload timing
        t0 = time.perf_counter()
        cid = pin_json(settings.pinata_jwt, rationale_obj, name=f"benchmark_match_donor_{donor_id}_run_{i+1}")
        t1 = time.perf_counter()
        pin_ms = _ms(t1 - t0)
        pinata_times_ms.append(pin_ms)

        # On-chain receipt timing (send_create_match already waits; we time it outside)
        primary = int(llm_out["primary_recipient_id"])
        backup = int(llm_out["backup_recipient_id"])

        t2 = time.perf_counter()
        txr = send_create_match(
            ctx=ctx,
            llm_private_key=settings.llm_private_key,
            donor_id=donor_id,
            primary_id=primary,
            backup_id=backup,
            match_cid=cid,
        )
        t3 = time.perf_counter()
        chain_ms = _ms(t3 - t2)
        chain_receipt_times_ms.append(chain_ms)
        gas_used_list.append(int(txr.gas_used))

        append_tx(
            network=settings.network,
            role="LLM",
            function="createMatch",
            tx_hash=txr.tx_hash,
            gas_used=txr.gas_used,
            notes=f"benchmark donor={donor_id},primary={primary},backup={backup},cid={cid}",
        )

        rows.append({
            "run": i + 1,
            "donor_id": donor_id,
            "pinata_upload_ms": pin_ms,
            "tx_receipt_ms": chain_ms,
            "match_cid": cid,
            "tx_hash": txr.tx_hash,
            "match_id": "" if txr.match_id is None else int(txr.match_id),
            "gas_used": int(txr.gas_used),
        })

        print(f" Run {i+1}/{runs}: Pinata={pin_ms}ms | TxReceipt={chain_ms}ms | Gas={txr.gas_used} | matchId={txr.match_id}")

    summary = {
        "network": settings.network,
        "contract_address": ctx.contract_address,
        "donor_id": donor_id,
        "runs": runs,
        "model_id": settings.openai_model_id,
        "pinata_upload": summarize_ms(pinata_times_ms),
        "tx_receipt": summarize_ms(chain_receipt_times_ms),
        "gas_used": {
            "avg": round(stats.mean(gas_used_list), 2),
            "min": int(min(gas_used_list)),
            "max": int(max(gas_used_list)),
            "runs": len(gas_used_list),
        },
    }

    stamp = ts_utc()
    csv_path = write_csv(
        filename=f"benchmark_chain_{stamp}.csv",
        rows=rows,
        fieldnames=["run","donor_id","pinata_upload_ms","tx_receipt_ms","gas_used","match_id","tx_hash","match_cid"],
    )
    json_path = write_json(
        filename=f"benchmark_chain_{stamp}.json",
        obj=summary,
    )

    print("\nSummary:")
    print(" Pinata upload:", summary["pinata_upload"])
    print(" Tx receipt:", summary["tx_receipt"])
    print(" Gas used:", summary["gas_used"])
    print(f"\nSaved:\n {csv_path}\n {json_path}")

if __name__ == "__main__":
    main()