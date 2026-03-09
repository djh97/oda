from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from .run_logger import append_match_row, make_row

from .config import get_settings
from .web3_client import make_web3_context, read_contract_health, ConfigError
from .schemas import MatchRequest, MatchResponse, BaselineCandidate, LLMDecision, OnChainRecord, RiskFlag
from .chain_reader import (
    get_donor,
    get_all_recipients,
    require_eligible_donor,
    filter_eligible_recipients,
    ChainDataError,
)
from .ipfs_client import fetch_json_from_ipfs, IPFSError
from .baseline import rank_recipients_baseline
from .llm_client import call_llm_decision_support, LLMError
from .pinata_client import pin_json, PinataError
from .chain_writer import send_create_match, ChainWriteError
from .tx_logger import append_tx

APP_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(title="Organ Donation Allocation (Local)")

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.get("/health", response_class=JSONResponse)
def health():
    try:
        settings = get_settings()
        ctx = make_web3_context(settings)
        data = read_contract_health(ctx)
        data["network"] = settings.network
        return data
    except ConfigError as e:
        return JSONResponse(status_code=500, content={"connected": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"connected": False, "error": f"Unexpected error: {e}"})


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    contract_address = None
    regulator = None

    try:
        settings = get_settings()
        ctx = make_web3_context(settings)
        h = read_contract_health(ctx)
        contract_address = h.get("contract_address")
        regulator = h.get("regulator")
    except Exception:
        pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": "Organ Donation Allocation (Local)",
            "subtitle": "Baseline ranking + LLM decision support + on-chain match recording (Sepolia)",
            "contract_address": contract_address,
            "regulator": regulator,
        },
    )


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """
    Full pipeline:
    - Fetch donor + recipients from chain
    - Fetch JSON from IPFS
    - Baseline ranking
    - LLM decision support (override + risk flags)
    - Upload rationale JSON to Pinata -> matchCID
    - Record match on-chain via createMatch() from LLM EOA
    """
    settings = get_settings()
    ctx = make_web3_context(settings)

    # Required env
    if not settings.openai_api_key or not settings.openai_model_id:
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY or OPENAI_MODEL_ID in .env"})
    if not settings.pinata_jwt:
        return JSONResponse(status_code=500, content={"error": "Missing PINATA_JWT in .env"})
    if not settings.llm_private_key:
        return JSONResponse(status_code=500, content={"error": "Missing LLM_PRIVATE_KEY in .env"})

    donor_id = req.donor_id

    try:
        donor_struct = get_donor(ctx, donor_id)
        require_eligible_donor(donor_struct)

        recipients_structs = get_all_recipients(ctx)
        eligible_recipients = filter_eligible_recipients(recipients_structs)

        if len(eligible_recipients) < 2:
            raise ChainDataError("Need at least 2 ethically approved recipients on-chain.")

        donor_json = fetch_json_from_ipfs(settings.pinata_gateway, donor_struct["ipfsHash"])

        recipients_json = []
        for r in eligible_recipients:
            recipients_json.append(fetch_json_from_ipfs(settings.pinata_gateway, r["ipfsHash"]))

        ranked = rank_recipients_baseline(donor_json, recipients_json)
        ranked_top = ranked[:10]

        baseline = [
            BaselineCandidate(
                rank=int(x["rank"]),
                recipient_id=int(x["recipient_id"]),
                score=float(x["score"]),
                factors=x["factors"],
            )
            for x in ranked_top
        ]

        llm_out = call_llm_decision_support(
            model_id=settings.openai_model_id,
            api_key=settings.openai_api_key,
            donor_id=donor_id,
            donor_json=donor_json,
            recipients_json=recipients_json,
            baseline_ranked=ranked_top,
        )

        # Convert risk_flags to Pydantic
        rf = []
        for item in llm_out.get("risk_flags", []):
            rf.append(RiskFlag(
                recipient_id=int(item["recipient_id"]),
                risk_level=str(item["risk_level"]),
                flags=list(item.get("flags", [])),
            ))

        llm = LLMDecision(
            donor_id=int(llm_out["donor_id"]),
            primary_recipient_id=int(llm_out["primary_recipient_id"]),
            backup_recipient_id=int(llm_out["backup_recipient_id"]),
            overrode_baseline=bool(llm_out.get("overrode_baseline", False)),
            override_reason=llm_out.get("override_reason"),
            risk_flags=rf,
            explanation=str(llm_out.get("explanation", "")),
        )

        # ---- 7B: Upload rationale JSON to Pinata ----
        rationale_obj = {
            "donor_id": donor_id,
            "baseline_top1": int(ranked_top[0]["recipient_id"]),
            "baseline_top2": int(ranked_top[1]["recipient_id"]),
            "baseline_top10": ranked_top,
            "llm_decision": llm_out,
        }
        match_cid = pin_json(settings.pinata_jwt, rationale_obj, name=f"match_donor_{donor_id}")

        # ---- 7C: Record on-chain createMatch ----
        txr = send_create_match(
            ctx=ctx,
            llm_private_key=settings.llm_private_key,
            donor_id=donor_id,
            primary_id=int(llm.primary_recipient_id),
            backup_id=int(llm.backup_recipient_id),
            match_cid=match_cid,
        )

        append_tx(
            network=settings.network,
            role="LLM",
            function="createMatch",
            tx_hash=txr.tx_hash,
            gas_used=txr.gas_used,
            notes=f"donor={donor_id},primary={llm.primary_recipient_id},backup={llm.backup_recipient_id},cid={match_cid}",
        )        

        onchain = OnChainRecord(
            tx_hash=txr.tx_hash,
            match_id=txr.match_id,
            gas_used=txr.gas_used,
            contract_address=ctx.contract_address,
        )

        # ---- Log run to CSV (pipeline-output/match_runs.csv) ----
        baseline_top1 = int(ranked_top[0]["recipient_id"])
        row = make_row(
            donor_id=donor_id,
            baseline_top1=baseline_top1,
            llm_primary=int(llm.primary_recipient_id),
            llm_backup=int(llm.backup_recipient_id),
            overrode_baseline=bool(llm.overrode_baseline),
            override_reason=llm.override_reason,
            match_cid=match_cid,
            tx_hash=txr.tx_hash,
            match_id=txr.match_id,
            gas_used=txr.gas_used,
        )
        append_match_row(row)

        return MatchResponse(
            donor_id=donor_id,
            baseline_top=baseline,
            llm_decision=llm,
            match_cid=match_cid,
            onchain=onchain,
        )

    except (ChainDataError, IPFSError, LLMError, PinataError, ChainWriteError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})