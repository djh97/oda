from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .chain_reader import (
    ChainDataError,
    filter_eligible_recipients,
    get_all_recipients,
    get_donor,
    require_eligible_donor,
)
from .chain_writer import ChainWriteError, send_create_match
from .config import get_settings
from .decision_guard import GuardError, apply_guarded_policy
from .evidence_view import EvidenceViewError, load_latest_evidence
from .ipfs_client import IPFSError, fetch_encrypted_json_batch, fetch_encrypted_json_from_ipfs
from .llm_client import LLMError, SYSTEM_PROMPT
from .local_llm_client import LocalNoteReviewClient, validate_local_training_state
from .pinata_client import PinataError, pin_encrypted_json
from .policy import PolicyInputError, load_protocol, rank_recipients_baseline
from .run_logger import append_match_row, make_row
from .schemas import BaselineCandidate, MatchRequest, MatchResponse, OnChainRecord
from .secure_storage import SecureStorageError, canonical_json_sha256
from .tx_logger import append_tx
from .web3_client import ConfigError, make_web3_context, read_contract_health


APP_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol()

app = FastAPI(title="Organ Donation and Transplantation Workflow Prototype")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@lru_cache(maxsize=1)
def _get_local_runtime() -> tuple[str, LocalNoteReviewClient]:
    state = validate_local_training_state()
    model_id = str(state["adapter"]["model_id"])
    return model_id, LocalNoteReviewClient(model_id, adapter=True)


@app.get("/health", response_class=JSONResponse)
def health():
    try:
        settings = get_settings()
        context = make_web3_context(settings)
        data = read_contract_health(context)
        data["network"] = settings.network
        data["protocol_id"] = PROTOCOL["protocol_id"]
        return data
    except ConfigError as exc:
        return JSONResponse(status_code=500, content={"connected": False, "error": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"connected": False, "error": "Unexpected internal health-check error"},
        )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    capture_mode = request.query_params.get("capture", "").strip().lower()
    if capture_mode not in {"full", "decision"}:
        capture_mode = ""
    canonical_evidence = None
    contract_address = None
    regulator = None
    network = "unconfigured"
    if capture_mode:
        try:
            canonical_evidence = load_latest_evidence().model_dump(mode="json")
            network = "sepolia"
            contract_address = canonical_evidence["onchain"]["contract_address"]
        except EvidenceViewError:
            canonical_evidence = None
    else:
        try:
            settings = get_settings()
            network = settings.network
            context = make_web3_context(settings)
            status = read_contract_health(context)
            contract_address = status.get("contract_address")
            regulator = status.get("regulator")
        except Exception:
            pass
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "app_title": "Transplant Workflow Prototype",
            "subtitle": "Transparent ranking, synthetic note review, deterministic guard, and on-chain governance",
            "network": network,
            "contract_address": contract_address,
            "regulator": regulator,
            "protocol_id": PROTOCOL["protocol_id"],
            "capture_mode": capture_mode,
            "canonical_evidence": canonical_evidence,
        },
    )


@app.get("/evidence/latest", response_model=MatchResponse)
def latest_evidence():
    try:
        return load_latest_evidence()
    except EvidenceViewError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    settings = get_settings()
    missing = []
    if not settings.pinata_jwt:
        missing.append("PINATA_JWT")
    if not settings.pinata_gateway:
        missing.append("PINATA_GATEWAY")
    if not settings.decision_service_private_key:
        missing.append("DECISION_SERVICE_PRIVATE_KEY")
    if not settings.offchain_encryption_key:
        missing.append("OFFCHAIN_ENCRYPTION_KEY")
    if missing:
        return JSONResponse(status_code=500, content={"error": "Missing configuration: " + ", ".join(missing)})

    try:
        context = make_web3_context(settings)
        donor_id = int(req.donor_id)
        donor_record = get_donor(context, donor_id)
        require_eligible_donor(donor_record)
        recipient_records = filter_eligible_recipients(get_all_recipients(context))
        if len(recipient_records) < 2:
            raise ChainDataError("At least two eligible, unreserved recipient profiles are required")

        donor = fetch_encrypted_json_from_ipfs(
            settings.pinata_gateway,
            donor_record["profileCID"],
            settings.offchain_encryption_key,
            expected_aad=f"donor:{donor_id}",
        )
        if int(donor.get("donor_id", 0)) != donor_id:
            raise ChainDataError("Decrypted donor profile does not match its on-chain donor ID")

        recipients = fetch_encrypted_json_batch(
            settings.pinata_gateway,
            [
                (record["profileCID"], f"recipient:{record['recipientId']}")
                for record in recipient_records
            ],
            settings.offchain_encryption_key,
        )
        for record, recipient in zip(recipient_records, recipients):
            if int(recipient.get("recipient_id", 0)) != int(record["recipientId"]):
                raise ChainDataError(
                    f"Decrypted recipient profile does not match on-chain recipient {record['recipientId']}"
                )
        if len({int(item["recipient_id"]) for item in recipients}) != len(recipients):
            raise ChainDataError("Decrypted recipient profiles contain duplicate IDs")
        ranked = rank_recipients_baseline(donor, recipients, PROTOCOL)
        selectable = [item for item in ranked if item["selectable"]]
        if len(selectable) < 2:
            raise PolicyInputError("Fewer than two recipients pass the structured compatibility gates")

        case_id = f"live-donor-{donor_id}"
        model_id, model_caller = _get_local_runtime()
        model_result = model_caller(
            model_id=model_id,
            api_key="",
            case_id=case_id,
            organ_type=str(donor["organ_type"]),
            recipients=recipients,
        )
        if model_result.metadata.model_id != model_id:
            raise LLMError("The runtime returned a model identifier different from the configured model")
        decision = apply_guarded_policy(
            donor_id,
            ranked,
            model_result.review,
            [recipient["recipient_id"] for recipient in recipients],
        )
        rationale = {
            "protocol_id": PROTOCOL["protocol_id"],
            "case_id": case_id,
            "donor_id": donor_id,
            "profile_cids": {
                "donor": donor_record["profileCID"],
                "recipients": {
                    str(record["recipientId"]): record["profileCID"] for record in recipient_records
                },
            },
            "baseline_ranking": ranked,
            "note_review": model_result.review.model_dump(mode="json"),
            "model_run": model_result.metadata.model_dump(mode="json"),
            "model_request_config": {
                "model_id_requested": model_id,
                "temperature": PROTOCOL["model_evaluation"]["temperature"],
                "seed": PROTOCOL["model_evaluation"]["seed"],
                "provider": PROTOCOL["model_evaluation"]["provider"],
                "do_sample": PROTOCOL["model_evaluation"]["do_sample"],
                "maximum_new_tokens": PROTOCOL["model_evaluation"]["maximum_new_tokens"],
                "system_prompt_sha256": hashlib.sha256(
                    SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
            "guarded_decision": decision.model_dump(mode="json"),
        }
        decision_cid = pin_encrypted_json(
            settings.pinata_jwt,
            rationale,
            settings.offchain_encryption_key,
            aad=f"decision:donor:{donor_id}",
            name=f"encrypted_decision_donor_{donor_id}",
        )
        retrieved_rationale = fetch_encrypted_json_from_ipfs(
            settings.pinata_gateway,
            decision_cid,
            settings.offchain_encryption_key,
            expected_aad=f"decision:donor:{donor_id}",
        )
        if canonical_json_sha256(retrieved_rationale) != canonical_json_sha256(rationale):
            raise SecureStorageError("Retrieved decision artifact does not match the encrypted source object")
        transaction = send_create_match(
            context,
            settings.decision_service_private_key,
            donor_id,
            decision.primary_recipient_id,
            decision.backup_recipient_id,
            decision_cid,
        )
        append_tx(
            network=settings.network,
            role="DecisionService",
            function="createMatch",
            tx_hash=transaction.tx_hash,
            gas_used=transaction.gas_used,
            notes=(
                f"donor={donor_id},primary={decision.primary_recipient_id},"
                f"backup={decision.backup_recipient_id},encrypted_cid={decision_cid}"
            ),
        )
        append_match_row(make_row(
            donor_id=donor_id,
            baseline_primary=decision.baseline_primary_recipient_id,
            guarded_primary=decision.primary_recipient_id,
            guarded_backup=decision.backup_recipient_id,
            guard_changed_primary=decision.overrode_baseline,
            temporary_hold_count=len(decision.temporary_hold_recipient_ids),
            review_required_count=len(decision.review_required_recipient_ids),
            model_id=model_result.metadata.model_id,
            model_latency_ms=model_result.metadata.latency_ms,
            decision_cid=decision_cid,
            tx_hash=transaction.tx_hash,
            match_id=transaction.match_id,
            gas_used=transaction.gas_used,
        ))

        baseline_output = [
            BaselineCandidate(
                rank=item["rank"],
                recipient_id=item["recipient_id"],
                score=item["score"],
                priority_tier=item["priority_tier"],
                selectable=item["selectable"],
                exclusion_reasons=item["exclusion_reasons"],
                factors=item["factors"],
            )
            for item in ranked
        ]
        return MatchResponse(
            donor_id=donor_id,
            baseline_top=baseline_output,
            guarded_decision=decision,
            model_run=model_result.metadata,
            match_cid=decision_cid,
            onchain=OnChainRecord(
                tx_hash=transaction.tx_hash,
                match_id=transaction.match_id,
                gas_used=transaction.gas_used,
                contract_address=context.contract_address,
            ),
        )
    except (
        ChainDataError,
        ChainWriteError,
        GuardError,
        IPFSError,
        LLMError,
        PinataError,
        PolicyInputError,
        SecureStorageError,
    ) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected internal error. Review the local server output."},
        )
