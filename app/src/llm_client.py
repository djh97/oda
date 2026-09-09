"""Strict language-model interface for synthetic note-readiness review."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from pydantic import ValidationError

from .policy import SUPPORTED_ORGANS, load_protocol
from .schemas import ModelRunMetadata, NoteReviewBatch


SYSTEM_PROMPT = """You review wholly synthetic transplant-candidate notes for a technical benchmark. You do not rank candidates, choose recipients, estimate outcomes, or make clinical decisions. Classify the current readiness signal expressed in each note using exactly one state.

eligible means no current protocol-defined hold is stated. This includes uncomplicated notes, explicit negation, and historical or resolved conditions.

review_required means the note is incomplete or ambiguous and explicitly requires further workup, clarification, or specialist clearance. It is not a temporary hold unless the note says transplantation is deferred.

temporary_hold means the note explicitly reports a current active infection, sepsis or instability, active malignancy treatment, or a current decision to defer transplantation.

When a note contains more than one signal, apply this precedence: temporary_hold, then review_required, then eligible. Return every explicitly supported evidence code that belongs to the selected highest-priority state. Do not return lower-priority, historical, negated, resolved, quoted, or superseded signals as evidence for that state.

The allowed evidence codes for eligible are no_current_concern, negated_condition, resolved_condition, and historical_condition_cleared. The allowed evidence codes for review_required are infection_workup_pending, indeterminate_lesion, specialist_clearance_pending, and missing_or_conflicting_information. The allowed evidence codes for temporary_hold are active_infection, sepsis_or_instability, active_malignancy_treatment, and explicit_temporary_deferral.

Return JSON only, without markdown fences or commentary, in this exact structure: {"case_id":"the supplied case ID","assessments":[{"recipient_id":1,"state":"eligible","evidence_codes":["no_current_concern"]}]}. Include one assessment for every supplied recipient ID, preserve the supplied case ID, and do not infer facts that are absent from the note."""
OPENAI_API_BASE = "https://api.openai.com/v1"

MODEL_EVALUATION = load_protocol()["model_evaluation"]
HOSTED_COMPARATOR = MODEL_EVALUATION["hosted_comparator"]
DEFAULT_TEMPERATURE = float(MODEL_EVALUATION["temperature"])
DEFAULT_SEED = int(MODEL_EVALUATION["seed"])
DEFAULT_TIMEOUT_SECONDS = float(HOSTED_COMPARATOR["timeout_seconds"])


class LLMError(RuntimeError):
    pass


class LLMInputError(LLMError):
    pass


class LLMClientConfigurationError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class LLMRefusalError(LLMError):
    pass


class LLMEmptyResponseError(LLMError):
    pass


class LLMOutputSchemaError(LLMError):
    pass


class LLMOutputCoverageError(LLMError):
    pass


@dataclass(frozen=True)
class NoteReviewResult:
    review: NoteReviewBatch
    metadata: ModelRunMetadata
    raw_text: str


def build_note_review_payload(
    case_id: str,
    organ_type: str,
    recipients: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(case_id, str) or not case_id.strip():
        raise LLMInputError("A non-empty string case_id is required")
    if not isinstance(organ_type, str):
        raise LLMInputError("organ_type must be a string")
    normalized_organ = organ_type.strip().lower()
    if normalized_organ not in SUPPORTED_ORGANS:
        raise LLMInputError(f"Unsupported organ_type {organ_type!r}")

    candidates: List[Dict[str, Any]] = []
    for recipient in recipients:
        raw_recipient_id = (
            recipient.get("recipient_id")
            if "recipient_id" in recipient
            else recipient.get("recipientId")
        )
        if type(raw_recipient_id) is not int or raw_recipient_id <= 0:
            raise LLMInputError("Each candidate must have a positive recipient_id")
        raw_note = recipient.get("medical_notes")
        if not isinstance(raw_note, str) or not raw_note.strip():
            raise LLMInputError(f"Candidate {raw_recipient_id} has no medical_notes")
        candidates.append(
            {"recipient_id": raw_recipient_id, "medical_notes": raw_note.strip()}
        )
    if not candidates:
        raise LLMInputError("At least one candidate note is required")
    if len({item["recipient_id"] for item in candidates}) != len(candidates):
        raise LLMInputError("Candidate recipient IDs must be unique")
    return {
        "case_id": case_id.strip(),
        "organ_type": normalized_organ,
        "candidates": candidates,
    }


def parse_note_review(
    value: NoteReviewBatch | Mapping[str, Any] | str,
    case_id: str,
    expected_recipient_ids: Iterable[int],
) -> NoteReviewBatch:
    try:
        if isinstance(value, NoteReviewBatch):
            review = value
        elif isinstance(value, str):
            review = NoteReviewBatch.model_validate_json(value)
        else:
            review = NoteReviewBatch.model_validate(value)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMOutputSchemaError(f"Model output failed strict schema validation: {exc}") from exc

    if review.case_id != str(case_id):
        raise LLMOutputCoverageError(
            f"Model returned case_id {review.case_id!r}; expected {case_id!r}"
        )
    expected = {int(value) for value in expected_recipient_ids}
    observed = [int(item.recipient_id) for item in review.assessments]
    if len(observed) != len(set(observed)):
        raise LLMOutputCoverageError("Model returned duplicate recipient assessments")
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise LLMOutputCoverageError(
            f"Model returned incomplete coverage; missing={missing}, extra={extra}"
        )
    return review


def call_note_review(
    model_id: str,
    api_key: str,
    case_id: str,
    organ_type: str,
    recipients: List[Dict[str, Any]],
    *,
    client: Any | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> NoteReviewResult:
    if not model_id or not str(model_id).strip():
        raise LLMClientConfigurationError("A model ID is required")
    if client is None and (not api_key or not str(api_key).strip()):
        raise LLMClientConfigurationError("An OpenAI API key is required")

    payload = build_note_review_payload(case_id, organ_type, recipients)
    expected_ids = [item["recipient_id"] for item in payload["candidates"]]
    if client is None:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_API_BASE,
                timeout=timeout_seconds,
                max_retries=int(HOSTED_COMPARATOR["sdk_max_retries"]),
            )
        except Exception as exc:
            raise LLMClientConfigurationError(
                f"Unable to initialize the OpenAI client ({type(exc).__name__})"
            ) from exc

    started = time.perf_counter()
    try:
        completion = client.beta.chat.completions.parse(
            model=str(model_id),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True, sort_keys=True)},
            ],
            response_format=NoteReviewBatch,
            temperature=float(temperature),
            seed=int(seed),
            max_completion_tokens=int(HOSTED_COMPARATOR["maximum_completion_tokens"]),
            store=bool(HOSTED_COMPARATOR["store"]),
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMOutputSchemaError(
            f"Provider response failed structured-output parsing ({type(exc).__name__})"
        ) from exc
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        details = [type(exc).__name__]
        if status is not None:
            details.append(f"status={status}")
        if request_id:
            details.append(f"request_id={request_id}")
        raise LLMAPIError(f"Note-review API call failed ({', '.join(details)})") from exc
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

    usage = getattr(completion, "usage", None)
    metadata = ModelRunMetadata(
        model_id=str(getattr(completion, "model", None) or model_id),
        provider=str(HOSTED_COMPARATOR["provider"]),
        response_id=str(getattr(completion, "id", "") or "") or None,
        system_fingerprint=(
            str(getattr(completion, "system_fingerprint", "") or "") or None
        ),
        service_tier=str(getattr(completion, "service_tier", "") or "") or None,
        latency_ms=latency_ms,
        input_tokens=int(usage.prompt_tokens) if usage is not None else None,
        output_tokens=int(usage.completion_tokens) if usage is not None else None,
    )
    if not completion.choices:
        error = LLMEmptyResponseError("Model response contained no choices")
        error.raw_text = ""
        error.model_run_metadata = metadata
        raise error
    message = completion.choices[0].message
    raw_text = str(message.content or "")
    if getattr(message, "refusal", None):
        error = LLMRefusalError("Model refused the synthetic note-review request")
        error.raw_text = raw_text
        error.model_run_metadata = metadata
        raise error
    if not raw_text.strip():
        error = LLMEmptyResponseError("Model response contained no structured content")
        error.raw_text = raw_text
        error.model_run_metadata = metadata
        raise error
    parsed = getattr(message, "parsed", None)
    try:
        review = parse_note_review(
            parsed if parsed is not None else raw_text,
            case_id,
            expected_ids,
        )
    except (LLMOutputSchemaError, LLMOutputCoverageError) as exc:
        exc.raw_text = raw_text
        exc.model_run_metadata = metadata
        raise
    return NoteReviewResult(review=review, metadata=metadata, raw_text=raw_text)
