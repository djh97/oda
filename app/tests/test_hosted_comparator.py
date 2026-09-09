from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluation import run_model_evaluation, smoke_test_model
from src.llm_client import LLMOutputCoverageError, call_note_review
from src.schemas import NoteReviewBatch


def _completion(review: NoteReviewBatch, *, raw_text: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(
        parsed=review,
        content=raw_text if raw_text is not None else review.model_dump_json(),
        refusal=None,
    )
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o-mini-2024-07-18",
        system_fingerprint="fp-test",
        service_tier="default",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25),
        choices=[SimpleNamespace(message=message)],
    )


def _client(completion: SimpleNamespace) -> tuple[SimpleNamespace, MagicMock]:
    parse = MagicMock(return_value=completion)
    client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )
    )
    return client, parse


def test_hosted_call_uses_the_frozen_structured_output_settings() -> None:
    review = NoteReviewBatch.model_validate(
        {
            "case_id": "validation-case",
            "assessments": [
                {
                    "recipient_id": 1,
                    "state": "eligible",
                    "evidence_codes": ["no_current_concern"],
                }
            ],
        }
    )
    client, parse = _client(_completion(review))

    result = call_note_review(
        "gpt-4o-mini-2024-07-18",
        "unused-with-injected-client",
        "validation-case",
        "kidney",
        [{"recipient_id": 1, "medical_notes": "No current concern."}],
        client=client,
    )

    request = parse.call_args.kwargs
    assert request["model"] == "gpt-4o-mini-2024-07-18"
    assert request["temperature"] == 0
    assert request["seed"] == 20260907
    assert request["max_completion_tokens"] == 512
    assert request["store"] is False
    assert request["response_format"] is NoteReviewBatch
    assert result.metadata.provider == "OpenAI"
    assert result.metadata.input_tokens == 100
    assert result.metadata.output_tokens == 25


def test_hosted_schema_failure_retains_raw_text_and_usage() -> None:
    wrong_case = NoteReviewBatch.model_validate(
        {
            "case_id": "wrong-case",
            "assessments": [
                {
                    "recipient_id": 1,
                    "state": "eligible",
                    "evidence_codes": ["no_current_concern"],
                }
            ],
        }
    )
    raw = json.dumps(wrong_case.model_dump(mode="json"), sort_keys=True)
    client, _parse = _client(_completion(wrong_case, raw_text=raw))

    with pytest.raises(LLMOutputCoverageError) as raised:
        call_note_review(
            "gpt-4o-mini-2024-07-18",
            "unused-with-injected-client",
            "validation-case",
            "kidney",
            [{"recipient_id": 1, "medical_notes": "No current concern."}],
            client=client,
        )

    assert raised.value.raw_text == raw
    assert raised.value.model_run_metadata.input_tokens == 100
    assert raised.value.model_run_metadata.output_tokens == 25


def test_hosted_condition_is_pinned_in_runners() -> None:
    expected = run_model_evaluation.load_protocol()["model_evaluation"][
        "hosted_comparator"
    ]["model_id"]
    run_model_evaluation._validate_condition_model("openai", expected)
    with pytest.raises(RuntimeError, match="preserved expected model"):
        run_model_evaluation._validate_condition_model("openai", "gpt-4o-mini")

    settings = smoke_test_model._condition_runtime_settings("openai_preflight")
    assert settings == {
        "temperature": 0,
        "seed": 20260907,
        "provider": "OpenAI",
        "do_sample": False,
        "maximum_new_tokens": 512,
        "endpoint": "chat_completions_structured_outputs",
        "sdk_max_retries": 0,
        "store": False,
    }
