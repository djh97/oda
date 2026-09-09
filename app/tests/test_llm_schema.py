from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.llm_client import (
    LLMInputError,
    LLMOutputCoverageError,
    LLMOutputSchemaError,
    build_note_review_payload,
    parse_note_review,
)
from src.schemas import NoteAssessment


class LLMContractTests(unittest.TestCase):
    def test_payload_contains_only_id_and_note_per_candidate(self) -> None:
        payload = build_note_review_payload(
            "abc",
            "kidney",
            [{"recipient_id": 1, "medical_notes": "Stable.", "reference_note_state": "temporary_hold"}],
        )
        self.assertEqual(payload["candidates"], [{"recipient_id": 1, "medical_notes": "Stable."}])

    def test_payload_rejects_coerced_identifiers_and_invalid_context(self) -> None:
        for recipient_id in ("1", True, 1.0):
            with self.subTest(recipient_id=recipient_id):
                with self.assertRaises(LLMInputError):
                    build_note_review_payload(
                        "abc",
                        "kidney",
                        [{"recipient_id": recipient_id, "medical_notes": "Stable."}],
                    )
        with self.assertRaises(LLMInputError):
            build_note_review_payload(
                "",
                "kidney",
                [{"recipient_id": 1, "medical_notes": "Stable."}],
            )
        with self.assertRaises(LLMInputError):
            build_note_review_payload(
                "abc",
                "pancreas",
                [{"recipient_id": 1, "medical_notes": "Stable."}],
            )
        with self.assertRaises(LLMInputError):
            build_note_review_payload(
                "abc",
                "kidney",
                [{"recipient_id": 1, "medical_notes": 123}],
            )

    def test_state_and_evidence_code_must_agree(self) -> None:
        with self.assertRaises(ValidationError):
            NoteAssessment.model_validate({
                "recipient_id": 1,
                "state": "eligible",
                "evidence_codes": ["active_infection"],
            })

    def test_schema_does_not_coerce_quoted_or_duplicate_recipient_ids(self) -> None:
        with self.assertRaises(ValidationError):
            NoteAssessment.model_validate({
                "recipient_id": "1",
                "state": "eligible",
                "evidence_codes": ["no_current_concern"],
            })
        with self.assertRaises(LLMOutputSchemaError):
            parse_note_review(
                {
                    "case_id": "abc",
                    "assessments": [
                        {
                            "recipient_id": 1,
                            "state": "eligible",
                            "evidence_codes": ["no_current_concern"],
                        },
                        {
                            "recipient_id": 1,
                            "state": "eligible",
                            "evidence_codes": ["no_current_concern"],
                        },
                    ],
                },
                "abc",
                [1],
            )

    def test_parser_requires_exact_coverage(self) -> None:
        value = {
            "case_id": "abc",
            "assessments": [
                {"recipient_id": 1, "state": "eligible", "evidence_codes": ["no_current_concern"]}
            ],
        }
        with self.assertRaises(LLMOutputCoverageError):
            parse_note_review(value, "abc", [1, 2])

    def test_parser_rejects_wrong_case_id(self) -> None:
        value = {
            "case_id": "wrong",
            "assessments": [
                {"recipient_id": 1, "state": "eligible", "evidence_codes": ["no_current_concern"]}
            ],
        }
        with self.assertRaises(LLMOutputCoverageError):
            parse_note_review(value, "abc", [1])


if __name__ == "__main__":
    unittest.main()
