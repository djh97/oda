"""Run post hoc boundary and failure-path challenges without external services."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence
from unittest.mock import patch

import requests

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.decision_guard import GuardAbstention, GuardError, apply_guarded_policy
from src.llm_client import (
    LLMOutputCoverageError,
    LLMOutputSchemaError,
    parse_note_review,
)
from src.pinata_client import PinataError, pin_json
from src.schemas import NoteReviewBatch
from src.secure_storage import (
    SecureStorageError,
    decrypt_json,
    encrypt_json,
    generate_encryption_key,
)


DEFAULT_OUTPUT = (
    APP_DIR
    / "pipeline-output"
    / "current"
    / "robustness"
    / "failure_path_challenges.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ranked(total: int, selectable_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
    selectable = set(range(1, total + 1) if selectable_ids is None else selectable_ids)
    rank = 0
    rows: list[dict[str, Any]] = []
    for recipient_id in range(1, total + 1):
        is_selectable = recipient_id in selectable
        if is_selectable:
            rank += 1
        rows.append(
            {
                "recipient_id": recipient_id,
                "selectable": is_selectable,
                "rank": rank if is_selectable else None,
                "score": float(100 - recipient_id),
            }
        )
    return rows


def _review(
    total: int,
    *,
    holds: Iterable[int] = (),
    reviews: Iterable[int] = (),
) -> NoteReviewBatch:
    held = set(holds)
    review_required = set(reviews)
    assessments = []
    for recipient_id in range(1, total + 1):
        if recipient_id in held:
            state = "temporary_hold"
            codes = ["active_infection"]
        elif recipient_id in review_required:
            state = "review_required"
            codes = ["specialist_clearance_pending"]
        else:
            state = "eligible"
            codes = ["no_current_concern"]
        assessments.append(
            {
                "recipient_id": recipient_id,
                "state": state,
                "evidence_codes": codes,
            }
        )
    return NoteReviewBatch.model_validate(
        {"case_id": "robustness-case", "assessments": assessments}
    )


def _selected(
    total: int,
    expected_pair: Sequence[int],
    *,
    holds: Iterable[int] = (),
    reviews: Iterable[int] = (),
    selectable_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    decision = apply_guarded_policy(
        1,
        _ranked(total, selectable_ids),
        _review(total, holds=holds, reviews=reviews),
        range(1, total + 1),
    )
    observed = [decision.primary_recipient_id, decision.backup_recipient_id]
    if observed != list(expected_pair):
        raise AssertionError(f"Expected pair {list(expected_pair)}, observed {observed}")
    return {"outcome": "selected", "pair": observed}


def _guard_abstains(
    total: int,
    *,
    holds: Iterable[int] = (),
    selectable_ids: Iterable[int] | None = None,
) -> dict[str, str]:
    try:
        apply_guarded_policy(
            1,
            _ranked(total, selectable_ids),
            _review(total, holds=holds),
            range(1, total + 1),
        )
    except GuardAbstention as exc:
        return {"outcome": "abstained", "reason": str(exc)}
    raise AssertionError("The guard selected a pair instead of abstaining")


def _guard_rejects_incomplete_coverage(*, extra: bool = False) -> dict[str, str]:
    total = 3
    review = _review(total + 1 if extra else total - 1)
    try:
        apply_guarded_policy(
            1,
            _ranked(total),
            review,
            range(1, total + 1),
        )
    except GuardError as exc:
        return {"outcome": "rejected", "reason": str(exc)}
    raise AssertionError("Incomplete or extra assessment coverage was accepted")


def _parse_rejects(payload: dict[str, Any], error_type: type[Exception]) -> dict[str, str]:
    try:
        parse_note_review(payload, "robustness-case", [1, 2])
    except error_type as exc:
        return {"outcome": "rejected", "reason": str(exc)}
    raise AssertionError(f"Expected {error_type.__name__}")


def _storage_rejects(mode: str) -> dict[str, str]:
    key = generate_encryption_key()
    envelope = encrypt_json({"synthetic": True}, key, aad="recipient:1")
    candidate_key = key
    expected_aad = "recipient:1"
    if mode == "tampered_ciphertext":
        raw = bytearray(base64.urlsafe_b64decode(envelope["ciphertext_b64"]))
        raw[0] ^= 1
        envelope["ciphertext_b64"] = base64.urlsafe_b64encode(raw).decode("ascii")
    elif mode == "wrong_key":
        candidate_key = generate_encryption_key()
    elif mode == "wrong_context":
        expected_aad = "donor:1"
    elif mode == "malformed_envelope":
        envelope["unexpected"] = True
    else:
        raise ValueError(f"Unknown storage mode {mode}")
    try:
        decrypt_json(envelope, candidate_key, expected_aad=expected_aad)
    except SecureStorageError as exc:
        return {"outcome": "rejected", "reason": str(exc)}
    raise AssertionError(f"Storage challenge {mode} was accepted")


def _gateway_unavailable() -> dict[str, str]:
    with patch(
        "src.pinata_client.requests.post",
        side_effect=requests.RequestException("simulated outage"),
    ):
        try:
            pin_json("redacted-test-token", {"encrypted": True})
        except PinataError as exc:
            return {"outcome": "failed_closed", "reason": str(exc)}
    raise AssertionError("The simulated gateway outage was not propagated")


def _gateway_invalid_cid() -> dict[str, str]:
    response = SimpleNamespace(status_code=200, json=lambda: {"IpfsHash": "../unsafe"})
    with patch("src.pinata_client.requests.post", return_value=response):
        try:
            pin_json("redacted-test-token", {"encrypted": True})
        except PinataError as exc:
            return {"outcome": "rejected", "reason": str(exc)}
    raise AssertionError("An invalid storage CID was accepted")


def _scenario_definitions() -> list[tuple[str, str, Callable[[], dict[str, Any]]]]:
    valid_two = {
        "case_id": "robustness-case",
        "assessments": [
            {"recipient_id": 1, "state": "eligible", "evidence_codes": ["no_current_concern"]},
            {"recipient_id": 2, "state": "eligible", "evidence_codes": ["no_current_concern"]},
        ],
    }
    return [
        ("guard_boundary", "two_candidates_both_available", lambda: _selected(2, (1, 2))),
        ("guard_boundary", "two_candidates_one_held", lambda: _guard_abstains(2, holds=(1,))),
        ("guard_boundary", "two_candidates_both_held", lambda: _guard_abstains(2, holds=(1, 2))),
        ("guard_boundary", "three_candidates_top_held", lambda: _selected(3, (2, 3), holds=(1,))),
        ("guard_boundary", "three_candidates_two_held", lambda: _guard_abstains(3, holds=(1, 2))),
        ("guard_boundary", "ten_candidates_first_eight_held", lambda: _selected(10, (9, 10), holds=range(1, 9))),
        ("guard_boundary", "twenty_candidates_mixed_holds", lambda: _selected(20, (2, 4), holds=(1, 3, 5, 8, 13))),
        ("guard_boundary", "one_structurally_selectable_candidate", lambda: _guard_abstains(4, selectable_ids=(1,))),
        ("guard_boundary", "zero_structurally_selectable_candidates", lambda: _guard_abstains(4, selectable_ids=())),
        ("guard_boundary", "review_required_candidates_remain_ordered", lambda: _selected(3, (1, 2), reviews=(1,))),
        ("model_output_validation", "missing_candidate_assessment", _guard_rejects_incomplete_coverage),
        ("model_output_validation", "extra_candidate_assessment", lambda: _guard_rejects_incomplete_coverage(extra=True)),
        (
            "model_output_validation",
            "wrong_case_identifier",
            lambda: _parse_rejects({**valid_two, "case_id": "wrong-case"}, LLMOutputCoverageError),
        ),
        (
            "model_output_validation",
            "state_evidence_mismatch",
            lambda: _parse_rejects(
                {
                    **valid_two,
                    "assessments": [
                        {
                            "recipient_id": 1,
                            "state": "eligible",
                            "evidence_codes": ["active_infection"],
                        },
                        valid_two["assessments"][1],
                    ],
                },
                LLMOutputSchemaError,
            ),
        ),
        ("storage_failure", "tampered_ciphertext", lambda: _storage_rejects("tampered_ciphertext")),
        ("storage_failure", "wrong_encryption_key", lambda: _storage_rejects("wrong_key")),
        ("storage_failure", "wrong_record_context", lambda: _storage_rejects("wrong_context")),
        ("storage_failure", "malformed_encryption_envelope", lambda: _storage_rejects("malformed_envelope")),
        ("storage_failure", "storage_gateway_unavailable", _gateway_unavailable),
        ("storage_failure", "storage_gateway_returns_invalid_cid", _gateway_invalid_cid),
    ]


def run_challenges() -> dict[str, Any]:
    results = []
    for category, name, challenge in _scenario_definitions():
        try:
            observed = challenge()
            results.append(
                {"category": category, "name": name, "status": "passed", "observed": observed}
            )
        except Exception as exc:  # Keep a complete machine-readable failure ledger.
            results.append(
                {
                    "category": category,
                    "name": name,
                    "status": "failed",
                    "observed": {"exception": type(exc).__name__, "reason": str(exc)},
                }
            )

    category_counts: dict[str, dict[str, int]] = {}
    for result in results:
        counts = category_counts.setdefault(result["category"], {"passed": 0, "failed": 0})
        counts[result["status"]] += 1
    passed = sum(result["status"] == "passed" for result in results)
    report = {
        "schema_version": "1.0",
        "evaluation_type": "post_hoc_component_robustness_challenge",
        "frozen_model_comparison_modified": False,
        "external_services_contacted": False,
        "status": "passed" if passed == len(results) else "failed",
        "scenario_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "category_counts": category_counts,
        "candidate_set_sizes_exercised": [2, 3, 4, 10, 20],
        "source_sha256": {
            "runner": _sha256(Path(__file__)),
            "decision_guard": _sha256(APP_DIR / "src" / "decision_guard.py"),
            "llm_client": _sha256(APP_DIR / "src" / "llm_client.py"),
            "schemas": _sha256(APP_DIR / "src" / "schemas.py"),
            "secure_storage": _sha256(APP_DIR / "src" / "secure_storage.py"),
            "pinata_client": _sha256(APP_DIR / "src" / "pinata_client.py"),
        },
        "scenarios": results,
        "interpretation": (
            "This post hoc suite tests deterministic component behavior under specified boundary "
            "and injected-failure conditions. It does not evaluate clinical validity, live-service "
            "availability, key compromise, chain reorganizations, or model accuracy on independently "
            "authored notes."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_challenges()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "scenario_count", "passed", "failed", "category_counts")}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
