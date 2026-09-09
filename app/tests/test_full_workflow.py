from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import evaluation.paper_full_workflow as workflow
from evaluation.build_manuscript_tables import (
    _address_rows,
    _listing_json,
    _metric_cell,
    _selected_recipient_roles,
)
from evaluation.paper_full_workflow import (
    DEMO_CASE_PATH,
    _aggregate_transaction_stages,
    _baseline_snapshot,
    _completed_full_workflow_runs,
    _decode_contract_struct,
    _expected_full_transaction_plan,
    _prepare_full_decision,
    _require_final_model_evidence,
    _runtime_profile,
    _validate_full_transaction_records,
    _verify_deployment,
    _write_completion,
    MATCH_FIELDS,
)
from src.llm_client import NoteReviewResult
from src.schemas import ModelRunMetadata, NoteReviewBatch
from src.secure_storage import generate_encryption_key
from src.transactions import TransactionReceipt


class FullWorkflowTests(unittest.TestCase):
    def test_journaled_pin_retains_one_envelope_and_verifies_readback(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        captured: dict[str, object] = {}

        class FakeLedger:
            retry_of = None

            def record(self, event_type, details):
                events.append((event_type, dict(details)))

        def fake_pin(_jwt, envelope, *, name):
            captured["envelope"] = envelope
            captured["name"] = name
            return "bafy-journaled-test"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            key = generate_encryption_key()
            with (
                patch("evaluation.paper_full_workflow.pin_json", side_effect=fake_pin),
                patch(
                    "evaluation.paper_full_workflow.fetch_json_from_ipfs",
                    side_effect=lambda *_args, **_kwargs: captured["envelope"],
                ),
            ):
                cid = workflow._pin_and_verify_encrypted_artifact(
                    run_dir,
                    FakeLedger(),
                    artifact_id="donor-1",
                    value={"donor_id": 1, "medical_notes": "synthetic note"},
                    pinata_jwt="test-jwt",
                    gateway="https://example.invalid/ipfs/",
                    encryption_key=key,
                    aad="donor:1",
                    name="test-donor",
                    readback_attempts=1,
                )

            self.assertEqual(cid, "bafy-journaled-test")
            self.assertTrue((run_dir / "encrypted_envelopes" / "donor-1.json").is_file())
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "pin_prepared",
                "pin_accepted",
                "pin_readback_started",
                "pin_readback_verified",
            ],
        )
        serialized = repr(events)
        self.assertNotIn(key, serialized)
        self.assertNotIn("synthetic note", serialized)

    def test_public_baseline_snapshot_omits_internal_tie_break_fields(self) -> None:
        with (
            patch.object(
                workflow,
                "assert_test_seed_not_retired",
                side_effect=ValueError("retired test seed"),
            ),
            patch.object(workflow, "load_authoritative_project_env") as load_environment,
            self.assertRaisesRegex(ValueError, "retired test seed"),
        ):
            workflow.run(stop_after_seed=True)
        load_environment.assert_not_called()

        snapshot = _baseline_snapshot(
            [
                {
                    "rank": 1,
                    "recipient_id": 4,
                    "score": 88.5,
                    "priority_tier": 0,
                    "selectable": True,
                    "exclusion_reasons": [],
                    "factors": {"distance": 0.8},
                    "waiting_time_days": 120,
                }
            ]
        )
        self.assertEqual(snapshot[0]["recipient_id"], 4)
        self.assertNotIn("waiting_time_days", snapshot[0])

    def test_canonical_workflow_requires_complete_frozen_model_evidence(self) -> None:
        summaries = {"fine_tuned": {"model_id_requested_values": ["ft:test:model"]}}
        expected_hashes = {"test_lock_sha256": "a" * 64}
        with (
            patch(
                "evaluation.build_manuscript_tables._validate_analysis_summaries",
                return_value=summaries,
            ) as validate_summaries,
            patch(
                "evaluation.build_manuscript_tables._validate_pretest_artifacts",
            ) as validate_pretests,
            patch(
                "evaluation.build_manuscript_tables._final_model_evidence_hashes",
                return_value=expected_hashes,
            ) as evidence_hashes,
        ):
            self.assertEqual(
                _require_final_model_evidence("ft:test:model"),
                expected_hashes,
            )

        validate_summaries.assert_called_once_with()
        validate_pretests.assert_called_once_with(summaries)
        evidence_hashes.assert_called_once_with()

        summaries["fine_tuned"]["model_id_requested_values"] = ["ft:other:model"]
        with (
            patch(
                "evaluation.build_manuscript_tables._validate_analysis_summaries",
                return_value=summaries,
            ),
            patch("evaluation.build_manuscript_tables._validate_pretest_artifacts"),
            self.assertRaisesRegex(RuntimeError, "not the fine-tuned model"),
        ):
            _require_final_model_evidence("ft:test:model")

    def test_transaction_stage_summary_preserves_stage_order(self) -> None:
        records = [
            {
                "category": "Deployment",
                "gas_used": 100,
                "fee_wei": 200,
                "confirmation_seconds": 4.0,
            },
            {
                "category": "Governance",
                "gas_used": 50,
                "fee_wei": 100,
                "confirmation_seconds": 6.0,
            },
            {
                "category": "Governance",
                "gas_used": 70,
                "fee_wei": 140,
                "confirmation_seconds": 8.0,
            },
        ]

        stages = _aggregate_transaction_stages(records)

        self.assertEqual([row["category"] for row in stages], ["Deployment", "Governance"])
        self.assertEqual(stages[1]["transaction_count"], 2)
        self.assertEqual(stages[1]["gas_used_total"], 120)
        self.assertEqual(stages[1]["confirmation_seconds_median"], 7.0)

        summary = {
            "contract_address": "0xcontract",
            "guarded_decision": {
                "primary_recipient_id": 4,
                "backup_recipient_id": 5,
                "baseline_primary_recipient_id": 4,
            },
        }
        self.assertEqual(
            _selected_recipient_roles(summary),
            [(4, "guarded primary; baseline primary"), (5, "guarded backup")],
        )
        addresses = {
            "REGULATOR_PRIVATE_KEY": "0xregulator",
            "HOSPITAL_PRIVATE_KEY": "0xhospital",
            "ETHICS_PRIVATE_KEY": "0xethics",
            "MEDICAL_PRIVATE_KEY": "0xmedical",
            "DECISION_SERVICE_PRIVATE_KEY": "0xservice",
            "DONOR_PRIVATE_KEY": "0xdonor",
            "RECIPIENT4_PRIVATE_KEY": "0xrecipient4",
            "RECIPIENT5_PRIVATE_KEY": "0xrecipient5",
        }
        address_text = _address_rows(summary, addresses)
        self.assertIn("Recipient 4 (guarded primary; baseline primary)", address_text)
        self.assertNotIn("Recipient 6", address_text)
        self.assertIn("123/400", _metric_cell(0.3075, [0.26, 0.35], count=123, denominator=400))

        listing_summary = {
            "demo_case_id": "oda-demo-kidney-001",
            "guarded_decision": {
                "baseline_primary_recipient_id": 10,
                "primary_recipient_id": 6,
                "backup_recipient_id": 8,
                "temporary_hold_recipient_ids": [10],
                "review_required_recipient_ids": [8],
                "note_assessments": [
                    {
                        "recipient_id": recipient_id,
                        "state": "temporary_hold" if recipient_id == 10 else "eligible",
                        "evidence_codes": [
                            "explicit_temporary_deferral"
                            if recipient_id == 10
                            else "no_current_concern"
                        ],
                    }
                    for recipient_id in range(1, 11)
                ],
            },
        }
        listing = json.loads(_listing_json(listing_summary))
        self.assertEqual(
            [item["recipient_id"] for item in listing["assessments"]],
            [10, 6, 8],
        )

        decoded_match = _decode_contract_struct(
            (
                1, 1, 4, 5, 4, False, "0xservice", "decision-cid", "",
                True, True, True, True, True, True, False,
            ),
            MATCH_FIELDS,
            label="Match",
        )
        self.assertTrue(decoded_match["finalized"])
        self.assertFalse(decoded_match["cancelled"])
        with self.assertRaisesRegex(RuntimeError, "expected 16"):
            _decode_contract_struct(tuple(range(15)), MATCH_FIELDS, label="Match")

        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current"
            run_dir = current / "full_workflow" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
            with patch("evaluation.paper_full_workflow.CURRENT_OUTPUT_DIR", current):
                _write_completion(run_dir, mode="full")
                self.assertEqual(_completed_full_workflow_runs(), [run_dir.resolve()])

    def test_full_transaction_plan_requires_all_45_ordered_role_calls(self) -> None:
        primary_id = 4
        backup_id = 5
        match_id = 1
        addresses = {
            "REGULATOR_PRIVATE_KEY": "0x" + "1" * 40,
            "HOSPITAL_PRIVATE_KEY": "0x" + "2" * 40,
            "ETHICS_PRIVATE_KEY": "0x" + "3" * 40,
            "MEDICAL_PRIVATE_KEY": "0x" + "4" * 40,
            "DONOR_PRIVATE_KEY": "0x" + "5" * 40,
            "DECISION_SERVICE_PRIVATE_KEY": "0x" + "6" * 40,
            "RECIPIENT4_PRIVATE_KEY": "0x" + "7" * 40,
        }
        records = []
        for index, step in enumerate(
            _expected_full_transaction_plan(primary_id),
            start=1,
        ):
            if step["arguments"] == "dynamic":
                arguments = (
                    f"donor_id=1,primary={primary_id},backup={backup_id},"
                    "encrypted_decision_cid=<recorded separately>"
                    if step["function"] == "createMatch"
                    else f"match_id={match_id}"
                )
            else:
                arguments = step["arguments"]
            gas_used = 50_000
            gas_price = 10_000_000_000
            fee_wei = gas_used * gas_price
            records.append({
                "category": step["category"],
                "role": step["role"],
                "function": step["function"],
                "arguments": arguments,
                "sender": addresses[step["sender_key"]],
                "tx_hash": f"0x{index:064x}",
                "status": 1,
                "block_number": index,
                "gas_used": gas_used,
                "effective_gas_price_wei": gas_price,
                "fee_wei": fee_wei,
                "fee_native": format(Decimal(fee_wei) / Decimal(10**18), ".18f"),
                "confirmation_seconds": 1.0,
            })

        validation = _validate_full_transaction_records(
            records,
            addresses,
            primary_id,
            backup_id,
            match_id,
        )
        self.assertTrue(validation["complete"])
        self.assertEqual(validation["observed_transaction_count"], 45)

        with self.assertRaisesRegex(RuntimeError, "exactly 45"):
            _validate_full_transaction_records(
                records[:-1],
                addresses,
                primary_id,
                backup_id,
                match_id,
            )
        altered = deepcopy(records)
        altered[-1]["sender"] = addresses["HOSPITAL_PRIVATE_KEY"]
        with self.assertRaisesRegex(RuntimeError, "wrong synthetic role"):
            _validate_full_transaction_records(
                altered,
                addresses,
                primary_id,
                backup_id,
                match_id,
            )

    def test_deployment_bytecode_must_match_the_pinned_artifact(self) -> None:
        contract_address = "0x" + "8" * 40
        deployment = TransactionReceipt(
            tx_hash="0x" + "9" * 64,
            sender="0x" + "1" * 40,
            status=1,
            gas_used=100,
            effective_gas_price_wei=10,
            block_number=1,
            submitted_at_utc="2026-09-07T00:00:00Z",
            confirmed_at_utc="2026-09-07T00:00:01Z",
            confirmation_seconds=1.0,
            contract_address=contract_address,
        )
        artifact = {"deployedBytecode": {"object": "0x60016000"}}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            w3 = type(
                "FakeWeb3",
                (),
                {"eth": type("FakeEth", (), {"get_code": lambda _self, _address: bytes.fromhex("60016000")})()},
            )()
            record = _verify_deployment(run_dir, w3, artifact, deployment)
            self.assertTrue(record["matches"])
            self.assertTrue((run_dir / "deployment_verification.json").is_file())

            bad_w3 = type(
                "FakeWeb3",
                (),
                {"eth": type("FakeEth", (), {"get_code": lambda _self, _address: bytes.fromhex("60026000")})()},
            )()
            with self.assertRaisesRegex(RuntimeError, "differs from"):
                _verify_deployment(run_dir, bad_w3, artifact, deployment)

    def test_decision_artifact_is_prepared_without_latent_profile_fields(self) -> None:
        case = json.loads(DEMO_CASE_PATH.read_text(encoding="utf-8"))
        donor = _runtime_profile(case["donor"])
        recipients = [_runtime_profile(item) for item in case["recipients"]]
        review = NoteReviewBatch.model_validate({
            "case_id": case["case_id"],
            "assessments": [
                {
                    "recipient_id": item["recipient_id"],
                    "state": item["reference_note_state"],
                    "evidence_codes": item["reference_evidence_codes"],
                }
                for item in case["recipients"]
            ],
        })
        model_result = NoteReviewResult(
            review=review,
            metadata=ModelRunMetadata(model_id="ft:test", latency_ms=8.0),
            raw_text=review.model_dump_json(),
        )
        captured: dict[str, object] = {}

        def fake_pin(_jwt, value, _key, **_kwargs):
            captured["decision"] = value
            return "decision-cid"

        def fake_fetch(*_args, **_kwargs):
            return captured["decision"]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "evaluation.paper_full_workflow.call_note_review",
                    return_value=model_result,
                ) as model_call,
                patch(
                    "evaluation.paper_full_workflow.pin_encrypted_json",
                    side_effect=fake_pin,
                ),
                patch(
                    "evaluation.paper_full_workflow._fetch_profile_with_retry",
                    side_effect=fake_fetch,
                ),
            ):
                ranked, _model, guarded, comparison, cid = _prepare_full_decision(
                    Path(directory),
                    case,
                    donor,
                    recipients,
                    "donor-cid",
                    {index: f"recipient-{index}-cid" for index in range(1, 11)},
                    model_id="ft:test",
                    pinata_jwt="test-jwt",
                    pinata_gateway="https://example.invalid/ipfs/",
                    encryption_key="test-key",
                )

            self.assertTrue((Path(directory) / "model_and_guard_record.json").exists())

        supplied_profiles = model_call.call_args.kwargs["recipients"]
        self.assertTrue(all("reference_note_state" not in item for item in supplied_profiles))
        self.assertTrue(all("reference_evidence_codes" not in item for item in supplied_profiles))
        self.assertEqual(cid, "decision-cid")
        self.assertEqual(guarded.primary_recipient_id, case["reference"]["primary_recipient_id"])
        self.assertTrue(comparison["primary_conforms"])
        self.assertTrue(any(item["priority_tier"] >= 0 for item in ranked))

    def test_nonconformant_demo_is_recorded_and_persisted_without_oracle_override(self) -> None:
        case = json.loads(DEMO_CASE_PATH.read_text(encoding="utf-8"))
        donor = _runtime_profile(case["donor"])
        recipients = [_runtime_profile(item) for item in case["recipients"]]
        review = NoteReviewBatch.model_validate({
            "case_id": case["case_id"],
            "assessments": [
                {
                    "recipient_id": item["recipient_id"],
                    "state": "eligible",
                    "evidence_codes": ["no_current_concern"],
                }
                for item in case["recipients"]
            ],
        })
        model_result = NoteReviewResult(
            review=review,
            metadata=ModelRunMetadata(model_id="ft:test", latency_ms=8.0),
            raw_text=review.model_dump_json(),
        )

        captured: dict[str, object] = {}

        def fake_pin(_jwt, value, _key, **_kwargs):
            captured["decision"] = value
            return "decision-cid"

        def fake_fetch(*_args, **_kwargs):
            return captured["decision"]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "evaluation.paper_full_workflow.call_note_review",
                    return_value=model_result,
                ),
                patch(
                    "evaluation.paper_full_workflow.pin_encrypted_json",
                    side_effect=fake_pin,
                ),
                patch(
                    "evaluation.paper_full_workflow._fetch_profile_with_retry",
                    side_effect=fake_fetch,
                ),
            ):
                _, _, guarded, comparison, cid = _prepare_full_decision(
                    Path(directory),
                    case,
                    donor,
                    recipients,
                    "donor-cid",
                    {index: f"recipient-{index}-cid" for index in range(1, 11)},
                    model_id="ft:test",
                    pinata_jwt="test-jwt",
                    pinata_gateway="https://example.invalid/ipfs/",
                    encryption_key="test-key",
                )

            record = json.loads(
                (Path(directory) / "model_and_guard_record.json").read_text(encoding="utf-8")
            )

        self.assertEqual(cid, "decision-cid")
        self.assertEqual(guarded.primary_recipient_id, case["reference"]["baseline_order"][0])
        self.assertFalse(comparison["primary_conforms"])
        self.assertFalse(record["reference_comparison"]["primary_conforms"])
        self.assertNotIn("reference_comparison", captured["decision"])


if __name__ == "__main__":
    unittest.main()
