from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
from starlette.requests import Request

from evaluation.paper_full_workflow import DEMO_CASE_PATH, _runtime_profile
from src.chain_writer import TxResult
from src.evidence_view import EvidenceViewError, load_latest_evidence
from src.llm_client import NoteReviewResult
from src.main import home, match
from src.schemas import MatchRequest, MatchResponse, ModelRunMetadata, NoteReviewBatch


class MainWorkflowTests(unittest.TestCase):
    def test_endpoint_records_the_deterministic_guard_selection(self) -> None:
        case = json.loads(Path(DEMO_CASE_PATH).read_text(encoding="utf-8"))
        donor = _runtime_profile(case["donor"])
        recipients = [_runtime_profile(item) for item in case["recipients"]]
        review = NoteReviewBatch.model_validate({
            "case_id": "live-donor-1",
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
            metadata=ModelRunMetadata(model_id="ft:test", latency_ms=12.5),
            raw_text=review.model_dump_json(),
        )
        settings = SimpleNamespace(
            pinata_jwt="test-jwt",
            decision_service_private_key="0x" + "11" * 32,
            offchain_encryption_key="test-encryption-key",
            pinata_gateway="https://example.invalid/ipfs/",
            network="sepolia",
        )
        context = SimpleNamespace(contract_address="0x" + "22" * 20)
        donor_record = {
            "donorId": 1,
            "profileCID": "donor-cid",
            "registered": True,
            "ethicallyEligible": True,
            "finalized": False,
            "hasOpenMatch": False,
        }
        recipient_records = [
            {
                "recipientId": index,
                "profileCID": f"recipient-{index}-cid",
                "registered": True,
                "ethicallyEligible": True,
                "reserved": False,
                "transplanted": False,
            }
            for index in range(1, 11)
        ]
        pinned: dict[str, dict] = {}
        model_call = MagicMock(return_value=model_result)

        def fake_pin(_jwt: str, value: dict, _key: str, **_kwargs: object) -> str:
            pinned["decision"] = value
            return "decision-cid"

        def fake_fetch(_gateway: str, cid: str, _key: str, **_kwargs: object) -> dict:
            if cid == "donor-cid":
                return donor
            return pinned["decision"]

        with (
            patch("src.main.get_settings", return_value=settings),
            patch("src.main.make_web3_context", return_value=context),
            patch("src.main.get_donor", return_value=donor_record),
            patch("src.main.get_all_recipients", return_value=recipient_records),
            patch("src.main.fetch_encrypted_json_from_ipfs", side_effect=fake_fetch),
            patch("src.main.fetch_encrypted_json_batch", return_value=recipients),
            patch("src.main._get_local_runtime", return_value=("ft:test", model_call)),
            patch("src.main.pin_encrypted_json", side_effect=fake_pin),
            patch(
                "src.main.send_create_match",
                return_value=TxResult(tx_hash="0x" + "a" * 64, gas_used=123456, match_id=1),
            ) as chain_call,
            patch("src.main.append_tx"),
            patch("src.main.append_match_row"),
        ):
            response = match(MatchRequest(donor_id=1))

        self.assertIsInstance(response, MatchResponse)
        self.assertEqual(response.guarded_decision.primary_recipient_id, case["reference"]["primary_recipient_id"])
        self.assertEqual(response.guarded_decision.backup_recipient_id, case["reference"]["backup_recipient_id"])
        model_recipients = model_call.call_args.kwargs["recipients"]
        self.assertTrue(all("reference_note_state" not in item for item in model_recipients))
        self.assertEqual(chain_call.call_args.args[3], response.guarded_decision.primary_recipient_id)
        self.assertEqual(chain_call.call_args.args[4], response.guarded_decision.backup_recipient_id)
        self.assertEqual(pinned["decision"]["model_request_config"]["model_id_requested"], "ft:test")

        malformed = response.model_dump(mode="json")
        selected_id = response.guarded_decision.primary_recipient_id
        selected_assessment = next(
            item
            for item in malformed["guarded_decision"]["note_assessments"]
            if item["recipient_id"] == selected_id
        )
        selected_assessment["state"] = "temporary_hold"
        selected_assessment["evidence_codes"] = ["active_infection"]
        with self.assertRaisesRegex(ValidationError, "selected candidate"):
            MatchResponse.model_validate(malformed)

        misordered = response.model_dump(mode="json")
        misordered["baseline_top"][0], misordered["baseline_top"][1] = (
            misordered["baseline_top"][1],
            misordered["baseline_top"][0],
        )
        with self.assertRaisesRegex(ValidationError, "consecutive ranked prefix"):
            MatchResponse.model_validate(misordered)

        with tempfile.TemporaryDirectory() as directory:
            app_dir = Path(directory) / "app"
            current_dir = app_dir / "pipeline-output" / "current"
            run_dir = current_dir / "full_workflow" / "canonical"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "run_summary.json"
            snapshot_path = run_dir / "ui_evidence_snapshot.json"
            artifact_manifest_path = run_dir / "artifact_manifest.json"
            completion_path = run_dir / "completion.json"
            summary_path.write_text(json.dumps({"complete": True}), encoding="utf-8")
            snapshot_path.write_text(response.model_dump_json(), encoding="utf-8")
            artifact_manifest_path.write_text("{}\n", encoding="utf-8")
            completion_path.write_text(
                json.dumps({
                    "schema_version": "1.0",
                    "status": "completed",
                    "mode": "full",
                    "run_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "artifact_manifest_sha256": hashlib.sha256(
                        artifact_manifest_path.read_bytes()
                    ).hexdigest(),
                }),
                encoding="utf-8",
            )
            (current_dir / "latest_full_workflow.json").write_text(
                json.dumps({
                    "run_directory": str(run_dir.relative_to(app_dir)).replace("\\", "/"),
                    "run_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "ui_evidence_snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                }),
                encoding="utf-8",
            )
            loaded = load_latest_evidence(
                app_dir=app_dir,
                current_output_dir=current_dir,
            )
            completion_path.write_text(
                json.dumps({"schema_version": "1.0", "status": "completed", "mode": "seed_only"}),
                encoding="utf-8",
            )
            with self.assertRaises(EvidenceViewError):
                load_latest_evidence(app_dir=app_dir, current_output_dir=current_dir)
        self.assertEqual(loaded.onchain.tx_hash, "0x" + "a" * 64)
        self.assertEqual(
            loaded.guarded_decision.primary_recipient_id,
            response.guarded_decision.primary_recipient_id,
        )

        with (
            patch("src.main.load_latest_evidence", return_value=response),
            patch("src.main.get_settings") as settings_call,
        ):
            request = Request({
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/",
                "raw_path": b"/",
                "query_string": b"capture=decision",
                "headers": [],
                "client": ("test", 1),
                "server": ("test", 80),
                "root_path": "",
            })
            capture_page = home(request)
        html = capture_page.body.decode("utf-8")
        self.assertEqual(capture_page.status_code, 200)
        self.assertIn('class="capture-decision"', html)
        self.assertIn('const canonicalEvidence = {', html)
        settings_call.assert_not_called()

    def test_endpoint_rejects_provider_model_substitution_before_persistence(self) -> None:
        settings = SimpleNamespace(
            pinata_jwt="test-jwt",
            decision_service_private_key="0x" + "11" * 32,
            offchain_encryption_key="test-encryption-key",
            pinata_gateway="https://example.invalid/ipfs/",
            network="sepolia",
        )
        context = SimpleNamespace(contract_address="0x" + "22" * 20)
        donor = {
            "donor_id": 1,
            "organ_type": "lung",
            "blood_type": "O",
            "age_years": 40,
            "weight_kg": 80,
            "height_cm": 175,
        }
        recipient = {
            "recipient_id": 1,
            "organ_type": "lung",
            "blood_type": "O",
            "crossmatch_result": "negative",
            "age_years": 40,
            "weight_kg": 80,
            "height_cm": 175,
            "waiting_time_days": 10,
            "synthetic_lung_priority_score": 50,
            "medical_notes": "No current concern.",
        }
        model_result = NoteReviewResult(
            review=NoteReviewBatch.model_validate(
                {
                    "case_id": "live-donor-1",
                    "assessments": [
                        {
                            "recipient_id": 1,
                            "state": "eligible",
                            "evidence_codes": ["no_current_concern"],
                        }
                    ],
                }
            ),
            metadata=ModelRunMetadata(model_id="ft:substituted", latency_ms=1.0),
            raw_text="{}",
        )
        donor_record = {
            "donorId": 1,
            "profileCID": "donor-cid",
            "registered": True,
            "ethicallyEligible": True,
            "finalized": False,
            "hasOpenMatch": False,
        }
        recipient_records = [
            {
                "recipientId": 1,
                "profileCID": "recipient-cid",
                "registered": True,
                "ethicallyEligible": True,
                "reserved": False,
                "transplanted": False,
            },
            {
                "recipientId": 2,
                "profileCID": "recipient-2-cid",
                "registered": True,
                "ethicallyEligible": True,
                "reserved": False,
                "transplanted": False,
            },
        ]
        recipients = [recipient, {**recipient, "recipient_id": 2, "waiting_time_days": 9}]
        model_call = MagicMock(return_value=model_result)
        with (
            patch("src.main.get_settings", return_value=settings),
            patch("src.main.make_web3_context", return_value=context),
            patch("src.main.get_donor", return_value=donor_record),
            patch("src.main.get_all_recipients", return_value=recipient_records),
            patch("src.main.fetch_encrypted_json_from_ipfs", return_value=donor),
            patch("src.main.fetch_encrypted_json_batch", return_value=recipients),
            patch(
                "src.main._get_local_runtime",
                return_value=("ft:expected", model_call),
            ),
            patch("src.main.pin_encrypted_json") as pin,
            patch("src.main.send_create_match") as send,
        ):
            response = match(MatchRequest(donor_id=1))

        self.assertEqual(response.status_code, 400)
        self.assertIn("different from the configured model", response.body.decode("utf-8"))
        pin.assert_not_called()
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
