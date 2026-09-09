from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from src.chain_writer import ChainWriteError, send_create_match
from src.pinata_client import PinataError, pin_json
from src.web3_client import ConfigError, make_web3_context


VALID_CID = "b" + base64.b32encode(
    b"\x01\x70\x12\x20" + b"\0" * 32
).decode("ascii").rstrip("=").lower()


class RuntimeBoundaryTests(unittest.TestCase):
    def test_web3_context_rejects_wrong_chain_before_loading_contract(self) -> None:
        settings = SimpleNamespace(
            network="sepolia",
            rpc_url="https://example.invalid",
            abi_path=Path("unused-abi.json"),
            address_path=Path("unused-address.json"),
        )
        web3_factory = MagicMock()
        web3_factory.return_value.is_connected.return_value = True
        web3_factory.return_value.eth.chain_id = 1
        with (
            patch("src.web3_client.Web3", web3_factory),
            self.assertRaisesRegex(ConfigError, "Expected sepolia chain ID"),
        ):
            make_web3_context(settings)

    def test_web3_context_rejects_address_without_contract_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abi_path = root / "abi.json"
            address_path = root / "address.json"
            runtime = bytes.fromhex("60016000")
            source_sha256 = "1" * 64
            artifact_sha256 = "2" * 64
            abi_path.write_text(
                json.dumps(
                    {
                        "abi": [],
                        "source_sha256": source_sha256,
                        "artifact_sha256": artifact_sha256,
                    }
                ),
                encoding="utf-8",
            )
            address_path.write_text(
                json.dumps(
                    {
                        "address": "0x" + "11" * 20,
                        "chain_id": 11155111,
                        "network": "sepolia",
                        "protocol_id": "ODA-SYNTH-MULTIORGAN-1.0",
                        "source_sha256": source_sha256,
                        "artifact_sha256": artifact_sha256,
                        "deployed_runtime_sha256": hashlib.sha256(runtime).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                network="sepolia",
                rpc_url="https://example.invalid",
                abi_path=abi_path,
                address_path=address_path,
            )
            web3_factory = MagicMock()
            web3_factory.return_value.is_connected.return_value = True
            web3_factory.return_value.eth.chain_id = 11155111
            web3_factory.return_value.eth.get_code.return_value = b""
            web3_factory.is_address.return_value = True
            web3_factory.to_checksum_address.side_effect = lambda value: value
            with (
                patch("src.web3_client.Web3", web3_factory),
                self.assertRaisesRegex(ConfigError, "No contract bytecode"),
            ):
                make_web3_context(settings)

    def test_web3_context_rejects_bytecode_not_bound_to_deployment_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abi_path = root / "abi.json"
            address_path = root / "address.json"
            source_sha256 = "1" * 64
            artifact_sha256 = "2" * 64
            abi_path.write_text(
                json.dumps(
                    {
                        "abi": [],
                        "source_sha256": source_sha256,
                        "artifact_sha256": artifact_sha256,
                    }
                ),
                encoding="utf-8",
            )
            address_path.write_text(
                json.dumps(
                    {
                        "address": "0x" + "11" * 20,
                        "chain_id": 11155111,
                        "network": "sepolia",
                        "protocol_id": "ODA-SYNTH-MULTIORGAN-1.0",
                        "source_sha256": source_sha256,
                        "artifact_sha256": artifact_sha256,
                        "deployed_runtime_sha256": hashlib.sha256(b"expected").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                network="sepolia",
                rpc_url="https://example.invalid",
                abi_path=abi_path,
                address_path=address_path,
            )
            web3_factory = MagicMock()
            web3_factory.return_value.is_connected.return_value = True
            web3_factory.return_value.eth.chain_id = 11155111
            web3_factory.return_value.eth.get_code.return_value = b"different"
            web3_factory.is_address.return_value = True
            web3_factory.to_checksum_address.side_effect = lambda value: value
            with (
                patch("src.web3_client.Web3", web3_factory),
                self.assertRaisesRegex(ConfigError, "canonical deployment record"),
            ):
                make_web3_context(settings)

    def test_create_match_requires_event_payload_to_match_submission(self) -> None:
        sender = "0x" + "22" * 20
        w3 = MagicMock()
        contract = MagicMock()
        contract.functions.authorizedDecisionServices.return_value.call.return_value = True
        receipt = SimpleNamespace(tx_hash="0xabc", gas_used=123)
        event = {
            "args": {
                "matchId": 7,
                "donorId": 1,
                "primaryRecipientId": 4,
                "backupRecipientId": 8,
                "decisionCID": VALID_CID,
                "recordedBy": sender,
            }
        }
        contract.events.MatchCreated.return_value.process_receipt.return_value = [event]
        context = SimpleNamespace(w3=w3, contract=contract)
        with (
            patch("src.chain_writer.private_key_address", return_value=sender),
            patch("src.chain_writer.send_signed_transaction", return_value=receipt),
        ):
            result = send_create_match(
                context,
                "0x" + "33" * 32,
                1,
                4,
                8,
                VALID_CID,
            )
            self.assertEqual(result.match_id, 7)

            event["args"]["decisionCID"] = "bafydifferent"
            with self.assertRaisesRegex(ChainWriteError, "does not match"):
                send_create_match(
                    context,
                    "0x" + "33" * 32,
                    1,
                    4,
                    8,
                    VALID_CID,
                )

    def test_create_match_rejects_a_non_cid_before_signing(self) -> None:
        context = SimpleNamespace(w3=MagicMock(), contract=MagicMock())
        with (
            patch("src.chain_writer.send_signed_transaction") as send,
            self.assertRaisesRegex(ChainWriteError, "decision CID is invalid"),
        ):
            send_create_match(context, "0x" + "33" * 32, 1, 4, 8, "not-a-cid")
        send.assert_not_called()

    def test_create_match_rejects_malformed_ids_and_authorization_before_signing(self) -> None:
        context = SimpleNamespace(w3=MagicMock(), contract=MagicMock())
        with (
            patch("src.chain_writer.send_signed_transaction") as send,
            self.assertRaisesRegex(ChainWriteError, "positive integers"),
        ):
            send_create_match(context, "0x" + "33" * 32, True, 4, 8, VALID_CID)
        send.assert_not_called()

        with (
            patch("src.chain_writer.private_key_address", return_value="0x" + "22" * 20),
            patch("src.chain_writer.send_signed_transaction") as send,
            self.assertRaisesRegex(ChainWriteError, "must be different"),
        ):
            send_create_match(context, "0x" + "33" * 32, 1, 4, 4, VALID_CID)
        send.assert_not_called()

        context.contract.functions.authorizedDecisionServices.return_value.call.return_value = "false"
        with (
            patch("src.chain_writer.private_key_address", return_value="0x" + "22" * 20),
            patch("src.chain_writer.send_signed_transaction") as send,
            self.assertRaisesRegex(ChainWriteError, "malformed value"),
        ):
            send_create_match(context, "0x" + "33" * 32, 1, 4, 8, VALID_CID)
        send.assert_not_called()

    def test_pinata_errors_are_wrapped_and_invalid_cids_are_rejected(self) -> None:
        with patch(
            "src.pinata_client.requests.post",
            side_effect=requests.RequestException("secret transport detail"),
        ):
            with self.assertRaisesRegex(PinataError, "request failed") as raised:
                pin_json("secret-jwt", {"encrypted": True})
        self.assertNotIn("secret", str(raised.exception))

        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"IpfsHash": "../unsafe"},
        )
        with patch("src.pinata_client.requests.post", return_value=response):
            with self.assertRaisesRegex(PinataError, "invalid IpfsHash"):
                pin_json("secret-jwt", {"encrypted": True})


if __name__ == "__main__":
    unittest.main()
