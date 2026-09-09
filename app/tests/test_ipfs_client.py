from __future__ import annotations

import base64
import time
import unittest
from unittest.mock import patch

from src.ipfs_client import (
    IPFSError,
    fetch_encrypted_json_batch,
    fetch_json_from_ipfs,
    validate_cid,
)


VALID_CID = "b" + base64.b32encode(
    b"\x01\x70\x12\x20" + b"\0" * 32
).decode("ascii").rstrip("=").lower()


class IPFSClientTests(unittest.TestCase):
    def test_batch_fetch_preserves_input_order(self) -> None:
        entries = [("slow", "recipient:1"), ("fast", "recipient:2")]

        def fake_fetch(_gateway, cid, _key, **_kwargs):
            if cid == "slow":
                time.sleep(0.02)
            return {"cid": cid}

        with patch("src.ipfs_client.fetch_encrypted_json_from_ipfs", side_effect=fake_fetch):
            result = fetch_encrypted_json_batch("https://example.invalid/", entries, "key", max_workers=2)

        self.assertEqual(result, [{"cid": "slow"}, {"cid": "fast"}])

    def test_batch_fetch_rejects_nonpositive_worker_count(self) -> None:
        with self.assertRaises(ValueError):
            fetch_encrypted_json_batch("https://example.invalid/", [("cid", "aad")], "key", max_workers=0)

    def test_fetch_error_does_not_echo_gateway_credentials(self) -> None:
        response = type("Response", (), {"status_code": 403})()
        with patch("src.ipfs_client.requests.get", return_value=response):
            with self.assertRaises(IPFSError) as raised:
                fetch_json_from_ipfs(
                    "https://secret-token@example.invalid/ipfs/",
                    VALID_CID,
                    retries=0,
                )
        self.assertIn(VALID_CID, str(raised.exception))
        self.assertNotIn("secret-token", str(raised.exception))

    def test_fetch_rejects_cid_path_injection_without_requesting_it(self) -> None:
        with patch("src.ipfs_client.requests.get") as request:
            with self.assertRaisesRegex(IPFSError, "canonical base32"):
                fetch_json_from_ipfs(
                    "https://example.invalid/ipfs/",
                    "../private",
                    retries=0,
                )
        request.assert_not_called()

    def test_cid_parser_validates_multicodec_and_multihash_structure(self) -> None:
        self.assertEqual(validate_cid(VALID_CID), VALID_CID)
        malformed = "b" + base64.b32encode(b"\x01\x70\x12\x20\0").decode(
            "ascii"
        ).rstrip("=").lower()
        with self.assertRaisesRegex(IPFSError, "multicodec or multihash"):
            validate_cid(malformed)


if __name__ == "__main__":
    unittest.main()
