from __future__ import annotations

import unittest

from src.secure_storage import (
    SecureStorageError,
    assert_encrypted_envelope,
    canonical_json_sha256,
    decrypt_json,
    encrypt_json,
    generate_encryption_key,
)


class SecureStorageTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": {"d": 4, "c": 3}}),
            canonical_json_sha256({"a": {"c": 3, "d": 4}, "b": 2}),
        )

    def test_authenticated_round_trip(self) -> None:
        key = generate_encryption_key()
        source = {"blood_type": "A", "hla_typing": ["A*01"], "medical_notes": "Synthetic."}
        envelope = encrypt_json(source, key, aad="recipient:7")
        assert_encrypted_envelope(envelope)
        self.assertEqual(decrypt_json(envelope, key, expected_aad="recipient:7"), source)
        serialized = str(envelope)
        self.assertNotIn("blood_type", serialized)
        self.assertNotIn("Synthetic", serialized)

    def test_tampering_fails_authentication(self) -> None:
        key = generate_encryption_key()
        envelope = encrypt_json({"value": 1}, key, aad="test:1")
        envelope["ciphertext_b64"] = envelope["ciphertext_b64"][:-2] + "AA"
        with self.assertRaises(SecureStorageError):
            decrypt_json(envelope, key, expected_aad="test:1")

    def test_wrong_context_is_rejected(self) -> None:
        key = generate_encryption_key()
        envelope = encrypt_json({"value": 1}, key, aad="donor:1")
        with self.assertRaises(SecureStorageError):
            decrypt_json(envelope, key, expected_aad="recipient:1")

    def test_malformed_envelope_is_rejected_before_decryption(self) -> None:
        key = generate_encryption_key()
        envelope = encrypt_json({"value": 1}, key, aad="recipient:1")
        envelope["unexpected"] = True
        with self.assertRaisesRegex(SecureStorageError, "unexpected or missing fields"):
            decrypt_json(envelope, key, expected_aad="recipient:1")

        envelope.pop("unexpected")
        envelope["nonce_b64"] = "not*base64"
        with self.assertRaisesRegex(SecureStorageError, "URL-safe base64"):
            decrypt_json(envelope, key, expected_aad="recipient:1")


if __name__ == "__main__":
    unittest.main()
