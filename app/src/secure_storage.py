"""Authenticated encryption envelope for synthetic off-chain JSON records."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping


ENVELOPE_FORMAT = "oda-aesgcm-v1"
ALGORITHM = "AES-256-GCM"


class SecureStorageError(ValueError):
    pass


def _decode_urlsafe_base64(value: object, *, label: str) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise SecureStorageError(f"{label} is missing")
    try:
        padding = "=" * ((4 - len(text) % 4) % 4)
        return base64.b64decode(
            (text + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecureStorageError(f"{label} must be URL-safe base64") from exc


def canonical_json_sha256(obj: Mapping[str, Any]) -> str:
    encoded = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_encryption_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def decode_encryption_key(encoded: str) -> bytes:
    key = _decode_urlsafe_base64(encoded, label="OFFCHAIN_ENCRYPTION_KEY")
    if len(key) != 32:
        raise SecureStorageError("OFFCHAIN_ENCRYPTION_KEY must decode to exactly 32 bytes")
    return key


def key_identifier(encoded_key: str) -> str:
    return hashlib.sha256(decode_encryption_key(encoded_key)).hexdigest()[:16]


def encrypt_json(
    obj: Mapping[str, Any],
    encoded_key: str,
    *,
    aad: str,
) -> Dict[str, Any]:
    if not aad or not str(aad).strip():
        raise SecureStorageError("A nonempty encryption context is required")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise SecureStorageError("Install the cryptography package to use encrypted storage") from exc

    key = decode_encryption_key(encoded_key)
    nonce = os.urandom(12)
    plaintext = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    associated_data = str(aad).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return {
        "format": ENVELOPE_FORMAT,
        "algorithm": ALGORITHM,
        "key_id": key_identifier(encoded_key),
        "aad": str(aad),
        "nonce_b64": base64.urlsafe_b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
    }


def decrypt_json(
    envelope: Mapping[str, Any],
    encoded_key: str,
    *,
    expected_aad: str | None = None,
) -> Dict[str, Any]:
    assert_encrypted_envelope(envelope)
    aad = str(envelope.get("aad") or "")
    if not aad:
        raise SecureStorageError("Encrypted record has no authentication context")
    if expected_aad is not None and aad != expected_aad:
        raise SecureStorageError(f"Encryption context mismatch; expected {expected_aad!r}, found {aad!r}")
    if str(envelope.get("key_id") or "") != key_identifier(encoded_key):
        raise SecureStorageError("Encrypted record key identifier does not match the configured key")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = _decode_urlsafe_base64(envelope["nonce_b64"], label="Encrypted-record nonce")
        ciphertext = _decode_urlsafe_base64(
            envelope["ciphertext_b64"],
            label="Encrypted-record ciphertext",
        )
        plaintext = AESGCM(decode_encryption_key(encoded_key)).decrypt(
            nonce,
            ciphertext,
            aad.encode("utf-8"),
        )
        value = json.loads(plaintext.decode("utf-8"))
    except SecureStorageError:
        raise
    except Exception as exc:
        raise SecureStorageError("Encrypted record failed authentication or decoding") from exc
    if not isinstance(value, dict):
        raise SecureStorageError("Decrypted JSON record must be an object")
    return value


def assert_encrypted_envelope(value: Mapping[str, Any]) -> None:
    expected = {"format", "algorithm", "key_id", "aad", "nonce_b64", "ciphertext_b64"}
    if set(value) != expected:
        raise SecureStorageError("Encrypted record envelope has unexpected or missing fields")
    if value.get("format") != ENVELOPE_FORMAT or value.get("algorithm") != ALGORITHM:
        raise SecureStorageError("Encrypted record envelope has an unsupported format")
    if not re.fullmatch(r"[0-9a-f]{16}", str(value.get("key_id", ""))):
        raise SecureStorageError("Encrypted record has an invalid key identifier")
    if not str(value.get("aad") or "").strip():
        raise SecureStorageError("Encrypted record has no authentication context")
    nonce = _decode_urlsafe_base64(value.get("nonce_b64"), label="Encrypted-record nonce")
    ciphertext = _decode_urlsafe_base64(
        value.get("ciphertext_b64"),
        label="Encrypted-record ciphertext",
    )
    if len(nonce) != 12:
        raise SecureStorageError("Encrypted-record nonce must contain exactly 12 bytes")
    if len(ciphertext) < 16:
        raise SecureStorageError("Encrypted-record ciphertext is shorter than the authentication tag")
