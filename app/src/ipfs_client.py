"""Authenticated retrieval helpers for encrypted off-chain records."""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Any, Dict, Iterable, Sequence, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from .secure_storage import decrypt_json


class IPFSError(RuntimeError):
    pass


_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _decode_base58(value: str) -> bytes:
    number = 0
    for character in value:
        position = _BASE58_ALPHABET.find(character)
        if position < 0:
            raise IPFSError("IPFS CID is missing or malformed")
        number = number * 58 + position
    encoded = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\0" * (len(value) - len(value.lstrip("1"))) + encoded


def _read_uvarint(value: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    start = offset
    while offset < len(value) and offset - start < 10:
        byte = value[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if byte < 0x80:
            if offset - start > 1 and byte == 0:
                raise IPFSError("IPFS CID contains a noncanonical integer")
            return result, offset
        shift += 7
    raise IPFSError("IPFS CID contains an invalid integer")


def validate_cid(cid: object) -> str:
    value = str(cid or "").strip()
    if not value or len(value) > 128:
        raise IPFSError("IPFS CID is missing or malformed")
    if value.startswith("Qm"):
        decoded = _decode_base58(value)
        if len(value) != 46 or len(decoded) != 34 or decoded[:2] != b"\x12\x20":
            raise IPFSError("IPFS CIDv0 is malformed")
        return value
    if not re.fullmatch(r"b[a-z2-7]+", value):
        raise IPFSError("IPFS CID must be canonical base32 CIDv1 or base58btc CIDv0")
    payload = value[1:]
    try:
        padding = "=" * ((8 - len(payload) % 8) % 8)
        decoded = base64.b32decode((payload.upper() + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise IPFSError("IPFS CIDv1 has invalid base32 encoding") from exc
    version, offset = _read_uvarint(decoded, 0)
    codec, offset = _read_uvarint(decoded, offset)
    hash_code, offset = _read_uvarint(decoded, offset)
    digest_length, offset = _read_uvarint(decoded, offset)
    if (
        version != 1
        or codec <= 0
        or hash_code <= 0
        or digest_length <= 0
        or digest_length != len(decoded) - offset
    ):
        raise IPFSError("IPFS CIDv1 has an invalid multicodec or multihash payload")
    return value


def _gateway_base(value: object, *, cid: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise IPFSError(f"Invalid IPFS gateway configuration for CID {cid}")
    path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def fetch_json_from_ipfs(gateway_base: str, cid: str, timeout: int = 10, retries: int = 2) -> Dict[str, Any]:
    """
    Fetch JSON from IPFS via an HTTP gateway.
    gateway_base example: "https://gateway.pinata.cloud/ipfs/"
    """
    cid = validate_cid(cid)
    if timeout <= 0 or retries < 0:
        raise ValueError("IPFS timeout must be positive and retries cannot be negative")
    gateway_base = _gateway_base(gateway_base, cid=cid)
    url = f"{gateway_base}{quote(cid, safe='')}"

    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
            else:
                value = r.json()
                if not isinstance(value, dict):
                    raise ValueError("response JSON is not an object")
                return value
        except (requests.RequestException, ValueError) as exc:
            last_error = type(exc).__name__
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
        else:
            raise IPFSError(
                f"IPFS fetch failed for CID {cid} after {retries + 1} attempts ({last_error})"
            )

    raise IPFSError(f"IPFS fetch failed for CID {cid}")


def fetch_encrypted_json_from_ipfs(
    gateway_base: str,
    cid: str,
    encryption_key: str,
    *,
    expected_aad: str,
    timeout: int = 10,
    retries: int = 2,
) -> Dict[str, Any]:
    envelope = fetch_json_from_ipfs(gateway_base, cid, timeout=timeout, retries=retries)
    try:
        return decrypt_json(envelope, encryption_key, expected_aad=expected_aad)
    except Exception as exc:
        raise IPFSError(f"Encrypted IPFS record could not be authenticated: {exc}") from exc


def fetch_encrypted_json_batch(
    gateway_base: str,
    records: Iterable[Tuple[str, str]],
    encryption_key: str,
    *,
    timeout: int = 10,
    retries: int = 2,
    max_workers: int = 8,
) -> list[Dict[str, Any]]:
    """Retrieve encrypted records concurrently while preserving input order."""
    entries: Sequence[Tuple[str, str]] = tuple(records)
    if not entries:
        return []
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    def fetch(entry: Tuple[str, str]) -> Dict[str, Any]:
        cid, expected_aad = entry
        return fetch_encrypted_json_from_ipfs(
            gateway_base,
            cid,
            encryption_key,
            expected_aad=expected_aad,
            timeout=timeout,
            retries=retries,
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(entries))) as executor:
        return list(executor.map(fetch, entries))
