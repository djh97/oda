import json
import requests
from typing import Any, Dict

from .ipfs_client import IPFSError, validate_cid
from .secure_storage import encrypt_json

PIN_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"

class PinataError(RuntimeError):
    pass

def pin_json(jwt: str, obj: Dict[str, Any], name: str = "match_rationale") -> str:
    """
    Uploads JSON to Pinata and returns CID (IpfsHash).
    """
    if not jwt:
        raise PinataError("PINATA_JWT is missing")

    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }

    payload = {
        "pinataMetadata": {"name": name},
        "pinataContent": obj,
        "pinataOptions": {"cidVersion": 1},
    }

    try:
        r = requests.post(PIN_JSON_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise PinataError("Pinata pinJSONToIPFS request failed") from exc
    if r.status_code not in (200, 201):
        raise PinataError(f"Pinata pinJSONToIPFS failed with HTTP {r.status_code}")

    try:
        data = r.json()
    except ValueError as exc:
        raise PinataError("Pinata response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise PinataError("Pinata response was not a JSON object")
    cid = data.get("IpfsHash")
    if not cid:
        raise PinataError("Pinata response did not contain an IpfsHash")
    try:
        return validate_cid(cid)
    except IPFSError as exc:
        raise PinataError("Pinata response contained an invalid IpfsHash") from exc


def pin_encrypted_json(
    jwt: str,
    obj: Dict[str, Any],
    encryption_key: str,
    *,
    aad: str,
    name: str,
) -> str:
    envelope = encrypt_json(obj, encryption_key, aad=aad)
    return pin_json(jwt, envelope, name=name)
