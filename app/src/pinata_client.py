import json
import requests
from typing import Any, Dict

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

    r = requests.post(PIN_JSON_URL, headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code not in (200, 201):
        raise PinataError(f"Pinata pinJSONToIPFS failed ({r.status_code}): {r.text}")

    data = r.json()
    cid = data.get("IpfsHash")
    if not cid:
        raise PinataError(f"Pinata response missing IpfsHash: {data}")
    return cid