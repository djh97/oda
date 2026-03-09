import time
import requests
from typing import Any, Dict, Optional, Tuple

class IPFSError(RuntimeError):
    pass

def fetch_json_from_ipfs(gateway_base: str, cid: str, timeout: int = 10, retries: int = 2) -> Dict[str, Any]:
    """
    Fetch JSON from IPFS via an HTTP gateway.
    gateway_base example: "https://gateway.pinata.cloud/ipfs/"
    """
    if not gateway_base.endswith("/"):
        gateway_base += "/"
    url = f"{gateway_base}{cid}"

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                raise IPFSError(f"IPFS fetch failed ({r.status_code}): {url}")
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise IPFSError(f"IPFS fetch error: {e}") from e

    raise IPFSError(f"IPFS fetch error: {last_err}")
