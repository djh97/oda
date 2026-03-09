import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = APP_DIR / "seed-data"
PINATA_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

def pin_file(jwt: str, file_path: Path, name: str) -> str:
    headers = {"Authorization": f"Bearer {jwt}"}
    metadata = {"name": name}

    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f)}
        data = {
            "pinataMetadata": json.dumps(metadata),
            "pinataOptions": json.dumps({"cidVersion": 1})
        }
        r = requests.post(PINATA_URL, headers=headers, files=files, data=data, timeout=60)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Pinata upload failed ({r.status_code}): {r.text}")

    return r.json()["IpfsHash"]

def main():
    load_dotenv(APP_DIR / ".env")
    jwt = os.getenv("PINATA_JWT", "").strip()
    if not jwt:
        raise RuntimeError("PINATA_JWT missing in app/.env")

    donor_path = SEED_DIR / "donor_1.json"
    if not donor_path.exists():
        raise RuntimeError(f"Missing {donor_path}")

    print("Uploading donor + 10 recipients to Pinata...")
    donor_cid = pin_file(jwt, donor_path, "ODA_seed_donor_1")
    print(f"SEED_DONOR_CID={donor_cid}")

    for i in range(1, 11):
        p = SEED_DIR / f"recipient_{i}.json"
        if not p.exists():
            raise RuntimeError(f"Missing {p}")
        cid = pin_file(jwt, p, f"ODA_seed_recipient_{i}")
        print(f"SEED_RECIPIENT{i}_CID={cid}")

    print("\n✅ Done. Copy these lines into app/.env")

if __name__ == "__main__":
    main()