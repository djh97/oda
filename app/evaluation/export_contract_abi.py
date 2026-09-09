"""Export the ABI from the current Foundry artifact with source checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
ARTIFACT_PATH = (
    IMPLEMENTATION_DIR
    / "smart-contracts"
    / "out"
    / "TransplantManagement.sol"
    / "TransplantManagement.json"
)
SOURCE_PATH = IMPLEMENTATION_DIR / "smart-contracts" / "src" / "TransplantManagement.sol"
OUTPUT_PATH = IMPLEMENTATION_DIR / "integration" / "abi" / "TransplantManagement.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8-sig"))
    abi = artifact.get("abi")
    if not isinstance(abi, list):
        raise RuntimeError(f"No ABI array found in {ARTIFACT_PATH}")
    value = {
        "contract": "TransplantManagement",
        "source_sha256": _sha256(SOURCE_PATH),
        "artifact_sha256": _sha256(ARTIFACT_PATH),
        "abi": abi,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported current ABI to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
