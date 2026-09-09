from dataclasses import dataclass
from pathlib import Path
import os

from evaluation.project_environment import load_authoritative_project_env

APP_DIR = Path(__file__).resolve().parents[1]
load_authoritative_project_env(APP_DIR / ".env")

REPO_ROOT = APP_DIR.parents[0]  # app/ is under repo root
INTEGRATION_DIR = REPO_ROOT / "integration"
ABI_PATH = INTEGRATION_DIR / "abi" / "TransplantManagement.json"

def _network_name() -> str:
    return os.getenv("NETWORK", "sepolia").strip().lower()

def _address_path() -> Path:
    return INTEGRATION_DIR / "addresses" / f"{_network_name()}.json"

@dataclass(frozen=True)
class Settings:
    network: str
    rpc_url: str

    # Keys / services (not required for /health, required later)
    decision_service_private_key: str | None
    pinata_jwt: str | None
    pinata_gateway: str
    offchain_encryption_key: str | None

    abi_path: Path
    address_path: Path

def get_settings() -> Settings:
    network = _network_name()
    network_rpc = os.getenv(f"{network.upper()}_RPC_URL", "").strip()
    return Settings(
        network=network,
        rpc_url=network_rpc or os.getenv("RPC_URL", "").strip(),
        decision_service_private_key=os.getenv("DECISION_SERVICE_PRIVATE_KEY"),
        pinata_jwt=os.getenv("PINATA_JWT"),
        pinata_gateway=os.getenv("PINATA_GATEWAY", "").strip(),
        offchain_encryption_key=os.getenv("OFFCHAIN_ENCRYPTION_KEY"),
        abi_path=ABI_PATH,
        address_path=_address_path(),
    )
