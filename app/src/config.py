from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from the app/ directory
APP_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=str(APP_DIR / ".env"), override=False)

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
    sepolia_rpc_url: str

    # Keys / services (not required for /health, required later)
    llm_private_key: str | None
    openai_api_key: str | None
    openai_model_id: str | None
    pinata_jwt: str | None
    pinata_gateway: str

    abi_path: Path
    address_path: Path

def get_settings() -> Settings:
    return Settings(
        network=_network_name(),
        sepolia_rpc_url=os.getenv("SEPOLIA_RPC_URL", "").strip(),
        llm_private_key=os.getenv("LLM_PRIVATE_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model_id=os.getenv("OPENAI_MODEL_ID"),
        pinata_jwt=os.getenv("PINATA_JWT"),
        pinata_gateway=os.getenv("PINATA_GATEWAY", "https://gateway.pinata.cloud/ipfs/").strip(),
        abi_path=ABI_PATH,
        address_path=_address_path(),
    )
