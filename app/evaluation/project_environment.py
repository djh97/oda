"""Load study runtime settings from app/.env as the authoritative source."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


APP_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = APP_DIR / ".env"

CONTROLLED_ENV_NAMES = (
    "NETWORK",
    "RPC_URL",
    "SEPOLIA_RPC_URL",
    "ETHERSCAN_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_MODEL",
    "OPENAI_MODEL_ID",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "PINATA_JWT",
    "PINATA_GATEWAY",
    "OFFCHAIN_ENCRYPTION_KEY",
    "TX_PRIORITY_FEE_GWEI",
    "TX_RECEIPT_TIMEOUT_S",
    "TX_POLL_LATENCY_S",
    "REGULATOR_PRIVATE_KEY",
    "HOSPITAL_PRIVATE_KEY",
    "ETHICS_PRIVATE_KEY",
    "MEDICAL_PRIVATE_KEY",
    "DONOR_PRIVATE_KEY",
    "DECISION_SERVICE_PRIVATE_KEY",
    *(f"RECIPIENT{index}_PRIVATE_KEY" for index in range(1, 11)),
)


class ProjectEnvironmentError(RuntimeError):
    pass


def load_authoritative_project_env(path: Path = ENV_PATH) -> Path:
    """Replace controlled shell values with literal values from the project file."""
    if os.getenv("ODA_DISABLE_DOTENV", "").strip() == "1":
        return path

    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    unexpected = sorted(set(values).difference(CONTROLLED_ENV_NAMES))
    if unexpected:
        raise ProjectEnvironmentError(
            "The project environment file contains unsupported variable names: "
            + ", ".join(unexpected)
        )
    for name in CONTROLLED_ENV_NAMES:
        os.environ.pop(name, None)
    for name, value in values.items():
        if value is not None:
            os.environ[name] = str(value)
    return path
