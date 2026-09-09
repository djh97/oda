"""Validate required .env entries without printing secrets or contacting services."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from dotenv import dotenv_values
from web3 import Web3

from evaluation.project_environment import CONTROLLED_ENV_NAMES
from src.secure_storage import SecureStorageError, decode_encryption_key


APP_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = APP_DIR / ".env"
ACTOR_KEYS = (
    "REGULATOR_PRIVATE_KEY",
    "HOSPITAL_PRIVATE_KEY",
    "ETHICS_PRIVATE_KEY",
    "MEDICAL_PRIVATE_KEY",
    "DONOR_PRIVATE_KEY",
    "DECISION_SERVICE_PRIVATE_KEY",
    *(f"RECIPIENT{i}_PRIVATE_KEY" for i in range(1, 11)),
)


def _is_placeholder(value: str) -> bool:
    upper = value.upper()
    return not value or "YOUR_" in upper or "..." in value


def _required_for(mode: str) -> tuple[str, ...]:
    if mode == "fine-tune":
        return ()
    if mode == "model-eval":
        return ()
    if mode == "sepolia":
        return (
            "NETWORK",
            "SEPOLIA_RPC_URL",
            "PINATA_JWT",
            "PINATA_GATEWAY",
            "OFFCHAIN_ENCRYPTION_KEY",
            *ACTOR_KEYS,
        )
    return tuple(dict.fromkeys((*_required_for("fine-tune"), *_required_for("model-eval"), *_required_for("sepolia"))))


def validate(mode: str) -> tuple[list[str], list[str]]:
    if not ENV_PATH.exists():
        return [], [f"Missing file {ENV_PATH}"]
    raw = dotenv_values(ENV_PATH, interpolate=False)
    values = {key: str(value or "").strip() for key, value in raw.items()}
    present: list[str] = []
    issues: list[str] = []
    unexpected = sorted(set(values).difference(CONTROLLED_ENV_NAMES))
    if unexpected:
        issues.append("Unsupported environment variable names: " + ", ".join(unexpected))
    for name in _required_for(mode):
        value = values.get(name, "")
        if _is_placeholder(value):
            issues.append(f"{name} is missing or contains a placeholder")
        else:
            present.append(name)

    api_key = values.get("OPENAI_API_KEY", "")
    if api_key and not _is_placeholder(api_key) and not re.fullmatch(
        r"sk-[A-Za-z0-9_-]{16,}",
        api_key,
    ):
        issues.append("OPENAI_API_KEY does not have expected OpenAI key syntax")

    if mode in {"sepolia", "all"}:
        if values.get("NETWORK", "").lower() != "sepolia":
            issues.append("NETWORK must be sepolia for the manuscript evidence run")
        rpc = values.get("SEPOLIA_RPC_URL", "")
        if rpc and not _is_placeholder(rpc):
            parsed_rpc = urlsplit(rpc)
            if parsed_rpc.scheme.lower() not in {"http", "https"} or not parsed_rpc.netloc:
                issues.append("SEPOLIA_RPC_URL must be a complete HTTP or HTTPS URL")
        gateway = values.get("PINATA_GATEWAY", "")
        if gateway and not _is_placeholder(gateway):
            parsed_gateway = urlsplit(gateway)
            if (
                parsed_gateway.scheme.lower() not in {"http", "https"}
                or not parsed_gateway.netloc
                or not parsed_gateway.path.rstrip("/").endswith("/ipfs")
                or parsed_gateway.username is not None
                or parsed_gateway.password is not None
                or bool(parsed_gateway.query)
                or bool(parsed_gateway.fragment)
            ):
                issues.append(
                    "PINATA_GATEWAY must be an HTTP or HTTPS gateway URL ending in /ipfs/ "
                    "without embedded credentials, a query, or a fragment"
                )
        try:
            if values.get("OFFCHAIN_ENCRYPTION_KEY") and not _is_placeholder(values["OFFCHAIN_ENCRYPTION_KEY"]):
                decode_encryption_key(values["OFFCHAIN_ENCRYPTION_KEY"])
        except SecureStorageError as exc:
            issues.append(str(exc))

        jwt = values.get("PINATA_JWT", "")
        if jwt and not _is_placeholder(jwt) and not re.fullmatch(
            r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
            jwt,
        ):
            issues.append("PINATA_JWT does not have JWT syntax")

        addresses: dict[str, list[str]] = {}
        for name in ACTOR_KEYS:
            value = values.get(name, "")
            if _is_placeholder(value):
                continue
            try:
                address = Web3().eth.account.from_key(value).address.lower()
            except Exception:
                issues.append(f"{name} is not a valid EVM private key")
                continue
            addresses.setdefault(address, []).append(name)
        for names in addresses.values():
            if len(names) > 1:
                issues.append(f"Synthetic actor keys must be distinct; one address is used by {', '.join(names)}")
    base_url = values.get("OPENAI_BASE_URL", "")
    if base_url and not _is_placeholder(base_url):
        normalized = base_url.rstrip("/").lower()
        if normalized != "https://api.openai.com/v1":
            issues.append("OPENAI_BASE_URL must be absent or use the official OpenAI API endpoint")
    for name, allow_zero in (
        ("TX_PRIORITY_FEE_GWEI", True),
        ("TX_RECEIPT_TIMEOUT_S", False),
        ("TX_POLL_LATENCY_S", False),
    ):
        value = values.get(name, "")
        if not value:
            continue
        try:
            numeric = float(value)
        except ValueError:
            issues.append(f"{name} must be numeric")
            continue
        if not math.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
            qualifier = "nonnegative" if allow_zero else "positive"
            issues.append(f"{name} must be a finite {qualifier} number")
    return sorted(set(present)), sorted(set(issues))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fine-tune", "model-eval", "sepolia", "all"), default="all")
    args = parser.parse_args(list(argv) if argv is not None else None)
    present, issues = validate(args.mode)
    print(f"Checked {ENV_PATH} for mode {args.mode}.")
    print(f"Configured entries: {len(present)}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("All required entries are syntactically configured. No service was contacted.")


if __name__ == "__main__":
    main()
