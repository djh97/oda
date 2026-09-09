import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from web3 import Web3
from web3.contract import Contract

from .config import Settings


class ConfigError(RuntimeError):
    pass


EXPECTED_CHAIN_IDS = {"sepolia": 11155111}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_abi(abi_path: Path) -> Any:
    data = _load_json(abi_path)
    # Accept either:
    # - raw ABI array: [ {...}, ... ]
    # - object with "abi": [ ... ]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "abi" in data and isinstance(data["abi"], list):
        return data["abi"]
    raise ConfigError(f"ABI file format not recognized: {abi_path}")


def load_contract_address(address_path: Path) -> str:
    data = load_deployment_record(address_path)
    return str(data["address"])


def _require_sha256(value: Any, *, name: str, path: Path) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ConfigError(f"Deployment metadata has an invalid {name} in {path}")
    return normalized


def load_deployment_record(address_path: Path) -> Dict[str, Any]:
    data = _load_json(address_path)
    if not isinstance(data, dict) or "address" not in data:
        raise ConfigError(f"Address file missing 'address' key: {address_path}")
    addr = str(data["address"]).strip()
    if not Web3.is_address(addr):
        raise ConfigError(f"Invalid contract address in {address_path}: {addr}")
    try:
        chain_id = int(data["chain_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Deployment metadata has an invalid chain_id in {address_path}") from exc
    network = str(data.get("network", "")).strip().lower()
    if not network:
        raise ConfigError(f"Deployment metadata is missing network in {address_path}")
    if str(data.get("protocol_id", "")).strip() != "ODA-SYNTH-MULTIORGAN-1.0":
        raise ConfigError(f"Deployment metadata has an unsupported protocol_id in {address_path}")
    return {
        **data,
        "address": Web3.to_checksum_address(addr),
        "chain_id": chain_id,
        "network": network,
        "source_sha256": _require_sha256(data.get("source_sha256"), name="source_sha256", path=address_path),
        "artifact_sha256": _require_sha256(
            data.get("artifact_sha256"), name="artifact_sha256", path=address_path
        ),
        "deployed_runtime_sha256": _require_sha256(
            data.get("deployed_runtime_sha256"),
            name="deployed_runtime_sha256",
            path=address_path,
        ),
    }


def _validate_abi_provenance(
    abi_path: Path,
    deployment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    data = _load_json(abi_path)
    if not isinstance(data, dict) or not isinstance(data.get("abi"), list):
        raise ConfigError(f"Canonical ABI package format not recognized: {abi_path}")
    for name in ("source_sha256", "artifact_sha256"):
        observed = _require_sha256(data.get(name), name=name, path=abi_path)
        if observed != deployment[name]:
            raise ConfigError(f"ABI and deployment metadata disagree on {name}")
    return data["abi"]


@dataclass(frozen=True)
class Web3Context:
    w3: Web3
    contract: Contract
    contract_address: str


def make_web3_context(settings: Settings) -> Web3Context:
    expected_chain_id = EXPECTED_CHAIN_IDS.get(settings.network)
    if expected_chain_id is None:
        raise ConfigError(f"Unsupported EVM network: {settings.network}")
    if not settings.rpc_url:
        raise ConfigError(f"{settings.network.upper()}_RPC_URL is empty. Set it in app/.env")

    w3 = Web3(Web3.HTTPProvider(settings.rpc_url))
    if not w3.is_connected():
        raise ConfigError(f"Web3 failed to connect. Check {settings.network.upper()}_RPC_URL")
    try:
        chain_id = int(w3.eth.chain_id)
    except Exception as exc:
        raise ConfigError("Web3 could not read the connected chain ID") from exc
    if chain_id != expected_chain_id:
        raise ConfigError(
            f"Expected {settings.network} chain ID {expected_chain_id}, found {chain_id}"
        )

    deployment = load_deployment_record(settings.address_path)
    if deployment["network"] != settings.network or deployment["chain_id"] != expected_chain_id:
        raise ConfigError("Configured deployment metadata does not match the selected network")
    abi = _validate_abi_provenance(settings.abi_path, deployment)
    contract_address = str(deployment["address"])
    try:
        deployed_code = bytes(w3.eth.get_code(contract_address))
    except Exception as exc:
        raise ConfigError("Web3 could not read the configured contract bytecode") from exc
    if not deployed_code:
        raise ConfigError(f"No contract bytecode is deployed at {contract_address}")
    observed_runtime_sha256 = hashlib.sha256(deployed_code).hexdigest()
    if observed_runtime_sha256 != deployment["deployed_runtime_sha256"]:
        raise ConfigError("Deployed contract bytecode does not match the canonical deployment record")
    contract = w3.eth.contract(address=contract_address, abi=abi)

    return Web3Context(w3=w3, contract=contract, contract_address=contract_address)


def read_contract_health(ctx: Web3Context) -> Dict[str, Any]:
    """
    Reads a few view functions to confirm the contract is reachable.
    """
    regulator = ctx.contract.functions.regulator().call()
    match_counter = ctx.contract.functions.matchCounter().call()
    donor_counter = ctx.contract.functions.donorCounter().call()
    recipient_counter = ctx.contract.functions.recipientCounter().call()

    return {
        "connected": True,
        "contract_address": ctx.contract_address,
        "regulator": regulator,
        "matchCounter": int(match_counter),
        "donorCounter": int(donor_counter),
        "recipientCounter": int(recipient_counter),
    }
