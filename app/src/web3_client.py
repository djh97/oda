import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from web3 import Web3
from web3.contract import Contract

from .config import Settings


class ConfigError(RuntimeError):
    pass


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
    data = _load_json(address_path)
    if not isinstance(data, dict) or "address" not in data:
        raise ConfigError(f"Address file missing 'address' key: {address_path}")
    addr = str(data["address"]).strip()
    if not Web3.is_address(addr):
        raise ConfigError(f"Invalid contract address in {address_path}: {addr}")
    return Web3.to_checksum_address(addr)


@dataclass(frozen=True)
class Web3Context:
    w3: Web3
    contract: Contract
    contract_address: str


def make_web3_context(settings: Settings) -> Web3Context:
    if not settings.sepolia_rpc_url:
        raise ConfigError("SEPOLIA_RPC_URL is empty. Set it in app/.env")

    w3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    if not w3.is_connected():
        raise ConfigError("Web3 failed to connect. Check SEPOLIA_RPC_URL")

    abi = load_abi(settings.abi_path)
    contract_address = load_contract_address(settings.address_path)
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
