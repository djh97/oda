"""Build transparent scenario-based USD estimates from measured transaction gas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from evaluation.benchmark_offchain import _resolve_run_dir
from evaluation.paper_full_workflow import DEFAULT_PROTOCOL_PATH, _sha256
from evaluation.synthetic_dataset import PROTOCOL


SCRIPT_PATH = Path(__file__).resolve()
COST_SETTINGS = PROTOCOL["performance_evaluation"]["cost_evaluation"]
EXPECTED_GAS_PRICES_GWEI = tuple(
    Decimal(str(value)) for value in COST_SETTINGS["gas_price_scenarios_gwei"]
)
INTERPRETATION = (
    "USD values are illustrative mainnet-style scenarios obtained by applying the stated gas prices "
    "and ETH/USD observation to measured gas units. They are not costs paid on Sepolia."
)
TABLE7_OBSERVATION_DATE = "2026-03-09"
TABLE7_NETWORK_SCENARIOS = (
    ("ethereum", "Ethereum", "ETH", Decimal("1944.53"), Decimal("0.04")),
    ("polygon", "Polygon", "POL", Decimal("0.177"), Decimal("146.00")),
    ("arbitrum", "Arbitrum", "ETH", Decimal("1944.53"), Decimal("0.02")),
    ("optimism", "Optimism", "ETH", Decimal("1944.53"), Decimal("0.01")),
    ("zksync", "zkSync Era", "ETH", Decimal("1944.53"), Decimal("0.05")),
)
TABLE7_INTERPRETATION = (
    "Function-level USD values reproduce the manuscript's fixed 9 March 2026 "
    "network scenarios by applying each stated gas price and token price to the "
    "measured Sepolia gas units. They are arithmetic comparisons, not fees "
    "measured on those networks, and do not model chain-specific L1-data or fee components."
)


def _read_transactions(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No transactions found in {path}")
    required = {
        "category",
        "function",
        "tx_hash",
        "status",
        "gas_used",
        "effective_gas_price_wei",
        "fee_wei",
        "fee_native",
    }
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("The transaction manifest omits required cost fields")
    transaction_hashes: set[str] = set()
    for row in rows:
        try:
            status = int(row["status"])
            gas_used = int(row["gas_used"])
            gas_price = int(row["effective_gas_price_wei"])
            fee_wei = int(row["fee_wei"])
            fee_native = Decimal(row["fee_native"])
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise RuntimeError("The transaction manifest contains an invalid numeric value") from exc
        if status != 1:
            raise RuntimeError("The transaction manifest contains a failed transaction")
        if gas_used <= 0 or gas_price <= 0 or fee_wei != gas_used * gas_price:
            raise RuntimeError("The transaction manifest contains inconsistent receipt costs")
        if not fee_native.is_finite() or fee_native != Decimal(fee_wei) / Decimal(10**18):
            raise RuntimeError("The transaction manifest contains an inconsistent native fee")
        if not row["category"].strip() or not row["function"].strip():
            raise RuntimeError("The transaction manifest contains an unnamed stage or function")
        tx_hash = row["tx_hash"].strip().lower()
        if not tx_hash or tx_hash in transaction_hashes:
            raise RuntimeError("The transaction manifest contains a missing or duplicate transaction hash")
        transaction_hashes.add(tx_hash)
    return rows


def _validate_timestamp(value: str) -> str:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("price_observed_at must include a time-zone offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_public_source_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("price_source_url must be a complete HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("price_source_url must not contain credentials")
    return value.strip()


def _scenario_name(gwei: Decimal) -> str:
    value = format(gwei.normalize(), "f").replace(".", "_")
    return f"gas_{value}_gwei"


def _fee_wei(row: Mapping[str, str]) -> int:
    if row.get("fee_wei") not in (None, ""):
        return int(row["fee_wei"])
    return int(Decimal(row["fee_native"]) * Decimal(10**18))


def _cost_fields(
    values: Sequence[Mapping[str, str]],
    eth_usd: Decimal,
    gas_prices_gwei: Sequence[Decimal],
) -> Dict[str, Any]:
    gas_used = sum(int(row["gas_used"]) for row in values)
    fee_wei = sum(_fee_wei(row) for row in values)
    output: Dict[str, Any] = {
        "transaction_count": len(values),
        "measured_gas_used": gas_used,
        "measured_sepolia_fee_wei": fee_wei,
        "measured_sepolia_fee_test_eth": format(
            Decimal(fee_wei) / Decimal(10**18), ".18f"
        ),
    }
    for gas_price in gas_prices_gwei:
        name = _scenario_name(gas_price)
        native = Decimal(gas_used) * gas_price / Decimal(10**9)
        output[f"{name}_eth"] = format(native, ".12f")
        output[f"{name}_usd"] = format(native * eth_usd, ".4f")
    for key, _network, _token, token_usd, gas_price_gwei in TABLE7_NETWORK_SCENARIOS:
        native = Decimal(gas_used) * gas_price_gwei / Decimal(10**9)
        output[f"table7_{key}_usd"] = format(native * token_usd, ".6f")
    return output


def _table7_scenario_records() -> list[Dict[str, str]]:
    return [
        {
            "key": key,
            "network": network,
            "token": token,
            "token_price_usd": format(token_usd, "f"),
            "gas_price_gwei": format(gas_price_gwei, "f"),
        }
        for key, network, token, token_usd, gas_price_gwei in TABLE7_NETWORK_SCENARIOS
    ]


def _validate_source_run(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": _sha256(DEFAULT_PROTOCOL_PATH),
        "network": "sepolia",
        "chain_id": 11155111,
        "transaction_count": len(rows),
        "gas_used_total": sum(int(row["gas_used"]) for row in rows),
        "fee_wei_total": sum(_fee_wei(row) for row in rows),
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise RuntimeError(f"Source run summary mismatch for {field}")
    final_checks = summary.get("final_checks")
    if not isinstance(final_checks, dict) or not final_checks or not all(final_checks.values()):
        raise RuntimeError("Source run does not contain a complete set of passing final checks")


def _string_row(value: Mapping[str, Any]) -> Dict[str, str]:
    return {str(key): str(item) for key, item in value.items()}


def validate_cost_artifacts(run_dir: Path) -> tuple[list[Dict[str, str]], Dict[str, Any]]:
    """Recompute every retained cost value from the canonical transaction manifest."""
    manifest_path = run_dir / "transaction_manifest.csv"
    summary_path = run_dir / "run_summary.json"
    rows = _read_transactions(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_source_run(summary, rows)

    output_dir = run_dir / "cost_estimate"
    function_path = output_dir / "cost_by_function.csv"
    stage_path = output_dir / "cost_by_stage.csv"
    assumptions_path = output_dir / "cost_assumptions.json"
    hashes_path = output_dir / "artifact_hashes.json"
    assumptions = json.loads(assumptions_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashed_paths = (function_path, stage_path, assumptions_path)
    if not isinstance(hashes, dict) or set(hashes) != {path.name for path in hashed_paths}:
        raise RuntimeError("Cost evidence artifact-hash map has missing or extra entries")
    for path in hashed_paths:
        record = hashes.get(path.name)
        if not isinstance(record, dict):
            raise RuntimeError(f"Cost evidence does not record {path.name}")
        try:
            recorded_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cost evidence has an invalid byte count for {path.name}") from exc
        if record.get("sha256") != _sha256(path) or recorded_bytes != path.stat().st_size:
            raise RuntimeError(f"Cost evidence hash verification failed for {path.name}")

    if assumptions.get("source_run_summary_sha256") != _sha256(summary_path):
        raise RuntimeError("Cost evidence belongs to a different canonical run")
    if assumptions.get("transaction_manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("Cost evidence does not match the transaction manifest")
    if assumptions.get("cost_script_sha256") != _sha256(SCRIPT_PATH):
        raise RuntimeError("Cost evidence was produced by a different cost script")
    if assumptions.get("network") != "sepolia" or assumptions.get("sepolia_fee_has_market_value") is not False:
        raise RuntimeError("Cost evidence does not preserve the Sepolia interpretation boundary")

    try:
        eth_usd = Decimal(str(assumptions.get("eth_usd", "0")))
    except ArithmeticError as exc:
        raise RuntimeError("Cost evidence has an invalid ETH/USD observation") from exc
    if not eth_usd.is_finite() or eth_usd <= 0:
        raise RuntimeError("Cost evidence has an invalid ETH/USD observation")
    observed_at = _validate_timestamp(str(assumptions.get("eth_usd_observed_at", "")))
    if assumptions.get("eth_usd_observed_at") != observed_at:
        raise RuntimeError("Cost evidence timestamp is not normalized to UTC")
    source_url = str(assumptions.get("eth_usd_source_url", ""))
    try:
        if _validate_public_source_url(source_url) != source_url:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("Cost evidence has an invalid ETH/USD source URL") from exc
    try:
        gas_prices = tuple(
            Decimal(str(value)) for value in assumptions.get("gas_price_scenarios_gwei", [])
        )
    except ArithmeticError as exc:
        raise RuntimeError("Cost evidence has invalid gas-price scenarios") from exc
    if gas_prices != EXPECTED_GAS_PRICES_GWEI:
        raise RuntimeError("Cost evidence does not use the prespecified gas-price scenarios")
    if assumptions.get("contract_address") != summary.get("contract_address"):
        raise RuntimeError("Cost evidence identifies a different contract")
    if assumptions.get("interpretation") != INTERPRETATION:
        raise RuntimeError("Cost evidence has an incorrect interpretation statement")
    if assumptions.get("table7_observation_date") != TABLE7_OBSERVATION_DATE:
        raise RuntimeError("Cost evidence has an incorrect Table 7 observation date")
    if assumptions.get("table7_network_scenarios") != _table7_scenario_records():
        raise RuntimeError("Cost evidence has incorrect Table 7 network assumptions")
    if assumptions.get("table7_interpretation") != TABLE7_INTERPRETATION:
        raise RuntimeError("Cost evidence has an incorrect Table 7 interpretation")

    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    grouped_by_stage: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped[(row["category"], row["function"])].append(row)
        grouped_by_stage.setdefault(row["category"], []).append(row)
    expected_function = {
        (category, function): _string_row({
            "category": category,
            "function": function,
            **_cost_fields(values, eth_usd, gas_prices),
        })
        for (category, function), values in grouped.items()
    }
    observed_function_rows = _read_csv_rows(function_path)
    observed_function = {
        (row["category"], row["function"]): row for row in observed_function_rows
    }
    if observed_function != expected_function or len(observed_function_rows) != len(expected_function):
        raise RuntimeError("Function-level cost evidence does not reproduce from transactions")

    expected_stage = {
        category: _string_row({
            "category": category,
            **_cost_fields(values, eth_usd, gas_prices),
        })
        for category, values in grouped_by_stage.items()
    }
    observed_stage_rows = _read_csv_rows(stage_path)
    observed_stage = {row["category"]: row for row in observed_stage_rows}
    if observed_stage != expected_stage or len(observed_stage_rows) != len(expected_stage):
        raise RuntimeError("Stage-level cost evidence does not reproduce from transactions")

    totals = _cost_fields(rows, eth_usd, gas_prices)
    expected_scenarios = {
        _scenario_name(gas_price): {
            "eth": totals[f"{_scenario_name(gas_price)}_eth"],
            "usd": totals[f"{_scenario_name(gas_price)}_usd"],
        }
        for gas_price in gas_prices
    }
    expected_totals = {
        "measured_transaction_count": totals["transaction_count"],
        "measured_gas_used_total": totals["measured_gas_used"],
        "measured_sepolia_fee_wei": totals["measured_sepolia_fee_wei"],
        "measured_sepolia_fee_test_eth": totals["measured_sepolia_fee_test_eth"],
        "scenario_totals": expected_scenarios,
        "table7_scenario_totals_usd": {
            key: totals[f"table7_{key}_usd"]
            for key, _network, _token, _token_usd, _gas_price in TABLE7_NETWORK_SCENARIOS
        },
    }
    for field, expected_value in expected_totals.items():
        if assumptions.get(field) != expected_value:
            raise RuntimeError(f"Cost evidence total does not reproduce for {field}")
    return observed_stage_rows, assumptions


def _read_csv_rows(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build(
    run_dir: Path,
    *,
    eth_usd: Decimal,
    price_observed_at: str,
    price_source_url: str,
    gas_prices_gwei: Sequence[Decimal],
) -> Path:
    if not eth_usd.is_finite() or eth_usd <= 0:
        raise ValueError("eth_usd must be positive")
    if tuple(gas_prices_gwei) != EXPECTED_GAS_PRICES_GWEI:
        expected = ", ".join(format(value, "f") for value in EXPECTED_GAS_PRICES_GWEI)
        raise ValueError(f"gas_prices_gwei must match the prespecified scenarios: {expected}")
    price_source_url = _validate_public_source_url(price_source_url)
    observed_at = _validate_timestamp(price_observed_at)
    manifest_path = run_dir / "transaction_manifest.csv"
    rows = _read_transactions(manifest_path)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_source_run(summary, rows)

    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    grouped_by_stage: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped[(row["category"], row["function"])].append(row)
        grouped_by_stage.setdefault(row["category"], []).append(row)

    table_rows: list[Dict[str, Any]] = []
    for (category, function), values in sorted(grouped.items()):
        output = {
            "category": category,
            "function": function,
            **_cost_fields(values, eth_usd, gas_prices_gwei),
        }
        table_rows.append(output)

    stage_rows = [
        {
            "category": category,
            **_cost_fields(values, eth_usd, gas_prices_gwei),
        }
        for category, values in grouped_by_stage.items()
    ]

    total_gas = sum(int(row["gas_used"]) for row in rows)
    actual_sepolia_fee_wei = sum(_fee_wei(row) for row in rows)
    actual_sepolia_fee = Decimal(actual_sepolia_fee_wei) / Decimal(10**18)
    assumptions = {
        "contract_address": summary["contract_address"],
        "source_run_summary_sha256": _sha256(summary_path),
        "transaction_manifest_sha256": _sha256(manifest_path),
        "cost_script_sha256": _sha256(SCRIPT_PATH),
        "network": "sepolia",
        "measured_transaction_count": len(rows),
        "measured_gas_used_total": total_gas,
        "measured_sepolia_fee_wei": actual_sepolia_fee_wei,
        "measured_sepolia_fee_test_eth": format(actual_sepolia_fee, ".18f"),
        "sepolia_fee_has_market_value": False,
        "eth_usd": format(eth_usd, "f"),
        "eth_usd_observed_at": observed_at,
        "eth_usd_source_url": price_source_url,
        "gas_price_scenarios_gwei": [format(value, "f") for value in gas_prices_gwei],
        "interpretation": INTERPRETATION,
        "scenario_totals": {},
        "table7_observation_date": TABLE7_OBSERVATION_DATE,
        "table7_network_scenarios": _table7_scenario_records(),
        "table7_interpretation": TABLE7_INTERPRETATION,
        "table7_scenario_totals_usd": {},
    }
    for gas_price in gas_prices_gwei:
        name = _scenario_name(gas_price)
        native = Decimal(total_gas) * gas_price / Decimal(10**9)
        assumptions["scenario_totals"][name] = {
            "eth": format(native, ".12f"),
            "usd": format(native * eth_usd, ".4f"),
        }
    table7_totals = _cost_fields(rows, eth_usd, gas_prices_gwei)
    assumptions["table7_scenario_totals_usd"] = {
        key: table7_totals[f"table7_{key}_usd"]
        for key, _network, _token, _token_usd, _gas_price in TABLE7_NETWORK_SCENARIOS
    }

    output_dir = run_dir / "cost_estimate"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "cost_by_function.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    with (output_dir / "cost_by_stage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    (output_dir / "cost_assumptions.json").write_text(
        json.dumps(assumptions, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (
                    output_dir / "cost_by_function.csv",
                    output_dir / "cost_by_stage.csv",
                    output_dir / "cost_assumptions.json",
                )
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_cost_artifacts(run_dir)
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--eth-usd", type=Decimal, required=True)
    parser.add_argument("--price-observed-at", required=True)
    parser.add_argument("--price-source-url", required=True)
    parser.add_argument(
        "--gas-price-gwei",
        type=Decimal,
        nargs="+",
        default=[
            Decimal(str(value))
            for value in COST_SETTINGS["gas_price_scenarios_gwei"]
        ],
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_dir = build(
        _resolve_run_dir(args.run_dir),
        eth_usd=args.eth_usd,
        price_observed_at=args.price_observed_at,
        price_source_url=args.price_source_url,
        gas_prices_gwei=args.gas_price_gwei,
    )
    print(f"Cost-estimation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
