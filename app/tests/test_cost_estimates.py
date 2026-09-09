from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from evaluation.build_cost_estimates import build, validate_cost_artifacts
from evaluation.paper_full_workflow import DEFAULT_PROTOCOL_PATH


class CostEstimateTests(unittest.TestCase):
    def test_measured_gas_is_kept_separate_from_scenario_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [
                {
                    "category": "Deployment",
                    "function": "constructor",
                    "tx_hash": "0xaaa",
                    "status": "1",
                    "gas_used": "100",
                    "effective_gas_price_wei": "1000000000",
                    "fee_wei": "100000000000",
                    "fee_native": "0.0000001",
                },
                {
                    "category": "Workflow",
                    "function": "createMatch",
                    "tx_hash": "0xbbb",
                    "status": "1",
                    "gas_used": "200",
                    "effective_gas_price_wei": "1000000000",
                    "fee_wei": "200000000000",
                    "fee_native": "0.0000002",
                },
            ]
            with (run_dir / "transaction_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (run_dir / "run_summary.json").write_text(
                json.dumps({
                    "contract_address": "0x" + "11" * 20,
                    "protocol_id": "ODA-SYNTH-MULTIORGAN-1.0",
                    "protocol_sha256": hashlib.sha256(
                        DEFAULT_PROTOCOL_PATH.read_bytes()
                    ).hexdigest(),
                    "network": "sepolia",
                    "chain_id": 11155111,
                    "transaction_count": 2,
                    "gas_used_total": 300,
                    "fee_wei_total": 300000000000,
                    "final_checks": {"all_transactions_succeeded": True},
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "prespecified scenarios"):
                build(
                    run_dir,
                    eth_usd=Decimal("2000"),
                    price_observed_at="2026-09-07T00:00:00Z",
                    price_source_url="https://example.invalid/price",
                    gas_prices_gwei=[Decimal("10")],
                )
            output = build(
                run_dir,
                eth_usd=Decimal("2000"),
                price_observed_at="2026-09-07T00:00:00Z",
                price_source_url="https://example.invalid/price",
                gas_prices_gwei=[Decimal("5"), Decimal("15"), Decimal("30")],
            )
            assumptions = json.loads((output / "cost_assumptions.json").read_text(encoding="utf-8"))
            with (output / "cost_by_stage.csv").open("r", encoding="utf-8", newline="") as handle:
                stage_rows = list(csv.DictReader(handle))
            validated_rows, validated_assumptions = validate_cost_artifacts(run_dir)
            assert validated_rows == stage_rows
            assert validated_assumptions == assumptions

            hashes_path = output / "artifact_hashes.json"
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            hashes["unexpected.txt"] = {"bytes": 0, "sha256": "0" * 64}
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or extra entries"):
                validate_cost_artifacts(run_dir)
            hashes.pop("unexpected.txt")
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")

            (output / "cost_by_stage.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash verification failed"):
                validate_cost_artifacts(run_dir)

        self.assertEqual(assumptions["measured_gas_used_total"], 300)
        self.assertEqual(assumptions["measured_sepolia_fee_wei"], 300000000000)
        self.assertFalse(assumptions["sepolia_fee_has_market_value"])
        self.assertEqual(assumptions["scenario_totals"]["gas_15_gwei"]["usd"], "0.0090")
        self.assertEqual(assumptions["table7_observation_date"], "2026-03-09")
        self.assertEqual(
            assumptions["table7_scenario_totals_usd"],
            {
                "ethereum": "0.000023",
                "polygon": "0.000008",
                "arbitrum": "0.000012",
                "optimism": "0.000006",
                "zksync": "0.000029",
            },
        )
        self.assertEqual([row["category"] for row in stage_rows], ["Deployment", "Workflow"])
        self.assertEqual(stage_rows[1]["measured_gas_used"], "200")
        self.assertEqual(stage_rows[1]["measured_sepolia_fee_wei"], "200000000000")
        self.assertEqual(stage_rows[1]["gas_15_gwei_usd"], "0.0060")
        self.assertEqual(stage_rows[1]["table7_ethereum_usd"], "0.000016")


if __name__ == "__main__":
    unittest.main()
