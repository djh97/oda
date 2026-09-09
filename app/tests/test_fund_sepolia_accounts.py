from __future__ import annotations

import unittest

from evaluation.fund_sepolia_accounts import (
    EXPECTED_GAS_BY_KEY,
    FundingError,
    TRANSFER_GAS,
    calculate_targets,
    required_regulator_balance,
)


class FundingPlanTests(unittest.TestCase):
    def test_targets_use_gas_budget_headroom_and_round_up(self) -> None:
        targets = calculate_targets(5_000_000_000)
        self.assertEqual(set(targets), set(EXPECTED_GAS_BY_KEY))
        self.assertEqual(targets["REGULATOR_PRIVATE_KEY"], 31_800_000_000_000_000)
        self.assertEqual(targets["RECIPIENT1_PRIVATE_KEY"], 500_000_000_000_000)
        self.assertTrue(all(value % 100_000_000_000_000 == 0 for value in targets.values()))

    def test_invalid_planning_fee_is_rejected(self) -> None:
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(FundingError):
                calculate_targets(value)

    def test_regulator_requirement_includes_deficits_and_transfer_fees(self) -> None:
        fee_cap = 3_000_000_000
        required = required_regulator_balance(
            10_000,
            [20_000, 0, 30_000],
            fee_cap,
        )
        self.assertEqual(required, 10_000 + 20_000 + 30_000 + 2 * TRANSFER_GAS * fee_cap)


if __name__ == "__main__":
    unittest.main()
