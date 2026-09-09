from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from evaluation.verify_etherscan_contract import (
    _constructor_args,
    _verification_environment,
    check_status,
)


class EtherscanVerificationTests(unittest.TestCase):
    def test_constructor_address_is_abi_encoded(self) -> None:
        address = "0x0DEcB4d4A5946462823C83AAf5c7dfaE560B6193"
        encoded = _constructor_args(address)
        self.assertEqual(len(encoded), 66)
        self.assertEqual(encoded[-40:], address[2:].lower())
        self.assertEqual(encoded[2:26], "0" * 24)

    def test_verification_environment_excludes_project_secrets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "REGULATOR_PRIVATE_KEY": "private",
                "OPENAI_API_KEY": "openai",
                "PATH": "retained",
            },
            clear=True,
        ):
            environment = _verification_environment("etherscan")
        self.assertEqual(environment["ETHERSCAN_API_KEY"], "etherscan")
        self.assertEqual(environment["PATH"], "retained")
        self.assertNotIn("REGULATOR_PRIVATE_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)

    def test_status_rejects_an_invalid_guid_before_network_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid characters"):
            check_status("invalid-guid!")


if __name__ == "__main__":
    unittest.main()
