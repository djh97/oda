from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.paper_full_workflow import DEMO_CASE_PATH, IMPLEMENTATION_DIR, _runtime_profile


class ContractPrivacyTests(unittest.TestCase):
    def test_contract_does_not_store_plaintext_clinical_fields(self) -> None:
        source = (
            IMPLEMENTATION_DIR / "smart-contracts" / "src" / "TransplantManagement.sol"
        ).read_text(encoding="utf-8").lower()
        forbidden_identifiers = (
            "bloodtype",
            "blood_type",
            "hlatyping",
            "hla_typing",
            "medicalnotes",
            "medical_notes",
            "organtype",
            "organ_type",
        )
        for identifier in forbidden_identifiers:
            self.assertNotIn(identifier, source)

    def test_exported_abi_has_only_ids_addresses_flags_and_cids(self) -> None:
        path = IMPLEMENTATION_DIR / "integration" / "abi" / "TransplantManagement.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(value["abi"]).lower()
        for identifier in ("blood", "hla", "medical_notes", "organ_type"):
            self.assertNotIn(identifier, serialized)

    def test_runtime_demo_profiles_remove_reference_labels(self) -> None:
        case = json.loads(Path(DEMO_CASE_PATH).read_text(encoding="utf-8"))
        recipient = _runtime_profile(case["recipients"][0])
        for field in (
            "reference_note_state",
            "reference_evidence_codes",
            "note_style",
            "template_family_id",
            "note_instance_id",
        ):
            self.assertNotIn(field, recipient)


if __name__ == "__main__":
    unittest.main()
