from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evaluation.synthetic_dataset import DATASET_DIR
from src.policy import (
    RETIRED_TEST_SEEDS,
    PolicyInputError,
    abo_compatible,
    active_test_seed,
    assert_protocol_frozen,
    assert_test_seed_not_retired,
    load_protocol,
    rank_recipients_baseline,
    selectable_order,
)


class PolicyTests(unittest.TestCase):
    def test_retired_or_malformed_test_seed_cannot_execute(self) -> None:
        protocol = load_protocol()
        for retired_seed in RETIRED_TEST_SEEDS:
            retired = deepcopy(protocol)
            retired["dataset"]["splits"]["test"]["seed"] = retired_seed
            with self.assertRaisesRegex(PolicyInputError, "retired"):
                assert_test_seed_not_retired(retired)

        replacement = deepcopy(protocol)
        replacement["dataset"]["splits"]["test"]["seed"] = 2026090791
        self.assertEqual(assert_test_seed_not_retired(replacement), 2026090791)

        for invalid in (None, True, 0, "2026090791"):
            malformed = deepcopy(replacement)
            malformed["dataset"]["splits"]["test"]["seed"] = invalid
            with self.assertRaisesRegex(PolicyInputError, "positive integer"):
                active_test_seed(malformed)

    def test_protocol_freeze_binds_protocol_and_seed(self) -> None:
        protocol = load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            freeze_path = root / "protocol_freeze.json"
            protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
            seed = active_test_seed(protocol)
            freeze_path.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "protocol_id": protocol["protocol_id"],
                        "protocol_version": protocol["version"],
                        "active_test_seed_sha256": hashlib.sha256(
                            str(seed).encode("ascii")
                        ).hexdigest(),
                        "source_hashes": {
                            "protocol_json_sha256": hashlib.sha256(
                                protocol_path.read_bytes()
                            ).hexdigest()
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            assert_protocol_frozen(
                protocol,
                protocol_path=protocol_path,
                freeze_path=freeze_path,
            )
            freeze_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PolicyInputError, "freeze record is invalid"):
                assert_protocol_frozen(
                    protocol,
                    protocol_path=protocol_path,
                    freeze_path=freeze_path,
                )

    def test_abo_table(self) -> None:
        expected = {
            "O": {"O", "A", "B", "AB"},
            "A": {"A", "AB"},
            "B": {"B", "AB"},
            "AB": {"AB"},
        }
        for donor in expected:
            for recipient in ("O", "A", "B", "AB"):
                self.assertEqual(abo_compatible(donor, recipient), recipient in expected[donor])

    def test_generated_reference_order_recomputes(self) -> None:
        path = DATASET_DIR / "validation_cases.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            cases = [json.loads(next(handle)) for _ in range(20)]
        for case in cases:
            ranked = rank_recipients_baseline(case["donor"], case["recipients"])
            self.assertEqual(selectable_order(ranked), case["reference"]["baseline_order"])
            self.assertGreaterEqual(len(selectable_order(ranked)), 4)

    def test_incompatible_candidates_have_no_rank(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        ranked = rank_recipients_baseline(case["donor"], case["recipients"])
        self.assertTrue(all(item["rank"] is not None for item in ranked if item["selectable"]))
        self.assertTrue(all(item["rank"] is None for item in ranked if not item["selectable"]))

    def test_kidney_hla_requires_six_distinct_antigens(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        kidney_case["recipients"][0]["hla_typing"] = ["A*01"]
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(kidney_case["donor"], kidney_case["recipients"])

    def test_kidney_hla_requires_two_values_per_locus(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        kidney_case["recipients"][0]["hla_typing"] = [
            "A*01",
            "A*02",
            "A*03",
            "B*07",
            "B*08",
            "DRB1*01",
        ]
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(kidney_case["donor"], kidney_case["recipients"])

    def test_negative_distance_is_rejected(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        case["recipients"][0]["distance_km"] = -1
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(case["donor"], case["recipients"])

    def test_string_boolean_is_rejected(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            liver_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "liver")
        liver_case["recipients"][0]["synthetic_status_one"] = "false"
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(liver_case["donor"], liver_case["recipients"])

    def test_liver_status_one_outranks_score_based_candidates(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "liver")
        donor = case["donor"]
        recipients = case["recipients"]
        for recipient in recipients:
            recipient["synthetic_status_one"] = False
            recipient["crossmatch_result"] = "positive"

        status_one, score_based = recipients[:2]
        for recipient in (status_one, score_based):
            recipient["organ_type"] = "liver"
            recipient["blood_type"] = donor["blood_type"]
            recipient["crossmatch_result"] = "negative"

        status_one.update({
            "synthetic_status_one": True,
            "synthetic_liver_urgency_score": 6,
            "distance_km": 5000,
            "waiting_time_days": 20,
            "weight_kg": round(float(donor["weight_kg"]) * 0.56, 1),
        })
        score_based.update({
            "synthetic_liver_urgency_score": 40,
            "distance_km": 0,
            "waiting_time_days": 1095,
            "weight_kg": donor["weight_kg"],
        })

        ranked = rank_recipients_baseline(donor, recipients)
        self.assertEqual(selectable_order(ranked)[:2], [status_one["recipient_id"], score_based["recipient_id"]])
        self.assertEqual(ranked[0]["priority_tier"], 1)

    def test_organ_mismatch_does_not_require_irrelevant_organ_fields(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        recipient = kidney_case["recipients"][0]
        recipient["organ_type"] = "liver"
        recipient.pop("hla_typing")
        recipient.pop("cpra_percent")
        recipient.pop("synthetic_epts_percent", None)
        recipient["synthetic_liver_urgency_score"] = 20
        recipient["synthetic_status_one"] = False

        ranked = rank_recipients_baseline(kidney_case["donor"], kidney_case["recipients"])
        observed = next(item for item in ranked if item["recipient_id"] == recipient["recipient_id"])
        self.assertFalse(observed["selectable"])
        self.assertEqual(observed["score"], 0.0)
        self.assertIn("organ_type_mismatch", observed["exclusion_reasons"])

    def test_synthetic_epts_is_omitted_for_pediatric_kidney_candidates(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        recipient = kidney_case["recipients"][0]
        recipient["age_years"] = 12
        recipient["synthetic_epts_percent"] = 20
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(kidney_case["donor"], kidney_case["recipients"])

    def test_adult_kidney_candidate_requires_synthetic_epts(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        recipient = kidney_case["recipients"][0]
        recipient["age_years"] = 30
        recipient.pop("synthetic_epts_percent", None)
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(kidney_case["donor"], kidney_case["recipients"])

    def test_kidney_longevity_priority_uses_threshold_pair(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            kidney_case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "kidney")
        donor = kidney_case["donor"]
        recipient = kidney_case["recipients"][0]
        donor["synthetic_kdpi_percent"] = 20
        recipient["age_years"] = 30
        recipient["synthetic_epts_percent"] = 20
        ranked = rank_recipients_baseline(donor, kidney_case["recipients"])
        observed = next(item for item in ranked if item["recipient_id"] == recipient["recipient_id"])
        self.assertEqual(observed["factors"]["longevity_priority"], 1.0)

        recipient["synthetic_epts_percent"] = 20.1
        ranked = rank_recipients_baseline(donor, kidney_case["recipients"])
        observed = next(item for item in ranked if item["recipient_id"] == recipient["recipient_id"])
        self.assertEqual(observed["factors"]["longevity_priority"], 0.0)

    def test_heart_status_tier_outranks_within_tier_score(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "heart")
        donor = case["donor"]
        recipients = case["recipients"]
        for recipient in recipients:
            recipient["crossmatch_result"] = "positive"

        urgent, less_urgent = recipients[:2]
        for recipient in (urgent, less_urgent):
            recipient["organ_type"] = "heart"
            recipient["blood_type"] = donor["blood_type"]
            recipient["crossmatch_result"] = "negative"

        urgent.update({
            "synthetic_adult_heart_status": 1,
            "distance_km": 5000,
            "waiting_time_days": 20,
            "weight_kg": round(float(donor["weight_kg"]) / 0.71, 1),
        })
        less_urgent.update({
            "synthetic_adult_heart_status": 2,
            "distance_km": 0,
            "waiting_time_days": 1095,
            "weight_kg": donor["weight_kg"],
        })

        ranked = rank_recipients_baseline(donor, recipients)
        self.assertEqual(selectable_order(ranked)[:2], [urgent["recipient_id"], less_urgent["recipient_id"]])
        self.assertEqual(ranked[0]["priority_tier"], 6)

    def test_lung_ranking_uses_supplied_synthetic_priority_index(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = next(json.loads(line) for line in handle if json.loads(line)["organ_type"] == "lung")
        donor = case["donor"]
        recipients = case["recipients"]
        for recipient in recipients:
            recipient["crossmatch_result"] = "positive"

        higher, lower = recipients[:2]
        for recipient in (higher, lower):
            recipient["organ_type"] = "lung"
            recipient["blood_type"] = donor["blood_type"]
            recipient["crossmatch_result"] = "negative"
            recipient["height_cm"] = donor["height_cm"]
        higher["synthetic_lung_priority_score"] = 90
        higher["waiting_time_days"] = 20
        lower["synthetic_lung_priority_score"] = 10
        lower["waiting_time_days"] = 1095

        ranked = rank_recipients_baseline(donor, recipients)
        self.assertEqual(selectable_order(ranked)[:2], [higher["recipient_id"], lower["recipient_id"]])
        self.assertEqual(ranked[0]["factors"], {"synthetic_priority_index": 0.9})

    def test_blood_group_parser_rejects_prefix_matches(self) -> None:
        with self.assertRaises(PolicyInputError):
            abo_compatible("Apple", "A")

    def test_nonfinite_numeric_input_is_rejected(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        case["recipients"][0]["distance_km"] = float("inf")
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(case["donor"], case["recipients"])

    def test_fractional_waiting_days_are_rejected(self) -> None:
        with (DATASET_DIR / "validation_cases.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        case["recipients"][0]["waiting_time_days"] = 1.5
        with self.assertRaises(PolicyInputError):
            rank_recipients_baseline(case["donor"], case["recipients"])


if __name__ == "__main__":
    unittest.main()
