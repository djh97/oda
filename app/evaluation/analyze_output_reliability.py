"""Post hoc analysis separating model-output validity from decision outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.synthetic_dataset import DATASET_DIR, PROTOCOL, STATE_CODES
from src.decision_guard import GuardAbstention, apply_guarded_policy
from src.llm_client import parse_note_review
from src.policy import rank_recipients_baseline
from src.schemas import NoteReviewBatch


APP_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
TEST_PATH = DATASET_DIR / "test_cases.jsonl"
OUTPUT_PATH = EVALUATION_DIR / "posthoc_output_reliability.json"
SCRIPT_PATH = Path(__file__).resolve()
STATES = ("eligible", "review_required", "temporary_hold")
CONDITIONS = ("tfidf_logistic", "untuned", "fine_tuned", "openai")
BOOTSTRAP_SEED = int(PROTOCOL["model_evaluation"]["bootstrap"]["seed"])
BOOTSTRAP_RESAMPLES = int(PROTOCOL["model_evaluation"]["bootstrap"]["resamples"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stratified_groups(
    cases: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        groups[(str(case["organ_type"]), str(case["scenario_family"]))].append(index)
    return [groups[key] for key in sorted(groups)]


def _draw_stratified_indices(
    rng: random.Random,
    groups: Sequence[Sequence[int]],
) -> list[int]:
    return [
        group[rng.randrange(len(group))]
        for group in groups
        for _ in range(len(group))
    ]


def _interval(values: Sequence[float]) -> list[float]:
    return [
        round(_percentile(values, 0.025), 6),
        round(_percentile(values, 0.975), 6),
    ]


def _classification_vector(
    pairs: Iterable[tuple[str, str]],
) -> tuple[int, ...]:
    positions = {state: index for index, state in enumerate(STATES)}
    values = [0] * (len(STATES) * 3)
    for truth, prediction in pairs:
        truth_index = positions[truth] * 3
        if prediction == truth:
            values[truth_index] += 1
        else:
            values[truth_index + 2] += 1
        if prediction in positions and prediction != truth:
            values[positions[prediction] * 3 + 1] += 1
    return tuple(values)


def _sum_vectors(
    vectors: Iterable[tuple[int, ...]],
) -> tuple[int, ...]:
    total = [0] * (len(STATES) * 3)
    for vector in vectors:
        for index, value in enumerate(vector):
            total[index] += value
    return tuple(total)


def _macro_f1_from_vector(vector: Sequence[int]) -> float:
    values = []
    for index in range(len(STATES)):
        tp, fp, fn = vector[index * 3 : index * 3 + 3]
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def _stratified_intervals(
    cases: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    seed_offset: int,
) -> dict[str, list[float] | None]:
    groups = _stratified_groups(cases)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    samples: dict[str, list[float]] = {
        "strict_schema_complete_output": [],
        "conditional_macro_f1": [],
        "failure_penalized_macro_f1": [],
        "temporary_hold_sensitivity": [],
        "complete_primary_backup_pair": [],
        "held_primary": [],
        "unavailable_primary": [],
        "reference_primary_agreement": [],
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = _draw_stratified_indices(rng, groups)
        selected = [units[index] for index in indices]
        denominator = len(selected)
        samples["strict_schema_complete_output"].append(
            sum(bool(unit["strict_valid"]) for unit in selected) / denominator
        )
        conditional_vectors = [
            unit["conditional_vector"]
            for unit in selected
            if unit["conditional_vector"] is not None
        ]
        if conditional_vectors:
            samples["conditional_macro_f1"].append(
                _macro_f1_from_vector(_sum_vectors(conditional_vectors))
            )
        samples["failure_penalized_macro_f1"].append(
            _macro_f1_from_vector(
                _sum_vectors(unit["failure_vector"] for unit in selected)
            )
        )
        hold_support = sum(int(unit["hold_support"]) for unit in selected)
        samples["temporary_hold_sensitivity"].append(
            sum(int(unit["hold_true_positive"]) for unit in selected) / hold_support
        )
        for field in (
            "complete_primary_backup_pair",
            "held_primary",
            "unavailable_primary",
            "reference_primary_agreement",
        ):
            samples[field].append(
                sum(bool(unit[field]) for unit in selected) / denominator
            )
    return {
        name: _interval(values) if values else None
        for name, values in samples.items()
    }


def _stratified_boolean_intervals(
    cases: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    seed_offset: int,
) -> dict[str, list[float]]:
    groups = _stratified_groups(cases)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    samples = {field: [] for field in fields}
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = _draw_stratified_indices(rng, groups)
        for field in fields:
            samples[field].append(
                sum(bool(units[index][field]) for index in indices) / len(indices)
            )
    return {field: _interval(values) for field, values in samples.items()}


def _classification_metrics(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, Any] | None:
    pairs = list(pairs)
    if not pairs:
        return None
    counts = {
        state: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for state in STATES
    }
    for truth, prediction in pairs:
        if truth not in counts:
            raise ValueError(f"Unknown reference state: {truth}")
        counts[truth]["support"] += 1
        if prediction == truth:
            counts[truth]["tp"] += 1
        else:
            counts[truth]["fn"] += 1
        if prediction in counts and prediction != truth:
            counts[prediction]["fp"] += 1

    by_class: dict[str, Any] = {}
    f1_values = []
    correct = 0
    for state, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[state] = {
            **values,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)
        correct += tp
    return {
        "candidate_count": len(pairs),
        "accuracy": round(correct / len(pairs), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "by_class": by_class,
    }


def _validate_retained_run(
    records_path: Path,
    config_path: Path,
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = _load_jsonl(records_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("raw_file_sha256") != _sha256(records_path):
        raise RuntimeError(f"Raw-output hash mismatch: {records_path.name}")
    if config.get("test_file_sha256") != _sha256(TEST_PATH):
        raise RuntimeError(f"Test-file hash mismatch: {config_path.name}")
    if len(records) != len(cases) or int(config.get("raw_record_count", -1)) != len(records):
        raise RuntimeError(f"Record-count mismatch: {records_path.name}")
    expected = [str(case["case_id"]) for case in cases]
    observed = [str(record.get("case_id", "")) for record in records]
    if observed != expected:
        raise RuntimeError(f"Case order differs from the frozen test set: {records_path.name}")
    return records


def _parse_strict_review(
    record: Mapping[str, Any],
    case: Mapping[str, Any],
) -> NoteReviewBatch | None:
    raw_output = record.get("raw_model_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None
    expected_ids = [int(item["recipient_id"]) for item in case["recipients"]]
    try:
        review = parse_note_review(raw_output, str(case["case_id"]), expected_ids)
    except Exception:
        return None
    retained = record.get("review")
    if retained is not None and review.model_dump(mode="json") != retained:
        raise RuntimeError(f"Retained and reparsed reviews differ for {case['case_id']}")
    return review


def _retained_successful_review(
    record: Mapping[str, Any],
    case: Mapping[str, Any],
) -> NoteReviewBatch | None:
    if record.get("status") != "success" or record.get("review") is None:
        return None
    review = NoteReviewBatch.model_validate(record["review"])
    expected_ids = {int(item["recipient_id"]) for item in case["recipients"]}
    observed_ids = {int(item.recipient_id) for item in review.assessments}
    if observed_ids != expected_ids or len(review.assessments) != len(expected_ids):
        raise RuntimeError(f"Retained review has incomplete coverage for {case['case_id']}")
    return review


def _reconstruct_tfidf_reviews(
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, NoteReviewBatch]:
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    from evaluation.run_classical_baseline import (
        SCRIPT_PATH as CLASSICAL_SCRIPT_PATH,
        TRAIN_PATH,
        _feature_union,
        _load_jsonl as load_classical_jsonl,
        _logistic_regression,
        _predict_evidence_sets,
        _training_rows,
    )

    config = json.loads(
        (EVALUATION_DIR / "tfidf_logistic_config.json").read_text(encoding="utf-8")
    )
    if config.get("script_sha256") != _sha256(CLASSICAL_SCRIPT_PATH):
        raise RuntimeError("The TF-IDF implementation differs from the retained run")
    if config.get("training_file_sha256") != _sha256(TRAIN_PATH):
        raise RuntimeError("The TF-IDF training data differ from the retained run")

    training_cases = load_classical_jsonl(TRAIN_PATH)
    train_notes, train_states, train_evidence = _training_rows(training_cases)
    features = _feature_union()
    training_matrix = features.fit_transform(train_notes)
    state_model = _logistic_regression()
    state_model.fit(training_matrix, train_states)
    evidence_binarizer = MultiLabelBinarizer(
        classes=sorted(code for values in STATE_CODES.values() for code in values)
    )
    evidence_matrix = evidence_binarizer.fit_transform(train_evidence)
    evidence_model = OneVsRestClassifier(_logistic_regression(), n_jobs=1)
    evidence_model.fit(training_matrix, evidence_matrix)

    notes = [str(recipient["medical_notes"]) for case in cases for recipient in case["recipients"]]
    test_matrix = features.transform(notes)
    predicted_states = [str(value) for value in state_model.predict(test_matrix)]
    predicted_evidence = _predict_evidence_sets(
        predicted_states,
        evidence_model.predict_proba(test_matrix),
        evidence_binarizer.classes_,
    )

    reviews: dict[str, NoteReviewBatch] = {}
    offset = 0
    record_by_id = {str(record["case_id"]): record for record in records}
    for case in cases:
        case_id = str(case["case_id"])
        count = len(case["recipients"])
        review = NoteReviewBatch.model_validate({
            "case_id": case_id,
            "assessments": [
                {
                    "recipient_id": int(recipient["recipient_id"]),
                    "state": state,
                    "evidence_codes": evidence,
                }
                for recipient, state, evidence in zip(
                    case["recipients"],
                    predicted_states[offset : offset + count],
                    predicted_evidence[offset : offset + count],
                )
            ],
        })
        offset += count
        retained = record_by_id[case_id].get("review")
        if retained is not None and review.model_dump(mode="json") != retained:
            raise RuntimeError(f"Reconstructed TF-IDF review differs for {case_id}")
        reviews[case_id] = review
    return reviews


def _analyze_condition(
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    seed_offset: int,
    reconstructed_reviews: Mapping[str, NoteReviewBatch] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records_by_id = {str(record["case_id"]): record for record in records}
    conditional_pairs: list[tuple[str, str]] = []
    failure_penalized_pairs: list[tuple[str, str]] = []
    strict_valid_count = 0
    completed_count = 0
    held_primary_count = 0
    reference_primary_count = 0
    units: list[dict[str, Any]] = []
    failure_types = Counter(
        str(record.get("error_type", "unknown_failure"))
        for record in records
        if record.get("status") == "failure"
    )

    for case in cases:
        case_id = str(case["case_id"])
        record = records_by_id[case_id]
        truth = {
            int(recipient["recipient_id"]): str(recipient["reference_note_state"])
            for recipient in case["recipients"]
        }
        conditional_review = (
            reconstructed_reviews.get(case_id)
            if reconstructed_reviews is not None
            else _parse_strict_review(record, case)
        )
        conditional_case_pairs: list[tuple[str, str]] = []
        if conditional_review is not None:
            strict_valid_count += 1
            predictions = {
                int(item.recipient_id): str(item.state.value)
                for item in conditional_review.assessments
            }
            conditional_case_pairs = [
                (truth[recipient_id], predictions[recipient_id])
                for recipient_id in truth
            ]
            conditional_pairs.extend(conditional_case_pairs)

        retained_review = _retained_successful_review(record, case)
        if retained_review is None:
            failure_case_pairs = [
                (state, "__failure__") for state in truth.values()
            ]
        else:
            retained_predictions = {
                int(item.recipient_id): str(item.state.value)
                for item in retained_review.assessments
            }
            failure_case_pairs = [
                (truth[recipient_id], retained_predictions[recipient_id])
                for recipient_id in truth
            ]
        failure_penalized_pairs.extend(failure_case_pairs)

        completed = False
        held_primary = False
        reference_primary = False
        if conditional_review is not None:
            ranked = rank_recipients_baseline(case["donor"], case["recipients"], PROTOCOL)
            try:
                guarded = apply_guarded_policy(
                    int(case["donor"]["donor_id"]),
                    ranked,
                    conditional_review,
                    truth,
                )
            except GuardAbstention:
                guarded = None
            if guarded is not None:
                completed = True
                primary = int(guarded.primary_recipient_id)
                held_primary = truth[primary] == "temporary_hold"
                reference_primary = (
                    primary == int(case["reference"]["primary_recipient_id"])
                )
                retained_guard = record.get("guarded_decision")
                if (
                    retained_guard is not None
                    and guarded.model_dump(mode="json") != retained_guard
                ):
                    raise RuntimeError(
                        f"Retained and recomputed guarded decisions differ for {case_id}"
                    )

        completed_count += int(completed)
        held_primary_count += int(held_primary)
        reference_primary_count += int(reference_primary)
        units.append({
            "strict_valid": conditional_review is not None,
            "conditional_vector": (
                _classification_vector(conditional_case_pairs)
                if conditional_case_pairs
                else None
            ),
            "failure_vector": _classification_vector(failure_case_pairs),
            "hold_support": sum(truth_state == "temporary_hold" for truth_state, _ in failure_case_pairs),
            "hold_true_positive": sum(
                truth_state == "temporary_hold" and predicted_state == "temporary_hold"
                for truth_state, predicted_state in failure_case_pairs
            ),
            "complete_primary_backup_pair": completed,
            "held_primary": held_primary,
            "unavailable_primary": not completed,
            "reference_primary_agreement": reference_primary,
        })

    case_count = len(cases)
    unavailable_count = case_count - completed_count
    summary = {
        "case_count": case_count,
        "strict_schema_complete_output": {
            "count": strict_valid_count,
            "denominator": case_count,
            "proportion": round(strict_valid_count / case_count, 6),
        },
        "conditional_state_classification": _classification_metrics(conditional_pairs),
        "failure_penalized_state_classification": _classification_metrics(failure_penalized_pairs),
        "decision_outcomes": {
            "complete_primary_backup_pair": {
                "count": completed_count,
                "denominator": case_count,
                "proportion": round(completed_count / case_count, 6),
            },
            "held_primary": {
                "count": held_primary_count,
                "denominator": case_count,
                "proportion": round(held_primary_count / case_count, 6),
            },
            "unavailable_primary": {
                "count": unavailable_count,
                "denominator": case_count,
                "proportion": round(unavailable_count / case_count, 6),
            },
            "held_or_unavailable_primary": {
                "count": held_primary_count + unavailable_count,
                "denominator": case_count,
                "proportion": round((held_primary_count + unavailable_count) / case_count, 6),
            },
            "reference_primary_agreement": {
                "count": reference_primary_count,
                "denominator": case_count,
                "proportion": round(reference_primary_count / case_count, 6),
            },
        },
        "retained_failure_types": dict(sorted(failure_types.items())),
        "stratified_bootstrap_95ci": _stratified_intervals(
            cases,
            units,
            seed_offset,
        ),
    }
    return summary, units


def _deterministic_outcomes(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held_primary = 0
    reference_agreement = 0
    units = []
    for case in cases:
        primary = int(case["reference"]["baseline_order"][0])
        truth = {
            int(recipient["recipient_id"]): str(recipient["reference_note_state"])
            for recipient in case["recipients"]
        }
        held = truth[primary] == "temporary_hold"
        agrees = primary == int(case["reference"]["primary_recipient_id"])
        held_primary += int(held)
        reference_agreement += int(agrees)
        units.append({
            "complete_primary_backup_pair": True,
            "held_primary": held,
            "unavailable_primary": False,
            "reference_primary_agreement": agrees,
        })
    count = len(cases)
    return {
        "case_count": count,
        "state_classification": None,
        "decision_outcomes": {
            "complete_primary_backup_pair": {"count": count, "denominator": count, "proportion": 1.0},
            "held_primary": {
                "count": held_primary,
                "denominator": count,
                "proportion": round(held_primary / count, 6),
            },
            "unavailable_primary": {"count": 0, "denominator": count, "proportion": 0.0},
            "held_or_unavailable_primary": {
                "count": held_primary,
                "denominator": count,
                "proportion": round(held_primary / count, 6),
            },
            "reference_primary_agreement": {
                "count": reference_agreement,
                "denominator": count,
                "proportion": round(reference_agreement / count, 6),
            },
        },
        "stratified_bootstrap_95ci": _stratified_boolean_intervals(
            cases,
            units,
            (
                "complete_primary_backup_pair",
                "held_primary",
                "unavailable_primary",
                "reference_primary_agreement",
            ),
            100,
        ),
    }


def _paired_strict_output_comparison(
    cases: Sequence[Mapping[str, Any]],
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_by_id = {str(record["case_id"]): record for record in left_records}
    right_by_id = {str(record["case_id"]): record for record in right_records}
    counts = {
        "both_valid": 0,
        "left_only_valid": 0,
        "right_only_valid": 0,
        "neither_valid": 0,
    }
    left_pairs: list[tuple[str, str]] = []
    right_pairs: list[tuple[str, str]] = []
    for case in cases:
        case_id = str(case["case_id"])
        left_review = _parse_strict_review(left_by_id[case_id], case)
        right_review = _parse_strict_review(right_by_id[case_id], case)
        if left_review is None and right_review is None:
            counts["neither_valid"] += 1
            continue
        if left_review is None:
            counts["right_only_valid"] += 1
            continue
        if right_review is None:
            counts["left_only_valid"] += 1
            continue
        counts["both_valid"] += 1
        truth = {
            int(recipient["recipient_id"]): str(recipient["reference_note_state"])
            for recipient in case["recipients"]
        }
        left_predictions = {
            int(item.recipient_id): str(item.state.value)
            for item in left_review.assessments
        }
        right_predictions = {
            int(item.recipient_id): str(item.state.value)
            for item in right_review.assessments
        }
        left_pairs.extend(
            (truth[recipient_id], left_predictions[recipient_id])
            for recipient_id in truth
        )
        right_pairs.extend(
            (truth[recipient_id], right_predictions[recipient_id])
            for recipient_id in truth
        )
    return {
        "case_validity_counts": counts,
        "left_classification": _classification_metrics(left_pairs),
        "right_classification": _classification_metrics(right_pairs),
    }


def _paired_stratified_intervals(
    cases: Sequence[Mapping[str, Any]],
    left_units: Sequence[Mapping[str, Any]],
    right_units: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    groups = _stratified_groups(cases)
    rng = random.Random(BOOTSTRAP_SEED + 500)
    samples = {
        "strict_output_right_minus_left": [],
        "common_valid_macro_f1_right_minus_left": [],
        "failure_penalized_macro_f1_right_minus_left": [],
        "complete_pair_right_minus_left": [],
        "reference_primary_agreement_right_minus_left": [],
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = _draw_stratified_indices(rng, groups)
        left_selected = [left_units[index] for index in indices]
        right_selected = [right_units[index] for index in indices]
        denominator = len(indices)
        samples["strict_output_right_minus_left"].append(
            (
                sum(bool(unit["strict_valid"]) for unit in right_selected)
                - sum(bool(unit["strict_valid"]) for unit in left_selected)
            )
            / denominator
        )
        common_indices = [
            index
            for index in indices
            if left_units[index]["conditional_vector"] is not None
            and right_units[index]["conditional_vector"] is not None
        ]
        left_common = _sum_vectors(
            left_units[index]["conditional_vector"] for index in common_indices
        )
        right_common = _sum_vectors(
            right_units[index]["conditional_vector"] for index in common_indices
        )
        samples["common_valid_macro_f1_right_minus_left"].append(
            _macro_f1_from_vector(right_common) - _macro_f1_from_vector(left_common)
        )
        left_failure = _sum_vectors(
            unit["failure_vector"] for unit in left_selected
        )
        right_failure = _sum_vectors(
            unit["failure_vector"] for unit in right_selected
        )
        samples["failure_penalized_macro_f1_right_minus_left"].append(
            _macro_f1_from_vector(right_failure)
            - _macro_f1_from_vector(left_failure)
        )
        for field, output_name in (
            ("complete_primary_backup_pair", "complete_pair_right_minus_left"),
            (
                "reference_primary_agreement",
                "reference_primary_agreement_right_minus_left",
            ),
        ):
            samples[output_name].append(
                (
                    sum(bool(unit[field]) for unit in right_selected)
                    - sum(bool(unit[field]) for unit in left_selected)
                )
                / denominator
            )
    return {name: _interval(values) for name, values in samples.items()}


def analyze() -> dict[str, Any]:
    cases = _load_jsonl(TEST_PATH)
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    input_files: dict[str, Any] = {
        "test_cases": {"path": str(TEST_PATH), "sha256": _sha256(TEST_PATH)}
    }
    for condition in CONDITIONS:
        records_path = EVALUATION_DIR / f"{condition}_raw.jsonl"
        config_path = EVALUATION_DIR / f"{condition}_config.json"
        records_by_condition[condition] = _validate_retained_run(
            records_path, config_path, cases
        )
        input_files[condition] = {
            "records_path": str(records_path),
            "records_sha256": _sha256(records_path),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        }

    tfidf_reviews = _reconstruct_tfidf_reviews(
        cases, records_by_condition["tfidf_logistic"]
    )
    conditions = {"deterministic_ranker_only": _deterministic_outcomes(cases)}
    condition_units: dict[str, list[dict[str, Any]]] = {}
    tfidf_summary, condition_units["tfidf_logistic"] = _analyze_condition(
        cases,
        records_by_condition["tfidf_logistic"],
        200,
        reconstructed_reviews=tfidf_reviews,
    )
    conditions["tfidf_logistic"] = tfidf_summary
    for index, condition in enumerate(("untuned", "fine_tuned", "openai"), start=1):
        summary, condition_units[condition] = _analyze_condition(
            cases,
            records_by_condition[condition],
            200 + index,
        )
        conditions[condition] = summary
    paired_comparison = _paired_strict_output_comparison(
        cases,
        records_by_condition["fine_tuned"],
        records_by_condition["openai"],
    )
    paired_comparison.update({
        "left": "fine_tuned",
        "right": "openai",
        "stratified_bootstrap_95ci": _paired_stratified_intervals(
            cases,
            condition_units["fine_tuned"],
            condition_units["openai"],
        ),
    })
    return {
        "analysis_type": "post_hoc_output_reliability_decomposition",
        "generated_at_utc": _utc_now(),
        "analysis_script_sha256": _sha256(SCRIPT_PATH),
        "definitions": {
            "strict_schema_complete_output": (
                "A response that satisfies the complete prespecified schema for all ten candidates."
            ),
            "conditional_state_classification": (
                "State-classification metrics calculated only from strict schema-valid complete outputs."
            ),
            "failure_penalized_state_classification": (
                "State-classification metrics over all cases, with every candidate in an invalid or "
                "unavailable output treated as an incorrect abstention."
            ),
            "held_primary": (
                "A completed pair whose selected primary has the study-generated temporary-hold state."
            ),
            "unavailable_primary": "A case for which no complete primary-backup pair was produced.",
            "stratified_bootstrap": (
                "Percentile intervals from case resampling within each organ-by-scenario "
                "stratum, preserving the fixed benchmark composition in every resample."
            ),
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "strata": ["organ_type", "scenario_family"],
            "stratum_count": len(_stratified_groups(cases)),
            "estimand": "design-standardized performance for the fixed synthetic benchmark mixture",
        },
        "input_files": input_files,
        "conditions": conditions,
        "paired_strict_output_comparison": paired_comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
