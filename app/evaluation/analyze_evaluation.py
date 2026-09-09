"""Compute prespecified metrics from raw locked-test model records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.synthetic_dataset import DATASET_DIR, PROTOCOL
from evaluation.run_model_evaluation import (
    _attempted_case_ids,
    validate_outcome_attempt_provenance,
    validate_test_lock,
)
from src.decision_guard import apply_guarded_policy
from src.llm_client import parse_note_review
from src.policy import rank_recipients_baseline
from src.schemas import ModelRunMetadata


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
ANALYSIS_PATH = Path(__file__).resolve()
TEST_PATH = DATASET_DIR / "test_cases.jsonl"
STATES = ("eligible", "review_required", "temporary_hold")
BOOTSTRAP_SEED = int(PROTOCOL["model_evaluation"]["bootstrap"]["seed"])
BOOTSTRAP_RESAMPLES = int(PROTOCOL["model_evaluation"]["bootstrap"]["resamples"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    case_values: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    seed_offset: int,
) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    count = len(case_values)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [case_values[rng.randrange(count)] for _ in range(count)]
        samples.append(statistic(resample))
    return [round(_percentile(samples, 0.025), 6), round(_percentile(samples, 0.975), 6)]


def _bootstrap_classification_intervals(
    case_counts: Sequence[Mapping[str, Mapping[str, int]]],
) -> Dict[str, list[float]]:
    rng = random.Random(BOOTSTRAP_SEED + 1)
    sample_values: Dict[str, list[float]] = {
        "macro_f1": [],
        **{
            f"{state}_{metric}": []
            for state in STATES
            for metric in ("precision", "recall", "f1")
        },
    }
    count = len(case_counts)
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [case_counts[rng.randrange(count)] for _ in range(count)]
        merged = _merge_classification_counts(sampled)
        metrics = _metrics_from_counts(merged, sum(merged[state]["support"] for state in STATES))
        sample_values["macro_f1"].append(float(metrics["macro_f1"]))
        for state in STATES:
            for metric in ("precision", "recall", "f1"):
                sample_values[f"{state}_{metric}"].append(
                    float(metrics["by_class"][state][metric])
                )
    return {
        name: [round(_percentile(values, 0.025), 6), round(_percentile(values, 0.975), 6)]
        for name, values in sample_values.items()
    }


def _classification_counts(pairs: Iterable[tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    pairs = list(pairs)
    counts = {
        state: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for state in STATES
    }
    for true, predicted in pairs:
        if true in counts:
            counts[true]["support"] += 1
            if predicted == true:
                counts[true]["tp"] += 1
            else:
                counts[true]["fn"] += 1
        if predicted in counts and predicted != true:
            counts[predicted]["fp"] += 1
    return counts


def _merge_classification_counts(
    values: Sequence[Mapping[str, Mapping[str, int]]],
) -> Dict[str, Dict[str, int]]:
    merged = {
        state: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for state in STATES
    }
    for item in values:
        for state in STATES:
            for key in ("tp", "fp", "fn", "support"):
                merged[state][key] += int(item[state][key])
    return merged


def _metrics_from_counts(counts: Mapping[str, Mapping[str, int]], candidate_count: int) -> Dict[str, Any]:
    by_class: Dict[str, Any] = {}
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
        correct += int(tp)
    return {
        "candidate_count": candidate_count,
        "accuracy": round(correct / candidate_count, 6) if candidate_count else 0.0,
        "macro_f1": round(statistics.mean(f1_values), 6),
        "by_class": by_class,
    }


def _classification_metrics(pairs: Iterable[tuple[str, str]]) -> Dict[str, Any]:
    pairs = list(pairs)
    counts = _classification_counts(pairs)
    return _metrics_from_counts(counts, len(pairs))


def _mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> Dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(not a and b for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_only_correct": left_only,
        "guarded_only_correct": right_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": round(p_value, 8),
    }


def _proportion(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _counted_proportion(values: Sequence[bool]) -> Dict[str, Any]:
    return {
        "count": sum(values),
        "denominator": len(values),
        "proportion": round(_proportion(values), 6) if values else None,
    }


def _conditional_proportion(
    rows: Sequence[Mapping[str, Any]],
    numerator_field: str,
    denominator_predicate: Callable[[Mapping[str, Any]], bool],
    seed_offset: int,
) -> Dict[str, Any]:
    eligible = [row for row in rows if denominator_predicate(row)]
    values = [bool(row[numerator_field]) for row in eligible]
    return {
        "count": sum(values),
        "denominator": len(values),
        "proportion": round(_proportion(values), 6) if values else None,
        "proportion_95ci": _bootstrap_ci(values, _proportion, seed_offset) if values else None,
    }


def _summarize_latency(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def analyze(
    records_path: Path,
    allow_incomplete: bool = False,
    *,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    records_path = records_path.resolve()
    cases = _load_jsonl(TEST_PATH)
    records = _load_jsonl(records_path)
    invalid_statuses = sorted(
        {str(record.get("status", "")) for record in records}.difference({"success", "failure"})
    )
    if invalid_statuses:
        raise RuntimeError(f"Raw evaluation contains invalid status values: {invalid_statuses}")
    record_map: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    for record in records:
        case_id = str(record.get("case_id"))
        if case_id in record_map:
            duplicates.append(case_id)
        record_map[case_id] = record
    if duplicates:
        raise RuntimeError(f"Duplicate raw records for cases: {sorted(set(duplicates))[:5]}")
    expected_case_ids = {str(case["case_id"]) for case in cases}
    unexpected = sorted(set(record_map).difference(expected_case_ids))
    if unexpected:
        raise RuntimeError(f"Raw evaluation contains {len(unexpected)} unknown case IDs: {unexpected[:5]}")
    missing = [str(case["case_id"]) for case in cases if str(case["case_id"]) not in record_map]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Raw evaluation is incomplete; missing {len(missing)} of {len(cases)} cases")

    case_rows: list[Dict[str, Any]] = []
    candidate_pairs_by_case: list[list[tuple[str, str]]] = []
    selectable_pairs_by_case: list[list[tuple[str, str]]] = []
    evidence_exact_by_case: list[list[bool]] = []
    failure_counts = Counter()
    confusion = Counter()
    latencies: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    evidence_code_state_results: Dict[str, list[bool]] = defaultdict(list)
    note_style_pairs: Dict[str, list[tuple[str, str]]] = defaultdict(list)
    note_complexity_pairs: Dict[str, list[tuple[str, str]]] = defaultdict(list)

    for case in cases:
        case_id = str(case["case_id"])
        record = record_map.get(case_id)
        true_by_id = {
            int(recipient["recipient_id"]): (
                str(recipient["reference_note_state"]),
                tuple(recipient["reference_evidence_codes"]),
            )
            for recipient in case["recipients"]
        }
        pred_by_id: Dict[int, tuple[str, tuple[str, ...]]] = {}
        selectable_ids = {int(value) for value in case["reference"]["baseline_order"]}
        success = bool(record and record.get("status") == "success")
        if success:
            if (
                record.get("organ_type") != case["organ_type"]
                or record.get("scenario_family") != case["scenario_family"]
            ):
                raise RuntimeError(f"Successful record metadata differs from case {case_id}")
            expected_ids = [int(item["recipient_id"]) for item in case["recipients"]]
            review = parse_note_review(record.get("review", {}), case_id, expected_ids)
            raw_output = record.get("raw_model_output")
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise RuntimeError(f"Successful record has no raw model output for case {case_id}")
            raw_review = parse_note_review(raw_output, case_id, expected_ids)
            if raw_review.model_dump(mode="json") != review.model_dump(mode="json"):
                raise RuntimeError(f"Parsed and raw reviews differ for case {case_id}")
            ranked = rank_recipients_baseline(case["donor"], case["recipients"], PROTOCOL)
            recomputed_guard = apply_guarded_policy(
                int(case["donor"]["donor_id"]),
                ranked,
                review,
                expected_ids,
            ).model_dump(mode="json")
            if record.get("guarded_decision") != recomputed_guard:
                raise RuntimeError(f"Guarded decision does not reproduce for case {case_id}")
            for item in review.assessments:
                pred_by_id[int(item.recipient_id)] = (
                    str(item.state.value),
                    tuple(code.value for code in item.evidence_codes),
                )
            metadata = ModelRunMetadata.model_validate(record.get("model_run", {}))
            if metadata.model_id != str(record.get("model_id_requested", "")):
                raise RuntimeError(f"Observed model differs from the requested model for case {case_id}")
            latencies.append(float(metadata.latency_ms))
            if metadata.input_tokens is not None:
                input_tokens.append(int(metadata.input_tokens))
            if metadata.output_tokens is not None:
                output_tokens.append(int(metadata.output_tokens))
        else:
            failure_counts[str(record.get("error_type", "missing_record") if record else "missing_record")] += 1

        pairs: list[tuple[str, str]] = []
        evidence_matches: list[bool] = []
        for recipient_id, (true_state, true_codes) in true_by_id.items():
            predicted = pred_by_id.get(recipient_id, ("__failure__", tuple()))
            pairs.append((true_state, predicted[0]))
            evidence_matches.append(set(true_codes) == set(predicted[1]))
            confusion[(true_state, predicted[0])] += 1
            recipient = next(
                item for item in case["recipients"]
                if int(item["recipient_id"]) == recipient_id
            )
            note_style_pairs[str(recipient["note_style"])].append((true_state, predicted[0]))
            note_complexity_pairs[str(recipient["note_complexity"])].append(
                (true_state, predicted[0])
            )
            for code in true_codes:
                evidence_code_state_results[code].append(true_state == predicted[0])
        candidate_pairs_by_case.append(pairs)
        selectable_pairs_by_case.append([
            pair
            for recipient_id, pair in zip(true_by_id, pairs)
            if recipient_id in selectable_ids
        ])
        evidence_exact_by_case.append(evidence_matches)

        baseline_primary = int(case["reference"]["baseline_order"][0])
        baseline_backup = int(case["reference"]["baseline_order"][1])
        reference_primary = int(case["reference"]["primary_recipient_id"])
        reference_backup = int(case["reference"]["backup_recipient_id"])
        reference_primary_rank = case["reference"]["baseline_order"].index(reference_primary) + 1
        candidate_pool_difficulty = (
            "no_primary_displacement"
            if reference_primary_rank == 1
            else "one_position_displacement"
            if reference_primary_rank == 2
            else "multiple_position_displacement"
        )
        baseline_unsafe = true_by_id[baseline_primary][0] == "temporary_hold"
        guarded = record.get("guarded_decision", {}) if success else {}
        guarded_primary = guarded.get("primary_recipient_id")
        guarded_backup = guarded.get("backup_recipient_id")
        guarded_unsafe = (
            guarded_primary is not None
            and true_by_id[int(guarded_primary)][0] == "temporary_hold"
        )
        guarded_backup_unsafe = (
            guarded_backup is not None
            and true_by_id[int(guarded_backup)][0] == "temporary_hold"
        )
        selection_made = guarded_primary is not None and guarded_backup is not None
        changed = guarded_primary is not None and int(guarded_primary) != baseline_primary
        state_exact = success and all(true == predicted for true, predicted in pairs)
        evidence_exact = success and all(evidence_matches)
        case_rows.append({
            "case_id": case_id,
            "organ_type": case["organ_type"],
            "scenario_family": case["scenario_family"],
            "model_success": success,
            "failure_type": "" if success else str(record.get("error_type", "missing_record") if record else "missing_record"),
            "selection_made": selection_made,
            "all_ten_states_exact": bool(state_exact),
            "all_ten_evidence_sets_exact": bool(evidence_exact),
            "reference_temporary_hold_count": sum(
                state == "temporary_hold" for state, _ in true_by_id.values()
            ),
            "reference_review_required_count": sum(
                state == "review_required" for state, _ in true_by_id.values()
            ),
            "baseline_primary": baseline_primary,
            "baseline_backup": baseline_backup,
            "reference_primary": reference_primary,
            "reference_backup": reference_backup,
            "reference_primary_baseline_rank": reference_primary_rank,
            "candidate_pool_difficulty": candidate_pool_difficulty,
            "guarded_primary": guarded_primary,
            "guarded_backup": guarded_backup,
            "baseline_top1_conformant": baseline_primary == reference_primary,
            "guarded_top1_conformant": guarded_primary is not None and int(guarded_primary) == reference_primary,
            "baseline_top2_set_conformant": {baseline_primary, baseline_backup} == {reference_primary, reference_backup},
            "guarded_top2_set_conformant": (
                guarded_primary is not None
                and guarded_backup is not None
                and {int(guarded_primary), int(guarded_backup)} == {reference_primary, reference_backup}
            ),
            "baseline_primary_unsafe": baseline_unsafe,
            "guarded_primary_unsafe": bool(guarded_unsafe),
            "guarded_backup_unsafe": bool(guarded_backup_unsafe),
            "unsafe_or_missing_primary": bool(not selection_made or guarded_unsafe),
            "unsafe_or_missing_backup": bool(not selection_made or guarded_backup_unsafe),
            "appropriate_override": bool(baseline_unsafe and changed and not guarded_unsafe),
            "missed_hold": bool(baseline_unsafe and not changed),
            "false_override": bool(not baseline_unsafe and changed),
            "unsafe_introduction": bool(not baseline_unsafe and changed and guarded_unsafe),
            "changed_primary": bool(changed),
        })

    flat_pairs = [pair for case_pairs in candidate_pairs_by_case for pair in case_pairs]
    classification = _classification_metrics(flat_pairs)
    count_vectors = [_classification_counts(values) for values in candidate_pairs_by_case]
    classification_intervals = _bootstrap_classification_intervals(count_vectors)
    classification["macro_f1_95ci"] = classification_intervals["macro_f1"]
    for state in STATES:
        for metric in ("precision", "recall", "f1"):
            classification["by_class"][state][f"{metric}_95ci"] = classification_intervals[
                f"{state}_{metric}"
            ]
    selectable_flat_pairs = [pair for case_pairs in selectable_pairs_by_case for pair in case_pairs]
    selectable_classification = _classification_metrics(selectable_flat_pairs)
    selectable_count_vectors = [
        _classification_counts(values) for values in selectable_pairs_by_case
    ]
    selectable_intervals = _bootstrap_classification_intervals(selectable_count_vectors)
    selectable_classification["macro_f1_95ci"] = selectable_intervals["macro_f1"]
    for state in STATES:
        for metric in ("precision", "recall", "f1"):
            selectable_classification["by_class"][state][f"{metric}_95ci"] = (
                selectable_intervals[f"{state}_{metric}"]
            )
    classification["evidence_code_exact_match"] = round(
        sum(value for values in evidence_exact_by_case for value in values) / sum(len(values) for values in evidence_exact_by_case),
        6,
    )
    classification["all_ten_states_exact"] = {
        "count": sum(bool(row["all_ten_states_exact"]) for row in case_rows),
        "denominator": len(case_rows),
        "proportion": round(_proportion([bool(row["all_ten_states_exact"]) for row in case_rows]), 6),
    }
    classification["all_ten_evidence_sets_exact"] = {
        "count": sum(bool(row["all_ten_evidence_sets_exact"]) for row in case_rows),
        "denominator": len(case_rows),
        "proportion": round(_proportion([bool(row["all_ten_evidence_sets_exact"]) for row in case_rows]), 6),
    }

    metric_fields = (
        "baseline_top1_conformant",
        "guarded_top1_conformant",
        "baseline_top2_set_conformant",
        "guarded_top2_set_conformant",
        "baseline_primary_unsafe",
        "guarded_primary_unsafe",
        "guarded_backup_unsafe",
        "unsafe_or_missing_primary",
        "unsafe_or_missing_backup",
        "appropriate_override",
        "missed_hold",
        "false_override",
        "unsafe_introduction",
        "changed_primary",
        "model_success",
        "selection_made",
        "all_ten_states_exact",
        "all_ten_evidence_sets_exact",
    )
    case_metrics: Dict[str, Any] = {}
    for offset, field in enumerate(metric_fields, start=10):
        values = [bool(row[field]) for row in case_rows]
        case_metrics[field] = {
            "count": sum(values),
            "denominator": len(values),
            "proportion": round(_proportion(values), 6),
            "proportion_95ci": _bootstrap_ci(values, _proportion, offset),
        }

    conditional_metrics = {
        "appropriate_override_among_baseline_holds": _conditional_proportion(
            case_rows,
            "appropriate_override",
            lambda row: bool(row["baseline_primary_unsafe"]),
            301,
        ),
        "missed_hold_among_baseline_holds": _conditional_proportion(
            case_rows,
            "missed_hold",
            lambda row: bool(row["baseline_primary_unsafe"]),
            302,
        ),
        "false_override_among_baseline_safe_cases": _conditional_proportion(
            case_rows,
            "false_override",
            lambda row: not bool(row["baseline_primary_unsafe"]),
            303,
        ),
        "unsafe_introduction_among_baseline_safe_cases": _conditional_proportion(
            case_rows,
            "unsafe_introduction",
            lambda row: not bool(row["baseline_primary_unsafe"]),
            306,
        ),
        "unsafe_primary_among_completed_selections": _conditional_proportion(
            case_rows,
            "guarded_primary_unsafe",
            lambda row: bool(row["selection_made"]),
            304,
        ),
        "unsafe_backup_among_completed_selections": _conditional_proportion(
            case_rows,
            "guarded_backup_unsafe",
            lambda row: bool(row["selection_made"]),
            305,
        ),
    }

    by_organ: Dict[str, Any] = {}
    for organ in sorted({str(row["organ_type"]) for row in case_rows}):
        indices = [index for index, row in enumerate(case_rows) if row["organ_type"] == organ]
        organ_pairs = [pair for index in indices for pair in candidate_pairs_by_case[index]]
        rows = [case_rows[index] for index in indices]
        by_organ[organ] = {
            "case_count": len(rows),
            "classification": _classification_metrics(organ_pairs),
            "guarded_top1_conformance": _counted_proportion(
                [bool(row["guarded_top1_conformant"]) for row in rows]
            ),
            "guarded_primary_unsafe": _counted_proportion(
                [bool(row["guarded_primary_unsafe"]) for row in rows]
            ),
            "unsafe_or_missing_primary": _counted_proportion(
                [bool(row["unsafe_or_missing_primary"]) for row in rows]
            ),
        }

    by_scenario: Dict[str, Any] = {}
    for scenario in sorted({str(row["scenario_family"]) for row in case_rows}):
        rows = [row for row in case_rows if row["scenario_family"] == scenario]
        by_scenario[scenario] = {
            "case_count": len(rows),
            "guarded_top1_conformance": _counted_proportion(
                [bool(row["guarded_top1_conformant"]) for row in rows]
            ),
            "guarded_primary_unsafe": _counted_proportion(
                [bool(row["guarded_primary_unsafe"]) for row in rows]
            ),
            "unsafe_or_missing_primary": _counted_proportion(
                [bool(row["unsafe_or_missing_primary"]) for row in rows]
            ),
            "false_override": _counted_proportion(
                [bool(row["false_override"]) for row in rows]
            ),
        }

    by_candidate_pool_difficulty: Dict[str, Any] = {}
    for difficulty in sorted({str(row["candidate_pool_difficulty"]) for row in case_rows}):
        indices = [
            index
            for index, row in enumerate(case_rows)
            if row["candidate_pool_difficulty"] == difficulty
        ]
        rows = [case_rows[index] for index in indices]
        pairs = [pair for index in indices for pair in candidate_pairs_by_case[index]]
        by_candidate_pool_difficulty[difficulty] = {
            "case_count": len(rows),
            "classification": _classification_metrics(pairs),
            "guarded_top1_conformance": _counted_proportion(
                [bool(row["guarded_top1_conformant"]) for row in rows]
            ),
            "unsafe_or_missing_primary": _counted_proportion(
                [bool(row["unsafe_or_missing_primary"]) for row in rows]
            ),
        }

    by_evidence_code = {
        code: {
            "candidate_count": len(values),
            "state_correct_count": sum(values),
            "state_accuracy": round(_proportion(values), 6),
        }
        for code, values in sorted(evidence_code_state_results.items())
    }
    by_note_style = {
        style: _classification_metrics(pairs)
        for style, pairs in sorted(note_style_pairs.items())
    }
    by_note_complexity = {
        complexity: _classification_metrics(pairs)
        for complexity, pairs in sorted(note_complexity_pairs.items())
    }

    condition_values = sorted({str(record.get("condition", "")) for record in records})
    requested_models = sorted({str(record.get("model_id_requested", "")) for record in records})
    if len(condition_values) > 1 or len(requested_models) > 1:
        raise RuntimeError("A raw evaluation file must contain exactly one condition and requested model")
    config_path = records_path.with_name(records_path.stem.removesuffix("_raw") + "_config.json")
    config = None
    if not config_path.exists() and not allow_incomplete:
        raise RuntimeError("A complete evaluation requires its generated configuration file")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("test_file_sha256") != _sha256(TEST_PATH):
            raise RuntimeError("Evaluation config test hash does not match the current locked test file")
        if config.get("raw_file_sha256") and config["raw_file_sha256"] != _sha256(records_path):
            raise RuntimeError("Evaluation config raw-file hash does not match the supplied record file")
        if condition_values and config.get("condition") != condition_values[0]:
            raise RuntimeError("Evaluation config condition does not match the raw records")
        if requested_models and config.get("model_id_requested") != requested_models[0]:
            raise RuntimeError("Evaluation config model does not match the raw records")
        if config.get("raw_record_count") is not None:
            if int(config["raw_record_count"]) != len(records):
                raise RuntimeError("Evaluation config record count does not match the raw records")
        elif not allow_incomplete:
            raise RuntimeError("A complete evaluation config has no raw-record count")
        if config.get("test_case_count") is not None:
            if int(config["test_case_count"]) != len(cases):
                raise RuntimeError("Evaluation config test-case count does not match the test file")
        elif not allow_incomplete:
            raise RuntimeError("A complete evaluation config has no test-case count")
        if not allow_incomplete:
            if config.get("evaluation_in_progress") is not False:
                raise RuntimeError("Evaluation configuration is not marked complete")
            successful_count = sum(record["status"] == "success" for record in records)
            failed_count = sum(record["status"] == "failure" for record in records)
            if (
                int(config.get("successful_record_count", -1)) != successful_count
                or int(config.get("failed_record_count", -1)) != failed_count
            ):
                raise RuntimeError("Evaluation config outcome counts do not match the raw records")
            attempt_log = config.get("attempt_log_file")
            if not isinstance(attempt_log, str) or not attempt_log.strip():
                raise RuntimeError("A complete evaluation must record its first-attempt ledger")
            attempt_path = config_path.with_name(
                records_path.stem.removesuffix("_raw") + "_attempts.jsonl"
            )
            if resolve_recorded_path(attempt_log) != attempt_path.resolve():
                raise RuntimeError("The first-attempt ledger path does not match the evaluation")
            if not attempt_path.is_file():
                raise RuntimeError("The first-attempt ledger is missing")
            if config.get("attempt_log_sha256") != _sha256(attempt_path):
                raise RuntimeError("The first-attempt ledger hash does not match the configuration")
            attempted, attempt_rows = _attempted_case_ids(
                attempt_path,
                condition=str(config.get("condition", "")),
                model_id=str(config.get("model_id_requested", "")),
            )
            if attempted != expected_case_ids or int(config.get("attempted_case_count", -1)) != len(cases):
                raise RuntimeError("The first-attempt ledger does not cover the complete test set")
            validate_outcome_attempt_provenance(
                records,
                attempt_rows,
                require_complete=True,
            )
            config_lock = config.get("test_lock")
            lock_path = config_path.parent / "test_lock.json"
            if not isinstance(config_lock, dict):
                raise RuntimeError("A complete evaluation config must contain the frozen test lock")
            if not lock_path.exists():
                raise RuntimeError("The frozen test-lock file is missing")
            with lock_path.open("r", encoding="utf-8") as handle:
                file_lock = json.load(handle)
            if file_lock != config_lock:
                raise RuntimeError("Evaluation config and test-lock file do not match")
            validate_test_lock(file_lock)
            if config_lock.get("test_file_sha256") != _sha256(TEST_PATH):
                raise RuntimeError("The frozen test lock does not match the supplied test file")
            if config_lock.get("analyzer_sha256") != _sha256(ANALYSIS_PATH):
                raise RuntimeError("The analysis script changed after the test set was locked")

    actual_models = sorted({
        str(record.get("model_run", {}).get("model_id", ""))
        for record in records
        if record.get("status") == "success"
    })
    if actual_models and actual_models != requested_models:
        raise RuntimeError("Observed successful model identifiers differ from the requested model")

    summary = {
        "generated_at_utc": _utc_now(),
        "analysis_script_sha256": _sha256(ANALYSIS_PATH),
        "condition_values": condition_values,
        "model_id_requested_values": requested_models,
        "model_id_observed_values": actual_models,
        "records_file": portable_path(records_path),
        "records_file_sha256": _sha256(records_path),
        "config_file": portable_path(config_path) if config_path.exists() else None,
        "config": config,
        "test_file": portable_path(TEST_PATH),
        "test_file_sha256": _sha256(TEST_PATH),
        "test_case_count": len(cases),
        "raw_record_count": len(records),
        "missing_record_count": len(missing),
        "successful_case_count": sum(row["model_success"] for row in case_rows),
        "failure_counts": dict(sorted(failure_counts.items())),
        "classification": classification,
        "classification_selectable_candidates": selectable_classification,
        "case_metrics": case_metrics,
        "conditional_case_metrics": conditional_metrics,
        "by_organ": by_organ,
        "by_scenario": by_scenario,
        "by_candidate_pool_difficulty": by_candidate_pool_difficulty,
        "by_evidence_code": by_evidence_code,
        "by_note_style": by_note_style,
        "by_note_complexity": by_note_complexity,
        "latency": _summarize_latency(latencies),
        "token_usage": {
            "input_total": sum(input_tokens),
            "output_total": sum(output_tokens),
            "input_mean_per_successful_case": round(statistics.mean(input_tokens), 3) if input_tokens else None,
            "output_mean_per_successful_case": round(statistics.mean(output_tokens), 3) if output_tokens else None,
        },
        "paired_top1_mcnemar": _mcnemar_exact(
            [row["baseline_top1_conformant"] for row in case_rows],
            [row["guarded_top1_conformant"] for row in case_rows],
        ),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }

    if write_outputs:
        stem = records_path.stem.removesuffix("_raw")
        output_dir = records_path.parent if records_path.parent.exists() else OUTPUT_DIR
        summary_path = output_dir / f"{stem}_summary.json"
        rows_path = output_dir / f"{stem}_case_metrics.csv"
        confusion_path = output_dir / f"{stem}_confusion.csv"
        with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
            writer.writeheader()
            writer.writerows(case_rows)
        with confusion_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reference_state", "predicted_state", "count"])
            writer.writeheader()
            for (true_state, predicted_state), count in sorted(confusion.items()):
                writer.writerow({"reference_state": true_state, "predicted_state": predicted_state, "count": count})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.records, allow_incomplete=args.allow_incomplete)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
