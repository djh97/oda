"""Paired comparison of two complete model runs on the locked test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.analyze_evaluation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    STATES,
    TEST_PATH,
    _classification_counts,
    _merge_classification_counts,
    _metrics_from_counts,
    _percentile,
)
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
COMPARISON_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_run_config(records_path: Path) -> Dict[str, Any]:
    config_path = records_path.with_name(records_path.stem.removesuffix("_raw") + "_config.json")
    if not config_path.exists():
        raise RuntimeError(f"Missing evaluation config for {records_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("raw_file_sha256") != _sha256(records_path):
        raise RuntimeError(f"Evaluation config raw-file hash mismatch for {records_path}")
    if config.get("test_file_sha256") != _sha256(TEST_PATH):
        raise RuntimeError(f"Evaluation config test-file hash mismatch for {records_path}")
    records = _load_jsonl(records_path)
    invalid_statuses = sorted({str(row.get("status", "")) for row in records}.difference({"success", "failure"}))
    if invalid_statuses:
        raise RuntimeError(f"Invalid evaluation status in {records_path}: {invalid_statuses}")
    conditions = {str(row.get("condition", "")) for row in records}
    models = {str(row.get("model_id_requested", "")) for row in records}
    if conditions != {str(config.get("condition", ""))}:
        raise RuntimeError(f"Evaluation config condition mismatch for {records_path}")
    if models != {str(config.get("model_id_requested", ""))}:
        raise RuntimeError(f"Evaluation config model mismatch for {records_path}")
    if int(config.get("raw_record_count", -1)) != len(records):
        raise RuntimeError(f"Evaluation config record-count mismatch for {records_path}")
    if int(config.get("test_case_count", -1)) != len(records):
        raise RuntimeError(f"Evaluation config test-case count mismatch for {records_path}")
    successful_count = sum(row["status"] == "success" for row in records)
    failed_count = sum(row["status"] == "failure" for row in records)
    if (
        int(config.get("successful_record_count", -1)) != successful_count
        or int(config.get("failed_record_count", -1)) != failed_count
    ):
        raise RuntimeError(f"Evaluation config outcome counts mismatch for {records_path}")
    if config.get("evaluation_in_progress") is not False:
        raise RuntimeError(f"Evaluation is not marked complete for {records_path}")
    attempt_log = config.get("attempt_log_file")
    if not isinstance(attempt_log, str) or not attempt_log.strip():
        raise RuntimeError(f"Evaluation attempt ledger is not recorded for {records_path}")
    attempt_path = records_path.with_name(
        records_path.stem.removesuffix("_raw") + "_attempts.jsonl"
    )
    if resolve_recorded_path(attempt_log) != attempt_path.resolve():
        raise RuntimeError(f"Evaluation attempt-ledger path mismatch for {records_path}")
    if not attempt_path.is_file() or config.get("attempt_log_sha256") != _sha256(attempt_path):
        raise RuntimeError(f"Evaluation attempt-ledger mismatch for {records_path}")
    attempted, attempt_rows = _attempted_case_ids(
        attempt_path,
        condition=str(config.get("condition", "")),
        model_id=str(config.get("model_id_requested", "")),
    )
    expected_case_ids = {str(row.get("case_id", "")) for row in records}
    if attempted != expected_case_ids or int(config.get("attempted_case_count", -1)) != len(records):
        raise RuntimeError(f"Evaluation attempt ledger is incomplete for {records_path}")
    validate_outcome_attempt_provenance(
        records,
        attempt_rows,
        require_complete=True,
    )
    config_lock = config.get("test_lock")
    lock_path = config_path.parent / "test_lock.json"
    if not isinstance(config_lock, dict) or not lock_path.exists():
        raise RuntimeError(f"Missing frozen test lock for {records_path}")
    with lock_path.open("r", encoding="utf-8") as handle:
        file_lock = json.load(handle)
    if file_lock != config_lock:
        raise RuntimeError(f"Evaluation config and test lock differ for {records_path}")
    validate_test_lock(file_lock)
    if config_lock.get("comparison_sha256") != _sha256(COMPARISON_PATH):
        raise RuntimeError("The comparison script changed after the test set was locked")
    return config


def _mcnemar(left: Sequence[bool], right: Sequence[bool]) -> Dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(not a and b for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": round(p_value, 8),
    }


def _condition_rows(records_path: Path, cases: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    records = _load_jsonl(records_path)
    record_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id"))
        if case_id in record_map:
            raise RuntimeError(f"Duplicate record for case {case_id} in {records_path}")
        record_map[case_id] = record
    expected = {str(case["case_id"]) for case in cases}
    if set(record_map) != expected:
        raise RuntimeError(
            f"{records_path} must contain exactly the {len(expected)} locked test cases; "
            f"found {len(record_map)}"
        )

    rows: list[Dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        record = record_map[case_id]
        success = record.get("status") == "success"
        predicted = {}
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
            ranked = rank_recipients_baseline(case["donor"], case["recipients"])
            recomputed_guard = apply_guarded_policy(
                int(case["donor"]["donor_id"]),
                ranked,
                review,
                expected_ids,
            ).model_dump(mode="json")
            if record.get("guarded_decision") != recomputed_guard:
                raise RuntimeError(f"Guarded decision does not reproduce for case {case_id}")
            metadata = ModelRunMetadata.model_validate(record.get("model_run", {}))
            if metadata.model_id != str(record.get("model_id_requested", "")):
                raise RuntimeError(f"Observed model differs from the requested model for case {case_id}")
            predicted = {
                int(item.recipient_id): str(item.state.value)
                for item in review.assessments
            }
        pairs = [
            (
                str(recipient["reference_note_state"]),
                predicted.get(int(recipient["recipient_id"]), "__failure__"),
            )
            for recipient in case["recipients"]
        ]
        guarded = recomputed_guard if success else {}
        guarded_primary = guarded.get("primary_recipient_id")
        guarded_backup = guarded.get("backup_recipient_id")
        baseline_primary = int(case["reference"]["baseline_order"][0])
        reference_primary = int(case["reference"]["primary_recipient_id"])
        true_by_id = {
            int(recipient["recipient_id"]): str(recipient["reference_note_state"])
            for recipient in case["recipients"]
        }
        rows.append({
            "counts": _classification_counts(pairs),
            "top1_correct": guarded_primary is not None and int(guarded_primary) == reference_primary,
            "all_ten_exact": success and all(left == right for left, right in pairs),
            "model_success": success,
            "selection_made": guarded_primary is not None and guarded_backup is not None,
            "unsafe_primary": (
                guarded_primary is not None
                and true_by_id[int(guarded_primary)] == "temporary_hold"
            ),
            "unsafe_or_missing_primary": (
                guarded_primary is None
                or guarded_backup is None
                or true_by_id[int(guarded_primary)] == "temporary_hold"
            ),
            "unsafe_introduction": (
                guarded_primary is not None
                and true_by_id[baseline_primary] != "temporary_hold"
                and int(guarded_primary) != baseline_primary
                and true_by_id[int(guarded_primary)] == "temporary_hold"
            ),
        })
    return rows


def _metric(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    counts = _merge_classification_counts([row["counts"] for row in rows])
    metrics = _metrics_from_counts(counts, sum(counts[state]["support"] for state in STATES))
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "temporary_hold_recall": float(metrics["by_class"]["temporary_hold"]["recall"]),
        "guarded_top1_conformance": sum(bool(row["top1_correct"]) for row in rows) / len(rows),
        "all_ten_states_exact": sum(bool(row["all_ten_exact"]) for row in rows) / len(rows),
        "model_success": sum(bool(row["model_success"]) for row in rows) / len(rows),
        "selection_rate": sum(bool(row["selection_made"]) for row in rows) / len(rows),
        "unsafe_or_missing_primary_rate": (
            sum(bool(row["unsafe_or_missing_primary"]) for row in rows) / len(rows)
        ),
        "unsafe_introduction_rate": (
            sum(bool(row["unsafe_introduction"]) for row in rows) / len(rows)
        ),
    }


def compare(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
    *,
    write_output: bool = True,
) -> Dict[str, Any]:
    left_path = left_path.resolve()
    right_path = right_path.resolve()
    if left_path == right_path:
        raise RuntimeError("Paired comparison requires two different raw evaluation files")
    left_config = _load_run_config(left_path)
    right_config = _load_run_config(right_path)
    if left_config["test_lock"] != right_config["test_lock"]:
        raise RuntimeError("Model runs were not produced under the same frozen test lock")
    if left_config.get("condition") == right_config.get("condition"):
        raise RuntimeError("Paired comparison requires two different conditions")
    if left_label != str(left_config.get("condition")):
        raise RuntimeError("The left label must match the left run's condition")
    if right_label != str(right_config.get("condition")):
        raise RuntimeError("The right label must match the right run's condition")
    cases = _load_jsonl(TEST_PATH)
    left = _condition_rows(left_path, cases)
    right = _condition_rows(right_path, cases)
    left_metrics = _metric(left)
    right_metrics = _metric(right)

    observed_delta = {
        key: round(right_metrics[key] - left_metrics[key], 6)
        for key in left_metrics
    }
    rng = random.Random(BOOTSTRAP_SEED + 500)
    samples = {key: [] for key in left_metrics}
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(len(cases)) for _ in cases]
        left_sample = [left[index] for index in indices]
        right_sample = [right[index] for index in indices]
        left_stat = _metric(left_sample)
        right_stat = _metric(right_sample)
        for key in samples:
            samples[key].append(right_stat[key] - left_stat[key])
    delta_intervals = {
        key: [round(_percentile(values, 0.025), 6), round(_percentile(values, 0.975), 6)]
        for key, values in samples.items()
    }

    summary = {
        "generated_at_utc": _utc_now(),
        "comparison_script_sha256": _sha256(COMPARISON_PATH),
        "test_file": portable_path(TEST_PATH),
        "test_file_sha256": _sha256(TEST_PATH),
        "case_count": len(cases),
        "left": {
            "label": left_label,
            "path": portable_path(left_path),
            "sha256": _sha256(left_path),
            "metrics": {key: round(value, 6) for key, value in left_metrics.items()},
            "config": left_config,
        },
        "right": {
            "label": right_label,
            "path": portable_path(right_path),
            "sha256": _sha256(right_path),
            "metrics": {key: round(value, 6) for key, value in right_metrics.items()},
            "config": right_config,
        },
        "delta_right_minus_left": observed_delta,
        "delta_95ci_cluster_bootstrap": delta_intervals,
        "paired_exact_mcnemar": {
            "guarded_top1_conformance": _mcnemar(
                [bool(row["top1_correct"]) for row in left],
                [bool(row["top1_correct"]) for row in right],
            ),
            "all_ten_states_exact": _mcnemar(
                [bool(row["all_ten_exact"]) for row in left],
                [bool(row["all_ten_exact"]) for row in right],
            ),
            "unsafe_or_missing_primary": _mcnemar(
                [bool(row["unsafe_or_missing_primary"]) for row in left],
                [bool(row["unsafe_or_missing_primary"]) for row in right],
            ),
        },
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED + 500},
    }
    if write_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{left_label}_vs_{right_label}_comparison.json"
        if output_path.exists():
            raise RuntimeError(f"Comparison output already exists and will not be replaced: {output_path}")
        output_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="untuned")
    parser.add_argument("--right-label", default="fine_tuned")
    args = parser.parse_args()
    summary = compare(args.left, args.right, args.left_label, args.right_label)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
