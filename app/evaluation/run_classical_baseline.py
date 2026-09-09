"""Train and evaluate the prespecified TF-IDF logistic text baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

from evaluation.artifact_paths import portable_path
import evaluation.run_model_evaluation as lock_manager
from evaluation.synthetic_dataset import (
    DATASET_DIR,
    PROTOCOL,
    STATE_CODES,
    validate_cases,
    validate_existing,
    validate_generation_manifest,
)
from src.decision_guard import apply_guarded_policy
from src.policy import rank_recipients_baseline
from src.schemas import ModelRunMetadata, NoteReviewBatch


APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "evaluation"
TRAIN_PATH = DATASET_DIR / "training_cases.jsonl"
VALIDATION_PATH = DATASET_DIR / "validation_cases.jsonl"
TEST_PATH = DATASET_DIR / "test_cases.jsonl"
SCRIPT_PATH = Path(__file__).resolve()
SETTINGS = PROTOCOL["model_evaluation"]["classical_text_baseline"]
STATES = tuple(PROTOCOL["note_taxonomy"]["states"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _training_rows(
    cases: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[tuple[str, ...]]]:
    notes: list[str] = []
    states: list[str] = []
    evidence: list[tuple[str, ...]] = []
    for case in cases:
        for recipient in case["recipients"]:
            notes.append(str(recipient["medical_notes"]))
            states.append(str(recipient["reference_note_state"]))
            evidence.append(tuple(str(code) for code in recipient["reference_evidence_codes"]))
    return notes, states, evidence


def _logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        C=float(SETTINGS["regularization_c"]),
        class_weight=str(SETTINGS["class_weight"]),
        max_iter=int(SETTINGS["maximum_iterations"]),
        random_state=int(SETTINGS["random_seed"]),
        solver="lbfgs",
    )


def _feature_union() -> FeatureUnion:
    minimum_frequency = int(SETTINGS["min_document_frequency"])
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(int(value) for value in SETTINGS["word_ngram_range"]),
                    min_df=minimum_frequency,
                    max_features=int(SETTINGS["word_max_features"]),
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(int(value) for value in SETTINGS["character_ngram_range"]),
                    min_df=minimum_frequency,
                    max_features=int(SETTINGS["character_max_features"]),
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ],
        n_jobs=1,
    )


def _predict_evidence_sets(
    predicted_states: Sequence[str],
    probabilities: Any,
    classes: Sequence[str],
) -> list[list[str]]:
    threshold = float(SETTINGS["evidence_probability_threshold"])
    class_names = [str(value) for value in classes]
    predictions: list[list[str]] = []
    for state, row in zip(predicted_states, probabilities):
        allowed = set(STATE_CODES[str(state)])
        allowed_indices = [index for index, code in enumerate(class_names) if code in allowed]
        selected = [
            class_names[index]
            for index in allowed_indices
            if float(row[index]) >= threshold
        ]
        if not selected:
            best_index = max(allowed_indices, key=lambda index: float(row[index]))
            selected = [class_names[best_index]]
        predictions.append(selected)
    return predictions


def _validation_metrics(
    true_states: Sequence[str],
    predicted_states: Sequence[str],
    true_evidence: Sequence[Sequence[str]],
    predicted_evidence: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_states,
        predicted_states,
        labels=list(STATES),
        zero_division=0,
    )
    return {
        "candidate_count": len(true_states),
        "state_accuracy": round(float(accuracy_score(true_states, predicted_states)), 6),
        "state_macro_f1": round(
            float(f1_score(true_states, predicted_states, labels=list(STATES), average="macro")),
            6,
        ),
        "state_by_class": {
            state: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, state in enumerate(STATES)
        },
        "evidence_exact_match": round(
            sum(set(left) == set(right) for left, right in zip(true_evidence, predicted_evidence))
            / len(true_evidence),
            6,
        ),
    }


def _fit_and_validate() -> Dict[str, Any]:
    training_cases = _load_jsonl(TRAIN_PATH)
    validation_cases = _load_jsonl(VALIDATION_PATH)
    validation_report = validate_cases(
        {"training": training_cases, "validation": validation_cases}
    )
    manifest_errors = validate_generation_manifest(DATASET_DIR)
    if not validation_report["valid"] or manifest_errors:
        raise RuntimeError("Synthetic training or validation data failed validation")
    train_notes, train_states, train_evidence = _training_rows(training_cases)
    validation_notes, validation_states, validation_evidence = _training_rows(validation_cases)

    features = _feature_union()
    training_started = time.perf_counter()
    train_matrix = features.fit_transform(train_notes)
    state_model = _logistic_regression()
    state_model.fit(train_matrix, train_states)
    evidence_binarizer = MultiLabelBinarizer(
        classes=sorted(code for values in STATE_CODES.values() for code in values)
    )
    train_evidence_matrix = evidence_binarizer.fit_transform(train_evidence)
    evidence_model = OneVsRestClassifier(_logistic_regression(), n_jobs=1)
    evidence_model.fit(train_matrix, train_evidence_matrix)
    training_seconds = time.perf_counter() - training_started

    validation_matrix = features.transform(validation_notes)
    validation_state_predictions = [str(value) for value in state_model.predict(validation_matrix)]
    validation_evidence_predictions = _predict_evidence_sets(
        validation_state_predictions,
        evidence_model.predict_proba(validation_matrix),
        evidence_binarizer.classes_,
    )
    diagnostics = _validation_metrics(
        validation_states,
        validation_state_predictions,
        validation_evidence,
        validation_evidence_predictions,
    )
    return {
        "features": features,
        "state_model": state_model,
        "evidence_model": evidence_model,
        "evidence_binarizer": evidence_binarizer,
        "train_matrix": train_matrix,
        "train_candidate_count": len(train_notes),
        "validation_candidate_count": len(validation_notes),
        "training_seconds": training_seconds,
        "diagnostics": diagnostics,
    }


def write_validation_diagnostics() -> Path:
    fitted = _fit_and_validate()
    output_dir = APP_DIR / "pipeline-output" / "current" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classical_validation_diagnostics.json"
    value = {
        "purpose": "Training-fit and validation-split diagnostics; no test predictions or metrics were generated",
        "generated_at_utc": _utc_now(),
        "protocol_id": PROTOCOL["protocol_id"],
        "script_sha256": _sha256(SCRIPT_PATH),
        "training_file_sha256": _sha256(TRAIN_PATH),
        "validation_file_sha256": _sha256(VALIDATION_PATH),
        "training_candidate_count": fitted["train_candidate_count"],
        "validation_candidate_count": fitted["validation_candidate_count"],
        "feature_count": int(fitted["train_matrix"].shape[1]),
        "training_seconds": round(float(fitted["training_seconds"]), 3),
        "validation_diagnostics": fitted["diagnostics"],
        "hyperparameters": SETTINGS,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "numpy", "scipy", "joblib")
        },
    }
    output_path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run(*, confirm_test_lock: bool = False) -> Path:
    output_path = OUTPUT_DIR / "tfidf_logistic_raw.jsonl"
    config_path = OUTPUT_DIR / "tfidf_logistic_config.json"
    attempt_path = OUTPUT_DIR / "tfidf_logistic_attempts.jsonl"
    if (output_path.exists() or attempt_path.exists()) and not config_path.exists():
        raise RuntimeError("Classical-baseline output exists without its configuration file")

    fitted = _fit_and_validate()
    features = fitted["features"]
    state_model = fitted["state_model"]
    evidence_model = fitted["evidence_model"]
    evidence_binarizer = fitted["evidence_binarizer"]
    train_matrix = fitted["train_matrix"]
    training_seconds = float(fitted["training_seconds"])
    diagnostics = fitted["diagnostics"]

    existing_lock = lock_manager._enforce_test_lock(create=False)
    if existing_lock is None and not confirm_test_lock:
        raise RuntimeError(
            "The first held-out evaluation requires explicit authorization. Review the frozen "
            "protocol, then rerun with --confirm-test-lock."
        )
    test_lock = existing_lock or lock_manager._enforce_test_lock(create=True)
    if test_lock is None:
        raise RuntimeError("Unable to create or read the frozen test lock")
    full_validation = validate_existing()
    if not full_validation["valid"]:
        raise RuntimeError("The locked synthetic dataset failed validation")
    test_cases = _load_jsonl(TEST_PATH)
    expected_case_order = [str(case["case_id"]) for case in test_cases]
    model_id = str(SETTINGS["model_id"])
    packages = {
        name: importlib.metadata.version(name)
        for name in ("scikit-learn", "numpy", "scipy", "joblib")
    }
    fixed_config = {
        "condition": "tfidf_logistic",
        "model_id_requested": model_id,
        "protocol_id": PROTOCOL["protocol_id"],
        "script_sha256": _sha256(SCRIPT_PATH),
        "training_file_sha256": _sha256(TRAIN_PATH),
        "validation_file_sha256": _sha256(VALIDATION_PATH),
        "test_file_sha256": _sha256(TEST_PATH),
        "test_case_count": len(test_cases),
        "training_candidate_count": fitted["train_candidate_count"],
        "validation_candidate_count": fitted["validation_candidate_count"],
        "feature_count": int(train_matrix.shape[1]),
        "validation_diagnostics": diagnostics,
        "hyperparameters": SETTINGS,
        "packages": packages,
        "attempt_log_file": portable_path(attempt_path),
        "test_lock": test_lock,
    }
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        for field, expected in fixed_config.items():
            if existing_config.get(field) != expected:
                raise RuntimeError(
                    f"Existing classical-baseline output has a different {field.replace('_', ' ')}"
                )
        if existing_config.get("evaluation_in_progress") is not True:
            if (
                existing_config.get("raw_file_sha256") != _sha256(output_path)
                or existing_config.get("attempt_log_sha256") != _sha256(attempt_path)
                or int(existing_config.get("raw_record_count", -1)) != len(test_cases)
            ):
                raise RuntimeError("Completed classical-baseline evidence is malformed or stale")
            return output_path
        config = existing_config
    else:
        config = {
            **fixed_config,
            "created_at_utc": _utc_now(),
            "training_seconds": round(training_seconds, 3),
            "evaluation_invocations": [],
        }

    config.setdefault("evaluation_invocations", []).append({"invoked_at_utc": _utc_now()})
    config["evaluation_in_progress"] = True
    lock_manager._write_json(config_path, config)
    completed = lock_manager._existing_case_ids(
        output_path,
        expected_order=expected_case_order,
    )
    attempted, attempt_rows = lock_manager._attempted_case_ids(
        attempt_path,
        condition="tfidf_logistic",
        model_id=model_id,
        expected_order=expected_case_order,
    )
    existing_records = _load_jsonl(output_path) if output_path.exists() else []
    lock_manager.validate_outcome_attempt_provenance(
        existing_records,
        attempt_rows,
        require_complete=False,
    )
    if not completed.issubset(attempted):
        raise RuntimeError("Classical-baseline outcome has no corresponding first attempt")

    case_by_id = {str(case["case_id"]): case for case in test_cases}
    case_order = {case_id: index for index, case_id in enumerate(expected_case_order)}
    for case_id in sorted(attempted.difference(completed), key=case_order.__getitem__):
        case = case_by_id[case_id]
        lock_manager._append_jsonl(
            output_path,
            {
                "case_id": case_id,
                "condition": "tfidf_logistic",
                "model_id_requested": model_id,
                "organ_type": str(case["organ_type"]),
                "scenario_family": str(case["scenario_family"]),
                "status": "failure",
                "attempt_started_at_utc": attempt_rows[case_id]["started_at_utc"],
                "completed_at_utc": _utc_now(),
                "error_type": "InterruptedAttempt",
                "error": (
                    "The first classical-baseline attempt ended before its outcome was durably "
                    "recorded; the case was retained as a failure and was not retried."
                ),
            },
        )
        completed.add(case_id)

    for case in test_cases:
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        attempt_started_at = _utc_now()
        lock_manager._append_jsonl(
            attempt_path,
            {
                "event": "attempt_started",
                "case_id": case_id,
                "condition": "tfidf_logistic",
                "model_id_requested": model_id,
                "started_at_utc": attempt_started_at,
            },
        )
        attempted.add(case_id)
        inference_started = time.perf_counter()
        try:
            notes = [str(recipient["medical_notes"]) for recipient in case["recipients"]]
            matrix = features.transform(notes)
            predicted_states = [str(value) for value in state_model.predict(matrix)]
            predicted_evidence = _predict_evidence_sets(
                predicted_states,
                evidence_model.predict_proba(matrix),
                evidence_binarizer.classes_,
            )
            latency_ms = round((time.perf_counter() - inference_started) * 1000.0, 3)
            review = NoteReviewBatch.model_validate({
                "case_id": case_id,
                "assessments": [
                    {
                        "recipient_id": int(recipient["recipient_id"]),
                        "state": state,
                        "evidence_codes": codes,
                    }
                    for recipient, state, codes in zip(
                        case["recipients"], predicted_states, predicted_evidence
                    )
                ],
            })
            ranked = rank_recipients_baseline(case["donor"], case["recipients"], PROTOCOL)
            guarded = apply_guarded_policy(
                int(case["donor"]["donor_id"]),
                ranked,
                review,
                [int(recipient["recipient_id"]) for recipient in case["recipients"]],
            )
            metadata = ModelRunMetadata(model_id=model_id, latency_ms=latency_ms)
            record = {
                "case_id": case_id,
                "condition": "tfidf_logistic",
                "model_id_requested": model_id,
                "organ_type": str(case["organ_type"]),
                "scenario_family": str(case["scenario_family"]),
                "status": "success",
                "attempt_started_at_utc": attempt_started_at,
                "completed_at_utc": _utc_now(),
                "review": review.model_dump(mode="json"),
                "guarded_decision": guarded.model_dump(mode="json"),
                "model_run": metadata.model_dump(mode="json"),
                "raw_model_output": review.model_dump_json(),
            }
        except Exception as exc:
            record = {
                "case_id": case_id,
                "condition": "tfidf_logistic",
                "model_id_requested": model_id,
                "organ_type": str(case["organ_type"]),
                "scenario_family": str(case["scenario_family"]),
                "status": "failure",
                "attempt_started_at_utc": attempt_started_at,
                "completed_at_utc": _utc_now(),
                "latency_ms": round((time.perf_counter() - inference_started) * 1000.0, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        lock_manager._append_jsonl(output_path, record)

    records = _load_jsonl(output_path)
    config.update({
        "completed_at_utc": _utc_now(),
        "evaluation_in_progress": False,
        "raw_record_count": len(records),
        "successful_record_count": sum(record["status"] == "success" for record in records),
        "failed_record_count": sum(record["status"] == "failure" for record in records),
        "raw_file_sha256": _sha256(output_path),
        "attempted_case_count": len(attempted),
        "attempt_log_sha256": _sha256(attempt_path),
    })
    lock_manager._write_json(config_path, config)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Fit on training data and save validation diagnostics without opening the test split.",
    )
    parser.add_argument(
        "--confirm-test-lock",
        action="store_true",
        help="Authorize creation of the irreversible held-out test lock if none exists.",
    )
    args = parser.parse_args()
    if args.validation_only:
        output_path = write_validation_diagnostics()
        print(f"Saved validation-only classical diagnostics to {output_path}")
    else:
        output_path = run(confirm_test_lock=args.confirm_test_lock)
        print(f"Saved classical-baseline records to {output_path}")


if __name__ == "__main__":
    main()
