"""Recalculate numerical evaluation results from retained raw outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
APP_DIR = REPOSITORY_DIR / "app"
EVALUATION_DIR = Path(__file__).resolve().parent / "evidence" / "evaluation"
sys.path.insert(0, str(APP_DIR))

from evaluation.analyze_evaluation import analyze  # noqa: E402
from evaluation import analyze_output_reliability as reliability  # noqa: E402


CONDITIONS = ("tfidf_logistic", "untuned", "fine_tuned", "openai")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def normalized_summary(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for field in ("generated_at_utc", "records_file", "config_file", "test_file"):
        result.pop(field, None)
    return result


def normalized_reliability(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("generated_at_utc", None)
    inputs = {}
    for name, record in result["input_files"].items():
        normalized = dict(record)
        normalized.pop("records_path", None)
        normalized.pop("config_path", None)
        normalized.pop("path", None)
        inputs[name] = normalized
    result["input_files"] = inputs
    return result


def differing_fields(left: dict[str, Any], right: dict[str, Any]) -> str:
    fields = sorted(
        key for key in set(left).union(right) if left.get(key) != right.get(key)
    )
    return ", ".join(fields) or "unknown"


def main() -> None:
    for condition in CONDITIONS:
        raw_path = EVALUATION_DIR / f"{condition}_raw.jsonl"
        retained_path = EVALUATION_DIR / f"{condition}_summary.json"
        recalculated = analyze(
            raw_path,
            allow_incomplete=True,
            write_outputs=False,
        )
        retained = load_json(retained_path)
        normalized_recalculated = normalized_summary(recalculated)
        normalized_retained = normalized_summary(retained)
        if normalized_recalculated != normalized_retained:
            raise SystemExit(
                f"Summary mismatch for {condition}: "
                f"{differing_fields(normalized_recalculated, normalized_retained)}"
            )
        print(f"Verified summary: {condition}")

    reliability.EVALUATION_DIR = EVALUATION_DIR
    recalculated_reliability = reliability.analyze()
    retained_reliability = load_json(EVALUATION_DIR / "posthoc_output_reliability.json")
    normalized_recalculated = normalized_reliability(recalculated_reliability)
    normalized_retained = normalized_reliability(retained_reliability)
    if normalized_recalculated != normalized_retained:
        raise SystemExit(
            "Output-reliability mismatch: "
            f"{differing_fields(normalized_recalculated, normalized_retained)}"
        )
    print("Verified post hoc output-reliability analysis")

    print("All retained model summaries and reliability results were reproduced.")


if __name__ == "__main__":
    main()
