from __future__ import annotations

import json
from pathlib import Path

from evaluation.synthetic_dataset import DATASET_DIR, generate_all
from src.policy import assert_protocol_frozen, assert_test_seed_not_retired


APP_DIR = Path(__file__).resolve().parents[1]
FINE_TUNING_STATE_PATH = APP_DIR / "pipeline-output" / "current" / "model" / "fine_tuning_job.json"
TEST_LOCK_PATH = APP_DIR / "pipeline-output" / "current" / "evaluation" / "test_lock.json"


def assert_generation_is_not_frozen() -> None:
    blockers = [path for path in (FINE_TUNING_STATE_PATH, TEST_LOCK_PATH) if path.exists()]
    if blockers:
        names = ", ".join(
            str(path.relative_to(APP_DIR)) if path.is_relative_to(APP_DIR) else str(path)
            for path in blockers
        )
        raise RuntimeError(
            "Dataset generation is frozen because a model or locked-test run has started. "
            f"Preserve this protocol version and create a new version instead. Found: {names}"
        )
    assert_test_seed_not_retired()
    assert_protocol_frozen()


def main() -> None:
    assert_generation_is_not_frozen()
    result = generate_all()
    print(f"Generated validated artifacts in {DATASET_DIR}")
    print(json.dumps(result["validation"]["split_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
