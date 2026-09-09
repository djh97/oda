from __future__ import annotations

import json
import sys

from evaluation.synthetic_dataset import DATASET_DIR, validate_existing


def main() -> None:
    report = validate_existing()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        sys.exit(1)
    print(f"Validated synthetic dataset in {DATASET_DIR}")


if __name__ == "__main__":
    main()
