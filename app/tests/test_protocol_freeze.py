from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.freeze_study_protocol import (
    EXPECTED_ACTIVE_SEED,
    OLD_STUDY_PROTOCOL_PARAGRAPH,
    freeze_protocol,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    protocol = root / "protocol.json"
    study = root / "STUDY_PROTOCOL.md"
    manuscript = root / "Manuscript.tex"
    freeze = root / "output" / "protocol_freeze.json"
    model = root / "output" / "fine_tuning_job.json"
    lock = root / "output" / "test_lock.json"
    _write(
        protocol,
        json.dumps(
            {
                "protocol_id": "ODA-SYNTH-MULTIORGAN-1.0",
                "version": "1.0.0",
                "dataset": {
                    "splits": {
                        "training": {"seed": 2026090701},
                        "validation": {"seed": 2026090702},
                        "test": {"seed": EXPECTED_ACTIVE_SEED},
                    }
                },
                "model_evaluation": {"seed": 20260907},
                "pre_freeze_history": [
                    {
                        "date": "2026-09-07",
                        "retired_test_seed": 2026090703,
                        "reason": "Inspected during development.",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _write(study, "| Test | 400 | 100 | Pending author-approved replacement | One final locked evaluation |\n\n" + OLD_STUDY_PROTOCOL_PARAGRAPH)
    _write(manuscript, f"seed {EXPECTED_ACTIVE_SEED}\nrow {EXPECTED_ACTIVE_SEED}\n")
    return protocol, study, manuscript, freeze, model, lock


def test_freeze_replaces_seed_and_writes_hash_bound_record(tmp_path: Path) -> None:
    protocol, study, manuscript, freeze, model, lock = _inputs(tmp_path)
    replacement = 1_234_567_891
    record = freeze_protocol(
        replacement_seed=replacement,
        protocol_path=protocol,
        study_protocol_path=study,
        manuscript_path=manuscript,
        freeze_path=freeze,
        fine_tuning_state_path=model,
        test_lock_path=lock,
    )

    value = json.loads(protocol.read_text(encoding="utf-8"))
    assert value["dataset"]["splits"]["test"]["seed"] == replacement
    assert {item["retired_test_seed"] for item in value["pre_freeze_history"]} == {
        2026090703,
        EXPECTED_ACTIVE_SEED,
    }
    assert str(replacement) in study.read_text(encoding="utf-8")
    assert manuscript.read_text(encoding="utf-8").count(str(replacement)) == 2
    assert str(replacement) not in json.dumps(record)
    assert record["active_test_seed_sha256"] == hashlib.sha256(
        str(replacement).encode("ascii")
    ).hexdigest()
    assert record["source_hashes"]["protocol_json_sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()


def test_freeze_refuses_reuse_or_post_start_replacement(tmp_path: Path) -> None:
    protocol, study, manuscript, freeze, model, lock = _inputs(tmp_path)
    with pytest.raises(ValueError, match="retired or already used"):
        freeze_protocol(
            replacement_seed=2026090703,
            protocol_path=protocol,
            study_protocol_path=study,
            manuscript_path=manuscript,
            freeze_path=freeze,
            fine_tuning_state_path=model,
            test_lock_path=lock,
        )

    _write(model, "{}\n")
    with pytest.raises(RuntimeError, match="after fine-tuning or test locking"):
        freeze_protocol(
            replacement_seed=1_234_567_891,
            protocol_path=protocol,
            study_protocol_path=study,
            manuscript_path=manuscript,
            freeze_path=freeze,
            fine_tuning_state_path=model,
            test_lock_path=lock,
        )
