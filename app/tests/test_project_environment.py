from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import evaluation.check_environment as environment_check
from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.project_environment import (
    ProjectEnvironmentError,
    load_authoritative_project_env,
)


def test_authoritative_loader_rejects_unsupported_names(tmp_path: Path) -> None:
    path = tmp_path / "study.env"
    path.write_text("OPENAI_API_KEY=test\nPATH=do-not-replace\n", encoding="utf-8")
    with patch.dict(os.environ, {"ODA_DISABLE_DOTENV": "0"}, clear=False):
        with pytest.raises(ProjectEnvironmentError, match="PATH"):
            load_authoritative_project_env(path)

    source_path = Path(__file__).resolve()
    recorded = portable_path(source_path)
    assert not Path(recorded).is_absolute()
    assert resolve_recorded_path(recorded) == source_path
    confined_root = tmp_path / "confined"
    confined_root.mkdir()
    with pytest.raises(ValueError, match="permitted directory"):
        resolve_recorded_path(
            "../outside.json",
            root=confined_root,
            permitted_root=confined_root,
        )


def test_environment_checker_validates_optional_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "study.env"
    path.write_text(
        "TX_RECEIPT_TIMEOUT_S=nan\n"
        "UNSUPPORTED_VALUE=test\n",
        encoding="utf-8",
    )
    with patch.object(environment_check, "ENV_PATH", path):
        _present, issues = environment_check.validate("fine-tune")
    assert "TX_RECEIPT_TIMEOUT_S must be a finite positive number" in issues
    assert "Unsupported environment variable names: UNSUPPORTED_VALUE" in issues
    assert environment_check._is_placeholder("ft:gpt-4o-mini-...")

    path.write_text(
        "PINATA_GATEWAY=https://example.invalid/ipfs/?mode=test\n",
        encoding="utf-8",
    )
    with patch.object(environment_check, "ENV_PATH", path):
        _present, issues = environment_check.validate("sepolia")
    assert any("without embedded credentials" in issue for issue in issues)
