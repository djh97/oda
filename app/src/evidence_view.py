"""Read the latest canonical UI snapshot with path and hash validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .schemas import MatchResponse


APP_DIR = Path(__file__).resolve().parents[1]
CURRENT_OUTPUT_DIR = APP_DIR / "pipeline-output" / "current"


class EvidenceViewError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise EvidenceViewError(f"Missing canonical evidence file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceViewError(f"Unable to read canonical evidence file: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvidenceViewError(f"Canonical evidence file is not a JSON object: {path.name}")
    return value


def load_latest_evidence(
    *,
    app_dir: Path = APP_DIR,
    current_output_dir: Path = CURRENT_OUTPUT_DIR,
) -> MatchResponse:
    pointer = _read_object(current_output_dir / "latest_full_workflow.json")
    workflow_root = (current_output_dir / "full_workflow").resolve()
    run_dir = (app_dir / str(pointer.get("run_directory", ""))).resolve()
    try:
        run_dir.relative_to(workflow_root)
    except ValueError as exc:
        raise EvidenceViewError("Canonical evidence pointer leaves the workflow directory") from exc

    summary_path = run_dir / "run_summary.json"
    snapshot_path = run_dir / "ui_evidence_snapshot.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    completion_path = run_dir / "completion.json"
    _read_object(summary_path)
    snapshot = _read_object(snapshot_path)
    _read_object(artifact_manifest_path)
    completion = _read_object(completion_path)
    expected_summary_hash = str(pointer.get("run_summary_sha256", "")).strip()
    expected_snapshot_hash = str(pointer.get("ui_evidence_snapshot_sha256", "")).strip()
    if not expected_summary_hash or _sha256(summary_path) != expected_summary_hash:
        raise EvidenceViewError("Canonical run-summary hash verification failed")
    if not expected_snapshot_hash or _sha256(snapshot_path) != expected_snapshot_hash:
        raise EvidenceViewError("Canonical UI-snapshot hash verification failed")
    if (
        completion.get("schema_version") != "1.0"
        or completion.get("status") != "completed"
        or completion.get("mode") != "full"
        or completion.get("run_summary_sha256") != _sha256(summary_path)
        or completion.get("artifact_manifest_sha256") != _sha256(artifact_manifest_path)
    ):
        raise EvidenceViewError("Canonical workflow completion verification failed")
    try:
        return MatchResponse.model_validate(snapshot)
    except Exception as exc:
        raise EvidenceViewError("Canonical UI snapshot failed schema validation") from exc
