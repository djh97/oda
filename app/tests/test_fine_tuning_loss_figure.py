from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from evaluation import build_evidence_manifest as evidence
from evaluation import build_fine_tuning_loss_figure as loss_figure


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "step,train_loss,valid_loss\n"
        "1,1.4,1.5\n"
        "2,1.0,\n"
        "3,0.7,0.8\n",
        encoding="utf-8",
    )


def _archive_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "file_id": "file-result",
        "local_path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_load_loss_series_accepts_sparse_validation_and_renders_legible_chart(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics(path)

    series = loss_figure.load_loss_series(path)

    assert series["training"] == [[1, 1.4], [2, 1.0], [3, 0.7]]
    assert series["validation"] == [[1, 1.5], [3, 0.8]]
    rendered = loss_figure.render_html(series)
    assert f'width="{loss_figure.WIDTH_PX}"' in rendered
    assert ">Loss</text>" in rendered
    assert "Training" in rendered and "Validation" in rendered
    assert "Update step" in rendered
    assert "font-size: 36px" in rendered


def test_render_aligns_epoch_summaries_at_their_training_steps() -> None:
    series = {
        "columns": {
            "step": "step",
            "training_loss": "loss",
            "validation_loss": "eval_loss",
        },
        "training": [[10, 0.8], [20, 0.5], [30, 0.4], [40, 0.3]],
        "validation": [[20, 0.6], [40, 0.4]],
        "steps_per_epoch": 20.0,
    }

    rendered = loss_figure.render_html(series)

    assert ">Training step</text>" in rendered
    assert 'aria-label="Training and validation loss by optimizer step"' in rendered
    assert loss_figure._epoch_summary(series) == {
        "training": [[1.0, 0.65], [2.0, 0.35]],
        "validation": [[1.0, 0.6], [2.0, 0.4]],
    }
    assert loss_figure._axis_metadata(series) == {
        "unit": "optimizer_step",
        "steps_per_epoch": 20.0,
        "training_aggregation": "step_weighted_mean_of_logged_intervals",
        "minimum": 20.0,
        "maximum": 40.0,
        "training_points": 2,
        "validation_points": 2,
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("step,train_loss\n1,0.5\n2,0.4\n", "validation-loss"),
        (
            "step,train_loss,valid_loss\n1,0.5,0.6\n1,0.4,0.6\n2,0.3,0.4\n",
            "conflicting training loss",
        ),
        ("step,train_loss,valid_loss\n1,nan,0.6\n2,0.4,0.5\n", "invalid train_loss"),
    ],
)
def test_load_loss_series_rejects_incomplete_or_invalid_metrics(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        loss_figure.load_loss_series(path)


def test_build_records_hashed_inputs_and_exact_output_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    metrics = model_dir / "result_files" / "metrics.csv"
    _write_metrics(metrics)
    state_path = model_dir / "fine_tuning_job.json"
    state = {"archived_result_files": [_archive_record(metrics)]}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    script_path = tmp_path / "renderer.py"
    script_path.write_text("# fixture\n", encoding="utf-8")
    output_path = tmp_path / "fine_tuning_loss.png"
    manifest_path = model_dir / "fine_tuning_loss_manifest.json"

    monkeypatch.setattr(loss_figure, "MODEL_DIR", model_dir)
    monkeypatch.setattr(loss_figure, "STATE_PATH", state_path)
    monkeypatch.setattr(loss_figure, "DEFAULT_PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(loss_figure, "SCRIPT_PATH", script_path)
    monkeypatch.setattr(loss_figure, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(loss_figure, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(loss_figure, "_validate_fine_tuning_state", lambda value: "ft:test")
    monkeypatch.setattr(loss_figure, "_find_edge", lambda value: Path("edge.exe"))

    def fake_capture(_browser: Path, _html: Path, output: Path) -> str:
        output.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\x0dIHDR"
            + struct.pack(">II", loss_figure.WIDTH_PX, loss_figure.HEIGHT_PX)
        )
        return "Microsoft Edge test"

    monkeypatch.setattr(loss_figure, "_capture_png", fake_capture)

    assert loss_figure.build() == output_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["env_file_read"] is False
    assert manifest["network_contacted"] is False
    assert manifest["observations"] == {
        "maximum_step": 3,
        "minimum_step": 1,
        "training": 3,
        "validation": 2,
    }
    assert manifest["axis"] == {
        "unit": "update_step",
        "minimum": 1.0,
        "maximum": 3.0,
    }
    assert manifest["output"]["dimensions_px"] == [
        loss_figure.WIDTH_PX,
        loss_figure.HEIGHT_PX,
    ]
    assert manifest["output"]["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()

    monkeypatch.setattr(evidence, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(evidence, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(evidence, "LOSS_FIGURE_FILE", output_path)
    monkeypatch.setattr(evidence, "LOSS_FIGURE_MANIFEST", manifest_path)
    evidence._validate_loss_figure_manifest()

    output_path.write_bytes(output_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="output record is stale"):
        evidence._validate_loss_figure_manifest()


def test_local_trainer_history_is_the_verified_loss_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    state_path = model_dir / "local_lora_training.json"
    state = {
        "training_result": {
            "log_history": [
                {"step": 10, "loss": 1.4},
                {"step": 20, "loss": 1.0},
                {"step": 20, "eval_loss": 0.9},
                {"step": 30, "loss": 0.7},
                {"step": 40, "eval_loss": 0.6},
            ]
        }
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    script_path = tmp_path / "renderer.py"
    script_path.write_text("# fixture\n", encoding="utf-8")
    output_path = tmp_path / "fine_tuning_loss.png"
    manifest_path = model_dir / "fine_tuning_loss_manifest.json"

    monkeypatch.setattr(loss_figure, "MODEL_DIR", model_dir)
    monkeypatch.setattr(loss_figure, "STATE_PATH", state_path)
    monkeypatch.setattr(loss_figure, "DEFAULT_PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(loss_figure, "SCRIPT_PATH", script_path)
    monkeypatch.setattr(loss_figure, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(loss_figure, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(loss_figure, "_validate_fine_tuning_state", lambda value: "local:test")
    monkeypatch.setattr(loss_figure, "_find_edge", lambda value: Path("edge.exe"))

    def fake_capture(_browser: Path, _html: Path, output: Path) -> str:
        output.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\x0dIHDR"
            + struct.pack(">II", loss_figure.WIDTH_PX, loss_figure.HEIGHT_PX)
        )
        return "Microsoft Edge test"

    monkeypatch.setattr(loss_figure, "_capture_png", fake_capture)

    assert loss_figure.build() == output_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["selected_metrics_file"]).resolve() == state_path.resolve()
    assert manifest["observations"] == {
        "maximum_step": 40,
        "minimum_step": 10,
        "training": 3,
        "validation": 2,
    }

    monkeypatch.setattr(evidence, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(evidence, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(evidence, "LOSS_FIGURE_FILE", output_path)
    monkeypatch.setattr(evidence, "LOSS_FIGURE_MANIFEST", manifest_path)
    evidence._validate_loss_figure_manifest()
