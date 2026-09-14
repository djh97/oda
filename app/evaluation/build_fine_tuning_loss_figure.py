"""Render a provenance-bound fine-tuning loss figure from the local Trainer log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.artifact_paths import portable_path, resolve_recorded_path
from evaluation.build_manuscript_tables import _validate_fine_tuning_state
from evaluation.publication_workspace import FIGURE_OUTPUT_DIR
from evaluation.train_local_lora import MODEL_OUTPUT_DIR as MODEL_DIR
from evaluation.train_local_lora import STATE_PATH
from src.policy import DEFAULT_PROTOCOL_PATH


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_PATH = FIGURE_OUTPUT_DIR / "fine_tuning_loss.png"
MANIFEST_PATH = MODEL_DIR / "fine_tuning_loss_manifest.json"
WIDTH_PX = 2126
HEIGHT_PX = 1200

TRAIN_LOSS_FIELDS = ("train_loss", "training_loss", "full_train_loss")
VALIDATION_LOSS_FIELDS = ("valid_loss", "validation_loss", "full_valid_loss")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read fine-tuning evidence file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Fine-tuning evidence file is not a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _field_name(fieldnames: list[str], aliases: tuple[str, ...], label: str) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames if name is not None}
    matches = [normalized[name] for name in aliases if name in normalized]
    if len(matches) != 1:
        expected = ", ".join(aliases)
        raise ValueError(f"Metrics CSV must contain exactly one {label} column from: {expected}")
    return matches[0]


def _loss(value: object, *, row_number: int, field: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(f"Metrics CSV row {row_number} has a nonnumeric {field}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Metrics CSV row {row_number} has an invalid {field}")
    return parsed


def load_loss_series(path: Path) -> dict[str, Any]:
    """Load strictly validated training and validation loss series from one CSV."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Metrics CSV has no header")
            fieldnames = [str(name) for name in reader.fieldnames]
            step_field = _field_name(fieldnames, ("step",), "step")
            train_field = _field_name(fieldnames, TRAIN_LOSS_FIELDS, "training-loss")
            validation_field = _field_name(
                fieldnames,
                VALIDATION_LOSS_FIELDS,
                "validation-loss",
            )
            by_step: dict[int, dict[str, float]] = {}
            for row_number, row in enumerate(reader, start=2):
                raw_step = str(row.get(step_field, "")).strip()
                try:
                    numeric_step = float(raw_step)
                    step = int(numeric_step)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Metrics CSV row {row_number} has an invalid step") from exc
                if not math.isfinite(numeric_step) or numeric_step != step or step < 0:
                    raise ValueError(f"Metrics CSV row {row_number} has an invalid step")
                values = {
                    "training": _loss(row.get(train_field), row_number=row_number, field=train_field),
                    "validation": _loss(
                        row.get(validation_field),
                        row_number=row_number,
                        field=validation_field,
                    ),
                }
                if values["training"] is None and values["validation"] is None:
                    continue
                target = by_step.setdefault(step, {})
                for label, value in values.items():
                    if value is None:
                        continue
                    if label in target and target[label] != value:
                        raise ValueError(
                            f"Metrics CSV contains conflicting {label} loss values at step {step}"
                        )
                    target[label] = value
    except OSError as exc:
        raise RuntimeError(f"Cannot read archived fine-tuning metrics: {path}") from exc

    training = [[step, values["training"]] for step, values in sorted(by_step.items()) if "training" in values]
    validation = [
        [step, values["validation"]]
        for step, values in sorted(by_step.items())
        if "validation" in values
    ]
    if len(training) < 2:
        raise ValueError("Metrics CSV contains fewer than two training-loss observations")
    if len(validation) < 2:
        raise ValueError("Metrics CSV contains fewer than two validation-loss observations")
    return {
        "columns": {
            "step": step_field,
            "training_loss": train_field,
            "validation_loss": validation_field,
        },
        "training": training,
        "validation": validation,
    }


def _find_metrics_file(state: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    training_result = state.get("training_result")
    if isinstance(training_result, Mapping):
        history = training_result.get("log_history")
        if not isinstance(history, list):
            raise RuntimeError("The local training state has no Trainer log history")
        training: list[list[float]] = []
        validation: list[list[float]] = []
        epoch_scales: list[float] = []
        epoch_scale_complete = True
        for record in history:
            if not isinstance(record, Mapping):
                raise RuntimeError("The local Trainer log contains a malformed event")
            step = record.get("step")
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                continue
            numeric_step = int(step)
            if float(step) != numeric_step or numeric_step < 0:
                raise RuntimeError("The local Trainer log contains an invalid step")
            records_loss = "loss" in record or "eval_loss" in record
            if records_loss:
                epoch = record.get("epoch")
                if (
                    isinstance(epoch, bool)
                    or not isinstance(epoch, (int, float))
                    or not math.isfinite(float(epoch))
                    or float(epoch) <= 0
                    or numeric_step <= 0
                ):
                    epoch_scale_complete = False
                else:
                    epoch_scales.append(numeric_step / float(epoch))
            if "loss" in record:
                loss = _loss(record["loss"], row_number=numeric_step, field="loss")
                if loss is not None:
                    training.append([numeric_step, loss])
            if "eval_loss" in record:
                loss = _loss(record["eval_loss"], row_number=numeric_step, field="eval_loss")
                if loss is not None:
                    validation.append([numeric_step, loss])
        if len(training) < 2 or len(validation) < 2:
            raise RuntimeError("The local Trainer log has insufficient training or validation losses")
        series = {
            "columns": {
                "step": "step",
                "training_loss": "loss",
                "validation_loss": "eval_loss",
            },
            "training": training,
            "validation": validation,
        }
        if epoch_scale_complete and len(epoch_scales) == len(training) + len(validation):
            steps_per_epoch = epoch_scales[0]
            if all(
                math.isclose(value, steps_per_epoch, rel_tol=1e-9, abs_tol=1e-9)
                for value in epoch_scales
            ):
                series["steps_per_epoch"] = steps_per_epoch
        return STATE_PATH, series

    archived = state.get("archived_result_files")
    if not isinstance(archived, list) or not archived:
        raise RuntimeError("The fine-tuning state has no archived result files")
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for record in archived:
        if not isinstance(record, dict):
            raise RuntimeError("The fine-tuning result-file archive is malformed")
        try:
            path = resolve_recorded_path(
                record.get("local_path"),
                permitted_root=MODEL_DIR / "result_files",
            )
        except ValueError as exc:
            raise RuntimeError("An archived fine-tuning result path is invalid") from exc
        if not path.is_file() or record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
            raise RuntimeError("An archived fine-tuning result file failed hash verification")
        if path.suffix.lower() != ".csv":
            continue
        try:
            series = load_loss_series(path)
        except ValueError:
            continue
        candidates.append((path, series))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one archived CSV containing both training and validation loss"
        )
    return candidates[0]


def _polyline(points: list[list[float]], *, x_min: float, x_max: float, y_max: float) -> str:
    left, right, top, bottom = 185.0, 2056.0, 75.0, 1010.0
    x_span = max(1, x_max - x_min)
    return " ".join(
        f"{left + (float(step) - x_min) * (right - left) / x_span:.2f},"
        f"{bottom - float(value) * (bottom - top) / y_max:.2f}"
        for step, value in points
    )


def _ticks(low: int, high: int, count: int = 6) -> list[int]:
    if high <= low:
        return [low]
    values = {round(low + index * (high - low) / (count - 1)) for index in range(count)}
    return sorted(values)


def _format_loss(value: float) -> str:
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2g}"


def _epoch_summary(series: Mapping[str, Any]) -> dict[str, list[list[float]]] | None:
    steps_per_epoch = series.get("steps_per_epoch")
    if not isinstance(steps_per_epoch, (int, float)) or isinstance(steps_per_epoch, bool):
        return None
    scale = float(steps_per_epoch)
    if not math.isfinite(scale) or scale <= 0:
        return None

    validation: list[list[float]] = []
    for step, loss in sorted(series["validation"]):
        epoch = float(step) / scale
        rounded_epoch = round(epoch)
        if rounded_epoch < 1 or not math.isclose(epoch, rounded_epoch, abs_tol=1e-9):
            raise RuntimeError("Validation loss was not recorded at an epoch boundary")
        validation.append([float(rounded_epoch), float(loss)])

    weighted_loss: dict[int, float] = {}
    weights: dict[int, float] = {}
    previous_step = 0.0
    for step, loss in sorted(series["training"]):
        numeric_step = float(step)
        interval = numeric_step - previous_step
        if interval <= 0:
            raise RuntimeError("Training-loss steps are not strictly increasing")
        epoch = int(math.ceil(numeric_step / scale - 1e-12))
        interval_start_epoch = int(math.floor(previous_step / scale + 1e-12)) + 1
        if epoch != interval_start_epoch:
            raise RuntimeError("A training-loss logging interval crosses an epoch boundary")
        weighted_loss[epoch] = weighted_loss.get(epoch, 0.0) + float(loss) * interval
        weights[epoch] = weights.get(epoch, 0.0) + interval
        previous_step = numeric_step

    validation_epochs = [int(point[0]) for point in validation]
    if sorted(weighted_loss) != validation_epochs:
        raise RuntimeError("Training and validation losses do not cover the same epochs")
    training = [
        [float(epoch), weighted_loss[epoch] / weights[epoch]]
        for epoch in validation_epochs
    ]
    return {"training": training, "validation": validation}


def _axis_metadata(series: Mapping[str, Any]) -> dict[str, Any]:
    epoch_summary = _epoch_summary(series)
    if epoch_summary is not None:
        scale = float(series["steps_per_epoch"])
        all_points = [
            [float(epoch) * scale, float(loss)]
            for epoch, loss in epoch_summary["training"] + epoch_summary["validation"]
        ]
        return {
            "unit": "optimizer_step",
            "steps_per_epoch": scale,
            "training_aggregation": "step_weighted_mean_of_logged_intervals",
            "minimum": min(float(point[0]) for point in all_points),
            "maximum": max(float(point[0]) for point in all_points),
            "training_points": len(epoch_summary["training"]),
            "validation_points": len(epoch_summary["validation"]),
        }
    all_points = series["training"] + series["validation"]
    return {
        "unit": "update_step",
        "minimum": min(float(point[0]) for point in all_points),
        "maximum": max(float(point[0]) for point in all_points),
    }


def render_html(series: Mapping[str, Any]) -> str:
    epoch_summary = _epoch_summary(series)
    plotted = epoch_summary if epoch_summary is not None else series
    training = [[float(x), float(y)] for x, y in plotted["training"]]
    validation = [[float(x), float(y)] for x, y in plotted["validation"]]
    if epoch_summary is not None:
        scale = float(series["steps_per_epoch"])
        training = [[epoch * scale, loss] for epoch, loss in training]
        validation = [[epoch * scale, loss] for epoch, loss in validation]
    axis = _axis_metadata(series)
    all_points = training + validation
    data_x_min = float(min(point[0] for point in all_points))
    data_x_max = float(max(point[0] for point in all_points))
    if epoch_summary is not None:
        x_padding = max(1.0, (data_x_max - data_x_min) * 0.05)
        x_min = data_x_min - x_padding
        x_max = data_x_max + x_padding
    else:
        x_min = data_x_min
        x_max = data_x_max
    observed_max = max(point[1] for point in all_points)
    if observed_max <= 0:
        raise ValueError("Loss observations cannot all be zero")
    y_max = observed_max * 1.08
    if epoch_summary is not None:
        x_ticks = [int(point[0]) for point in training]
    else:
        x_ticks = _ticks(int(x_min), int(x_max))
    y_ticks = [index * y_max / 5 for index in range(6)]
    left, right, top, bottom = 185.0, 2056.0, 75.0, 1010.0
    x_span = max(1, x_max - x_min)

    grid = []
    for value in y_ticks:
        y = bottom - value * (bottom - top) / y_max
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>'
            f'<text x="{left - 28}" y="{y + 12:.2f}" text-anchor="end" class="tick">'
            f'{html.escape(_format_loss(value))}</text>'
        )
    for value in x_ticks:
        x = left + (value - x_min) * (right - left) / x_span
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" class="grid"/>'
            f'<text x="{x:.2f}" y="{bottom + 58}" text-anchor="middle" class="tick">'
            f'{value:g}</text>'
        )

    train_polyline = _polyline(training, x_min=x_min, x_max=x_max, y_max=y_max)
    validation_polyline = _polyline(validation, x_min=x_min, x_max=x_max, y_max=y_max)
    validation_markers = []
    for step, value in validation:
        x = left + (step - x_min) * (right - left) / x_span
        y = bottom - value * (bottom - top) / y_max
        validation_markers.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="11" class="valid-marker"/>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
html, body {{ width: {WIDTH_PX}px; height: {HEIGHT_PX}px; margin: 0; overflow: hidden; background: #ffffff; }}
svg {{ display: block; font-family: Arial, Helvetica, sans-serif; }}
.axis {{ stroke: #202124; stroke-width: 9; }}
.grid {{ stroke: #d5d9dd; stroke-width: 4; }}
.tick {{ fill: #202124; font-size: 36px; }}
.axis-label {{ fill: #202124; font-size: 43px; font-weight: 600; }}
.legend {{ fill: #202124; font-size: 38px; }}
.train {{ fill: none; stroke: #0072b2; stroke-width: 10; stroke-linejoin: round; stroke-linecap: round; }}
.valid {{ fill: none; stroke: #d55e00; stroke-width: 10; stroke-dasharray: 28 18; stroke-linejoin: round; stroke-linecap: round; }}
.valid-marker {{ fill: #ffffff; stroke: #d55e00; stroke-width: 8; }}
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PX}" height="{HEIGHT_PX}" viewBox="0 0 {WIDTH_PX} {HEIGHT_PX}" role="img" aria-label="Training and validation loss by {html.escape(str(axis['unit']).replace('_', ' '))}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  {''.join(grid)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>
  <polyline points="{train_polyline}" class="train"/>
  <polyline points="{validation_polyline}" class="valid"/>
  {''.join(validation_markers)}
  <text x="{(left + right) / 2:.2f}" y="1125" text-anchor="middle" class="axis-label">{'Training step' if epoch_summary is not None else 'Update step'}</text>
  <text x="57" y="{(top + bottom) / 2:.2f}" text-anchor="middle" transform="rotate(-90 57 {(top + bottom) / 2:.2f})" class="axis-label">Loss</text>
  <line x1="1420" y1="55" x2="1510" y2="55" class="train"/>
  <text x="1535" y="68" class="legend">Training</text>
  <line x1="1740" y1="55" x2="1830" y2="55" class="valid"/>
  <circle cx="1785" cy="55" r="11" class="valid-marker"/>
  <text x="1855" y="68" class="legend">Validation</text>
</svg>
</body>
</html>
"""


def _find_edge(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    discovered = shutil.which("msedge") or shutil.which("msedge.exe")
    if discovered:
        candidates.append(Path(discovered))
    if os.name == "nt":
        candidates.extend([
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("Microsoft Edge was not found; pass --browser with its executable path")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError("The loss-figure renderer did not produce a valid PNG image")
    return struct.unpack(">II", header[16:24])


def _capture_png(browser: Path, html_path: Path, output: Path) -> str:
    if os.name == "nt":
        escaped_browser = str(browser).replace("'", "''")
        version = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"(Get-Item -LiteralPath '{escaped_browser}').VersionInfo.ProductVersion",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        version = subprocess.run(
            [str(browser), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    browser_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or not browser_version:
        raise RuntimeError("Unable to record the browser version used for the loss figure")
    profile_path = output.parent / "edge-profile"
    profile_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile_path}",
            "--force-device-scale-factor=1",
            f"--window-size={WIDTH_PX},{HEIGHT_PX}",
            f"--screenshot={output}",
            html_path.as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Edge failed to render the fine-tuning loss figure: {detail}")
    dimensions = _png_dimensions(output)
    if dimensions != (WIDTH_PX, HEIGHT_PX):
        raise RuntimeError(
            f"Loss figure has dimensions {dimensions}, expected {(WIDTH_PX, HEIGHT_PX)}"
        )
    return browser_version


def build(*, browser_path: Path | None = None, replace: bool = False) -> Path:
    if (OUTPUT_PATH.exists() or MANIFEST_PATH.exists()) and not replace:
        raise RuntimeError("Refusing to replace existing loss-figure evidence without --replace")
    state = _read_object(STATE_PATH)
    _validate_fine_tuning_state(state)
    metrics_path, series = _find_metrics_file(state)
    browser = _find_edge(browser_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="loss-figure-", dir=MODEL_DIR) as temporary_dir:
        temporary = Path(temporary_dir)
        html_path = temporary / "loss.html"
        png_path = temporary / OUTPUT_PATH.name
        html_path.write_text(render_html(series), encoding="utf-8", newline="\n")
        browser_version = _capture_png(browser, html_path, png_path)
        os.replace(png_path, OUTPUT_PATH)

    archived_paths = []
    for record in state.get("archived_result_files", []):
        archived_paths.append(
            resolve_recorded_path(
                record["local_path"],
                permitted_root=MODEL_DIR / "result_files",
            )
        )
    inputs = {
        portable_path(path): _record(path)
        for path in (STATE_PATH, DEFAULT_PROTOCOL_PATH, SCRIPT_PATH, *archived_paths)
    }
    output_record = _record(OUTPUT_PATH)
    output_record["dimensions_px"] = [WIDTH_PX, HEIGHT_PX]
    output_record["effective_dpi_at_180_mm"] = round(WIDTH_PX / (180 / 25.4), 2)
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "network_contacted": False,
        "env_file_read": False,
        "browser": {
            "executable_name": browser.name,
            "version": browser_version,
        },
        "inputs": inputs,
        "selected_metrics_file": portable_path(metrics_path),
        "columns": series["columns"],
        "observations": {
            "training": len(series["training"]),
            "validation": len(series["validation"]),
            "minimum_step": int(min(point[0] for point in series["training"] + series["validation"])),
            "maximum_step": int(max(point[0] for point in series["training"] + series["validation"])),
        },
        "axis": _axis_metadata(series),
        "output": output_record,
    }
    _write_json(MANIFEST_PATH, manifest)
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = build(browser_path=args.browser, replace=args.replace)
    print(f"Rendered {output}")
    print(f"Recorded {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
