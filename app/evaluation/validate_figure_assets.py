"""Validate publication-scale geometry for editable manuscript diagrams."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent.parent
JOURNAL_DIR = WORKSPACE_DIR / "Frontiers_Medical_Technology_2026-09-06"
EDITABLE_FIGURE_NAMES = (
    "system_architecture",
    "sequence_enrollment",
    "sequence_matching",
)
TERMINAL_CAPTURE_NAMES = (
    "foundry_tests",
    "slither_analysis",
)
FIGURE_NAMES = EDITABLE_FIGURE_NAMES + TERMINAL_CAPTURE_NAMES
FINAL_WIDTH_MM = 180.0
MIN_FONT_POINTS = 8.0
MIN_LINE_POINTS = 2.0
MIN_DPI = 300.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: str, *, label: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*", value)
    if not match:
        raise ValueError(f"Invalid {label}: {value!r}")
    parsed = float(match.group(1))
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _png_metadata(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG image: {path}")

    offset = 8
    ihdr: bytes | None = None
    compressed = bytearray()
    found_iend = False
    chunk_index = 0
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise ValueError(f"Truncated PNG chunk in {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError(f"Truncated PNG chunk in {path}")
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG chunk checksum mismatch in {path}")
        if chunk_index == 0 and (chunk_type != b"IHDR" or length != 13):
            raise ValueError(f"PNG does not start with a valid IHDR chunk: {path}")
        if chunk_type == b"IHDR":
            if ihdr is not None or chunk_index != 0 or length != 13:
                raise ValueError(f"Invalid PNG IHDR chunk in {path}")
            ihdr = chunk_data
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            if length != 0 or crc_end != len(payload):
                raise ValueError(f"Invalid PNG IEND chunk in {path}")
            found_iend = True
            break
        offset = crc_end
        chunk_index += 1

    if ihdr is None or not compressed or not found_iend:
        raise ValueError(f"Incomplete PNG image: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width <= 0 or height <= 0 or compression != 0 or filtering != 0:
        raise ValueError(f"Invalid PNG IHDR values in {path}")
    if interlace not in {0, 1}:
        raise ValueError(f"Invalid PNG interlace method in {path}")

    if bit_depth == 8 and color_type == 2 and interlace == 0:
        try:
            scanlines = zlib.decompress(bytes(compressed))
        except zlib.error as exc:
            raise ValueError(f"Invalid PNG image data in {path}") from exc
        row_bytes = width * 3
        expected_size = height * (row_bytes + 1)
        if len(scanlines) != expected_size:
            raise ValueError(f"PNG scanline size mismatch in {path}")
        if any(scanlines[index] > 4 for index in range(0, expected_size, row_bytes + 1)):
            raise ValueError(f"Invalid PNG scanline filter in {path}")

    return {
        "width_px": width,
        "height_px": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
    }


def _declared_sizes(source: str, pattern: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, source, flags=re.IGNORECASE)]


def inspect_pair(svg_path: Path, png_path: Path, *, final_width_mm: float = FINAL_WIDTH_MM) -> dict[str, Any]:
    source = svg_path.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SVG XML: {svg_path}") from exc
    view_box = re.split(r"[\s,]+", str(root.attrib.get("viewBox", "")).strip())
    if len(view_box) != 4:
        raise ValueError(f"SVG has no four-value viewBox: {svg_path}")
    _, _, logical_width, logical_height = (float(value) for value in view_box)
    if logical_width <= 0 or logical_height <= 0:
        raise ValueError(f"SVG viewBox is not positive: {svg_path}")
    declared_width = _number(root.attrib.get("width", ""), label="SVG width")
    declared_height = _number(root.attrib.get("height", ""), label="SVG height")

    font_sizes = _declared_sizes(
        source,
        r"\bfont\s*:\s*[^;{}]*?(\d+(?:\.\d+)?)px",
    )
    font_sizes.extend(_declared_sizes(source, r"\bfont-size\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)"))
    stroke_widths = _declared_sizes(
        source,
        r"\bstroke-width\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)",
    )
    if not font_sizes:
        raise ValueError(f"SVG has no measurable font sizes: {svg_path}")
    if not stroke_widths:
        raise ValueError(f"SVG has no measurable stroke widths: {svg_path}")

    points_per_unit = final_width_mm * 72.0 / (25.4 * logical_width)
    minimum_font_points = min(font_sizes) * points_per_unit
    minimum_line_points = min(stroke_widths) * points_per_unit
    png = _png_metadata(png_path)
    effective_dpi = png["width_px"] / (final_width_mm / 25.4)
    ratios_match = abs(
        png["width_px"] / png["height_px"] - logical_width / logical_height
    ) <= 1e-6
    dimensions_match = (
        png["width_px"] == round(declared_width)
        and png["height_px"] == round(declared_height)
    )
    checks = {
        "font_at_least_8pt": minimum_font_points + 1e-9 >= MIN_FONT_POINTS,
        "lines_at_least_2pt": minimum_line_points + 1e-9 >= MIN_LINE_POINTS,
        "effective_resolution_at_least_300dpi": effective_dpi + 1e-9 >= MIN_DPI,
        "svg_and_png_dimensions_match": dimensions_match,
        "svg_and_png_aspect_ratios_match": ratios_match,
        "png_is_8bit_rgb": png["bit_depth"] == 8 and png["color_type"] == 2,
    }
    return {
        "svg": svg_path.name,
        "png": png_path.name,
        "svg_bytes": svg_path.stat().st_size,
        "svg_sha256": _sha256(svg_path),
        "png_bytes": png_path.stat().st_size,
        "png_sha256": _sha256(png_path),
        "final_width_mm": final_width_mm,
        "logical_dimensions": [logical_width, logical_height],
        "raster_dimensions_px": [png["width_px"], png["height_px"]],
        "minimum_declared_font_px": min(font_sizes),
        "minimum_font_points_at_final_width": round(minimum_font_points, 3),
        "minimum_declared_stroke_px": min(stroke_widths),
        "minimum_line_points_at_final_width": round(minimum_line_points, 3),
        "effective_dpi_at_final_width": round(effective_dpi, 3),
        "png_bit_depth": png["bit_depth"],
        "png_color_type": png["color_type"],
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def inspect_terminal_capture(
    png_path: Path,
    *,
    final_width_mm: float = FINAL_WIDTH_MM,
) -> dict[str, Any]:
    png = _png_metadata(png_path)
    effective_dpi = png["width_px"] / (final_width_mm / 25.4)
    checks = {
        "effective_resolution_at_least_300dpi": effective_dpi + 1e-9 >= MIN_DPI,
        "png_is_8bit_rgb": png["bit_depth"] == 8 and png["color_type"] == 2,
    }
    return {
        "png": png_path.name,
        "png_bytes": png_path.stat().st_size,
        "png_sha256": _sha256(png_path),
        "capture_type": "native_terminal_screenshot",
        "final_width_mm": final_width_mm,
        "raster_dimensions_px": [png["width_px"], png["height_px"]],
        "effective_dpi_at_final_width": round(effective_dpi, 3),
        "png_bit_depth": png["bit_depth"],
        "png_color_type": png["color_type"],
        "manual_readability_review_required": True,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def validate(directory: Path, *, final_width_mm: float = FINAL_WIDTH_MM) -> dict[str, Any]:
    resolved = directory.resolve()
    figures: dict[str, Any] = {}
    for name in EDITABLE_FIGURE_NAMES:
        svg_path = resolved / f"{name}.svg"
        png_path = resolved / f"{name}.png"
        if not svg_path.is_file() or not png_path.is_file():
            raise FileNotFoundError(f"Missing SVG/PNG pair for {name} in {resolved}")
        figures[name] = inspect_pair(svg_path, png_path, final_width_mm=final_width_mm)
    for name in TERMINAL_CAPTURE_NAMES:
        png_path = resolved / f"{name}.png"
        if not png_path.is_file():
            raise FileNotFoundError(f"Missing terminal-capture PNG for {name} in {resolved}")
        figures[name] = inspect_terminal_capture(
            png_path,
            final_width_mm=final_width_mm,
        )
    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "directory": str(resolved),
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "network_contacted": False,
        "env_file_read": False,
        "requirements": {
            "final_width_mm": final_width_mm,
            "minimum_font_points": MIN_FONT_POINTS,
            "minimum_line_points": MIN_LINE_POINTS,
            "minimum_dpi": MIN_DPI,
            "png_color_mode": "8-bit RGB",
            "terminal_capture_readability": "manual review",
        },
        "figures": figures,
        "all_passed": all(item["all_passed"] for item in figures.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=JOURNAL_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-width-mm", type=float, default=FINAL_WIDTH_MM)
    args = parser.parse_args()
    if args.final_width_mm <= 0:
        raise ValueError("--final-width-mm must be positive")
    report = validate(args.directory, final_width_mm=args.final_width_mm)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
