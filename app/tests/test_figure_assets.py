from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from evaluation import build_evidence_manifest as evidence
from evaluation import validate_figure_assets as figures


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_rgb_png(path: Path, width: int, height: int) -> None:
    scanline = b"\x00" + (b"\xff\xff\xff" * width)
    image_data = zlib.compress(scanline * height, level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", image_data)
        + _chunk(b"IEND", b"")
    )


def _write_pair(
    directory: Path,
    name: str,
    *,
    width: int = 3600,
    height: int = 2000,
    font_px: float = 12,
    stroke_px: float = 3,
) -> tuple[Path, Path]:
    svg_path = directory / f"{name}.svg"
    png_path = directory / f"{name}.png"
    svg_path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            'viewBox="0 0 720 400">'
            f'<rect width="720" height="400" fill="white" stroke="black" '
            f'stroke-width="{stroke_px}"/>'
            f'<text x="20" y="40" font-size="{font_px}px">Test</text>'
            '</svg>\n'
        ),
        encoding="utf-8",
    )
    _write_rgb_png(png_path, width, height)
    return svg_path, png_path


def test_inspect_pair_accepts_valid_publication_geometry(tmp_path: Path) -> None:
    svg_path, png_path = _write_pair(tmp_path, "figure")

    result = figures.inspect_pair(svg_path, png_path)

    assert result["all_passed"] is True
    assert all(result["checks"].values())
    assert result["raster_dimensions_px"] == [3600, 2000]
    assert result["png_bit_depth"] == 8
    assert result["png_color_type"] == 2


def test_inspect_pair_reports_each_publication_geometry_failure(tmp_path: Path) -> None:
    svg_path, png_path = _write_pair(
        tmp_path,
        "figure",
        width=1000,
        height=1000,
        font_px=4,
        stroke_px=1,
    )

    result = figures.inspect_pair(svg_path, png_path)

    assert result["all_passed"] is False
    assert result["checks"] == {
        "font_at_least_8pt": False,
        "lines_at_least_2pt": False,
        "effective_resolution_at_least_300dpi": False,
        "svg_and_png_dimensions_match": True,
        "svg_and_png_aspect_ratios_match": False,
        "png_is_8bit_rgb": True,
    }


def test_png_parser_rejects_truncated_or_tampered_data(tmp_path: Path) -> None:
    svg_path, png_path = _write_pair(tmp_path, "figure")
    del svg_path
    payload = png_path.read_bytes()

    png_path.write_bytes(payload[:26])
    with pytest.raises(ValueError, match="Truncated|Incomplete|valid PNG"):
        figures._png_metadata(png_path)

    png_path.write_bytes(payload[:-13] + b"tampered")
    with pytest.raises(ValueError, match="Truncated|Incomplete|checksum|IEND"):
        figures._png_metadata(png_path)


def test_validate_requires_all_five_pairs_and_records_provenance(tmp_path: Path) -> None:
    for name in figures.EDITABLE_FIGURE_NAMES:
        _write_pair(tmp_path, name)
    for name in figures.TERMINAL_CAPTURE_NAMES:
        _write_rgb_png(tmp_path / f"{name}.png", 3600, 2000)

    result = figures.validate(tmp_path)

    assert result["all_passed"] is True
    assert result["network_contacted"] is False
    assert result["env_file_read"] is False
    assert list(result["figures"]) == list(figures.FIGURE_NAMES)
    assert result["validator"]["sha256"] == figures._sha256(Path(figures.__file__))
    json.dumps(result)

    (tmp_path / f"{figures.FIGURE_NAMES[-1]}.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing terminal-capture PNG"):
        figures.validate(tmp_path)


def test_final_evidence_gate_rejects_nonpublication_figure_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in figures.EDITABLE_FIGURE_NAMES:
        _write_pair(tmp_path, name)
    for name in figures.TERMINAL_CAPTURE_NAMES:
        _write_rgb_png(tmp_path / f"{name}.png", 3600, 2000)
    monkeypatch.setattr(evidence, "FIGURE_OUTPUT_DIR", tmp_path)

    evidence._validate_publication_figure_assets()

    _write_pair(tmp_path, figures.FIGURE_NAMES[0], font_px=4)
    with pytest.raises(RuntimeError, match="Publication-scale figure validation failed"):
        evidence._validate_publication_figure_assets()
