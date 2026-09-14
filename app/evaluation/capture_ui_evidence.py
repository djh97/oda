"""Render manuscript UI panels from the hash-verified canonical snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.project_environment import CONTROLLED_ENV_NAMES
from evaluation.publication_workspace import FIGURE_OUTPUT_DIR, REPOSITORY_DIR
from src.evidence_view import CURRENT_OUTPUT_DIR, load_latest_evidence


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
WORKSPACE_DIR = REPOSITORY_DIR
SCRIPT_PATH = Path(__file__).resolve()
POINTER_PATH = CURRENT_OUTPUT_DIR / "latest_full_workflow.json"

CAPTURES = (
    ("full", FIGURE_OUTPUT_DIR / "Full_UI.png", 1000, 980),
    ("decision", FIGURE_OUTPUT_DIR / "LLM_Decision.png", 1000, 960),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read canonical evidence file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Canonical evidence file is not a JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _resolve_latest_run() -> Path:
    pointer = _read_object(POINTER_PATH)
    run_dir = (APP_DIR / str(pointer.get("run_directory", ""))).resolve()
    workflow_root = (CURRENT_OUTPUT_DIR / "full_workflow").resolve()
    try:
        run_dir.relative_to(workflow_root)
    except ValueError as exc:
        raise RuntimeError("The canonical-run pointer leaves the evidence directory") from exc
    summary_path = run_dir / "run_summary.json"
    snapshot_path = run_dir / "ui_evidence_snapshot.json"
    if pointer.get("run_summary_sha256") != _sha256(summary_path):
        raise RuntimeError("The canonical-run pointer has a stale run-summary hash")
    if pointer.get("ui_evidence_snapshot_sha256") != _sha256(snapshot_path):
        raise RuntimeError("The canonical-run pointer has a stale UI-snapshot hash")
    load_latest_evidence(app_dir=APP_DIR, current_output_dir=CURRENT_OUTPUT_DIR)
    return run_dir


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


def _browser_version(browser: Path) -> str:
    if os.name == "nt":
        quoted_path = str(browser).replace("'", "''")
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"(Get-Item -LiteralPath '{quoted_path}').VersionInfo.ProductVersion",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        result = subprocess.run(
            [str(browser), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    version = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not version:
        raise RuntimeError("Unable to record the browser version used for UI capture")
    return version


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_capture_page(port: int, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/?capture=full"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = response.read().decode("utf-8")
            if response.status == 200 and "canonicalEvidence" in body:
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError("The local read-only capture page did not start") from last_error


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Browser output is not a valid PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def _capture_page(
    browser: Path,
    *,
    profile_dir: Path,
    port: int,
    mode: str,
    output: Path,
    width: int,
    height: int,
) -> tuple[int, int]:
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile_dir}",
            f"--window-size={width},{height}",
            "--virtual-time-budget=8000",
            f"--screenshot={output}",
            f"http://127.0.0.1:{port}/?capture={mode}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Edge failed to create the {mode} UI capture: {detail}")
    dimensions = _png_dimensions(output)
    if dimensions != (width, height):
        raise RuntimeError(
            f"The {mode} UI capture has dimensions {dimensions}, expected {(width, height)}"
        )
    return dimensions


def capture(*, browser_path: Path | None = None, replace: bool = False) -> Path:
    run_dir = _resolve_latest_run()
    output_paths = [path for _, path, _, _ in CAPTURES]
    existing = [path for path in output_paths if path.exists()]
    manifest_path = run_dir / "ui_capture_manifest.json"
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not replace:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(f"Refusing to replace existing UI evidence without --replace: {names}")

    browser = _find_edge(browser_path)
    version = _browser_version(browser)

    env = os.environ.copy()
    env["ODA_DISABLE_DOTENV"] = "1"
    env["PYTHON_DOTENV_DISABLED"] = "1"
    for name in CONTROLLED_ENV_NAMES:
        env.pop(name, None)
    port = _free_local_port()
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=APP_DIR,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    capture_records: dict[str, dict[str, object]] = {}
    try:
        _wait_for_capture_page(port)
        with tempfile.TemporaryDirectory(prefix="ui-capture-", dir=run_dir) as profile:
            profile_dir = Path(profile)
            for mode, output, width, height in CAPTURES:
                output.parent.mkdir(parents=True, exist_ok=True)
                dimensions = _capture_page(
                    browser,
                    profile_dir=profile_dir,
                    port=port,
                    mode=mode,
                    output=output,
                    width=width,
                    height=height,
                )
                capture_records[_relative(output)] = {
                    **_record(output),
                    "mode": mode,
                    "dimensions_px": [*dimensions],
                    "viewport_px": [width, height],
                }
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    input_paths = (
        POINTER_PATH,
        run_dir / "run_summary.json",
        run_dir / "ui_evidence_snapshot.json",
        APP_DIR / "templates" / "index.html",
        APP_DIR / "src" / "main.py",
        APP_DIR / "src" / "evidence_view.py",
        APP_DIR / "evaluation" / "project_environment.py",
        SCRIPT_PATH,
    )
    manifest = {
        "schema_version": "1.0",
        "captured_at_utc": _utc_now(),
        "canonical_run": _relative(run_dir),
        "capture_policy": "read_only_canonical_snapshot_no_model_storage_or_chain_call",
        "browser": {
            "executable_name": browser.name,
            "version": version,
        },
        "inputs": {_relative(path): _record(path) for path in input_paths},
        "captures": capture_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = capture(browser_path=args.browser, replace=args.replace)
    print(f"UI evidence manifest written to {path}")


if __name__ == "__main__":
    main()
