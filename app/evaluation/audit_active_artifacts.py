"""Scan active text artifacts for common credential shapes without reading .env."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
WORKSPACE_DIR = IMPLEMENTATION_DIR.parent
JOURNAL_DIR = WORKSPACE_DIR / "Frontiers_Medical_Technology_2026-09-06"
OUTPUT_PATH = APP_DIR / "pipeline-output" / "current" / "security_audit.json"

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".example",
    ".html",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sol",
    ".svg",
    ".tex",
    ".toml",
    ".txt",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "archive",
    "cache",
    "lib",
    "node_modules",
    "out",
}
PATTERNS = {
    "openai_api_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    "private_key_assignment": re.compile(
        r"(?im)\b(?:[A-Z0-9_]*PRIVATE_KEY|private[_-]?key)\b[\"']?\s*[:=]\s*[\"']?"
        r"(0x[0-9a-f]{64})\b"
    ),
    "infura_project_url": re.compile(
        r"https?://[^\s\"']+\.infura\.io/v3/[A-Za-z0-9_-]{20,}",
        re.IGNORECASE,
    ),
    "alchemy_project_url": re.compile(r"https?://[^\s]+\.g\.alchemy\.com/v2/[A-Za-z0-9_-]{20,}"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_DIR.resolve()).as_posix()


def _is_private_env_file(path: Path) -> bool:
    name = path.name.casefold()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def active_text_files() -> Iterable[Path]:
    for root in (IMPLEMENTATION_DIR, JOURNAL_DIR):
        for path in root.rglob("*"):
            if not path.is_file() or path == OUTPUT_PATH or _is_private_env_file(path):
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
                yield path


def scan_file(path: Path) -> list[dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    findings: list[dict[str, object]] = []
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            matched = match.group(1) if match.lastindex else match.group(0)
            findings.append({
                "file": _relative(path),
                "line": line,
                "pattern": name,
                "match_sha256_prefix": hashlib.sha256(matched.encode("utf-8")).hexdigest()[:12],
            })
    return findings


def audit() -> dict[str, object]:
    files = sorted(set(active_text_files()), key=_relative)
    findings = [finding for path in files for finding in scan_file(path)]
    report = {
        "generated_at_utc": _utc_now(),
        "files_scanned": len(files),
        "env_file_read": False,
        "finding_count": len(findings),
        "findings": findings,
        "limitations": (
            "Pattern scanning cannot prove that no secret exists. Review generated artifacts and repository "
            "history before public release."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = audit()
    print(f"Scanned {report['files_scanned']} active text files; findings={report['finding_count']}")
    print(f"Report written to {OUTPUT_PATH}")
    if report["finding_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
