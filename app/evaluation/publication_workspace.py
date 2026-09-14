"""Resolve optional publication artifacts outside the implementation repository."""

from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = APP_DIR.parent
DEFAULT_ROOT = APP_DIR / "pipeline-output" / "publication"


def _directory(variable: str, default_name: str) -> Path:
    configured = os.environ.get(variable, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_ROOT / default_name


ARTICLE_SOURCE_DIR = _directory("ODA_ARTICLE_SOURCE_DIR", "source")
ARTICLE_SUPPORT_DIR = _directory("ODA_ARTICLE_SUPPORT_DIR", "support")
ARTICLE_ARCHIVE_DIR = _directory("ODA_ARTICLE_ARCHIVE_DIR", "archive")
FIGURE_SOURCE_DIR = _directory("ODA_FIGURE_SOURCE_DIR", "figure-source")
FIGURE_OUTPUT_DIR = _directory("ODA_FIGURE_OUTPUT_DIR", "figures")


def external_directories() -> tuple[Path, ...]:
    """Return configured publication directories without duplicate paths."""
    values = (
        ARTICLE_SOURCE_DIR,
        ARTICLE_SUPPORT_DIR,
        ARTICLE_ARCHIVE_DIR,
        FIGURE_SOURCE_DIR,
        FIGURE_OUTPUT_DIR,
    )
    return tuple(dict.fromkeys(path.resolve() for path in values))
