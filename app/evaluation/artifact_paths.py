"""Portable path serialization for retained study artifacts."""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
WORKSPACE_DIR = IMPLEMENTATION_DIR.parent


def portable_path(path: Path | str, *, root: Path = WORKSPACE_DIR) -> str:
    """Serialize workspace files relatively while allowing external test fixtures."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_recorded_path(
    value: object,
    *,
    root: Path = WORKSPACE_DIR,
    permitted_root: Path | None = None,
) -> Path:
    """Resolve a retained relative path and optionally confine it to a directory."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Recorded artifact path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if permitted_root is not None:
        try:
            resolved.relative_to(permitted_root.resolve())
        except ValueError as exc:
            raise ValueError("Recorded artifact path leaves its permitted directory") from exc
    return resolved
