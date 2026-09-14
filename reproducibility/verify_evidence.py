"""Verify the checksums of the retained study evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
CHECKSUM_PATH = ROOT / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checksums() -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        CHECKSUM_PATH.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed checksum record on line {line_number}"
            ) from exc
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"Invalid SHA-256 digest on line {line_number}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.startswith("evidence/"):
            raise RuntimeError(f"Unsafe evidence path on line {line_number}")
        if relative in records:
            raise RuntimeError(f"Duplicate checksum record for {relative}")
        records[relative] = digest
    return records


def main() -> None:
    expected = expected_checksums()
    observed_paths = {
        path.relative_to(ROOT).as_posix()
        for path in EVIDENCE_DIR.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected) - observed_paths)
    unexpected = sorted(observed_paths - set(expected))
    changed = sorted(
        relative
        for relative, digest in expected.items()
        if relative in observed_paths and sha256(ROOT / relative) != digest
    )
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        if changed:
            details.append("changed: " + ", ".join(changed))
        raise SystemExit("Evidence verification failed (" + "; ".join(details) + ")")
    print(f"Verified {len(expected)} retained evidence files.")


if __name__ == "__main__":
    main()
