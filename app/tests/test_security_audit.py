from pathlib import Path

from evaluation.audit_active_artifacts import PATTERNS, _is_private_env_file


def test_private_environment_files_are_never_scanned() -> None:
    assert _is_private_env_file(Path(".env"))
    assert _is_private_env_file(Path(".ENV"))
    assert _is_private_env_file(Path(".env.local"))
    assert _is_private_env_file(Path(".env.backup"))
    assert not _is_private_env_file(Path(".env.example"))
    assert not _is_private_env_file(Path("settings.env"))


def test_private_key_pattern_covers_assignment_and_json_forms() -> None:
    secret = "0x" + "a" * 64
    samples = (
        f"DECISION_SERVICE_PRIVATE_KEY={secret}",
        f"DECISION_SERVICE_PRIVATE_KEY='{secret}'",
        f'{{"private_key": "{secret}"}}',
    )
    pattern = PATTERNS["private_key_assignment"]
    assert all(pattern.search(sample) for sample in samples)
