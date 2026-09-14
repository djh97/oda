from pathlib import Path

from evaluation import publication_workspace as workspace


def test_directory_uses_configured_external_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "article source"
    monkeypatch.setenv("ODA_TEST_PUBLICATION_DIR", str(target))

    assert workspace._directory("ODA_TEST_PUBLICATION_DIR", "fallback") == target.resolve()


def test_directory_uses_ignored_local_staging_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ODA_TEST_PUBLICATION_DIR", raising=False)

    assert workspace._directory("ODA_TEST_PUBLICATION_DIR", "fallback") == (
        workspace.DEFAULT_ROOT / "fallback"
    )
