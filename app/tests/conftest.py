from pathlib import Path

import pytest

from evaluation.publication_workspace import ARTICLE_SOURCE_DIR

APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
EVIDENCE_BOUND_MODULES = {
    "test_benchmark_offchain.py",
    "test_evaluation_lock.py",
    "test_fine_tuning.py",
    "test_fine_tuning_loss_figure.py",
    "test_manuscript_source.py",
    "test_manuscript_tables.py",
    "test_smoke_model.py",
    "test_software_verification_figure.py",
}
REQUIRED_LOCAL_ARTIFACTS = (
    APP_DIR / "pipeline-output" / "current" / "protocol" / "protocol_freeze.json",
    APP_DIR / "pipeline-output" / "current" / "model" / "local_lora_training.json",
    IMPLEMENTATION_DIR / "smart-contracts" / "out" / "TransplantManagement.sol" / "TransplantManagement.json",
    ARTICLE_SOURCE_DIR / "Manuscript.tex",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if all(path.is_file() for path in REQUIRED_LOCAL_ARTIFACTS):
        return

    marker = pytest.mark.skip(
        reason="requires the retained experiment bundle and manuscript workspace"
    )
    for item in items:
        if Path(str(item.fspath)).name in EVIDENCE_BOUND_MODULES:
            item.add_marker(marker)
