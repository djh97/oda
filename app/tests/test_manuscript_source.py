from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from evaluation import build_evidence_manifest as evidence


def _copy_manuscript_inputs(target: Path) -> tuple[Path, Path]:
    target.mkdir()
    manuscript = target / "Manuscript.tex"
    bibliography = target / "cas-refs.bib"
    manuscript.write_text(
        (evidence.JOURNAL_DIR / "Manuscript.tex").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    bibliography.write_text(
        (evidence.JOURNAL_DIR / "cas-refs.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return manuscript, bibliography


def test_current_manuscript_source_passes_preserved_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = evidence._validate_manuscript_source()
    assert report["running_title_words"] == 5
    assert report["abstract_words"] <= 350
    assert 5 <= report["keywords"] <= 8
    assert report["combined_figures_tables"] == 15
    assert report["float_environments"] == {"figure": 8, "table": 8, "algorithm": 7}

    manuscript = (evidence.JOURNAL_DIR / "Manuscript.tex").read_text(encoding="utf-8")
    protocol = json.loads(
        (evidence.APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json").read_text(
            encoding="utf-8"
        )
    )
    retired_strings = tuple(str(seed) for seed in evidence.RETIRED_TEST_SEEDS)
    named_retired_seeds = [seed for seed in retired_strings if seed in manuscript]
    if named_retired_seeds:
        with pytest.raises(RuntimeError, match="retired test seed"):
            evidence._validate_final_manuscript_source(manuscript)
    manuscript_without_retired_seed = manuscript
    for retired_seed in named_retired_seeds:
        manuscript_without_retired_seed = manuscript_without_retired_seed.replace(
            retired_seed,
            "replacement-seed",
        )
    manuscript_with_results_footnote = manuscript_without_retired_seed.replace(
        "Testing comprised three layers.",
        "Testing comprised three layers.\\footnote{Prohibited results note.}",
        1,
    )
    with pytest.raises(RuntimeError, match="Results sections still contain a footnote"):
        evidence._validate_final_manuscript_source(manuscript_with_results_footnote)
    evidence._validate_final_manuscript_source(manuscript_without_retired_seed)

    retired_protocol = copy.deepcopy(protocol)
    retired_protocol["dataset"]["splits"]["test"]["seed"] = evidence.RETIRED_TEST_SEEDS[-1]
    with pytest.raises(RuntimeError, match="active protocol still uses retired test seed"):
        evidence._validate_final_protocol_seed(retired_protocol)
    replacement_protocol = copy.deepcopy(protocol)
    replacement_protocol["dataset"]["splits"]["test"]["seed"] = 2026090791
    recorded_retired_seeds = {
        int(entry["retired_test_seed"])
        for entry in replacement_protocol["pre_freeze_history"]
    }
    for retired_seed in evidence.RETIRED_TEST_SEEDS:
        if retired_seed not in recorded_retired_seeds:
            replacement_protocol["pre_freeze_history"].append(
                {
                    "date": "2026-09-07",
                    "retired_test_seed": retired_seed,
                    "reason": "Exposed before the locked evaluation.",
                }
            )
    evidence._validate_final_protocol_seed(replacement_protocol)
    dependencies = {path.name for path in evidence._manuscript_dependencies(manuscript)}
    assert {
        "system_architecture.png",
        "sequence_enrollment.png",
        "sequence_matching.png",
        "foundry_tests.png",
        "slither_analysis.png",
        "Full_UI.png",
        "LLM_Decision.png",
        "fine_tuning_loss.png",
    } <= dependencies
    assert not any(Path(name).suffix in {".tex", ".json"} for name in dependencies)
    assert not re.search(
        r"\\(?:input|include)\s*\{|\\lstinputlisting(?:\[[^\]]*\])?\s*\{",
        manuscript,
    )

    bibliography = (evidence.JOURNAL_DIR / "cas-refs.bib").read_text(encoding="utf-8")
    assert not re.search(r"(?im)^\s*[\w-]+\s*=\s*\{\s*\}\s*,?", bibliography)
    assert not re.search(r"(?im)^\s*article[-_]?number\s*=", bibliography)
    assert not re.search(
        r"(?im)^\s*(?:doi\s*=\s*\{https?://|url\s*=\s*\{https?://(?:dx\.)?doi\.org/)",
        bibliography,
    )

    journal = tmp_path / "journal"
    manuscript_path, _ = _copy_manuscript_inputs(journal)
    manuscript_path.write_text(
        manuscript_path.read_text(encoding="utf-8").replace("\\toprule", "\\hline", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(evidence, "JOURNAL_DIR", journal)
    with pytest.raises(RuntimeError, match="booktabs"):
        evidence._validate_manuscript_source()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            evidence.EXPECTED_FUNDING,
            "This funding sentence was changed.",
            "funding statement",
        ),
        ("\\begin{figure}[!ht]", "\\begin{figure}[h]", "figure float"),
        (
            "Section~\\ref{sec:RelatedWork}",
            "Section~\\ref{sec:not-a-real-section}",
            "no matching label",
        ),
        (
            "\\bibliographystyle{Frontiers-Vancouver}",
            "\\bibliographystyle{Frontiers-Harvard}",
            "Vancouver bibliography style",
        ),
    ],
)
def test_manuscript_source_rejects_preserved_invariant_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    message: str,
) -> None:
    journal = tmp_path / "journal"
    manuscript, _ = _copy_manuscript_inputs(journal)
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(evidence, "JOURNAL_DIR", journal)
    with pytest.raises(RuntimeError, match=message):
        evidence._validate_manuscript_source()


def test_manuscript_source_rejects_changed_generated_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal"
    manuscript, _ = _copy_manuscript_inputs(journal)
    fragment = evidence.GENERATED_MANUSCRIPT_FILES[0].read_text(encoding="utf-8").strip()
    first_line = fragment.splitlines()[0]
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8").replace(first_line, first_line + " changed", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(evidence, "JOURNAL_DIR", journal)
    with pytest.raises(RuntimeError, match="Generated manuscript evidence"):
        evidence._validate_manuscript_source()
