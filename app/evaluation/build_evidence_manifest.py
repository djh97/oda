"""Build local provenance manifests without reading secrets or contacting services."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.policy import RETIRED_TEST_SEEDS


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
WORKSPACE_DIR = IMPLEMENTATION_DIR.parent
JOURNAL_DIR = WORKSPACE_DIR / "Frontiers_Medical_Technology_2026-09-06"
JOURNAL_DOCS_DIR = JOURNAL_DIR / "docs"
JOURNAL_ARCHIVE_DIR = JOURNAL_DIR / "archive"
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current"
MANUSCRIPT_STAGING_DIR = OUTPUT_DIR / "manuscript"
MANIFEST_PATH = OUTPUT_DIR / "evidence_manifest.json"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment_manifest.json"
PLACEHOLDER_MARKER = "% PLACEHOLDER:"
UI_CAPTURE_MANIFEST = "ui_capture_manifest.json"

GENERATED_MANUSCRIPT_FILES = (
    MANUSCRIPT_STAGING_DIR / "generated_tx_trace_rows.tex",
    MANUSCRIPT_STAGING_DIR / "generated_address_rows.tex",
    MANUSCRIPT_STAGING_DIR / "generated_primary_result_rows.tex",
    MANUSCRIPT_STAGING_DIR / "generated_model_summary.tex",
    MANUSCRIPT_STAGING_DIR / "generated_cost_rows.tex",
    MANUSCRIPT_STAGING_DIR / "generated_latency_rows.tex",
    MANUSCRIPT_STAGING_DIR / "generated_note_review_example.json",
)

UI_CAPTURE_FILES = (
    JOURNAL_DIR / "Full_UI.png",
    JOURNAL_DIR / "LLM_Decision.png",
)

LOSS_FIGURE_FILE = JOURNAL_DIR / "fine_tuning_loss.png"
LOSS_FIGURE_MANIFEST = OUTPUT_DIR / "model" / "fine_tuning_loss_manifest.json"

EXPECTED_TITLE = "Enhancing Organ Donation and Transplantation Workflows using Blockchain and LLMs"
EXPECTED_FUNDING = (
    "This research was funded by the Socio-Technical Systems Lab (STSL), "
    "Khalifa University of Science and Technology (KU-STSL)."
)
EXPECTED_AI_ACKNOWLEDGMENT = (
    "During the preparation of this work, the authors used ChatGPT in order to improve "
    "the readability and language. After using this tool, the authors reviewed and edited "
    "the content as needed and take full responsibility for the content of the published article."
)
EXPECTED_AUTHORS = (
    "Diana Hawashin\\,$^{1,*}$, Khaled Salah\\,$^{1}$, Hamda Al Breiki\\,$^{2}$, "
    "Raja Jayaraman\\,$^{3}$, Samer Ellahham\\,$^{4}$ and Ibrar Yaqoob\\,$^{5}$"
)
EXPECTED_ETHICS = (
    "This study used synthetic data and did not involve human participants, animals, "
    "or identifiable patient records."
)
RETIRED_TEST_SEED_STRINGS = tuple(str(seed) for seed in RETIRED_TEST_SEEDS)
REQUIRED_STANDALONE_SECTIONS = (
    "Introduction",
    "Related Work",
    "System Design",
    "System Implementation",
    "Testing and Validation",
    "System Evaluation",
    "Discussion",
    "Conclusion",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_DIR.resolve()).as_posix()


def _files_under(path: Path, pattern: str = "*") -> Iterable[Path]:
    if not path.exists():
        return ()
    return (
        candidate
        for candidate in path.rglob(pattern)
        if candidate.is_file()
        and ".venv" not in candidate.parts
        and "__pycache__" not in candidate.parts
        and candidate.name not in {".env", MANIFEST_PATH.name}
    )


def _manuscript_dependencies(manuscript: str) -> set[Path]:
    raw_paths: set[str] = set()
    for pattern in (
        r"\\input\{([^}]+)\}",
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        r"\\lstinputlisting(?:\[[^\]]*\])?\{([^}]+)\}",
        r"\\framedfitimage\{([^}]+)\}",
    ):
        raw_paths.update(re.findall(pattern, manuscript, flags=re.DOTALL))

    dependencies: set[Path] = set()
    for raw_path in raw_paths:
        if "#" in raw_path:
            continue
        candidate = Path(raw_path.strip())
        if candidate.is_absolute():
            raise RuntimeError(f"Manuscript dependency must use a relative path: {raw_path}")
        resolved = _inside(
            JOURNAL_DIR / candidate,
            JOURNAL_DIR,
            label="Manuscript dependency",
        )
        dependencies.add(resolved)
    return dependencies


def _validate_manuscript_source() -> dict[str, object]:
    manuscript_path = JOURNAL_DIR / "Manuscript.tex"
    bibliography_path = JOURNAL_DIR / "cas-refs.bib"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")
    errors: list[str] = []

    if "\\documentclass[utf8]{FrontiersinVancouver}" not in manuscript:
        errors.append("The Frontiers Vancouver document class is not selected")
    if (
        "\\linenumbers" not in manuscript
        or "\\usepackage[onehalfspacing]{setspace}" not in manuscript
    ):
        errors.append("Line numbering or the Frontiers template spacing is missing")

    title_match = re.search(r"\\title\[([^\]]+)\]\{([^}]+)\}", manuscript)
    if not title_match:
        errors.append("The manuscript title declaration is missing")
        running_title = ""
        full_title = ""
    else:
        running_title, full_title = (value.strip() for value in title_match.groups())
        if full_title != EXPECTED_TITLE:
            errors.append("The author-approved full title changed")
        if len(running_title.split()) > 5:
            errors.append("The running title exceeds five words")

    for section in REQUIRED_STANDALONE_SECTIONS:
        if f"\\section{{{section}}}" not in manuscript:
            errors.append(f"Required standalone section is missing: {section}")
    if re.search(r"\btaken together\b", manuscript, flags=re.IGNORECASE):
        errors.append("The author-rejected phrase 'Taken together' is present")

    introduction_match = re.search(
        r"\\section\{Introduction\}(.*?)\\section\{Related Work\}",
        manuscript,
        flags=re.DOTALL,
    )
    if not introduction_match:
        errors.append("The Introduction boundary could not be identified")
    else:
        for sentence in re.split(r"(?<=[.!?])\s+", introduction_match.group(1)):
            citation_count = sum(
                len([key for key in group.split(",") if key.strip()])
                for group in re.findall(r"\\cite\w*\{([^}]+)\}", sentence)
            )
            if citation_count > 2:
                errors.append("An Introduction sentence contains more than two references")
                break

    if manuscript.count(EXPECTED_FUNDING) != 1:
        errors.append("The exact author-approved funding statement is missing or duplicated")
    if manuscript.count(EXPECTED_AI_ACKNOWLEDGMENT) != 1:
        errors.append("The exact author-approved AI acknowledgment is missing or duplicated")
    if f"\\def\\Authors{{{EXPECTED_AUTHORS}}}" not in manuscript:
        errors.append("The author-approved author order or metadata changed")
    if "\\def\\corrAuthor{Diana Hawashin}" not in manuscript:
        errors.append("The corresponding author changed")
    if "\\def\\corrEmail{diana.jhawashin@ku.ac.ae}" not in manuscript:
        errors.append("The corresponding-author email changed")
    if (
        "\\textbf{Hamda Al Breiki}: Validation, Writing--review \\& editing."
        not in manuscript
    ):
        errors.append("Hamda Al Breiki's approved contribution is missing")
    if manuscript.count(EXPECTED_ETHICS) != 1:
        errors.append("The exact concise ethics statement is missing or duplicated")

    float_counts: dict[str, int] = {}
    for environment in ("figure", "table", "algorithm"):
        starts = re.findall(
            rf"\\begin\{{{environment}\}}(?:\[([^\]]*)\])?",
            manuscript,
        )
        float_counts[environment] = len(starts)
        if any(specifier != "!ht" for specifier in starts):
            errors.append(f"Every {environment} float must use [!ht]")

    figure_blocks = re.findall(
        r"\\begin\{figure\}\[!ht\](.*?)\\end\{figure\}",
        manuscript,
        flags=re.DOTALL,
    )
    table_blocks = re.findall(
        r"\\begin\{table\}\[!ht\](.*?)\\end\{table\}",
        manuscript,
        flags=re.DOTALL,
    )
    figure_count = sum("\\caption{" in block for block in figure_blocks)
    table_count = sum("\\caption{" in block for block in table_blocks)
    listing_count = manuscript.count("\\begin{lstlisting}")
    if figure_count + table_count > 15:
        errors.append("The combined Frontiers figure/table limit of 15 is exceeded")
    for index, block in enumerate(table_blocks, start=1):
        if any(rule not in block for rule in ("\\toprule", "\\midrule", "\\bottomrule")):
            errors.append(f"Table {index} does not use complete booktabs rules")
        if re.search(r"\\(?:hline|cline|vline)\b", block):
            errors.append(f"Table {index} contains a non-booktabs rule")

    count_match = re.search(
        r"Main-text word count:\s*([\d,]+)\s*\\quad\s*Figures:\s*(\d+)\s*"
        r"\\quad\s*Tables:\s*(\d+)",
        manuscript,
    )
    if not count_match:
        errors.append("The required first-page counts are missing")
        reported_word_count = None
    else:
        reported_word_count = int(count_match.group(1).replace(",", ""))
        reported_counts = tuple(int(value) for value in count_match.groups()[1:])
        if reported_counts != (figure_count, table_count):
            errors.append("The first-page figure or table count is stale")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        manuscript,
        flags=re.DOTALL,
    )
    abstract_word_count = None
    keyword_count = None
    if not abstract_match:
        errors.append("The abstract environment is missing")
    else:
        abstract = abstract_match.group(1)
        if re.search(r"\\cite\w*\{", abstract):
            errors.append("The abstract contains a citation")
        if "\\noindent\\textbf{Background.}" in abstract:
            errors.append("The first abstract heading must not use \\noindent")
        for heading in ("Methods", "Results", "Conclusion"):
            if f"\\noindent\\textbf{{{heading}.}}" not in abstract:
                errors.append(
                    f"The {heading} abstract heading must begin with \\noindent"
                )
        keyword_match = re.search(r"\\section\{Keywords:\}\s*([^}\n]+)", abstract)
        if not keyword_match:
            errors.append("The keyword list is missing")
        else:
            keywords = [value.strip() for value in keyword_match.group(1).split(",")]
            keyword_count = len([value for value in keywords if value])
            if keyword_count < 5 or keyword_count > 8:
                errors.append("Frontiers requires five to eight keywords")
        abstract_body = re.sub(r"\\keyFont\{.*", "", abstract, flags=re.DOTALL)
        abstract_body = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", abstract_body)
        abstract_body = re.sub(r"\\[A-Za-z]+", " ", abstract_body)
        abstract_body = re.sub(r"[^A-Za-z0-9'-]+", " ", abstract_body)
        abstract_word_count = len(abstract_body.split())
        if abstract_word_count > 350:
            errors.append("The abstract exceeds 350 words")

    labels = re.findall(r"\\label\{([^}]+)\}", manuscript)
    labels.extend(re.findall(r"\blabel\s*=\s*\{([^}]+)\}", manuscript))
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        errors.append("Duplicate LaTeX labels: " + ", ".join(duplicate_labels))

    referenced_labels = {
        label.strip()
        for group in re.findall(
            r"\\(?:auto|page|eq)?ref\{([^}]+)\}",
            manuscript,
        )
        for label in group.split(",")
        if label.strip()
    }
    missing_labels = sorted(referenced_labels.difference(labels))
    if missing_labels:
        errors.append("LaTeX references have no matching label: " + ", ".join(missing_labels))
    unreferenced_displays = sorted(
        label
        for label in labels
        if label.startswith(("fig:", "tab:")) and label not in referenced_labels
    )
    if unreferenced_displays:
        errors.append(
            "Figures or tables are not cited in the manuscript text: "
            + ", ".join(unreferenced_displays)
        )

    if re.search(
        r"\\(?:input|include)\s*\{|\\lstinputlisting(?:\[[^\]]*\])?\s*\{",
        manuscript,
    ):
        errors.append("External manuscript content imports are prohibited")
    for generated_path in GENERATED_MANUSCRIPT_FILES:
        if not generated_path.is_file():
            errors.append(f"Generated manuscript evidence is missing: {generated_path.name}")
            continue
        generated_content = generated_path.read_text(encoding="utf-8").strip()
        if not generated_content or manuscript.count(generated_content) != 1:
            errors.append(
                "Generated manuscript evidence must occur exactly once: "
                + generated_path.name
            )

    if manuscript.count("\\bibliographystyle{Frontiers-Vancouver}") != 1:
        errors.append("The Frontiers Vancouver bibliography style is missing or duplicated")
    if manuscript.count("\\bibliography{cas-refs}") != 1:
        errors.append("The cas-refs bibliography command is missing or duplicated")

    bibliography_key_list = re.findall(
        r"@\w+\s*\{\s*([^,\s]+)\s*,",
        bibliography,
        flags=re.IGNORECASE,
    )
    duplicate_bibliography_keys = sorted(
        key for key, count in Counter(bibliography_key_list).items() if count > 1
    )
    if duplicate_bibliography_keys:
        errors.append(
            "Duplicate BibTeX keys: " + ", ".join(duplicate_bibliography_keys)
        )
    bibliography_keys = set(bibliography_key_list)
    if re.search(r"(?im)^\s*[\w-]+\s*=\s*\{\s*\}\s*,?", bibliography):
        errors.append("The bibliography contains an empty field")
    if re.search(r"(?im)^\s*article[-_]?number\s*=", bibliography):
        errors.append("Use the Frontiers-supported BibTeX eid field for article numbers")
    if re.search(
        r"(?im)^\s*(?:doi\s*=\s*\{https?://|url\s*=\s*\{https?://(?:dx\.)?doi\.org/)",
        bibliography,
    ):
        errors.append("The bibliography contains a DOI-form URL instead of a bare DOI field")
    citation_keys = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript)
        for key in group.split(",")
        if key.strip()
    }
    missing_citations = sorted(citation_keys.difference(bibliography_keys))
    if missing_citations:
        errors.append("Citation keys missing from cas-refs.bib: " + ", ".join(missing_citations))

    if errors:
        raise RuntimeError("Manuscript source validation failed: " + "; ".join(errors))
    return {
        "title": full_title,
        "running_title": running_title,
        "running_title_words": len(running_title.split()),
        "abstract_words": abstract_word_count,
        "keywords": keyword_count,
        "reported_main_text_words": reported_word_count,
        "figures": figure_count,
        "tables": table_count,
        "listings": listing_count,
        "combined_figures_tables": figure_count + table_count,
        "float_environments": float_counts,
        "labels": len(labels),
        "citations": len(citation_keys),
        "bibliography_entries": len(bibliography_keys),
    }


def _validate_final_manuscript_source(manuscript: str) -> None:
    retired = [seed for seed in RETIRED_TEST_SEED_STRINGS if seed in manuscript]
    if retired:
        raise RuntimeError(
            "Manuscript still names retired test seed(s): " + ", ".join(retired)
        )

    results_match = re.search(
        r"\\section\{Testing and Validation\}(.*?)\\section\{Discussion\}",
        manuscript,
        flags=re.DOTALL,
    )
    if not results_match:
        raise RuntimeError("The field-equivalent Results boundary could not be identified")
    if "\\footnote{" in results_match.group(1):
        raise RuntimeError(
            "The field-equivalent Results sections still contain a footnote"
        )


def _validate_final_protocol_seed(protocol: Mapping[str, object]) -> None:
    dataset = _record_map(protocol.get("dataset"), label="Protocol dataset")
    splits = _record_map(dataset.get("splits"), label="Protocol splits")
    test = _record_map(splits.get("test"), label="Protocol test split")
    active_seed_value = test.get("seed")
    if active_seed_value is None:
        raise RuntimeError("The protocol does not define an active test seed")
    active_seed = str(active_seed_value)
    if active_seed in RETIRED_TEST_SEED_STRINGS:
        raise RuntimeError(f"The active protocol still uses retired test seed {active_seed}")

    history = protocol.get("pre_freeze_history")
    if not isinstance(history, list):
        raise RuntimeError("The protocol does not retain its pre-freeze seed history")
    recorded = {
        str(entry.get("retired_test_seed"))
        for entry in history
        if isinstance(entry, Mapping) and entry.get("retired_test_seed") is not None
    }
    omitted = sorted(set(RETIRED_TEST_SEED_STRINGS).difference(recorded))
    if omitted:
        raise RuntimeError(
            "The protocol history omits retired test seed(s): " + ", ".join(omitted)
        )


def _artifact_paths() -> list[Path]:
    paths: set[Path] = {
        IMPLEMENTATION_DIR / "README.md",
        IMPLEMENTATION_DIR / "LICENSE",
        IMPLEMENTATION_DIR / ".gitignore",
        APP_DIR / ".env.example",
        APP_DIR / "requirements.txt",
        APP_DIR / "requirements-dev.txt",
        APP_DIR / "requirements-lock.txt",
        APP_DIR / "requirements-ml.txt",
        IMPLEMENTATION_DIR / "smart-contracts" / "foundry.toml",
        IMPLEMENTATION_DIR / "smart-contracts" / "foundry.lock",
        IMPLEMENTATION_DIR / "smart-contracts" / "requirements-slither.txt",
        JOURNAL_DIR / "Manuscript.tex",
        JOURNAL_ARCHIVE_DIR / "manuscript_backups" / "Manuscript.pre-redesign-2026-09-07.tex",
        JOURNAL_DIR / "cas-refs.bib",
        JOURNAL_DIR / "FrontiersinVancouver.cls",
        JOURNAL_DIR / "Frontiers-Vancouver.bst",
        JOURNAL_DIR / "logo1.eps",
        JOURNAL_DOCS_DIR / "STUDY_PROTOCOL.md",
        JOURNAL_DOCS_DIR / "CLAIM_EVIDENCE_MATRIX.md",
        JOURNAL_DOCS_DIR / "SUBMISSION_READINESS_PLAN.md",
        JOURNAL_DOCS_DIR / "MANUSCRIPT_ISSUES.md",
        JOURNAL_DOCS_DIR / "AUTHOR_DECISIONS_REQUIRED.md",
        JOURNAL_DOCS_DIR / "OVERLEAF_UPLOAD_MANIFEST.md",
        JOURNAL_DOCS_DIR / "Contribution_to_the_field.txt",
        JOURNAL_DIR / "system_architecture.png",
        JOURNAL_DIR / "system_architecture.svg",
        JOURNAL_DIR / "sequence_enrollment.png",
        JOURNAL_DIR / "sequence_enrollment.svg",
        JOURNAL_DIR / "sequence_matching.png",
        JOURNAL_DIR / "sequence_matching.svg",
        JOURNAL_DIR / "foundry_tests.png",
        JOURNAL_DIR / "foundry_tests[Original].png",
        JOURNAL_DIR / "slither_analysis.png",
        JOURNAL_DIR / "slither_analysis[Original].png",
        *UI_CAPTURE_FILES,
        LOSS_FIGURE_FILE,
        *GENERATED_MANUSCRIPT_FILES,
    }
    for directory, pattern in (
        (APP_DIR / "protocols", "*.json"),
        (APP_DIR / "seed-data", "*.json"),
        (APP_DIR / "src", "*.py"),
        (APP_DIR / "evaluation", "*.py"),
        (APP_DIR / "tests", "*.py"),
        (IMPLEMENTATION_DIR / "datasets" / "synthetic-v1", "*"),
        (IMPLEMENTATION_DIR / "integration", "*"),
        (IMPLEMENTATION_DIR / "smart-contracts" / "src", "*.sol"),
        (IMPLEMENTATION_DIR / "smart-contracts" / "test", "*.sol"),
        (IMPLEMENTATION_DIR / "smart-contracts" / "script", "*.sol"),
        (IMPLEMENTATION_DIR / "smart-contracts" / "security", "*"),
        (IMPLEMENTATION_DIR / "smart-contracts" / "test-output", "*"),
        (OUTPUT_DIR, "*"),
    ):
        paths.update(_files_under(directory, pattern))
    return sorted((path for path in paths if path.is_file()), key=_relative)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read required final JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Required final JSON artifact is not an object: {path}")
    return value


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} leaves its permitted directory: {path}") from exc
    return resolved


def _record_map(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{label} is missing or empty")
    return value


def _verify_record_map(
    value: object,
    *,
    base_dir: Path,
    permitted_root: Path,
    label: str,
) -> set[Path]:
    records = _record_map(value, label=label)
    verified: set[Path] = set()
    for raw_path, raw_metadata in records.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeError(f"{label} contains an invalid path")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        candidate = _inside(candidate, permitted_root, label=label)
        if candidate in verified:
            raise RuntimeError(f"{label} contains the same resolved path more than once")
        if not candidate.is_file():
            raise RuntimeError(f"{label} references a missing file: {candidate}")
        if not isinstance(raw_metadata, Mapping):
            raise RuntimeError(f"{label} has malformed metadata for {candidate}")
        try:
            recorded_bytes = int(raw_metadata.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} has an invalid byte count for {candidate}") from exc
        if recorded_bytes != candidate.stat().st_size:
            raise RuntimeError(f"{label} byte count is stale for {candidate}")
        if str(raw_metadata.get("sha256", "")) != _sha256(candidate):
            raise RuntimeError(f"{label} hash is stale for {candidate}")
        verified.add(candidate)
    return verified


def _resolve_manifest_run(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError("A final manifest does not identify its canonical run")
    path = Path(raw)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
    return _inside(
        path,
        OUTPUT_DIR / "full_workflow",
        label="Canonical run",
    )


def _validate_artifact_manifest(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "artifact_manifest.json")
    verified = _verify_record_map(
        manifest.get("artifacts"),
        base_dir=run_dir,
        permitted_root=run_dir,
        label="Canonical artifact manifest",
    )
    required = {
        run_dir / name
        for name in (
            "actor_addresses.json",
            "command.json",
            "contract_build.json",
            "deployment_verification.json",
            "encrypted_profile_cids.json",
            "environment_manifest.json",
            "model_and_guard_record.json",
            "run_summary.json",
            "seed_summary.json",
            "transaction_manifest.csv",
            "transaction_stage_summary.csv",
            "transaction_summary.csv",
            "ui_evidence_snapshot.json",
        )
    }
    missing = sorted(path.name for path in required - verified)
    if missing:
        raise RuntimeError("Canonical artifact manifest omits core files: " + ", ".join(missing))


def _validate_workflow_completions(run_dir: Path) -> None:
    from evaluation import paper_full_workflow as workflow

    root = OUTPUT_DIR / "full_workflow"
    completed_full_runs: list[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        completion_path = directory / workflow.COMPLETION_FILENAME
        if not completion_path.is_file():
            continue
        completion = _read_json(completion_path)
        if (
            completion.get("schema_version") != "1.0"
            or completion.get("status") != "completed"
            or completion.get("mode") not in {"full", "seed_only"}
            or not (directory / "artifact_manifest.json").is_file()
            or completion.get("artifact_manifest_sha256")
            != _sha256(directory / "artifact_manifest.json")
        ):
            raise RuntimeError(f"Workflow completion record is malformed or stale: {completion_path}")
        command = _read_json(directory / "command.json")
        if command.get("mode") != completion["mode"]:
            raise RuntimeError(f"Workflow completion mode differs from its command: {completion_path}")
        if completion["mode"] == "full":
            if not (directory / "run_summary.json").is_file():
                raise RuntimeError(f"Completed workflow has no run summary: {completion_path}")
            if completion.get("run_summary_sha256") != _sha256(directory / "run_summary.json"):
                raise RuntimeError(f"Workflow completion summary hash is stale: {completion_path}")
            completed_full_runs.append(directory.resolve())
    if completed_full_runs != [run_dir.resolve()]:
        raise RuntimeError(
            "The evidence freeze requires exactly one completed full-workflow run, matching the canonical pointer"
        )


def _validate_manuscript_table_manifest(run_dir: Path) -> None:
    from evaluation import build_manuscript_tables as tables

    summary = _read_json(run_dir / "run_summary.json")
    tables._validate_deployment_verification(run_dir, summary)
    tables._validate_canonical_decision(run_dir, summary)
    tables._validate_canonical_transactions(
        summary,
        tables._read_csv(run_dir / "transaction_manifest.csv"),
        _read_json(run_dir / "actor_addresses.json"),
    )
    analysis_summaries = tables._validate_analysis_summaries()
    pretest_sources = tables._validate_pretest_artifacts(analysis_summaries)
    fine_tuning_path = tables.LOCAL_LORA_TRAINING_PATH
    fine_tuning_sources = tables._fine_tuning_source_paths(_read_json(fine_tuning_path))
    _, cost_sources = tables._cost_rows(run_dir, summary)
    _, latency_sources = tables._latency_rows(run_dir)

    manifest = _read_json(run_dir / "manuscript_table_manifest.json")
    if manifest.get("protocol_id") != tables.PROTOCOL["protocol_id"]:
        raise RuntimeError("Manuscript table manifest belongs to a different protocol")
    if _resolve_manifest_run(manifest.get("canonical_run")) != run_dir.resolve():
        raise RuntimeError("Manuscript table manifest belongs to a different canonical run")

    sources = _verify_record_map(
        manifest.get("sources"),
        base_dir=WORKSPACE_DIR,
        permitted_root=WORKSPACE_DIR,
        label="Manuscript table source manifest",
    )
    expected_sources = {
        tables.SCRIPT_PATH,
        tables.DEFAULT_PROTOCOL_PATH,
        run_dir / "run_summary.json",
        run_dir / "contract_build.json",
        run_dir / "deployment_verification.json",
        run_dir / "model_and_guard_record.json",
        run_dir / "transaction_manifest.csv",
        run_dir / "actor_addresses.json",
        *cost_sources,
        *latency_sources,
        fine_tuning_path,
        *fine_tuning_sources,
        *pretest_sources,
        tables.EVALUATION_DIR / "test_lock.json",
        tables.MODEL_API_COST_PATH,
        *(tables.EVALUATION_DIR / f"{condition}_raw.jsonl" for condition, _ in tables.CONDITIONS),
        *(tables.EVALUATION_DIR / f"{condition}_config.json" for condition, _ in tables.CONDITIONS),
        *(tables.EVALUATION_DIR / f"{condition}_attempts.jsonl" for condition, _ in tables.CONDITIONS),
        *(tables.EVALUATION_DIR / f"{condition}_summary.json" for condition, _ in tables.CONDITIONS),
        *(
            tables.EVALUATION_DIR / f"{left}_vs_{right}_comparison.json"
            for left, right in tables.PAIRED_COMPARISONS
        ),
    }
    expected_sources = {path.resolve() for path in expected_sources}
    if sources != expected_sources:
        missing = sorted(_relative(path) for path in expected_sources - sources)
        extra = sorted(_relative(path) for path in sources - expected_sources)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise RuntimeError("Manuscript table source manifest is incomplete: " + "; ".join(details))

    generated = _verify_record_map(
        manifest.get("generated"),
        base_dir=WORKSPACE_DIR,
        permitted_root=MANUSCRIPT_STAGING_DIR,
        label="Generated manuscript manifest",
    )
    expected_generated = {path.resolve() for path in GENERATED_MANUSCRIPT_FILES}
    if generated != expected_generated:
        raise RuntimeError("Generated manuscript manifest does not contain the exact expected files")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"UI capture is not a valid PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def _validate_ui_capture_manifest(run_dir: Path) -> None:
    manifest = _read_json(run_dir / UI_CAPTURE_MANIFEST)
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("UI capture manifest has an unsupported schema version")
    if _resolve_manifest_run(manifest.get("canonical_run")) != run_dir.resolve():
        raise RuntimeError("UI captures belong to a different canonical run")

    expected_inputs = {
        OUTPUT_DIR / "latest_full_workflow.json",
        run_dir / "run_summary.json",
        run_dir / "ui_evidence_snapshot.json",
        APP_DIR / "templates" / "index.html",
        APP_DIR / "src" / "main.py",
        APP_DIR / "src" / "evidence_view.py",
        APP_DIR / "evaluation" / "project_environment.py",
        APP_DIR / "evaluation" / "capture_ui_evidence.py",
    }
    inputs = _verify_record_map(
        manifest.get("inputs"),
        base_dir=WORKSPACE_DIR,
        permitted_root=WORKSPACE_DIR,
        label="UI capture input manifest",
    )
    if inputs != {path.resolve() for path in expected_inputs}:
        raise RuntimeError("UI capture manifest does not contain the exact expected inputs")

    captures = _record_map(manifest.get("captures"), label="UI capture output manifest")
    verified = _verify_record_map(
        captures,
        base_dir=WORKSPACE_DIR,
        permitted_root=JOURNAL_DIR,
        label="UI capture output manifest",
    )
    if verified != {path.resolve() for path in UI_CAPTURE_FILES}:
        raise RuntimeError("UI capture manifest does not contain the exact expected images")
    expected_modes = {"Full_UI.png": "full", "LLM_Decision.png": "decision"}
    for raw_path, raw_metadata in captures.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = WORKSPACE_DIR / path
        path = path.resolve()
        if not isinstance(raw_metadata, Mapping):
            raise RuntimeError(f"UI capture metadata is malformed for {path.name}")
        dimensions = _png_dimensions(path)
        if raw_metadata.get("mode") != expected_modes[path.name]:
            raise RuntimeError(f"UI capture mode is incorrect for {path.name}")
        if [*dimensions] != raw_metadata.get("dimensions_px"):
            raise RuntimeError(f"UI capture dimensions are stale for {path.name}")


def _validate_loss_figure_manifest() -> None:
    from evaluation import build_fine_tuning_loss_figure as loss

    manifest = _read_json(LOSS_FIGURE_MANIFEST)
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("Fine-tuning loss-figure manifest has an unsupported schema version")
    if manifest.get("env_file_read") is not False or manifest.get("network_contacted") is not False:
        raise RuntimeError("Fine-tuning loss figure did not preserve its offline boundary")

    state = _read_json(loss.STATE_PATH)
    loss._validate_fine_tuning_state(state)
    local_training = isinstance(state.get("training_result"), Mapping)
    archived = state.get("archived_result_files", [])
    if not isinstance(archived, list) or (not local_training and not archived):
        raise RuntimeError("Fine-tuning loss figure has no retained loss evidence")
    archived_paths: set[Path] = set()
    for record in archived:
        if not isinstance(record, Mapping):
            raise RuntimeError("Fine-tuning loss figure has malformed provider result metadata")
        try:
            archived_paths.add(
                loss.resolve_recorded_path(
                    record.get("local_path"),
                    permitted_root=loss.MODEL_DIR / "result_files",
                )
            )
        except ValueError as exc:
            raise RuntimeError("Fine-tuning loss figure has an invalid provider result path") from exc

    inputs = _verify_record_map(
        manifest.get("inputs"),
        base_dir=WORKSPACE_DIR,
        permitted_root=WORKSPACE_DIR,
        label="Fine-tuning loss-figure input manifest",
    )
    expected_inputs = {
        loss.STATE_PATH.resolve(),
        loss.DEFAULT_PROTOCOL_PATH.resolve(),
        loss.SCRIPT_PATH.resolve(),
        *archived_paths,
    }
    if inputs != expected_inputs:
        raise RuntimeError("Fine-tuning loss-figure manifest does not contain the exact inputs")

    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise RuntimeError("Fine-tuning loss-figure output record is malformed")
    raw_output_path = output.get("path")
    candidate = Path(str(raw_output_path or ""))
    if not candidate.is_absolute():
        candidate = WORKSPACE_DIR / candidate
    candidate = _inside(candidate, JOURNAL_DIR, label="Fine-tuning loss figure")
    if candidate != LOSS_FIGURE_FILE.resolve() or not candidate.is_file():
        raise RuntimeError("Fine-tuning loss-figure output path is missing or incorrect")
    if output.get("bytes") != candidate.stat().st_size or output.get("sha256") != _sha256(candidate):
        raise RuntimeError("Fine-tuning loss-figure output record is stale")
    dimensions = _png_dimensions(candidate)
    if dimensions != (loss.WIDTH_PX, loss.HEIGHT_PX):
        raise RuntimeError("Fine-tuning loss figure has unexpected pixel dimensions")
    if output.get("dimensions_px") != [loss.WIDTH_PX, loss.HEIGHT_PX]:
        raise RuntimeError("Fine-tuning loss-figure dimensions are stale")
    try:
        effective_dpi = float(output.get("effective_dpi_at_180_mm", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Fine-tuning loss figure has invalid effective resolution") from exc
    if effective_dpi < 300:
        raise RuntimeError("Fine-tuning loss figure is below 300 dpi at 180 mm")

    selected = Path(str(manifest.get("selected_metrics_file", "")))
    if not selected.is_absolute():
        selected = WORKSPACE_DIR / selected
    expected_selected, series = loss._find_metrics_file(state)
    if local_training:
        selected = _inside(selected, loss.MODEL_DIR, label="Loss metrics file")
        if selected != loss.STATE_PATH.resolve():
            raise RuntimeError("Fine-tuning loss figure used the wrong local training record")
    else:
        selected = _inside(selected, loss.MODEL_DIR / "result_files", label="Loss metrics file")
        if selected not in archived_paths:
            raise RuntimeError("Fine-tuning loss figure used an unarchived metrics file")
    if selected != expected_selected.resolve():
        raise RuntimeError("Fine-tuning loss figure selected a different loss source")
    if manifest.get("columns") != series["columns"]:
        raise RuntimeError("Fine-tuning loss-figure column metadata is stale")
    if manifest.get("axis") != loss._axis_metadata(series):
        raise RuntimeError("Fine-tuning loss-figure axis metadata is stale")
    expected_observations = {
        "training": len(series["training"]),
        "validation": len(series["validation"]),
        "minimum_step": int(min(point[0] for point in series["training"] + series["validation"])),
        "maximum_step": int(max(point[0] for point in series["training"] + series["validation"])),
    }
    if manifest.get("observations") != expected_observations:
        raise RuntimeError("Fine-tuning loss-figure observation counts are stale")


def _validate_publication_figure_assets() -> None:
    from evaluation.validate_figure_assets import validate

    report = validate(JOURNAL_DIR)
    failed = [
        name
        for name, result in report["figures"].items()
        if not result.get("all_passed")
    ]
    if report.get("all_passed") is not True or failed:
        details = ", ".join(failed) if failed else "unknown figure"
        raise RuntimeError(
            "Publication-scale figure validation failed for: " + details
        )


def _validate_software_evidence(run_dir: Path, manuscript: str) -> None:
    from evaluation import refresh_software_evidence as software

    manifest = _read_json(software.MANIFEST_PATH)
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("Software verification manifest has an unsupported schema version")
    if manifest.get("env_file_read") is not False or manifest.get("network_contacted") is not False:
        raise RuntimeError("Software verification did not preserve its offline secret boundary")

    sources = _verify_record_map(
        manifest.get("sources"),
        base_dir=WORKSPACE_DIR,
        permitted_root=WORKSPACE_DIR,
        label="Software verification source manifest",
    )
    expected_sources = {path.resolve() for path in software.source_paths()}
    if sources != expected_sources:
        raise RuntimeError("Software verification manifest does not contain the exact tested sources")
    outputs = _verify_record_map(
        manifest.get("outputs"),
        base_dir=WORKSPACE_DIR,
        permitted_root=WORKSPACE_DIR,
        label="Software verification output manifest",
    )
    expected_outputs = {path.resolve() for path in software.output_paths()}
    if outputs != expected_outputs:
        raise RuntimeError("Software verification manifest does not contain the exact retained outputs")

    terminal_captures = manifest.get("terminal_captures")
    if not isinstance(terminal_captures, Mapping):
        raise RuntimeError("Software verification manifest has no terminal-capture provenance")
    expected_capture_sources = {
        "foundry": software.FOUNDRY_CAPTURE_SOURCE_PATH.resolve(),
        "slither": software.SLITHER_CAPTURE_SOURCE_PATH.resolve(),
    }
    for name, expected_source in expected_capture_sources.items():
        capture = terminal_captures.get(name)
        if not isinstance(capture, Mapping):
            raise RuntimeError(f"Software verification omits the {name} terminal capture")
        source = capture.get("source")
        if not isinstance(source, Mapping):
            raise RuntimeError(f"The {name} terminal-capture source record is malformed")
        verified = _verify_record_map(
            {source.get("path"): {"bytes": source.get("bytes"), "sha256": source.get("sha256")}},
            base_dir=WORKSPACE_DIR,
            permitted_root=JOURNAL_DIR,
            label=f"{name.title()} terminal-capture source",
        )
        if verified != {expected_source}:
            raise RuntimeError(f"The {name} terminal-capture source changed")

    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise RuntimeError("Software verification manifest has no result summary")
    python_result = results.get("python")
    foundry_result = results.get("foundry")
    gas_result = results.get("foundry_gas")
    slither_result = results.get("slither")
    slither_human_result = results.get("slither_human_summary")
    if not all(isinstance(item, Mapping) for item in (
        python_result,
        foundry_result,
        gas_result,
        slither_result,
        slither_human_result,
    )):
        raise RuntimeError("Software verification result records are malformed")

    parsed_python = software._parse_pytest(software.PYTEST_LOG_PATH.read_text(encoding="utf-8"))
    parsed_foundry = software._parse_foundry(software.FOUNDRY_LOG_PATH.read_text(encoding="utf-8"))
    parsed_gas = software._parse_foundry(software.FOUNDRY_GAS_LOG_PATH.read_text(encoding="utf-8"))
    slither_json = _read_json(software.SLITHER_JSON_PATH)
    parsed_slither = software._parse_slither(
        software.SLITHER_LOG_PATH.read_text(encoding="utf-8"),
        slither_json,
    )
    parsed_slither_human = software._parse_slither_human_summary(
        software.SLITHER_HUMAN_LOG_PATH.read_text(encoding="utf-8")
    )
    if (
        dict(python_result) != parsed_python
        or dict(foundry_result) != parsed_foundry
        or dict(gas_result) != parsed_gas
        or dict(slither_result) != parsed_slither
        or dict(slither_human_result) != parsed_slither_human
    ):
        raise RuntimeError("Software verification summaries do not match their retained logs")
    if (
        int(parsed_python["passed"]) < 1
        or int(parsed_python["failed"]) != 0
        or int(parsed_foundry["passed"]) != int(parsed_foundry["total"])
        or int(parsed_foundry["failed"]) != 0
        or int(parsed_foundry["skipped"]) != 0
        or parsed_gas != parsed_foundry
        or int(parsed_slither["detectors_executed"]) < 1
        or int(parsed_slither["findings"]) != 0
        or any(
            int(parsed_slither_human[key]) != 0
            for key in (
                "optimization_issues",
                "informational_issues",
                "low_issues",
                "medium_issues",
                "high_issues",
            )
        )
    ):
        raise RuntimeError("Final software verification is not clean")

    summary = _read_json(run_dir / "run_summary.json")
    contract_source = software.SMART_CONTRACTS_DIR / "src" / "TransplantManagement.sol"
    from evaluation.paper_full_workflow import _artifact_bytecode

    contract_runtime_sha256 = hashlib.sha256(
        _artifact_bytecode(
            _read_json(software.CONTRACT_ARTIFACT_PATH),
            "deployedBytecode",
        )
    ).hexdigest()
    if (
        manifest.get("contract_source_sha256") != _sha256(contract_source)
        or summary.get("contract_source_sha256") != _sha256(contract_source)
        or manifest.get("contract_runtime_sha256") != contract_runtime_sha256
        or summary.get("deployed_runtime_sha256") != contract_runtime_sha256
    ):
        raise RuntimeError("Software verification and canonical deployment use different contract artifacts")

    reported_counts = (
        (int(parsed_python["passed"]), "Python tests"),
        (int(parsed_foundry["passed"]), "Foundry tests"),
        (int(parsed_slither["detectors_executed"]), "detectors"),
    )
    for count, label in reported_counts:
        if not re.search(rf"\b{count}\s+{re.escape(label)}\b", manuscript, re.IGNORECASE):
            raise RuntimeError(f"The manuscript does not report the verified {count} {label}")


def _require_final_artifacts() -> None:
    from evaluation import build_manuscript_tables as tables

    required = {
        IMPLEMENTATION_DIR / "LICENSE",
        JOURNAL_DIR / "Manuscript.tex",
        JOURNAL_DIR / "cas-refs.bib",
        JOURNAL_DIR / "FrontiersinVancouver.cls",
        JOURNAL_DIR / "Frontiers-Vancouver.bst",
        JOURNAL_DIR / "logo1.eps",
        *UI_CAPTURE_FILES,
        LOSS_FIGURE_FILE,
        LOSS_FIGURE_MANIFEST,
        OUTPUT_DIR / "model" / "local_lora_training.json",
        OUTPUT_DIR / "evaluation" / "test_lock.json",
        OUTPUT_DIR / "software" / "software_verification_manifest.json",
        OUTPUT_DIR / "latest_full_workflow.json",
        *GENERATED_MANUSCRIPT_FILES,
    }
    for condition, _label in tables.CONDITIONS:
        required.add(OUTPUT_DIR / "evaluation" / f"{condition}_summary.json")
    for left, right in tables.PAIRED_COMPARISONS:
        required.add(OUTPUT_DIR / "evaluation" / f"{left}_vs_{right}_comparison.json")

    manuscript = (JOURNAL_DIR / "Manuscript.tex").read_text(encoding="utf-8")
    required.update(_manuscript_dependencies(manuscript))
    missing = sorted(_relative(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError("Missing final artifacts: " + ", ".join(missing))

    stale = [
        _relative(path)
        for path in GENERATED_MANUSCRIPT_FILES
        if PLACEHOLDER_MARKER in path.read_text(encoding="utf-8")
    ]
    if stale:
        raise RuntimeError("Generated manuscript files still contain placeholders: " + ", ".join(stale))

    gates = (
        "FINAL-EVIDENCE-GATE",
        "FINAL-METADATA-GATE",
    )
    remaining_gates = [gate for gate in gates if gate in manuscript]
    if remaining_gates:
        raise RuntimeError("Manuscript still contains unresolved final gates: " + ", ".join(remaining_gates))
    _validate_final_manuscript_source(manuscript)

    from evaluation.audit_active_artifacts import audit
    from evaluation.synthetic_dataset import validate_existing
    from src.evidence_view import EvidenceViewError, load_latest_evidence

    _validate_final_protocol_seed(
        _read_json(APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json")
    )

    dataset_report = validate_existing()
    if dataset_report.get("valid") is not True:
        errors = dataset_report.get("errors")
        details = ", ".join(str(item) for item in errors) if isinstance(errors, list) else "unknown error"
        raise RuntimeError("The frozen synthetic dataset is invalid: " + details)

    security = audit()
    if security.get("env_file_read") is not False or int(security.get("finding_count", -1)) != 0:
        raise RuntimeError("The final active-artifact credential scan is missing or has findings")

    pointer = _read_json(OUTPUT_DIR / "latest_full_workflow.json")
    run_dir = _inside(
        APP_DIR / str(pointer.get("run_directory", "")),
        OUTPUT_DIR / "full_workflow",
        label="Latest full-workflow pointer",
    )
    for name in (
        "run_summary.json",
        "artifact_manifest.json",
        "completion.json",
        "deployment_verification.json",
        "manuscript_table_manifest.json",
        "ui_evidence_snapshot.json",
        UI_CAPTURE_MANIFEST,
    ):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"The canonical full-workflow run is missing {name}")
    if pointer.get("run_summary_sha256") != _sha256(run_dir / "run_summary.json"):
        raise RuntimeError("The latest full-workflow pointer has a stale run-summary hash")
    if pointer.get("ui_evidence_snapshot_sha256") != _sha256(run_dir / "ui_evidence_snapshot.json"):
        raise RuntimeError("The latest full-workflow pointer has a stale UI-snapshot hash")
    try:
        load_latest_evidence(app_dir=APP_DIR, current_output_dir=OUTPUT_DIR)
    except EvidenceViewError as exc:
        raise RuntimeError("The canonical UI evidence fails path, hash, or schema validation") from exc

    _validate_artifact_manifest(run_dir)
    _validate_workflow_completions(run_dir)
    _validate_manuscript_table_manifest(run_dir)
    _validate_ui_capture_manifest(run_dir)
    _validate_loss_figure_manifest()
    _validate_publication_figure_assets()
    _validate_software_evidence(run_dir, manuscript)


def _command_version(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    output = (result.stdout or result.stderr).strip()
    return {
        "available": result.returncode == 0,
        "return_code": result.returncode,
        "version_output": output,
    }


def build(*, require_final: bool = False) -> tuple[dict[str, object], dict[str, object]]:
    manuscript_validation = _validate_manuscript_source()
    if require_final:
        _require_final_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()
    package_names = (
        "cryptography",
        "fastapi",
        "openai",
        "pydantic",
        "pytest",
        "python-dotenv",
        "requests",
        "scikit-learn",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "web3",
    )
    slither = IMPLEMENTATION_DIR / "smart-contracts" / ".venv" / "Scripts" / "slither.exe"
    environment = {
        "generated_at_utc": generated_at,
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in package_names
        },
        "tools": {
            "forge": _command_version(["forge", "--version"]),
            "solc": _command_version(["solc", "--version"]),
            "slither": _command_version([str(slither), "--version"]),
        },
        "secrets_recorded": False,
        "network_contacted": False,
    }
    ENVIRONMENT_PATH.write_text(
        json.dumps(environment, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [
        {
            "path": _relative(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _artifact_paths()
    ]
    evidence = {
        "generated_at_utc": generated_at,
        "workspace_name": WORKSPACE_DIR.name,
        "file_count": len(files),
        "files": files,
        "manuscript_source_validation": manuscript_validation,
        "exclusions": [
            "Implementation/app/.env",
            "virtual environments",
            "Python caches",
            "Foundry build/cache directories",
            "archived superseded evidence",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Fail unless every frozen submission artifact and manuscript gate is complete.",
    )
    args = parser.parse_args()
    evidence, _ = build(require_final=args.require_final)
    print(f"Recorded {evidence['file_count']} files in {MANIFEST_PATH}")
    print(f"Recorded the local software environment in {ENVIRONMENT_PATH}")


if __name__ == "__main__":
    main()
