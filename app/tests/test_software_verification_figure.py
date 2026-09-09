from __future__ import annotations

import pytest

from evaluation import refresh_software_evidence as software


def _slither_human_summary() -> str:
    return """Compiled with Foundry
Total number of contracts in source files: 1
Source lines of code (SLOC) in source files: 435
Number of  assembly lines: 0
Number of optimization issues: 0
Number of informational issues: 0
Number of low issues: 0
Number of medium issues: 0
Number of high issues: 0

+----------------------+-------------+------+------------+--------------+----------+
| Name                 | # functions | ERCS | ERC20 info | Complex code | Features |
+----------------------+-------------+------+------------+--------------+----------+
| TransplantManagement | 33          |      |            | No           |          |
+----------------------+-------------+------+------------+--------------+----------+
"""


def test_foundry_summary_parser_uses_complete_tool_result() -> None:
    output = (
        "Suite result: ok. 41 passed; 0 failed; 0 skipped\n"
        "Ran 1 test suite in 2.00ms: "
        "41 tests passed, 0 failed, 0 skipped (41 total tests)\n"
    )

    assert software._parse_foundry(output) == {
        "passed": 41,
        "failed": 0,
        "skipped": 0,
        "total": 41,
    }


def test_slither_parser_uses_human_readable_summary() -> None:
    assert software._parse_slither_human_summary(_slither_human_summary()) == {
        "contracts": 1,
        "source_lines": 435,
        "assembly_lines": 0,
        "optimization_issues": 0,
        "informational_issues": 0,
        "low_issues": 0,
        "medium_issues": 0,
        "high_issues": 0,
        "functions": 33,
    }


def test_slither_parser_rejects_incomplete_human_summary() -> None:
    with pytest.raises(
        software.SoftwareEvidenceError,
        match="human-summary output is incomplete",
    ):
        software._parse_slither_human_summary("Number of high issues: 0")


def test_terminal_capture_records_bind_originals_and_processed_images() -> None:
    records = software._validate_terminal_captures()

    assert set(records) == {"foundry", "slither"}
    assert records["foundry"]["capture_type"] == "native_terminal_screenshot"
    assert records["foundry"]["output"]["dimensions_px"] == [2126, 1882]
    assert records["slither"]["output"]["dimensions_px"] == [2126, 841]
    assert records["foundry"]["source"]["sha256"]
    assert records["slither"]["source"]["sha256"]
