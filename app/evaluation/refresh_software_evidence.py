"""Run offline software checks and bind their reports to the tested sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.publication_workspace import (
    FIGURE_OUTPUT_DIR,
    FIGURE_SOURCE_DIR,
    REPOSITORY_DIR,
)

APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
WORKSPACE_DIR = REPOSITORY_DIR
SMART_CONTRACTS_DIR = IMPLEMENTATION_DIR / "smart-contracts"
OUTPUT_DIR = APP_DIR / "pipeline-output" / "current" / "software"
MANIFEST_PATH = OUTPUT_DIR / "software_verification_manifest.json"
PYTEST_LOG_PATH = OUTPUT_DIR / "python_pytest.txt"
FOUNDRY_LOG_PATH = SMART_CONTRACTS_DIR / "test-output" / "foundry_v30_full.txt"
FOUNDRY_GAS_LOG_PATH = SMART_CONTRACTS_DIR / "test-output" / "foundry_v30_gas_report.txt"
SLITHER_JSON_PATH = SMART_CONTRACTS_DIR / "security" / "slither" / "slither_v30.json"
SLITHER_LOG_PATH = SMART_CONTRACTS_DIR / "security" / "slither" / "slither_v30.txt"
SLITHER_HUMAN_LOG_PATH = (
    SMART_CONTRACTS_DIR / "security" / "slither" / "slither_v30_human_summary.txt"
)
CONTRACT_ARTIFACT_PATH = (
    SMART_CONTRACTS_DIR / "out" / "TransplantManagement.sol" / "TransplantManagement.json"
)
SCRIPT_PATH = Path(__file__).resolve()
SOLC_VERSION = "0.8.26"
FOUNDRY_FIGURE_PNG_PATH = FIGURE_OUTPUT_DIR / "foundry_tests.png"
SLITHER_FIGURE_PNG_PATH = FIGURE_OUTPUT_DIR / "slither_analysis.png"
FOUNDRY_CAPTURE_SOURCE_PATH = FIGURE_SOURCE_DIR / "foundry_tests[Original].png"
SLITHER_CAPTURE_SOURCE_PATH = FIGURE_SOURCE_DIR / "slither_analysis[Original].png"
FOUNDRY_FIGURE_SIZE = (2126, 1882)
SLITHER_FIGURE_SIZE = (2126, 841)


class SoftwareEvidenceError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def source_paths() -> tuple[Path, ...]:
    paths: set[Path] = {
        SCRIPT_PATH,
        APP_DIR / "requirements-lock.txt",
        APP_DIR / "requirements-ml.txt",
        APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json",
        APP_DIR / "templates" / "index.html",
        SMART_CONTRACTS_DIR / "foundry.toml",
        SMART_CONTRACTS_DIR / "foundry.lock",
        SMART_CONTRACTS_DIR / "requirements-slither.txt",
    }
    for directory, pattern in (
        (APP_DIR / "src", "*.py"),
        (APP_DIR / "evaluation", "*.py"),
        (APP_DIR / "tests", "*.py"),
        (SMART_CONTRACTS_DIR / "src", "*.sol"),
        (SMART_CONTRACTS_DIR / "test", "*.sol"),
        (SMART_CONTRACTS_DIR / "script", "*.sol"),
    ):
        paths.update(path for path in directory.rglob(pattern) if path.is_file())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SoftwareEvidenceError(
            "Missing software-verification source: " + ", ".join(str(path) for path in missing)
        )
    return tuple(sorted(paths, key=_relative))


def output_paths() -> tuple[Path, ...]:
    return (
        PYTEST_LOG_PATH,
        FOUNDRY_LOG_PATH,
        FOUNDRY_GAS_LOG_PATH,
        SLITHER_JSON_PATH,
        SLITHER_LOG_PATH,
        SLITHER_HUMAN_LOG_PATH,
        CONTRACT_ARTIFACT_PATH,
        FOUNDRY_FIGURE_PNG_PATH,
        SLITHER_FIGURE_PNG_PATH,
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int = 600,
) -> tuple[subprocess.CompletedProcess[str], str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = "".join((result.stdout, result.stderr))
    return result, combined


def _version(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> str:
    result, output = _run(command, cwd=cwd, env=env, timeout=60)
    if result.returncode != 0 or not output.strip():
        raise SoftwareEvidenceError(f"Unable to record tool version: {' '.join(command)}")
    return output.strip()


def find_local_solc(*, env: Mapping[str, str]) -> Path:
    names = (f"solc-{SOLC_VERSION}", f"solc-{SOLC_VERSION}.exe")
    candidates = [
        *(Path.home() / ".svm" / SOLC_VERSION / name for name in names),
        *(
            Path.home() / ".solc-select" / "artifacts" / f"solc-{SOLC_VERSION}" / name
            for name in names
        ),
    ]
    discovered = shutil.which("solc")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        result, output = _run(
            [str(candidate), "--version"],
            cwd=SMART_CONTRACTS_DIR,
            env=env,
            timeout=60,
        )
        if result.returncode == 0 and re.search(
            rf"\bVersion:\s*{re.escape(SOLC_VERSION)}(?:\+|\b)",
            output,
        ):
            return candidate.resolve()
    raise SoftwareEvidenceError(f"A local solc {SOLC_VERSION} binary is not available")


def _parse_pytest(output: str) -> dict[str, int]:
    passed = re.search(r"(?m)(\d+) passed(?:,| in)", output)
    failed = re.search(r"(?m)(\d+) failed", output)
    subtests = re.search(r"(?m)(\d+) subtests passed", output)
    if not passed:
        raise SoftwareEvidenceError("Could not parse the pytest pass count")
    return {
        "passed": int(passed.group(1)),
        "failed": int(failed.group(1)) if failed else 0,
        "subtests_passed": int(subtests.group(1)) if subtests else 0,
    }


def _parse_foundry(output: str) -> dict[str, int]:
    matches = re.findall(
        r"(\d+) tests passed, (\d+) failed, (\d+) skipped \((\d+) total tests\)",
        output,
    )
    if not matches:
        raise SoftwareEvidenceError("Could not parse the Foundry test summary")
    passed, failed, skipped, total = (int(value) for value in matches[-1])
    return {"passed": passed, "failed": failed, "skipped": skipped, "total": total}


def _parse_slither(console: str, report: Mapping[str, Any]) -> dict[str, int]:
    matches = re.findall(
        r"analyzed \(\d+ contracts? with (\d+) detectors\), (\d+) result\(s\) found",
        console,
    )
    if not matches:
        raise SoftwareEvidenceError("Could not parse the Slither detector summary")
    detectors, findings = (int(value) for value in matches[-1])
    if report.get("success") is not True or report.get("error") not in (None, ""):
        raise SoftwareEvidenceError("Slither's JSON report does not record a successful run")
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise SoftwareEvidenceError("Slither's JSON report has malformed results")
    json_findings = results.get("detectors", [])
    if json_findings is None:
        json_findings = []
    if not isinstance(json_findings, list) or len(json_findings) != findings:
        raise SoftwareEvidenceError("Slither console and JSON finding counts differ")
    return {"detectors_executed": detectors, "findings": findings}


def _slither_human_summary_lines(output: str) -> list[str]:
    lines = [line.rstrip() for line in output.splitlines()]
    try:
        start = next(
            index for index, line in enumerate(lines) if line == "Compiled with Foundry"
        )
        contract_row = next(
            index
            for index, line in enumerate(lines[start:], start=start)
            if line.startswith("| TransplantManagement ")
        )
        end = next(
            index
            for index, line in enumerate(lines[contract_row + 1 :], start=contract_row + 1)
            if line.startswith("+")
        )
    except StopIteration as exc:
        raise SoftwareEvidenceError("Slither human-summary output is incomplete") from exc

    excerpt: list[str] = []
    for line in lines[start : end + 1]:
        if line or (excerpt and excerpt[-1]):
            excerpt.append(line)
    return excerpt


def _parse_slither_human_summary(output: str) -> dict[str, int]:
    excerpt = "\n".join(_slither_human_summary_lines(output))
    patterns = {
        "contracts": r"Total number of contracts in source files:\s*(\d+)",
        "source_lines": r"Source lines of code \(SLOC\) in source files:\s*(\d+)",
        "assembly_lines": r"Number of\s+assembly lines:\s*(\d+)",
        "optimization_issues": r"Number of optimization issues:\s*(\d+)",
        "informational_issues": r"Number of informational issues:\s*(\d+)",
        "low_issues": r"Number of low issues:\s*(\d+)",
        "medium_issues": r"Number of medium issues:\s*(\d+)",
        "high_issues": r"Number of high issues:\s*(\d+)",
    }
    parsed: dict[str, int] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, excerpt)
        if not match:
            raise SoftwareEvidenceError(f"Slither human summary omits {name}")
        parsed[name] = int(match.group(1))
    function_match = re.search(r"\| TransplantManagement\s+\|\s*(\d+)\s+\|", excerpt)
    if not function_match:
        raise SoftwareEvidenceError("Slither human summary omits the contract function count")
    parsed["functions"] = int(function_match.group(1))
    return parsed


def _validate_terminal_captures() -> dict[str, object]:
    from evaluation.build_fine_tuning_loss_figure import _png_dimensions

    captures = (
        (
            "foundry",
            FOUNDRY_CAPTURE_SOURCE_PATH,
            FOUNDRY_FIGURE_PNG_PATH,
            FOUNDRY_FIGURE_SIZE,
            [67, 0, 820, 726],
        ),
        (
            "slither",
            SLITHER_CAPTURE_SOURCE_PATH,
            SLITHER_FIGURE_PNG_PATH,
            SLITHER_FIGURE_SIZE,
            [0, 0, 685, 271],
        ),
    )
    records: dict[str, object] = {}
    for name, source_path, output_path, expected_size, crop in captures:
        if not source_path.is_file():
            raise SoftwareEvidenceError(
                f"The original {name} terminal screenshot is missing"
            )
        if not output_path.is_file():
            raise SoftwareEvidenceError(f"The processed {name} screenshot is missing")
        if _png_dimensions(output_path) != expected_size:
            raise SoftwareEvidenceError(
                f"{output_path.name} has unexpected dimensions"
            )
        records[name] = {
            "capture_type": "native_terminal_screenshot",
            "permitted_processing": ["crop", "proportional_resize", "RGB_conversion"],
            "crop_xywh": crop,
            "source": {"path": _relative(source_path), **_record(source_path)},
            "output": {
                "path": _relative(output_path),
                "dimensions_px": list(expected_size),
                **_record(output_path),
            },
        }
    return records


def refresh(
    *,
    prepare_only: bool = False,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOUNDRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SLITHER_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_env = os.environ.copy()
    clean_env["ODA_DISABLE_DOTENV"] = "1"
    clean_env["PYTHON_DOTENV_DISABLED"] = "1"
    sensitive_fragments = (
        "API_KEY",
        "CREDENTIAL",
        "ENCRYPTION_KEY",
        "MNEMONIC",
        "PASSWORD",
        "PRIVATE_KEY",
        "RPC_URL",
        "SECRET",
        "TOKEN",
        "OPENAI",
        "PINATA",
    )
    for name in tuple(clean_env):
        if any(fragment in name.upper() for fragment in sensitive_fragments):
            clean_env.pop(name, None)
    clean_env["FOUNDRY_OFFLINE"] = "true"

    pytest_command = [sys.executable, "-m", "pytest", "-q"]
    pytest_result, pytest_output = _run(pytest_command, cwd=APP_DIR, env=clean_env)
    pytest_counts = _parse_pytest(pytest_output)
    if pytest_result.returncode != 0 or pytest_counts["failed"]:
        raise SoftwareEvidenceError(
            "The Python test suite did not pass:\n" + pytest_output[-8000:]
        )

    forge = shutil.which("forge")
    if not forge:
        raise SoftwareEvidenceError("Foundry forge is not available on PATH")
    solc = find_local_solc(env=clean_env)
    clean_env["FOUNDRY_SOLC"] = str(solc)
    foundry_command = [forge, "test", "--offline", "--use", str(solc), "-vv"]
    foundry_result, foundry_output = _run(
        foundry_command,
        cwd=SMART_CONTRACTS_DIR,
        env=clean_env,
    )
    foundry_counts = _parse_foundry(foundry_output)
    if foundry_result.returncode != 0 or foundry_counts["failed"] or foundry_counts["skipped"]:
        raise SoftwareEvidenceError("The Foundry test suite did not pass cleanly")

    gas_command = [
        forge,
        "test",
        "--offline",
        "--use",
        str(solc),
        "--gas-report",
    ]
    gas_result, gas_output = _run(gas_command, cwd=SMART_CONTRACTS_DIR, env=clean_env)
    gas_counts = _parse_foundry(gas_output)
    if gas_result.returncode != 0 or gas_counts != foundry_counts:
        raise SoftwareEvidenceError("The Foundry gas-report run did not reproduce the test result")

    slither = SMART_CONTRACTS_DIR / ".venv" / "Scripts" / "slither.exe"
    if not slither.is_file():
        discovered = shutil.which("slither")
        if not discovered:
            raise SoftwareEvidenceError("Slither is not available")
        slither = Path(discovered)
    with tempfile.TemporaryDirectory(prefix="slither-evidence-", dir=SMART_CONTRACTS_DIR) as directory:
        temporary_json = Path(directory) / "slither.json"
        slither_command = [
            str(slither),
            ".",
            "--json",
            str(temporary_json),
            "--exclude-dependencies",
        ]
        slither_result, slither_console = _run(
            slither_command,
            cwd=SMART_CONTRACTS_DIR,
            env=clean_env,
            timeout=900,
        )
        if slither_result.returncode != 0 or not temporary_json.is_file():
            raise SoftwareEvidenceError("Slither did not complete successfully")
        slither_report = json.loads(temporary_json.read_text(encoding="utf-8"))
        if not isinstance(slither_report, Mapping):
            raise SoftwareEvidenceError("Slither's JSON report is not an object")
        slither_counts = _parse_slither(slither_console, slither_report)
        if slither_counts["findings"]:
            raise SoftwareEvidenceError("Slither reported findings")
        slither_json_text = json.dumps(
            slither_report,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"

    slither_human_command = [
        str(slither),
        ".",
        "--print",
        "human-summary",
        "--exclude-dependencies",
    ]
    slither_human_result, slither_human_output = _run(
        slither_human_command,
        cwd=SMART_CONTRACTS_DIR,
        env=clean_env,
        timeout=900,
    )
    if slither_human_result.returncode != 0:
        raise SoftwareEvidenceError("Slither human-summary generation failed")
    slither_human_counts = _parse_slither_human_summary(slither_human_output)
    issue_keys = (
        "optimization_issues",
        "informational_issues",
        "low_issues",
        "medium_issues",
        "high_issues",
    )
    if any(slither_human_counts[key] for key in issue_keys):
        raise SoftwareEvidenceError("Slither human summary reports one or more issues")

    PYTEST_LOG_PATH.write_text(pytest_output, encoding="utf-8", newline="\n")
    FOUNDRY_LOG_PATH.write_text(foundry_output, encoding="utf-8", newline="\n")
    FOUNDRY_GAS_LOG_PATH.write_text(gas_output, encoding="utf-8", newline="\n")
    SLITHER_JSON_PATH.write_text(slither_json_text, encoding="utf-8", newline="\n")
    SLITHER_HUMAN_LOG_PATH.write_text(
        slither_human_output,
        encoding="utf-8",
        newline="\n",
    )

    generated_at = _utc_now()
    contract_source = SMART_CONTRACTS_DIR / "src" / "TransplantManagement.sol"
    contract_test = SMART_CONTRACTS_DIR / "test" / "TransplantManagementV30.t.sol"
    slither_version = _version(
        [str(slither), "--version"],
        cwd=SMART_CONTRACTS_DIR,
        env=clean_env,
    )
    solc_version = _version(
        [str(solc), "--version"],
        cwd=SMART_CONTRACTS_DIR,
        env=clean_env,
    )
    forge_version = _version(
        [forge, "--version"],
        cwd=SMART_CONTRACTS_DIR,
        env=clean_env,
    )
    slither_log = "\n".join([
        slither_version,
        solc_version.splitlines()[1] if len(solc_version.splitlines()) > 1 else solc_version,
        "Analyzed contract: src/TransplantManagement.sol",
        f"Analyzed at UTC: {generated_at}",
        f"Contract SHA-256: {_sha256(contract_source)}",
        f"Detectors executed: {slither_counts['detectors_executed']}",
        f"Results: {slither_counts['findings']}",
        "",
        "Command",
        "-------",
        ".\\.venv\\Scripts\\slither.exe . --json security\\slither\\slither_v30.json --exclude-dependencies",
        "",
        "Console result",
        "--------------",
        slither_console.strip(),
        "",
    ])
    SLITHER_LOG_PATH.write_text(slither_log, encoding="utf-8", newline="\n")
    if prepare_only:
        return {
            "prepared": True,
            "results": {
                "python": pytest_counts,
                "foundry": foundry_counts,
                "foundry_gas": gas_counts,
                "slither": slither_counts,
                "slither_human_summary": slither_human_counts,
            },
        }
    terminal_captures = _validate_terminal_captures()

    missing_outputs = [path for path in output_paths() if not path.is_file()]
    if missing_outputs:
        raise SoftwareEvidenceError(
            "Missing software-verification output: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    from evaluation.paper_full_workflow import _artifact_bytecode

    contract_runtime_sha256 = hashlib.sha256(
        _artifact_bytecode(
            json.loads(CONTRACT_ARTIFACT_PATH.read_text(encoding="utf-8")),
            "deployedBytecode",
        )
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at,
        "env_file_read": False,
        "network_contacted": False,
        "environment_boundary": {
            "dotenv_disabled": True,
            "foundry_offline": True,
            "foundry_solc_sha256": _sha256(solc),
            "credential_like_variables_removed": True,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tools": {
            "forge": forge_version,
            "solc": solc_version,
            "slither": slither_version,
        },
        "terminal_captures": terminal_captures,
        "commands": {
            "python": [Path(sys.executable).name, "-m", "pytest", "-q"],
            "foundry": [
                "forge",
                "test",
                "--offline",
                "--use",
                str(solc),
                "-vv",
            ],
            "foundry_gas": [
                "forge",
                "test",
                "--offline",
                "--use",
                str(solc),
                "--gas-report",
            ],
            "slither": [
                ".\\.venv\\Scripts\\slither.exe",
                ".",
                "--json",
                "security\\slither\\slither_v30.json",
                "--exclude-dependencies",
            ],
            "slither_human_summary": [
                ".\\.venv\\Scripts\\slither.exe",
                ".",
                "--print",
                "human-summary",
                "--exclude-dependencies",
            ],
        },
        "results": {
            "python": pytest_counts,
            "foundry": foundry_counts,
            "foundry_gas": gas_counts,
            "slither": slither_counts,
            "slither_human_summary": slither_human_counts,
        },
        "contract_source_sha256": _sha256(contract_source),
        "contract_test_sha256": _sha256(contract_test),
        "contract_runtime_sha256": contract_runtime_sha256,
        "solc_binary": {
            "filename": solc.name,
            **_record(solc),
        },
        "sources": {_relative(path): _record(path) for path in source_paths()},
        "outputs": {_relative(path): _record(path) for path in output_paths()},
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = refresh(prepare_only=args.prepare_only)
    results = manifest["results"]
    if args.prepare_only:
        print(
            "Software verification prepared: "
            f"Python={results['python']['passed']}, "
            f"Foundry={results['foundry']['passed']}, "
            f"Slither findings={results['slither']['findings']}"
        )
        print("Tool outputs refreshed; native terminal screenshots were left unchanged")
        return
    print(
        "Software verification complete: "
        f"Python={results['python']['passed']}, "
        f"Foundry={results['foundry']['passed']}, "
        f"Slither findings={results['slither']['findings']}"
    )
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
