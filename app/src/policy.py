"""Transparent, protocol-versioned reference ranking for synthetic cases."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = APP_DIR / "protocols" / "oda_synth_multiorgan_v1.json"
DEFAULT_PROTOCOL_FREEZE_PATH = (
    APP_DIR / "pipeline-output" / "current" / "protocol" / "protocol_freeze.json"
)
DEFAULT_PROTOCOL_LINEAGE_PATH = (
    APP_DIR
    / "pipeline-output"
    / "current"
    / "protocol"
    / "protocol_lineage_v1.2.0.json"
)
SUPPORTED_ORGANS = ("kidney", "liver", "heart", "lung")
RETIRED_TEST_SEEDS = (2026090703, 2026090717)


class PolicyInputError(ValueError):
    pass


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != "ODA-SYNTH-MULTIORGAN-1.0":
        raise PolicyInputError("Unsupported or missing protocol identifier")
    return protocol


def active_test_seed(protocol: Mapping[str, Any] | None = None) -> int:
    policy = protocol or load_protocol()
    try:
        dataset = policy["dataset"]
        splits = dataset["splits"]
        test_split = splits["test"]
        seed = test_split["seed"]
    except (KeyError, TypeError) as exc:
        raise PolicyInputError("The protocol must define dataset.splits.test.seed") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise PolicyInputError("The protocol test seed must be a positive integer")
    return seed


def assert_test_seed_not_retired(protocol: Mapping[str, Any] | None = None) -> int:
    seed = active_test_seed(protocol)
    if seed in RETIRED_TEST_SEEDS:
        raise PolicyInputError(
            f"Test seed {seed} is retired and cannot be executed. Insert an author-approved "
            "replacement seed and record the retirement in pre_freeze_history first."
        )
    return seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protocol_frozen(
    protocol: Mapping[str, Any] | None = None,
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
    freeze_path: Path | str = DEFAULT_PROTOCOL_FREEZE_PATH,
) -> Dict[str, Any]:
    """Require a valid freeze record for the active machine-readable protocol."""
    policy = protocol or load_protocol(protocol_path)
    seed = assert_test_seed_not_retired(policy)
    record_path = Path(freeze_path)
    if not record_path.is_file():
        raise PolicyInputError("The study protocol has not been frozen")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyInputError("The study protocol freeze record is unreadable") from exc

    source_hashes = record.get("source_hashes")
    expected_seed_hash = hashlib.sha256(str(seed).encode("ascii")).hexdigest()
    checks = {
        "status": record.get("status") == "frozen",
        "protocol_id": record.get("protocol_id") == policy.get("protocol_id"),
        "protocol_version": record.get("protocol_version") == policy.get("version"),
        "seed_hash": record.get("active_test_seed_sha256") == expected_seed_hash,
        "protocol_hash": (
            isinstance(source_hashes, Mapping)
            and source_hashes.get("protocol_json_sha256") == _sha256(Path(protocol_path))
        ),
        "retired_history": set(RETIRED_TEST_SEEDS).issubset(
            {
                int(item["retired_test_seed"])
                for item in policy.get("pre_freeze_history", [])
                if isinstance(item, Mapping) and "retired_test_seed" in item
            }
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, valid in checks.items() if not valid)
        raise PolicyInputError(f"The study protocol freeze record is invalid: {failed}")
    return record


def _resolve_workspace_path(value: object) -> Path:
    workspace = APP_DIR.parents[1].resolve()
    candidate = (workspace / str(value or "")).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise PolicyInputError("A protocol-lineage path leaves the workspace") from exc
    return candidate


def assert_preserved_artifact_lineage(
    artifact_name: str,
    artifact_path: Path | str,
    *,
    artifact_protocol_version: object,
    protocol_source_sha256: object,
    lineage_path: Path | str = DEFAULT_PROTOCOL_LINEAGE_PATH,
) -> Dict[str, Any]:
    """Verify an immutable pre-test artifact inherited by a later protocol version."""
    protocol = load_protocol()
    record_path = Path(lineage_path)
    if not record_path.is_file():
        raise PolicyInputError("The active protocol has no preserved-artifact lineage record")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyInputError("The protocol-lineage record is unreadable") from exc

    active = record.get("active_protocol")
    parent = record.get("parent_protocol")
    amendment = record.get("amendment")
    artifacts = record.get("preserved_artifacts")
    preserved_sources = record.get("preserved_sources")
    artifact = artifacts.get(artifact_name) if isinstance(artifacts, Mapping) else None
    checks = {
        "status": record.get("status") == "verified_pretest_lineage",
        "protocol_id": record.get("protocol_id") == protocol.get("protocol_id"),
        "held_out_lock": record.get("held_out_test_lock_present") is False,
        "held_out_execution": record.get("held_out_model_execution_started") is False,
        "active_protocol": (
            isinstance(active, Mapping)
            and active.get("version") == protocol.get("version")
            and active.get("sha256") == _sha256(DEFAULT_PROTOCOL_PATH)
        ),
        "parent_protocol": (
            isinstance(parent, Mapping)
            and parent.get("version") == artifact_protocol_version
            and parent.get("sha256") == protocol_source_sha256
        ),
        "artifact": (
            isinstance(artifact, Mapping)
            and artifact.get("protocol_version") == artifact_protocol_version
            and artifact.get("protocol_source_sha256") == protocol_source_sha256
            and artifact.get("sha256") == _sha256(Path(artifact_path))
        ),
        "preserved_sources": isinstance(preserved_sources, Mapping) and bool(preserved_sources),
    }
    for name, value in (("parent_protocol", parent), ("amendment", amendment), ("artifact", artifact)):
        if not isinstance(value, Mapping):
            continue
        path = _resolve_workspace_path(value.get("path"))
        checks[f"{name}_path"] = (
            path.is_file()
            and value.get("sha256") == _sha256(path)
            and (name != "artifact" or path == Path(artifact_path).resolve())
        )
    if isinstance(preserved_sources, Mapping):
        for name, value in preserved_sources.items():
            if not isinstance(value, Mapping):
                checks[f"preserved_source_{name}"] = False
                continue
            path = _resolve_workspace_path(value.get("path"))
            checks[f"preserved_source_{name}"] = (
                path.is_file() and value.get("sha256") == _sha256(path)
            )
    if not all(checks.values()):
        failed = ", ".join(name for name, valid in checks.items() if not valid)
        raise PolicyInputError(f"The preserved-artifact lineage is invalid: {failed}")
    return record


def _required(record: Mapping[str, Any], name: str) -> Any:
    if name not in record or record[name] is None or record[name] == "":
        raise PolicyInputError(f"Missing required field: {name}")
    return record[name]


def _number(record: Mapping[str, Any], name: str) -> float:
    value = _required(record, name)
    if isinstance(value, bool):
        raise PolicyInputError(f"Field {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyInputError(f"Field {name} must be numeric") from exc
    if not math.isfinite(number):
        raise PolicyInputError(f"Field {name} must be finite")
    return number


def _number_range(
    record: Mapping[str, Any],
    name: str,
    lower: float,
    upper: float,
) -> float:
    value = _number(record, name)
    if not lower <= value <= upper:
        raise PolicyInputError(f"Field {name} must be between {lower} and {upper}")
    return value


def _number_min(record: Mapping[str, Any], name: str, lower: float) -> float:
    value = _number(record, name)
    if value < lower:
        raise PolicyInputError(f"Field {name} must be at least {lower}")
    return value


def _boolean(record: Mapping[str, Any], name: str, *, default: bool = False) -> bool:
    if name not in record:
        return default
    value = record[name]
    if not isinstance(value, bool):
        raise PolicyInputError(f"Field {name} must be boolean")
    return value


def _positive_integer(record: Mapping[str, Any], name: str) -> int:
    value = _required(record, name)
    if isinstance(value, bool):
        raise PolicyInputError(f"Field {name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyInputError(f"Field {name} must be a positive integer") from exc
    if number <= 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise PolicyInputError(f"Field {name} must be a positive integer")
    return number


def _nonnegative_integer(record: Mapping[str, Any], name: str) -> int:
    value = _required(record, name)
    if isinstance(value, bool):
        raise PolicyInputError(f"Field {name} must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyInputError(f"Field {name} must be a nonnegative integer") from exc
    if number < 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise PolicyInputError(f"Field {name} must be a nonnegative integer")
    return number


def _validate_demographics(record: Mapping[str, Any]) -> None:
    _number_range(record, "age_years", 0.0, 120.0)
    _number_range(record, "weight_kg", 1.0, 400.0)
    _number_range(record, "height_cm", 30.0, 250.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(value: float, upper: float) -> float:
    if upper <= 0:
        raise PolicyInputError("Normalization upper bound must be positive")
    return _clamp01(value / upper)


def _proximity(value: float, lower: float, upper: float) -> float:
    if not lower < 1.0 < upper:
        raise PolicyInputError("Size bounds must straddle 1.0")
    if value <= 1.0:
        return _clamp01((value - lower) / (1.0 - lower))
    return _clamp01((upper - value) / (upper - 1.0))


def normalize_blood_group(value: str) -> str:
    blood = str(value or "").strip().upper().replace(" ", "")
    match = re.fullmatch(r"(AB|A|B|O)[+-]?", blood)
    if match:
        return match.group(1)
    raise PolicyInputError(f"Unsupported blood group: {value!r}")


def abo_compatible(
    donor_blood: str,
    recipient_blood: str,
    protocol: Mapping[str, Any] | None = None,
) -> bool:
    policy = protocol or load_protocol()
    donor = normalize_blood_group(donor_blood)
    recipient = normalize_blood_group(recipient_blood)
    return recipient in policy["blood_compatibility"][donor]


def _hla_values(record: Mapping[str, Any]) -> List[str]:
    value = _required(record, "hla_typing")
    if not isinstance(value, list):
        raise PolicyInputError("hla_typing must be a list of antigen strings")
    values = [str(item).strip().upper() for item in value if str(item).strip()]
    if len(values) != 6 or len(set(values)) != 6:
        raise PolicyInputError("Kidney HLA typing must contain six distinct antigens")
    patterns = {
        "A": re.compile(r"^A\*\d{2,3}(?::\d{2,3})?$"),
        "B": re.compile(r"^B\*\d{2,3}(?::\d{2,3})?$"),
        "DR": re.compile(r"^DRB1\*\d{2,3}(?::\d{2,3})?$"),
    }
    counts = {name: sum(bool(pattern.fullmatch(item)) for item in values) for name, pattern in patterns.items()}
    if counts != {"A": 2, "B": 2, "DR": 2}:
        raise PolicyInputError("Kidney HLA typing must contain two A, two B, and two DRB1 antigens")
    return values


def hla_overlap(donor_hla: Iterable[str], recipient_hla: Iterable[str]) -> int:
    donor = {str(item).strip().upper() for item in donor_hla if str(item).strip()}
    recipient = {str(item).strip().upper() for item in recipient_hla if str(item).strip()}
    return len(donor.intersection(recipient))


def _common_gates(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    donor_organ = str(_required(donor, "organ_type")).strip().lower()
    recipient_organ = str(_required(recipient, "organ_type")).strip().lower()
    if donor_organ != recipient_organ:
        reasons.append("organ_type_mismatch")
    if not abo_compatible(
        str(_required(donor, "blood_type")),
        str(_required(recipient, "blood_type")),
        protocol,
    ):
        reasons.append("abo_incompatible")
    crossmatch = str(_required(recipient, "crossmatch_result")).strip().lower()
    if crossmatch not in {"negative", "positive"}:
        raise PolicyInputError("crossmatch_result must be 'negative' or 'positive'")
    if crossmatch != "negative":
        reasons.append("crossmatch_not_negative")
    return reasons


def _distance_score(recipient: Mapping[str, Any], maximum: float) -> float:
    return 1.0 - _scale(_number_min(recipient, "distance_km", 0.0), maximum)


def _waiting_score(recipient: Mapping[str, Any], maximum: float) -> float:
    return _scale(_nonnegative_integer(recipient, "waiting_time_days"), maximum)


def _weighted_score(weights: Mapping[str, float], factors: Mapping[str, float]) -> float:
    missing = set(weights).difference(factors)
    if missing:
        raise PolicyInputError(f"Missing score factors: {sorted(missing)}")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise PolicyInputError("Organ-adapter weights must sum to 1.0")
    return 100.0 * sum(float(weights[name]) * _clamp01(factors[name]) for name in weights)


def _kidney_factors(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> Dict[str, float]:
    norm = settings["normalization"]
    donor_hla = _hla_values(donor)
    recipient_hla = _hla_values(recipient)
    kdpi = _number_range(donor, "synthetic_kdpi_percent", 0.0, 100.0)
    age = _number_range(recipient, "age_years", 0.0, 120.0)
    pediatric = age < float(norm["adult_age_min"])
    if pediatric:
        if "synthetic_epts_percent" in recipient:
            raise PolicyInputError("synthetic_epts_percent must be omitted for pediatric kidney candidates")
        longevity_priority = 0.0
    else:
        epts = _number_range(recipient, "synthetic_epts_percent", 0.0, 100.0)
        longevity_priority = 1.0 if (
            kdpi <= float(norm["top_kdpi_percent"])
            and epts <= float(norm["top_epts_percent"])
        ) else 0.0
    cpra = _number_range(recipient, "cpra_percent", 0.0, 100.0)
    if cpra >= float(norm["cpra_national_priority_threshold"]):
        sensitization_priority = 1.0
    elif cpra >= float(norm["cpra_priority_threshold"]):
        sensitization_priority = float(norm["cpra_priority_value"])
    else:
        sensitization_priority = 0.0
    return {
        "hla_overlap": _scale(hla_overlap(donor_hla, recipient_hla), norm["hla_antigen_count"]),
        "waiting_time": _waiting_score(recipient, norm["max_waiting_days"]),
        "sensitization_priority": sensitization_priority,
        "longevity_priority": longevity_priority,
        "pediatric_priority": 1.0 if pediatric and kdpi <= float(norm["pediatric_kdpi_max"]) else 0.0,
        "distance": _distance_score(recipient, norm["max_distance_km"]),
    }


def _liver_factors(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[Dict[str, float], List[str], int]:
    norm = settings["normalization"]
    age = _number_range(recipient, "age_years", 0.0, 120.0)
    if age < float(norm["adult_age_min"]):
        raise PolicyInputError("The synthetic liver adapter is restricted to adult candidates")
    ratio = _number(recipient, "weight_kg") / _number(donor, "weight_kg")
    exclusions = [] if norm["weight_ratio_min"] <= ratio <= norm["weight_ratio_max"] else ["size_incompatible"]
    status_one = _boolean(recipient, "synthetic_status_one")
    raw_urgency = _number_range(
        recipient,
        "synthetic_liver_urgency_score",
        norm["synthetic_urgency_min"],
        norm["synthetic_urgency_max"],
    )
    urgency_span = float(norm["synthetic_urgency_max"]) - float(norm["synthetic_urgency_min"])
    urgency = 1.0 if status_one else (raw_urgency - float(norm["synthetic_urgency_min"])) / urgency_span
    return ({
        "medical_urgency": urgency,
        "abo_identity": 1.0 if normalize_blood_group(str(donor["blood_type"])) == normalize_blood_group(
            str(recipient["blood_type"])
        ) else 0.0,
        "distance": _distance_score(recipient, norm["max_distance_km"]),
        "size_compatibility": _proximity(ratio, norm["weight_ratio_min"], norm["weight_ratio_max"]),
    }, exclusions, 1 if status_one else 0)


def _heart_factors(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[Dict[str, float], List[str], int]:
    norm = settings["normalization"]
    if _number_range(recipient, "age_years", 0.0, 120.0) < float(norm["adult_age_min"]):
        raise PolicyInputError("The synthetic heart adapter is restricted to adult candidates")
    ratio = _number(donor, "weight_kg") / _number(recipient, "weight_kg")
    exclusions = [] if norm["weight_ratio_min"] <= ratio <= norm["weight_ratio_max"] else ["size_incompatible"]
    raw_status = _number_range(recipient, "synthetic_adult_heart_status", 1.0, 6.0)
    if not raw_status.is_integer():
        raise PolicyInputError("synthetic_adult_heart_status must be an integer")
    status = int(raw_status)
    if status < 1 or status > 6:
        raise PolicyInputError("synthetic_adult_heart_status must be from 1 through 6")
    return ({
        "size_compatibility": _proximity(ratio, norm["weight_ratio_min"], norm["weight_ratio_max"]),
        "distance": _distance_score(recipient, norm["max_distance_km"]),
        "waiting_time": _waiting_score(recipient, norm["max_waiting_days"]),
    }, exclusions, 7 - status)


def _lung_factors(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[Dict[str, float], List[str]]:
    norm = settings["normalization"]
    if _number_range(recipient, "age_years", 0.0, 120.0) < float(norm["adult_age_min"]):
        raise PolicyInputError("The synthetic lung adapter is restricted to adult candidates")
    ratio = _number(donor, "height_cm") / _number(recipient, "height_cm")
    exclusions = [] if norm["height_ratio_min"] <= ratio <= norm["height_ratio_max"] else ["size_incompatible"]
    return ({
        "synthetic_priority_index": _scale(
            _number_range(recipient, "synthetic_lung_priority_score", 0.0, norm["synthetic_priority_max"]),
            norm["synthetic_priority_max"],
        ),
    }, exclusions)


def rank_recipients_baseline(
    donor: Dict[str, Any],
    recipients: List[Dict[str, Any]],
    protocol: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Rank records under the frozen synthetic protocol.

    Structurally incompatible candidates are retained at the end for audit but
    never receive a selectable rank. The score is a protocol score on a 0-100
    scale, not an estimated clinical outcome.
    """
    policy = protocol or load_protocol()
    organ = str(_required(donor, "organ_type")).strip().lower()
    if organ not in SUPPORTED_ORGANS:
        raise PolicyInputError(f"Unsupported organ type: {organ!r}")
    _positive_integer(donor, "donor_id")
    _validate_demographics(donor)
    settings = policy["organ_adapters"][organ]
    if organ == "kidney":
        _hla_values(donor)
        _number_range(donor, "synthetic_kdpi_percent", 0.0, 100.0)

    scored: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for recipient in recipients:
        recipient_id = _positive_integer(recipient, "recipient_id")
        if recipient_id in seen_ids:
            raise PolicyInputError("recipient_id values must be unique positive integers")
        seen_ids.add(recipient_id)
        _validate_demographics(recipient)

        exclusions = _common_gates(donor, recipient, policy)
        priority_tier = 0
        if "organ_type_mismatch" in exclusions:
            factors = {name: 0.0 for name in settings["weights"]}
        elif organ == "kidney":
            factors = _kidney_factors(donor, recipient, settings)
        elif organ == "liver":
            factors, organ_exclusions, priority_tier = _liver_factors(donor, recipient, settings)
            exclusions.extend(organ_exclusions)
        elif organ == "heart":
            factors, organ_exclusions, priority_tier = _heart_factors(donor, recipient, settings)
            exclusions.extend(organ_exclusions)
        else:
            factors, organ_exclusions = _lung_factors(donor, recipient, settings)
            exclusions.extend(organ_exclusions)

        score = 0.0 if "organ_type_mismatch" in exclusions else _weighted_score(settings["weights"], factors)
        scored.append({
            "recipient_id": recipient_id,
            "score": round(score, 6),
            "priority_tier": priority_tier,
            "selectable": not exclusions,
            "exclusion_reasons": sorted(set(exclusions)),
            "factors": {name: round(value, 6) for name, value in factors.items()},
            "waiting_time_days": _nonnegative_integer(recipient, "waiting_time_days"),
            "rank": None,
        })

    scored.sort(
        key=lambda item: (
            not item["selectable"],
            -item["priority_tier"],
            -item["score"],
            -item["waiting_time_days"],
            item["recipient_id"],
        )
    )
    rank = 0
    for item in scored:
        if item["selectable"]:
            rank += 1
            item["rank"] = rank
    return scored


def selectable_order(ranked: Iterable[Mapping[str, Any]]) -> List[int]:
    return [int(item["recipient_id"]) for item in ranked if bool(item.get("selectable"))]
