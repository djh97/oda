"""Deterministic multi-organ synthetic benchmark generation and validation."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.llm_client import SYSTEM_PROMPT, build_note_review_payload
from src.policy import (
    DEFAULT_PROTOCOL_PATH,
    SUPPORTED_ORGANS,
    load_protocol,
    rank_recipients_baseline,
    selectable_order,
)


APP_DIR = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = APP_DIR.parent
DATASET_DIR = IMPLEMENTATION_DIR / "datasets" / "synthetic-v1"
PROTOCOL = load_protocol()

SPLIT_FILE_NAMES = {
    "training": "training_cases.jsonl",
    "validation": "validation_cases.jsonl",
    "test": "test_cases.jsonl",
}

SCENARIO_WEIGHTS = {
    str(name): int(weight)
    for name, weight in PROTOCOL["dataset"]["scenario_weights_percent"].items()
}
if sum(SCENARIO_WEIGHTS.values()) != 100 or any(weight <= 0 for weight in SCENARIO_WEIGHTS.values()):
    raise ValueError("Protocol scenario weights must be positive integers summing to 100")

COMPLEXITY_WEIGHTS = {
    str(name): int(weight)
    for name, weight in PROTOCOL["dataset"]["note_generation"]["complexity_weights_percent"].items()
}
if sum(COMPLEXITY_WEIGHTS.values()) != 100 or any(
    weight <= 0 for weight in COMPLEXITY_WEIGHTS.values()
):
    raise ValueError("Protocol note-complexity weights must be positive integers summing to 100")
COMPLEXITY_POOL = tuple(
    name
    for name, weight in COMPLEXITY_WEIGHTS.items()
    for _ in range(weight)
)

STATE_CODES = {
    "eligible": (
        "no_current_concern",
        "negated_condition",
        "resolved_condition",
        "historical_condition_cleared",
    ),
    "review_required": (
        "infection_workup_pending",
        "indeterminate_lesion",
        "specialist_clearance_pending",
        "missing_or_conflicting_information",
    ),
    "temporary_hold": (
        "active_infection",
        "sepsis_or_instability",
        "active_malignancy_treatment",
        "explicit_temporary_deferral",
    ),
}
STATE_PRECEDENCE = tuple(PROTOCOL["note_taxonomy"]["state_precedence"])
if set(STATE_PRECEDENCE) != set(STATE_CODES):
    raise ValueError("Protocol state precedence must contain each readiness state exactly once")

NOTE_TEMPLATES: Dict[str, Dict[str, Sequence[str]]] = {
    "training": {
        "no_current_concern": (
            "Current assessment documents stable status with no readiness concern.",
            "The present review records stable findings and no active barrier to proceeding.",
        ),
        "negated_condition": (
            "There is no evidence of active infection; the recent screening result is negative.",
            "Active infection was considered and explicitly ruled out during this assessment.",
        ),
        "resolved_condition": (
            "A previously treated infection has resolved, therapy is complete, and current cultures are negative.",
            "The earlier febrile illness is resolved with no current symptoms or positive cultures.",
        ),
        "historical_condition_cleared": (
            "A remote condition is documented in the history and has already received specialist clearance.",
            "The prior malignancy remains in remission and the note records completed specialist clearance.",
        ),
        "infection_workup_pending": (
            "A possible infection is under evaluation and the culture workup remains pending.",
            "New fever requires an infection assessment; readiness is not yet determined.",
        ),
        "indeterminate_lesion": (
            "Imaging identified an indeterminate lesion that requires characterization before readiness can be established.",
            "A newly observed lesion has uncertain significance and follow-up assessment is pending.",
        ),
        "specialist_clearance_pending": (
            "Specialist clearance requested for a recent finding has not yet been completed.",
            "The assessment remains incomplete while the requested specialty review is pending.",
        ),
        "missing_or_conflicting_information": (
            "Two recent records conflict, so current readiness cannot be determined from the available information.",
            "A required result is missing and the present note cannot resolve readiness.",
        ),
        "active_infection": (
            "A current bloodstream infection is being treated after a positive culture.",
            "The note documents an active bacterial infection requiring ongoing therapy.",
        ),
        "sepsis_or_instability": (
            "The candidate currently has sepsis with hemodynamic instability.",
            "Current assessment records unstable vital signs in the setting of suspected sepsis.",
        ),
        "active_malignancy_treatment": (
            "The candidate is receiving active treatment for a newly diagnosed malignancy.",
            "Current oncology treatment is underway for active malignant disease.",
        ),
        "explicit_temporary_deferral": (
            "The transplant team has explicitly placed the candidate on temporary hold pending recovery.",
            "The current plan states that transplantation is deferred until the acute issue resolves.",
        ),
    },
    "validation": {
        "no_current_concern": (
            "Today's review finds the candidate stable without a current concern affecting readiness.",
            "No active readiness issue is recorded in the latest assessment.",
        ),
        "negated_condition": (
            "The chart specifically denies ongoing infection, and repeat testing is negative.",
            "Evaluation shows no present infectious process despite an earlier concern.",
        ),
        "resolved_condition": (
            "The former infection cleared after treatment and is no longer active.",
            "Prior pneumonia is documented as resolved, with treatment concluded.",
        ),
        "historical_condition_cleared": (
            "A historical diagnosis is noted, but follow-up clearance has been documented.",
            "Remote treated disease is listed only as history and the relevant service has cleared it.",
        ),
        "infection_workup_pending": (
            "Cultures were ordered for a possible infection and final results are not available.",
            "An infectious concern remains unresolved while diagnostic testing is in progress.",
        ),
        "indeterminate_lesion": (
            "A lesion of unclear significance awaits additional imaging and review.",
            "Further characterization is required for a newly detected indeterminate mass.",
        ),
        "specialist_clearance_pending": (
            "A required specialty opinion remains outstanding before readiness can be confirmed.",
            "The requested consultant has not yet issued clearance for the recent finding.",
        ),
        "missing_or_conflicting_information": (
            "Readiness remains uncertain because the latest reports contain inconsistent findings.",
            "The assessment lacks a required result and cannot establish current readiness.",
        ),
        "active_infection": (
            "Ongoing antimicrobial treatment is documented for a current culture-confirmed infection.",
            "A present invasive infection remains active and under treatment.",
        ),
        "sepsis_or_instability": (
            "The latest note describes active sepsis and circulatory instability.",
            "The candidate is presently unstable with findings consistent with sepsis.",
        ),
        "active_malignancy_treatment": (
            "Active cancer therapy is in progress for current disease.",
            "The latest oncology note confirms ongoing treatment of an active malignancy.",
        ),
        "explicit_temporary_deferral": (
            "The current disposition is temporary transplant deferral until reassessment.",
            "The case is explicitly held from transplantation for the present interval.",
        ),
    },
    "test": {
        "no_current_concern": (
            "The most recent evaluation is stable and identifies no present obstacle under the review protocol.",
            "Current documentation contains no active issue that would pause candidacy.",
        ),
        "negated_condition": (
            "Although infection was queried, the note states that no active infection is present.",
            "Repeat evaluation excludes a current infectious process rather than confirming one.",
        ),
        "resolved_condition": (
            "An earlier infection is described as fully resolved after completion of therapy.",
            "The previous illness has cleared; it is historical rather than an active condition.",
        ),
        "historical_condition_cleared": (
            "The record mentions remote treated disease and confirms that clearance was obtained.",
            "A past malignancy appears in the history, with remission and specialty clearance documented.",
        ),
        "infection_workup_pending": (
            "Possible infection has not been excluded because confirmatory studies are still outstanding.",
            "The note leaves an infectious concern open while final testing is awaited.",
        ),
        "indeterminate_lesion": (
            "The significance of a recent lesion remains undetermined pending further evaluation.",
            "A new imaging abnormality cannot yet be classified and needs follow-up.",
        ),
        "specialist_clearance_pending": (
            "Readiness cannot yet be confirmed because a requested specialist decision is outstanding.",
            "The candidate awaits formal clearance from the consulting service.",
        ),
        "missing_or_conflicting_information": (
            "Available reports disagree on a current finding, leaving the readiness assessment unresolved.",
            "Essential documentation is absent, so the note does not support a final readiness classification.",
        ),
        "active_infection": (
            "A culture-confirmed infection is currently present and treatment remains ongoing.",
            "The assessment identifies an active infectious process that has not cleared.",
        ),
        "sepsis_or_instability": (
            "Current sepsis is accompanied by unstable hemodynamic findings.",
            "The record describes ongoing septic physiology with clinical instability.",
        ),
        "active_malignancy_treatment": (
            "Treatment for active malignant disease is currently being administered.",
            "The candidate has present malignancy for which oncology therapy is ongoing.",
        ),
        "explicit_temporary_deferral": (
            "The documented decision is to pause transplant consideration temporarily.",
            "Transplantation is expressly deferred at this time, with later reassessment planned.",
        ),
    },
    "demo": {
        "no_current_concern": (
            "The latest synthetic review records stable findings without a present readiness concern.",
            "No current issue that would pause the demonstration workflow is documented.",
        ),
        "negated_condition": (
            "A recent infectious concern was investigated, and the current note explicitly excludes active infection.",
            "The assessment mentions infection only to document that it is not currently present.",
        ),
        "resolved_condition": (
            "A prior infection completed treatment and is documented as resolved at this assessment.",
            "The earlier acute illness has cleared and is no longer active.",
        ),
        "historical_condition_cleared": (
            "A remote diagnosis remains in the history, with the required clearance already documented.",
            "The note records treated historical disease and completed specialist clearance.",
        ),
        "infection_workup_pending": (
            "A possible infection remains under evaluation because final studies are outstanding.",
            "The infectious workup is incomplete, so present readiness cannot yet be resolved.",
        ),
        "indeterminate_lesion": (
            "A newly observed lesion remains indeterminate and requires further characterization.",
            "The significance of a recent imaging finding is unresolved pending follow-up.",
        ),
        "specialist_clearance_pending": (
            "The requested specialist assessment has not yet provided clearance.",
            "A formal specialty decision remains outstanding before readiness can be confirmed.",
        ),
        "missing_or_conflicting_information": (
            "The available reports conflict and do not establish current readiness.",
            "A required result is absent from the record, leaving the assessment unresolved.",
        ),
        "active_infection": (
            "A current culture-confirmed infection remains active while treatment continues.",
            "The latest assessment documents an ongoing infectious process requiring therapy.",
        ),
        "sepsis_or_instability": (
            "Active sepsis with hemodynamic instability is documented in the current assessment.",
            "The candidate is presently unstable in the setting of sepsis.",
        ),
        "active_malignancy_treatment": (
            "Treatment is currently underway for active malignant disease.",
            "The latest oncology assessment confirms active cancer therapy.",
        ),
        "explicit_temporary_deferral": (
            "The current transplant plan explicitly records a temporary deferral.",
            "Transplant consideration is paused until the acute issue is reassessed.",
        ),
    },
}

NOTE_CONTEXTS: Dict[str, Sequence[str]] = {
    "training": (
        "Transplant coordination reviewed the candidate during the current encounter.",
        "The multidisciplinary team updated the readiness assessment.",
        "The latest progress entry documents the candidate's current condition.",
        "During scheduled candidate review, the following issue was recorded.",
        "The transplant service entered a new status note.",
        "The current chart update addresses transplant readiness.",
        "An interval assessment was completed by the transplant service.",
        "The candidate's status was reviewed after recent follow-up.",
    ),
    "validation": (
        "An interval coordination note records the latest assessment.",
        "The transplant program completed a current candidacy review.",
        "A new clinical update was added after team discussion.",
        "The current encounter includes a focused readiness assessment.",
        "The candidate record was updated following recent follow-up.",
        "A multidisciplinary review produced the present status entry.",
        "The transplant service reassessed the candidate at this visit.",
        "The latest status review contains the following finding.",
    ),
    "test": (
        "The current handoff summarizes a new readiness finding.",
        "A transplant follow-up note records the present assessment.",
        "The latest candidacy entry was completed after chart review.",
        "The transplant team documented an interval status change.",
        "A focused review of current readiness produced this update.",
        "The candidate's most recent encounter contains the following assessment.",
        "An updated transplant-service note addresses present candidacy.",
        "The current multidisciplinary summary records this finding.",
    ),
    "demo": (
        "The demonstration record contains a current status update.",
        "The latest synthetic demonstration note provides this assessment.",
        "The candidate's demonstration record was reviewed for current readiness.",
        "The demonstration workflow includes the following candidacy update.",
    ),
}

NOTE_QUALIFIERS: Dict[str, Sequence[str]] = {
    "training": (
        "The medication list was reconciled in a separate section.",
        "The encounter date and reviewing service were recorded.",
        "Other elements of the candidacy assessment were unchanged.",
        "The note was signed after routine chart reconciliation.",
        "Follow-up timing was documented elsewhere in the record.",
        "The coordinator retained the prior demographic information.",
        "No additional readiness-related change was entered in this note.",
        "The update supersedes the preceding status entry.",
    ),
    "validation": (
        "Medication reconciliation appears in the accompanying encounter note.",
        "The service and review interval were logged separately.",
        "No other candidacy update was added during this review.",
        "The remaining assessment fields were carried forward without change.",
        "The entry was finalized following multidisciplinary chart review.",
        "Administrative follow-up was documented in another section.",
        "The prior demographic record remains unchanged.",
        "This entry replaces the earlier interim status note.",
    ),
    "test": (
        "The medication record was reviewed in the same encounter.",
        "Scheduling details were placed in the coordination section.",
        "No further change in candidacy was recorded in this update.",
        "The remainder of the transplant assessment was unchanged.",
        "The note was closed after review of the available chart.",
        "Routine administrative fields were updated separately.",
        "Previously verified demographic details were retained.",
        "This assessment replaces the prior provisional entry.",
    ),
    "demo": (
        "Other demonstration fields remain unchanged.",
        "The sample note was finalized after routine record review.",
        "Administrative details are recorded separately.",
        "This entry replaces the earlier demonstration status.",
    ),
}

NOTE_TRANSITIONS: Dict[str, Sequence[str]] = {
    "training": (
        "The coordinator also documented the next review interval.",
        "The entry was routed to the transplant service for follow-up.",
        "A separate section contains the supporting administrative record.",
        "The current update was added to the longitudinal candidacy record.",
        "The multidisciplinary summary references this encounter.",
        "The status entry was communicated to the coordination team.",
    ),
    "validation": (
        "The coordination team recorded a subsequent review date.",
        "This update was incorporated into the longitudinal record.",
        "A related administrative entry documents follow-up scheduling.",
        "The transplant service acknowledged the updated assessment.",
        "The finding was carried into the multidisciplinary summary.",
        "The candidacy record now reflects this encounter.",
    ),
    "test": (
        "The team entered a new follow-up interval after this review.",
        "The status was added to the longitudinal transplant record.",
        "A separate coordination entry contains scheduling information.",
        "The updated assessment was forwarded to the transplant service.",
        "The multidisciplinary handoff includes this status entry.",
        "The candidacy record was revised after the encounter.",
    ),
    "demo": (
        "The sample workflow records a later review interval.",
        "The demonstration summary includes this update.",
        "A separate sample entry contains scheduling details.",
    ),
}

NOTE_CONTEXT_EVIDENCE_TEMPLATES: Dict[str, Dict[str, Sequence[str]]] = {
    "training": {
        "no_current_concern": ("The remainder of the current assessment is stable.",),
        "negated_condition": ("A separate suspected urinary infection was excluded by repeat testing.",),
        "resolved_condition": ("An earlier unrelated respiratory infection resolved after treatment.",),
        "historical_condition_cleared": ("A remote treated condition remains cleared on follow-up.",),
        "infection_workup_pending": ("A separate urine culture also remains pending.",),
        "indeterminate_lesion": ("An unrelated incidental lesion also awaits characterization.",),
        "specialist_clearance_pending": ("A separate specialty review has not yet issued clearance.",),
        "missing_or_conflicting_information": ("One unrelated laboratory result is not yet available.",),
    },
    "validation": {
        "no_current_concern": ("Other findings in the present review remain stable.",),
        "negated_condition": ("Repeat testing excluded a different suspected infectious process.",),
        "resolved_condition": ("A distinct earlier illness cleared after therapy was completed.",),
        "historical_condition_cleared": ("A separate remote diagnosis has documented follow-up clearance.",),
        "infection_workup_pending": ("Testing for another possible infection is still in progress.",),
        "indeterminate_lesion": ("A separate imaging abnormality still needs characterization.",),
        "specialist_clearance_pending": ("Another requested consultant opinion remains outstanding.",),
        "missing_or_conflicting_information": ("A different part of the assessment lacks a required result.",),
    },
    "test": {
        "no_current_concern": ("The other elements reviewed at this encounter are stable.",),
        "negated_condition": ("A different infectious concern was ruled out on repeat assessment.",),
        "resolved_condition": ("An unrelated prior illness is documented as resolved after treatment.",),
        "historical_condition_cleared": ("A distinct historical condition retains documented specialist clearance.",),
        "infection_workup_pending": ("A separate possible infection awaits completion of its workup.",),
        "indeterminate_lesion": ("An unrelated new lesion is also awaiting further assessment.",),
        "specialist_clearance_pending": ("A different specialty review has yet to provide clearance.",),
        "missing_or_conflicting_information": ("Documentation for another finding remains incomplete.",),
    },
    "demo": {
        "no_current_concern": ("Other findings in the demonstration review are stable.",),
        "negated_condition": ("A separate infectious concern was excluded on repeat review.",),
        "resolved_condition": ("An unrelated earlier illness is recorded as resolved.",),
        "historical_condition_cleared": ("A distinct historical diagnosis has documented clearance.",),
        "infection_workup_pending": ("A separate possible infection remains under evaluation.",),
        "indeterminate_lesion": ("An unrelated imaging finding still needs characterization.",),
        "specialist_clearance_pending": ("Another requested specialty opinion remains outstanding.",),
        "missing_or_conflicting_information": ("A different assessment item lacks a required result.",),
    },
}

NOTE_STYLE_ORDERS = {
    "context_first": ("context", "core", "qualifier", "transition"),
    "finding_first": ("core", "qualifier", "context", "transition"),
    "interleaved": ("context", "transition", "core", "qualifier"),
    "delayed_finding": ("qualifier", "context", "transition", "core"),
}

HLA_POOLS = {
    "A": ("A*01", "A*02", "A*03", "A*11", "A*24", "A*26"),
    "B": ("B*07", "B*08", "B*15", "B*27", "B*35", "B*44"),
    "DR": ("DRB1*01", "DRB1*03", "DRB1*04", "DRB1*11", "DRB1*13", "DRB1*15"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _case_id(seed: int, index: int) -> str:
    return hashlib.sha256(f"{seed}:{index}".encode("ascii")).hexdigest()[:16]


def _sample_hla(rng: random.Random) -> List[str]:
    return [
        *rng.sample(HLA_POOLS["A"], 2),
        *rng.sample(HLA_POOLS["B"], 2),
        *rng.sample(HLA_POOLS["DR"], 2),
    ]


def _recipient_hla(rng: random.Random, donor_hla: Sequence[str], overlap_target: int) -> List[str]:
    overlap_target = max(0, min(6, overlap_target))
    donor_by_locus = {
        locus: [value for value in donor_hla if value in pool]
        for locus, pool in HLA_POOLS.items()
    }
    overlap_by_locus = {locus: 0 for locus in HLA_POOLS}
    slots = [locus for locus in HLA_POOLS for _ in range(2)]
    rng.shuffle(slots)
    for locus in slots[:overlap_target]:
        overlap_by_locus[locus] += 1

    values: List[str] = []
    for locus, pool in HLA_POOLS.items():
        retained = rng.sample(donor_by_locus[locus], overlap_by_locus[locus])
        alternatives = [value for value in pool if value not in donor_by_locus[locus]]
        values.extend(retained)
        values.extend(rng.sample(alternatives, 2 - len(retained)))
    rng.shuffle(values)
    return values


def _other_organ(organ: str) -> str:
    return SUPPORTED_ORGANS[(SUPPORTED_ORGANS.index(organ) + 1) % len(SUPPORTED_ORGANS)]


def _compatible_blood_types(donor_blood: str) -> Sequence[str]:
    return tuple(PROTOCOL["blood_compatibility"][donor_blood])


def _incompatible_blood_types(donor_blood: str) -> Sequence[str]:
    allowed = set(_compatible_blood_types(donor_blood))
    return tuple(value for value in ("O", "A", "B", "AB") if value not in allowed)


def _scenario_allocation(count: int) -> Dict[str, int]:
    names = list(SCENARIO_WEIGHTS)
    raw = {name: count * SCENARIO_WEIGHTS[name] / 100.0 for name in names}
    assigned = {name: int(raw[name]) for name in names}
    for name in sorted(names, key=lambda item: raw[item] - assigned[item], reverse=True)[: count - sum(assigned.values())]:
        assigned[name] += 1
    return assigned


def _scenario_schedule(count: int, rng: random.Random) -> List[str]:
    names = list(SCENARIO_WEIGHTS)
    assigned = _scenario_allocation(count)
    schedule = [name for name in names for _ in range(assigned[name])]
    rng.shuffle(schedule)
    return schedule


def _make_donor(rng: random.Random, donor_id: int, organ: str) -> Dict[str, Any]:
    donor = {
        "donor_id": donor_id,
        "organ_type": organ,
        "blood_type": rng.choice(("O", "A", "B", "AB")),
        "age_years": rng.randint(18, 70),
        "weight_kg": round(rng.uniform(52.0, 105.0), 1),
        "height_cm": round(rng.uniform(150.0, 195.0), 1),
    }
    if organ == "kidney":
        donor["hla_typing"] = _sample_hla(rng)
        donor["synthetic_kdpi_percent"] = round(rng.uniform(5.0, 95.0), 1)
    return donor


def _make_recipient(
    rng: random.Random,
    donor: Mapping[str, Any],
    recipient_id: int,
    position: int,
) -> Dict[str, Any]:
    organ = str(donor["organ_type"])
    blood = rng.choice(_compatible_blood_types(str(donor["blood_type"])))
    crossmatch = "negative"
    recipient_organ = organ

    if position == 7:
        incompatible = _incompatible_blood_types(str(donor["blood_type"]))
        if incompatible:
            blood = rng.choice(incompatible)
        else:
            crossmatch = "positive"
    elif position == 8:
        crossmatch = "positive"
    elif position == 9:
        recipient_organ = _other_organ(organ)

    recipient: Dict[str, Any] = {
        "recipient_id": recipient_id,
        "organ_type": recipient_organ,
        "blood_type": blood,
        "crossmatch_result": crossmatch,
        "age_years": rng.randint(10, 75) if organ == "kidney" else rng.randint(18, 75),
        "weight_kg": round(rng.uniform(40.0, 115.0), 1),
        "height_cm": round(rng.uniform(140.0, 200.0), 1),
        "waiting_time_days": rng.randint(20, 3650 if organ == "kidney" else 1095),
        "distance_km": round(rng.uniform(5.0, 5000.0), 1),
    }

    if organ == "kidney":
        recipient["hla_typing"] = _recipient_hla(rng, donor["hla_typing"], position % 7)
        recipient["cpra_percent"] = round(rng.uniform(0.0, 100.0), 1)
        if recipient["age_years"] >= 18:
            recipient["synthetic_epts_percent"] = round(rng.uniform(1.0, 99.0), 1)
        if position == 6:
            recipient["crossmatch_result"] = "positive"
    elif organ == "liver":
        ratio = rng.uniform(0.70, 1.45)
        recipient["weight_kg"] = round(float(donor["weight_kg"]) * ratio, 1)
        if position == 6:
            recipient["weight_kg"] = round(float(donor["weight_kg"]) * 2.05, 1)
        recipient["synthetic_liver_urgency_score"] = rng.randint(6, 40)
        recipient["synthetic_status_one"] = rng.random() < 0.04
    elif organ == "heart":
        ratio = rng.uniform(0.78, 1.22)
        recipient["weight_kg"] = round(float(donor["weight_kg"]) / ratio, 1)
        if position == 6:
            recipient["weight_kg"] = round(float(donor["weight_kg"]) / 0.55, 1)
        recipient["synthetic_adult_heart_status"] = rng.randint(1, 6)
    else:
        ratio = rng.uniform(0.86, 1.14)
        recipient["height_cm"] = round(float(donor["height_cm"]) / ratio, 1)
        if position == 6:
            recipient["height_cm"] = round(float(donor["height_cm"]) / 0.78, 1)
        recipient["synthetic_lung_priority_score"] = round(rng.uniform(5.0, 95.0), 1)

    return recipient


def _set_reference(
    recipient: Dict[str, Any],
    state: str,
    rng: random.Random,
    preferred_code: str | None = None,
) -> None:
    code = preferred_code or rng.choice(STATE_CODES[state])
    recipient["reference_note_state"] = state
    recipient["reference_evidence_codes"] = [code]


def _assign_states(
    recipients: List[Dict[str, Any]],
    baseline_order: Sequence[int],
    scenario: str,
    rng: random.Random,
) -> None:
    by_id = {int(item["recipient_id"]): item for item in recipients}
    protected_holds: set[int] = set()
    for recipient in recipients:
        roll = rng.random()
        state = "eligible" if roll < 0.60 else "review_required" if roll < 0.80 else "temporary_hold"
        _set_reference(recipient, state, rng)

    top1 = by_id[int(baseline_order[0])]
    if scenario in {"top1_temporary_hold", "top2_temporary_hold"}:
        _set_reference(top1, "temporary_hold", rng)
        protected_holds.add(int(top1["recipient_id"]))
        second = by_id[int(baseline_order[1])]
        if scenario == "top2_temporary_hold":
            _set_reference(second, "temporary_hold", rng)
            protected_holds.add(int(second["recipient_id"]))
        else:
            _set_reference(second, "eligible", rng, "no_current_concern")
    elif scenario == "top1_review_required":
        _set_reference(top1, "review_required", rng)
    elif scenario == "top1_negated_or_resolved":
        _set_reference(top1, "eligible", rng, rng.choice(("negated_condition", "resolved_condition")))
    else:
        _set_reference(top1, "eligible", rng, "no_current_concern")

    lower_hold_id = int(baseline_order[3])
    if scenario == "non_top1_temporary_hold":
        _set_reference(by_id[lower_hold_id], "temporary_hold", rng)
        protected_holds.add(lower_hold_id)

    if scenario != "top1_review_required":
        _set_reference(by_id[int(baseline_order[2])], "review_required", rng)

    nonheld = [
        rid
        for rid in baseline_order
        if by_id[int(rid)]["reference_note_state"] != "temporary_hold"
    ]
    for rid in reversed(baseline_order):
        if len(nonheld) >= int(PROTOCOL["dataset"]["minimum_not_on_hold"]):
            break
        recipient = by_id[int(rid)]
        if int(rid) not in protected_holds and recipient["reference_note_state"] == "temporary_hold":
            _set_reference(recipient, "eligible", rng, "no_current_concern")
            nonheld.append(int(rid))


def _assign_note_composition(recipients: List[Dict[str, Any]], rng: random.Random) -> None:
    """Assign note complexity and evidence composition before rendering text."""
    for recipient in recipients:
        state = str(recipient["reference_note_state"])
        complexity = rng.choice(COMPLEXITY_POOL)
        reference_codes = list(recipient["reference_evidence_codes"])
        context_codes: List[str] = []

        if complexity in {"multi_evidence", "multi_evidence_contrast"}:
            available = [code for code in STATE_CODES[state] if code not in reference_codes]
            reference_codes.append(rng.choice(available))

        if complexity in {"contrast", "multi_evidence_contrast"}:
            state_index = STATE_PRECEDENCE.index(state)
            lower_states = STATE_PRECEDENCE[state_index + 1 :]
            if lower_states:
                context_state = rng.choice(lower_states)
                context_codes.append(rng.choice(STATE_CODES[context_state]))
            else:
                available = [code for code in STATE_CODES[state] if code not in reference_codes]
                reference_codes.append(rng.choice(available))

        recipient["reference_evidence_codes"] = reference_codes
        recipient["context_evidence_codes"] = context_codes
        recipient["note_complexity"] = complexity


def _render_notes(split: str, case_id: str, recipients: List[Dict[str, Any]], rng: random.Random) -> None:
    for index, recipient in enumerate(recipients, start=1):
        reference_codes = [str(code) for code in recipient["reference_evidence_codes"]]
        context_codes = [str(code) for code in recipient["context_evidence_codes"]]
        rendered_evidence = []
        for code in reference_codes:
            variants = NOTE_TEMPLATES[split][code]
            rendered_evidence.append((code, variants[rng.randrange(len(variants))], "reference"))
        for code in context_codes:
            variants = NOTE_CONTEXT_EVIDENCE_TEMPLATES[split][code]
            rendered_evidence.append((code, variants[rng.randrange(len(variants))], "context"))
        rng.shuffle(rendered_evidence)
        context_index = rng.randrange(len(NOTE_CONTEXTS[split]))
        qualifier_index = rng.randrange(len(NOTE_QUALIFIERS[split]))
        transition_index = rng.randrange(len(NOTE_TRANSITIONS[split]))
        style = rng.choice(tuple(NOTE_STYLE_ORDERS))
        components = {
            "context": NOTE_CONTEXTS[split][context_index],
            "core": " ".join(sentence for _, sentence, _ in rendered_evidence),
            "qualifier": NOTE_QUALIFIERS[split][qualifier_index],
            "transition": NOTE_TRANSITIONS[split][transition_index],
        }
        recipient["medical_notes"] = " ".join(
            components[name] for name in NOTE_STYLE_ORDERS[style]
        )
        recipient["note_style"] = style
        recipient["template_family_id"] = hashlib.sha256(
            "\0".join(
                f"{kind}:{code}:{sentence}" for code, sentence, kind in rendered_evidence
            ).encode("utf-8")
        ).hexdigest()[:12]
        recipient["note_instance_id"] = hashlib.sha256(
            f"{case_id}:{index}:{recipient['medical_notes']}".encode("utf-8")
        ).hexdigest()[:12]


def generate_case(
    split: str,
    split_seed: int,
    case_index: int,
    organ: str,
    scenario: str,
) -> Dict[str, Any]:
    rng = random.Random(split_seed * 1_000_003 + case_index)
    case_id = _case_id(split_seed, case_index)
    donor_id = int(PROTOCOL["dataset"]["id_offsets"][split]) + case_index + 1
    donor = _make_donor(rng, donor_id, organ)
    recipient_suffixes = list(range(1, int(PROTOCOL["dataset"]["recipients_per_case"]) + 1))
    rng.shuffle(recipient_suffixes)
    recipients = [
        _make_recipient(rng, donor, donor_id * 100 + recipient_suffixes[position], position)
        for position in range(len(recipient_suffixes))
    ]
    rng.shuffle(recipients)
    ranked = rank_recipients_baseline(donor, recipients, PROTOCOL)
    baseline_order = selectable_order(ranked)
    if len(baseline_order) < int(PROTOCOL["dataset"]["minimum_structurally_compatible"]):
        raise RuntimeError(f"Generator produced too few compatible candidates in case {case_id}")

    _assign_states(recipients, baseline_order, scenario, rng)
    _assign_note_composition(recipients, rng)
    _render_notes(split, case_id, recipients, rng)
    by_id = {int(item["recipient_id"]): item for item in recipients}
    guarded_order = [
        recipient_id
        for recipient_id in baseline_order
        if by_id[recipient_id]["reference_note_state"] != "temporary_hold"
    ]

    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_version": PROTOCOL["version"],
        "case_id": case_id,
        "split": split,
        "generation_seed": split_seed,
        "organ_type": organ,
        "scenario_family": scenario,
        "donor": donor,
        "recipients": recipients,
        "reference": {
            "baseline_order": baseline_order,
            "guarded_order": guarded_order,
            "primary_recipient_id": guarded_order[0],
            "backup_recipient_id": guarded_order[1],
        },
    }


def generate_split(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    if count % len(SUPPORTED_ORGANS) != 0:
        raise ValueError("Split size must be divisible by the number of organs")
    per_organ = count // len(SUPPORTED_ORGANS)
    cases: List[Dict[str, Any]] = []
    for organ_index, organ in enumerate(SUPPORTED_ORGANS):
        schedule_rng = random.Random(seed + organ_index * 7919)
        schedule = _scenario_schedule(per_organ, schedule_rng)
        for within_organ, scenario in enumerate(schedule):
            case_index = organ_index * per_organ + within_organ
            cases.append(generate_case(split, seed, case_index, organ, scenario))
    cases.sort(key=lambda item: item["case_id"])
    return cases


def generate_demo_case(seed: int = 2026090704) -> Dict[str, Any]:
    """Create a standalone demonstration case that is not part of any study split."""
    case = generate_case(
        split="demo",
        split_seed=seed,
        case_index=0,
        organ="kidney",
        scenario="top1_temporary_hold",
    )
    id_map: Dict[int, int] = {}
    case["donor"]["donor_id"] = 1
    for new_id, recipient in enumerate(case["recipients"], start=1):
        old_id = int(recipient["recipient_id"])
        id_map[old_id] = new_id
        recipient["recipient_id"] = new_id

    reference = case["reference"]
    reference["baseline_order"] = [id_map[int(value)] for value in reference["baseline_order"]]
    reference["guarded_order"] = [id_map[int(value)] for value in reference["guarded_order"]]
    reference["primary_recipient_id"] = id_map[int(reference["primary_recipient_id"])]
    reference["backup_recipient_id"] = id_map[int(reference["backup_recipient_id"])]
    case["case_id"] = "oda-demo-kidney-001"
    case["split"] = "demonstration"
    case["purpose"] = "Interface and end-to-end workflow demonstration only; excluded from model evaluation."
    for recipient in case["recipients"]:
        recipient["note_instance_id"] = hashlib.sha256(
            f"{case['case_id']}:{recipient['recipient_id']}:{recipient['medical_notes']}".encode("utf-8")
        ).hexdigest()[:12]

    recomputed = selectable_order(rank_recipients_baseline(case["donor"], case["recipients"], PROTOCOL))
    if recomputed != reference["baseline_order"]:
        raise RuntimeError("Demonstration case ID remapping changed the deterministic order")
    return case


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def _fine_tuning_row(case: Mapping[str, Any]) -> Dict[str, Any]:
    payload = build_note_review_payload(case["case_id"], case["organ_type"], case["recipients"])
    assistant = {
        "case_id": case["case_id"],
        "assessments": [
            {
                "recipient_id": int(recipient["recipient_id"]),
                "state": recipient["reference_note_state"],
                "evidence_codes": recipient["reference_evidence_codes"],
            }
            for recipient in case["recipients"]
        ],
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=True, sort_keys=True)},
        ]
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _note_component_pool(split: str) -> set[str]:
    components = {
        sentence
        for variants in NOTE_TEMPLATES[split].values()
        for sentence in variants
    }
    components.update(NOTE_CONTEXTS[split])
    components.update(NOTE_QUALIFIERS[split])
    components.update(NOTE_TRANSITIONS[split])
    components.update(
        sentence
        for variants in NOTE_CONTEXT_EVIDENCE_TEMPLATES[split].values()
        for sentence in variants
    )
    return components


def validate_cases(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    errors: List[str] = []
    case_ids: Dict[str, set[str]] = {}
    donor_ids: Dict[str, set[int]] = {}
    recipient_ids: Dict[str, set[int]] = {}
    note_hashes: Dict[str, set[str]] = {}
    template_families: Dict[str, set[str]] = {}
    split_summaries: Dict[str, Any] = {}
    kidney_overlap_counts = Counter()

    for split, cases in splits.items():
        expected = int(PROTOCOL["dataset"]["splits"][split]["cases"])
        if len(cases) != expected:
            errors.append(f"{split}: expected {expected} cases, found {len(cases)}")
        case_ids[split] = set()
        donor_ids[split] = set()
        recipient_ids[split] = set()
        note_hashes[split] = set()
        template_families[split] = set()
        organ_counts = Counter()
        scenario_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        primary_baseline_rank_counts: Counter[int] = Counter()
        state_counts = Counter()
        code_counts = Counter()
        style_counts = Counter()
        complexity_counts = Counter()
        context_code_counts = Counter()
        evidence_cardinality_counts = Counter()
        exclusion_reason_counts = Counter()
        organ_factor_coverage = Counter()
        split_kidney_overlap_counts = Counter()
        unique_notes_by_code: Dict[str, set[str]] = defaultdict(set)
        compatible_counts: List[int] = []
        selectable_by_payload_position: Dict[int, Counter[str]] = defaultdict(Counter)
        selectable_by_id_suffix: Dict[int, Counter[str]] = defaultdict(Counter)
        readiness_by_payload_position: Dict[int, Counter[str]] = defaultdict(Counter)
        readiness_by_id_suffix: Dict[int, Counter[str]] = defaultdict(Counter)
        payload_order_signatures: set[tuple[int, ...]] = set()

        for case in cases:
            case_id = str(case.get("case_id", ""))
            if not case_id or case_id in case_ids[split]:
                errors.append(f"{split}: missing or duplicate case_id {case_id!r}")
            case_ids[split].add(case_id)
            donor_id = int(case.get("donor", {}).get("donor_id", 0))
            if donor_id <= 0 or donor_id in donor_ids[split]:
                errors.append(f"{split}: missing or duplicate donor_id {donor_id!r}")
            donor_ids[split].add(donor_id)
            organ = str(case.get("organ_type", ""))
            organ_counts[organ] += 1
            scenario_counts[organ][str(case.get("scenario_family", ""))] += 1
            recipients = list(case.get("recipients", []))
            if len(recipients) != int(PROTOCOL["dataset"]["recipients_per_case"]):
                errors.append(f"{case_id}: incorrect recipient count")
                continue

            try:
                ranked = rank_recipients_baseline(case["donor"], recipients, PROTOCOL)
            except Exception as exc:
                errors.append(f"{case_id}: policy failure: {exc}")
                continue
            baseline_order = selectable_order(ranked)
            for item in ranked:
                exclusion_reason_counts.update(item["exclusion_reasons"])
            compatible_counts.append(len(baseline_order))
            if len(baseline_order) < int(PROTOCOL["dataset"]["minimum_structurally_compatible"]):
                errors.append(f"{case_id}: too few structurally compatible candidates")
            if baseline_order != case.get("reference", {}).get("baseline_order"):
                errors.append(f"{case_id}: stored baseline order differs from recomputation")

            by_id = {int(recipient["recipient_id"]): recipient for recipient in recipients}
            if len(by_id) != len(recipients):
                errors.append(f"{case_id}: duplicate recipient ID")
            suffixes = [int(recipient["recipient_id"]) - donor_id * 100 for recipient in recipients]
            expected_suffixes = list(
                range(1, int(PROTOCOL["dataset"]["recipients_per_case"]) + 1)
            )
            if sorted(suffixes) != expected_suffixes:
                errors.append(f"{case_id}: recipient identifiers do not form the expected randomized set")
            payload_order_signatures.add(tuple(suffixes))
            selectable_ids = set(baseline_order)
            for payload_position, recipient in enumerate(recipients, start=1):
                recipient_id = int(recipient["recipient_id"])
                readiness_state = str(recipient.get("reference_note_state", ""))
                recipient_id_suffix = recipient_id - donor_id * 100
                class_name = "selectable" if recipient_id in selectable_ids else "excluded"
                selectable_by_payload_position[payload_position][class_name] += 1
                selectable_by_id_suffix[recipient_id_suffix][class_name] += 1
                readiness_by_payload_position[payload_position][readiness_state] += 1
                readiness_by_id_suffix[recipient_id_suffix][readiness_state] += 1
            repeated_recipient_ids = recipient_ids[split].intersection(by_id)
            if repeated_recipient_ids:
                errors.append(
                    f"{split}: recipient IDs reused across cases: {sorted(repeated_recipient_ids)[:3]}"
                )
            recipient_ids[split].update(by_id)
            guarded = [
                rid for rid in baseline_order
                if by_id[rid].get("reference_note_state") != "temporary_hold"
            ]
            if len(guarded) < int(PROTOCOL["dataset"]["minimum_not_on_hold"]):
                errors.append(f"{case_id}: too few non-held compatible candidates")
            reference = case.get("reference", {})
            if guarded != reference.get("guarded_order"):
                errors.append(f"{case_id}: stored guarded order differs from recomputation")
            if len(guarded) >= 2 and (
                reference.get("primary_recipient_id") != guarded[0]
                or reference.get("backup_recipient_id") != guarded[1]
            ):
                errors.append(f"{case_id}: primary or backup reference is inconsistent")
            if guarded and guarded[0] in baseline_order:
                primary_baseline_rank_counts[baseline_order.index(guarded[0]) + 1] += 1

            for recipient in recipients:
                state = str(recipient.get("reference_note_state", ""))
                codes = recipient.get("reference_evidence_codes", [])
                context_codes = recipient.get("context_evidence_codes", [])
                state_counts[state] += 1
                if state not in STATE_CODES or not codes or any(code not in STATE_CODES[state] for code in codes):
                    errors.append(f"{case_id}: invalid state/evidence pair for recipient {recipient.get('recipient_id')}")
                if len(codes) != len(set(codes)):
                    errors.append(f"{case_id}: duplicate reference evidence code for recipient {recipient.get('recipient_id')}")
                code_counts.update(codes)
                context_code_counts.update(context_codes)
                evidence_cardinality_counts[len(codes)] += 1
                state_index = STATE_PRECEDENCE.index(state) if state in STATE_PRECEDENCE else -1
                lower_codes = {
                    code
                    for lower_state in STATE_PRECEDENCE[state_index + 1 :]
                    for code in STATE_CODES[lower_state]
                } if state_index >= 0 else set()
                if any(code not in lower_codes for code in context_codes):
                    errors.append(
                        f"{case_id}: context evidence is not lower priority for recipient {recipient.get('recipient_id')}"
                    )
                complexity = str(recipient.get("note_complexity", ""))
                complexity_counts[complexity] += 1
                expected_shape = {
                    "direct": (1, 0),
                    "multi_evidence": (2, 0),
                    "contrast": (2, 0) if state == "eligible" else (1, 1),
                    "multi_evidence_contrast": (3, 0) if state == "eligible" else (2, 1),
                }.get(complexity)
                if expected_shape is None or (len(codes), len(context_codes)) != expected_shape:
                    errors.append(
                        f"{case_id}: invalid {complexity!r} note composition for recipient {recipient.get('recipient_id')}"
                    )
                note = str(recipient.get("medical_notes", ""))
                if not note:
                    errors.append(f"{case_id}: empty note")
                style = str(recipient.get("note_style", ""))
                style_counts[style] += 1
                if style not in NOTE_STYLE_ORDERS:
                    errors.append(f"{case_id}: invalid note style {style!r}")
                if any(marker in note.lower() for marker in ("synthetic", "held-out", "validation set", "test set")):
                    errors.append(f"{case_id}: note text exposes benchmark-construction wording")
                note_hashes[split].add(hashlib.sha256(note.encode("utf-8")).hexdigest())
                template_families[split].add(str(recipient.get("template_family_id", "")))
                for code in codes:
                    unique_notes_by_code[str(code)].add(note)

                if organ == "kidney":
                    pediatric = float(recipient["age_years"]) < 18.0
                    donor_kdpi = float(case["donor"]["synthetic_kdpi_percent"])
                    has_epts = "synthetic_epts_percent" in recipient
                    organ_factor_coverage[
                        "kidney_pediatric_candidate" if pediatric else "kidney_adult_candidate"
                    ] += 1
                    organ_factor_coverage[
                        "kidney_epts_omitted_for_pediatric" if pediatric and not has_epts
                        else "kidney_epts_present_for_adult" if not pediatric and has_epts
                        else "kidney_epts_age_schema_error"
                    ] += 1
                    longevity = (
                        not pediatric
                        and donor_kdpi <= 20.0
                        and float(recipient["synthetic_epts_percent"]) <= 20.0
                    )
                    organ_factor_coverage[
                        "kidney_longevity_priority_true" if longevity else "kidney_longevity_priority_false"
                    ] += 1
                    pediatric_priority = pediatric and donor_kdpi <= 35.0
                    organ_factor_coverage[
                        "kidney_pediatric_priority_true"
                        if pediatric_priority else "kidney_pediatric_priority_false"
                    ] += 1
                    cpra = float(recipient["cpra_percent"])
                    organ_factor_coverage[
                        "kidney_cpra_98_or_higher"
                        if cpra >= 98.0 else "kidney_cpra_80_to_97_9"
                        if cpra >= 80.0 else "kidney_cpra_below_80"
                    ] += 1
                elif organ == "liver":
                    organ_factor_coverage[
                        "liver_status_one_true"
                        if recipient["synthetic_status_one"] else "liver_status_one_false"
                    ] += 1
                elif organ == "heart":
                    organ_factor_coverage[
                        f"heart_status_{int(recipient['synthetic_adult_heart_status'])}"
                    ] += 1
                elif organ == "lung":
                    score = float(recipient["synthetic_lung_priority_score"])
                    organ_factor_coverage[
                        "lung_priority_50_or_higher" if score >= 50.0 else "lung_priority_below_50"
                    ] += 1

            if organ == "kidney":
                donor_hla = set(case["donor"]["hla_typing"])
                donor_locus_counts = {
                    locus: sum(value in pool for value in donor_hla)
                    for locus, pool in HLA_POOLS.items()
                }
                if donor_locus_counts != {"A": 2, "B": 2, "DR": 2}:
                    errors.append(f"{case_id}: donor does not have two HLA values per locus")
                for recipient in recipients:
                    locus_counts = {
                        locus: sum(value in pool for value in recipient["hla_typing"])
                        for locus, pool in HLA_POOLS.items()
                    }
                    if any(count != 2 for count in locus_counts.values()):
                        errors.append(
                            f"{case_id}: recipient {recipient['recipient_id']} does not have two HLA values per locus"
                        )
                    overlap = len(donor_hla.intersection(recipient["hla_typing"]))
                    overlap_label = "zero" if overlap == 0 else "nonzero"
                    kidney_overlap_counts[overlap_label] += 1
                    split_kidney_overlap_counts[overlap_label] += 1

        split_summaries[split] = {
            "case_count": len(cases),
            "organ_counts": dict(sorted(organ_counts.items())),
            "scenario_counts_by_organ": {
                organ: dict(sorted(counts.items())) for organ, counts in sorted(scenario_counts.items())
            },
            "reference_primary_baseline_rank_counts": {
                str(rank): count for rank, count in sorted(primary_baseline_rank_counts.items())
            },
            "state_counts": dict(sorted(state_counts.items())),
            "evidence_code_counts": dict(sorted(code_counts.items())),
            "unique_note_count": len(note_hashes[split]),
            "unique_note_proportion": round(
                len(note_hashes[split]) / (len(cases) * int(PROTOCOL["dataset"]["recipients_per_case"])),
                6,
            ) if cases else 0.0,
            "unique_template_family_count": len(template_families[split]),
            "note_style_counts": dict(sorted(style_counts.items())),
            "note_complexity_counts": dict(sorted(complexity_counts.items())),
            "context_evidence_code_counts": dict(sorted(context_code_counts.items())),
            "reference_evidence_cardinality_counts": dict(sorted(evidence_cardinality_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
            "organ_factor_coverage": dict(sorted(organ_factor_coverage.items())),
            "kidney_hla_overlap_counts": dict(sorted(split_kidney_overlap_counts.items())),
            "unique_notes_by_evidence_code": {
                code: len(values) for code, values in sorted(unique_notes_by_code.items())
            },
            "minimum_structurally_compatible": min(compatible_counts) if compatible_counts else 0,
            "maximum_structurally_compatible": max(compatible_counts) if compatible_counts else 0,
            "unique_payload_order_signatures": len(payload_order_signatures),
            "selectability_by_payload_position": {
                str(position): dict(sorted(counts.items()))
                for position, counts in sorted(selectable_by_payload_position.items())
            },
            "selectability_by_recipient_id_suffix": {
                str(suffix): dict(sorted(counts.items()))
                for suffix, counts in sorted(selectable_by_id_suffix.items())
            },
            "readiness_by_payload_position": {
                str(position): dict(sorted(counts.items()))
                for position, counts in sorted(readiness_by_payload_position.items())
            },
            "readiness_by_recipient_id_suffix": {
                str(suffix): dict(sorted(counts.items()))
                for suffix, counts in sorted(readiness_by_id_suffix.items())
            },
        }
        expected_per_organ = expected // len(SUPPORTED_ORGANS)
        expected_scenario_counts = _scenario_allocation(expected_per_organ)
        for organ in SUPPORTED_ORGANS:
            if organ_counts[organ] != expected_per_organ:
                errors.append(f"{split}: {organ} count is not balanced")
            if dict(scenario_counts[organ]) != expected_scenario_counts:
                errors.append(
                    f"{split}: {organ} scenario counts do not match the protocol allocation"
                )
        if not {1, 2, 3}.issubset(primary_baseline_rank_counts):
            errors.append(
                f"{split}: reference-primary displacement does not exercise baseline ranks 1, 2, and 3"
            )
        for state in STATE_CODES:
            if state_counts[state] == 0:
                errors.append(f"{split}: readiness state {state} is absent")
        for style in NOTE_STYLE_ORDERS:
            if style_counts[style] == 0:
                errors.append(f"{split}: note style {style} is absent")
        note_count = len(cases) * int(PROTOCOL["dataset"]["recipients_per_case"])
        for complexity, expected_weight in COMPLEXITY_WEIGHTS.items():
            observed = complexity_counts[complexity] / note_count if note_count else 0.0
            expected_proportion = expected_weight / 100.0
            if abs(observed - expected_proportion) > 0.03:
                errors.append(
                    f"{split}: note complexity {complexity} proportion {observed:.3f} differs "
                    f"from {expected_proportion:.2f} by more than 0.03"
                )
        for reason in ("organ_type_mismatch", "abo_incompatible", "crossmatch_not_negative", "size_incompatible"):
            if exclusion_reason_counts[reason] == 0:
                errors.append(f"{split}: hard-gate reason {reason} is not exercised")
        for coverage_key in (
            "kidney_pediatric_candidate",
            "kidney_adult_candidate",
            "kidney_epts_omitted_for_pediatric",
            "kidney_epts_present_for_adult",
            "kidney_longevity_priority_true",
            "kidney_longevity_priority_false",
            "kidney_pediatric_priority_true",
            "kidney_pediatric_priority_false",
            "kidney_cpra_98_or_higher",
            "kidney_cpra_80_to_97_9",
            "kidney_cpra_below_80",
            "liver_status_one_true",
            "liver_status_one_false",
            "lung_priority_50_or_higher",
            "lung_priority_below_50",
            *(f"heart_status_{status}" for status in range(1, 7)),
        ):
            if organ_factor_coverage[coverage_key] == 0:
                errors.append(f"{split}: organ-factor condition {coverage_key} is not exercised")
        if organ_factor_coverage["kidney_epts_age_schema_error"]:
            errors.append(f"{split}: kidney EPTS fields do not follow the adult-only schema")
        if split_kidney_overlap_counts["zero"] == 0 or split_kidney_overlap_counts["nonzero"] == 0:
            errors.append(f"{split}: kidney cases lack zero or nonzero HLA overlaps")
        for code in (code for codes in STATE_CODES.values() for code in codes):
            if len(unique_notes_by_code[code]) < 80:
                errors.append(f"{split}: evidence code {code} has fewer than 80 distinct rendered notes")
        minimum_unique_proportion = float(
            PROTOCOL["dataset"]["note_generation"]["minimum_unique_note_proportion"][split]
        )
        observed_unique_proportion = (
            len(note_hashes[split])
            / (len(cases) * int(PROTOCOL["dataset"]["recipients_per_case"]))
            if cases
            else 0.0
        )
        if observed_unique_proportion < minimum_unique_proportion:
            errors.append(
                f"{split}: unique-note proportion {observed_unique_proportion:.3f} is below "
                f"{minimum_unique_proportion:.2f}"
            )
        if len(payload_order_signatures) < max(10, len(cases) // 2):
            errors.append(f"{split}: candidate payload order lacks sufficient randomization")
        for position, counts in selectable_by_payload_position.items():
            if not counts["selectable"] or not counts["excluded"]:
                errors.append(
                    f"{split}: payload position {position} reveals structural selectability"
                )
        for suffix, counts in selectable_by_id_suffix.items():
            if not counts["selectable"] or not counts["excluded"]:
                errors.append(
                    f"{split}: recipient ID suffix {suffix} reveals structural selectability"
                )
        expected_states = set(STATE_CODES)
        for position, counts in readiness_by_payload_position.items():
            if not expected_states.issubset(counts):
                errors.append(
                    f"{split}: payload position {position} reveals note-readiness state"
                )
        for suffix, counts in readiness_by_id_suffix.items():
            if not expected_states.issubset(counts):
                errors.append(
                    f"{split}: recipient ID suffix {suffix} reveals note-readiness state"
                )

    split_names = list(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if case_ids[left].intersection(case_ids[right]):
                errors.append(f"Case-ID leakage between {left} and {right}")
            if donor_ids[left].intersection(donor_ids[right]):
                errors.append(f"Donor-ID leakage between {left} and {right}")
            if recipient_ids[left].intersection(recipient_ids[right]):
                errors.append(f"Recipient-ID leakage between {left} and {right}")
            if note_hashes[left].intersection(note_hashes[right]):
                errors.append(f"Exact note leakage between {left} and {right}")
            if template_families[left].intersection(template_families[right]):
                errors.append(f"Template-family leakage between {left} and {right}")
            overlapping_components = _note_component_pool(left).intersection(_note_component_pool(right))
            if overlapping_components:
                errors.append(
                    f"Note-component leakage between {left} and {right}: "
                    f"{sorted(overlapping_components)[:3]}"
                )

    if kidney_overlap_counts["zero"] == 0 or kidney_overlap_counts["nonzero"] == 0:
        errors.append("Kidney cases do not contain both zero and nonzero HLA overlaps")

    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "validated_at_utc": _utc_now(),
        "valid": not errors,
        "errors": errors,
        "split_summaries": split_summaries,
        "kidney_hla_overlap_counts": dict(kidney_overlap_counts),
    }


def generate_all(output_dir: Path = DATASET_DIR) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split, settings in PROTOCOL["dataset"]["splits"].items():
        cases = generate_split(split, int(settings["cases"]), int(settings["seed"]))
        splits[split] = cases
        _write_jsonl(output_dir / SPLIT_FILE_NAMES[split], cases)

    _write_jsonl(output_dir / "fine_tuning_training.jsonl", (_fine_tuning_row(case) for case in splits["training"]))
    _write_jsonl(output_dir / "fine_tuning_validation.jsonl", (_fine_tuning_row(case) for case in splits["validation"]))

    demo_path = APP_DIR / "seed-data" / "demo_case.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    with demo_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(generate_demo_case(), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    report = validate_cases(splits)
    with (output_dir / "validation_report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["valid"]:
        raise RuntimeError("Generated dataset failed validation: " + "; ".join(report["errors"][:10]))

    artifact_names = [
        *SPLIT_FILE_NAMES.values(),
        "fine_tuning_training.jsonl",
        "fine_tuning_validation.jsonl",
        "validation_report.json",
    ]
    manifest = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_version": PROTOCOL["version"],
        "generated_at_utc": _utc_now(),
        "python_version": sys.version.split()[0],
        "source_inputs": {
            "protocol_json": {
                "path": str(DEFAULT_PROTOCOL_PATH.relative_to(IMPLEMENTATION_DIR)).replace("\\", "/"),
                "sha256": _sha256(DEFAULT_PROTOCOL_PATH),
            },
            "generator": {
                "path": str(Path(__file__).resolve().relative_to(IMPLEMENTATION_DIR)).replace("\\", "/"),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "artifacts": {
            name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in artifact_names
        },
        "demonstration_case": {
            "path": str(demo_path.relative_to(IMPLEMENTATION_DIR)).replace("\\", "/"),
            "sha256": _sha256(demo_path),
            "bytes": demo_path.stat().st_size,
            "excluded_from_model_evaluation": True,
        },
    }
    with (output_dir / "generation_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return {"manifest": manifest, "validation": report}


def validate_generation_manifest(output_dir: Path = DATASET_DIR) -> List[str]:
    errors: List[str] = []
    manifest_path = output_dir / "generation_manifest.json"
    if not manifest_path.exists():
        return [f"Missing generation manifest: {manifest_path}"]
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Unreadable generation manifest ({type(exc).__name__})"]

    if manifest.get("protocol_id") != PROTOCOL["protocol_id"]:
        errors.append("Generation manifest protocol ID does not match the active protocol")
    source_inputs = manifest.get("source_inputs", {})
    expected_sources = {
        "protocol_json": (DEFAULT_PROTOCOL_PATH, "Protocol"),
        "generator": (Path(__file__).resolve(), "Generator"),
    }
    for key, (path, label) in expected_sources.items():
        recorded = source_inputs.get(key, {})
        if not path.exists() or recorded.get("sha256") != _sha256(path):
            errors.append(f"{label} hash does not match the generation manifest")
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if source_inputs.get("system_prompt_sha256") != prompt_hash:
        errors.append("System-prompt hash does not match the generation manifest")

    artifacts = manifest.get("artifacts", {})
    expected_artifacts = {
        *SPLIT_FILE_NAMES.values(),
        "fine_tuning_training.jsonl",
        "fine_tuning_validation.jsonl",
        "validation_report.json",
    }
    if set(artifacts) != expected_artifacts:
        errors.append("Generation manifest artifact list is incomplete or contains unexpected entries")
    for name in sorted(expected_artifacts):
        path = output_dir / name
        recorded = artifacts.get(name, {})
        if not path.exists():
            errors.append(f"Manifest artifact is missing: {name}")
            continue
        if recorded.get("sha256") != _sha256(path):
            errors.append(f"Manifest artifact hash mismatch: {name}")
        if recorded.get("bytes") != path.stat().st_size:
            errors.append(f"Manifest artifact byte-count mismatch: {name}")

    demo_path = APP_DIR / "seed-data" / "demo_case.json"
    demo_record = manifest.get("demonstration_case", {})
    if not demo_path.exists() or demo_record.get("sha256") != _sha256(demo_path):
        errors.append("Demonstration-case hash does not match the generation manifest")
    elif demo_record.get("bytes") != demo_path.stat().st_size:
        errors.append("Demonstration-case byte count does not match the generation manifest")
    if demo_record.get("excluded_from_model_evaluation") is not True:
        errors.append("Generation manifest does not exclude the demonstration case from model evaluation")
    return errors


def validate_existing(output_dir: Path = DATASET_DIR) -> Dict[str, Any]:
    splits = {
        split: _load_jsonl(output_dir / file_name)
        for split, file_name in SPLIT_FILE_NAMES.items()
    }
    report = validate_cases(splits)
    manifest_errors = validate_generation_manifest(output_dir)
    report["manifest_valid"] = not manifest_errors
    report["errors"].extend(manifest_errors)
    report["valid"] = not report["errors"]
    return report
