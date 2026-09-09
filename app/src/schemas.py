from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReadinessState(str, Enum):
    ELIGIBLE = "eligible"
    REVIEW_REQUIRED = "review_required"
    TEMPORARY_HOLD = "temporary_hold"


class EvidenceCode(str, Enum):
    ACTIVE_INFECTION = "active_infection"
    SEPSIS_OR_INSTABILITY = "sepsis_or_instability"
    ACTIVE_MALIGNANCY_TREATMENT = "active_malignancy_treatment"
    EXPLICIT_TEMPORARY_DEFERRAL = "explicit_temporary_deferral"
    INFECTION_WORKUP_PENDING = "infection_workup_pending"
    INDETERMINATE_LESION = "indeterminate_lesion"
    SPECIALIST_CLEARANCE_PENDING = "specialist_clearance_pending"
    MISSING_OR_CONFLICTING_INFORMATION = "missing_or_conflicting_information"
    NO_CURRENT_CONCERN = "no_current_concern"
    NEGATED_CONDITION = "negated_condition"
    RESOLVED_CONDITION = "resolved_condition"
    HISTORICAL_CONDITION_CLEARED = "historical_condition_cleared"


class MatchRequest(StrictModel):
    donor_id: StrictInt = Field(..., ge=1, description="On-chain donor ID")


class BaselineCandidate(StrictModel):
    rank: Optional[StrictInt] = Field(default=None, ge=1)
    recipient_id: StrictInt = Field(..., ge=1)
    score: float = Field(..., ge=0, le=100)
    priority_tier: StrictInt = Field(..., ge=0)
    selectable: StrictBool
    exclusion_reasons: List[str] = Field(default_factory=list)
    factors: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ranking_fields_are_consistent(self) -> "BaselineCandidate":
        if self.selectable != (self.rank is not None):
            raise ValueError("Only selectable candidates may have a rank")
        if self.selectable and self.exclusion_reasons:
            raise ValueError("Selectable candidates cannot have exclusion reasons")
        if not self.selectable and not self.exclusion_reasons:
            raise ValueError("Unselectable candidates require at least one exclusion reason")
        for name, value in self.factors.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Factor {name} must be numeric")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"Factor {name} must be between zero and one")
        return self


class NoteAssessment(StrictModel):
    recipient_id: StrictInt = Field(..., ge=1)
    state: ReadinessState
    evidence_codes: List[EvidenceCode] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_matches_state(self) -> "NoteAssessment":
        allowed = {
            ReadinessState.ELIGIBLE: {
                EvidenceCode.NO_CURRENT_CONCERN,
                EvidenceCode.NEGATED_CONDITION,
                EvidenceCode.RESOLVED_CONDITION,
                EvidenceCode.HISTORICAL_CONDITION_CLEARED,
            },
            ReadinessState.REVIEW_REQUIRED: {
                EvidenceCode.INFECTION_WORKUP_PENDING,
                EvidenceCode.INDETERMINATE_LESION,
                EvidenceCode.SPECIALIST_CLEARANCE_PENDING,
                EvidenceCode.MISSING_OR_CONFLICTING_INFORMATION,
            },
            ReadinessState.TEMPORARY_HOLD: {
                EvidenceCode.ACTIVE_INFECTION,
                EvidenceCode.SEPSIS_OR_INSTABILITY,
                EvidenceCode.ACTIVE_MALIGNANCY_TREATMENT,
                EvidenceCode.EXPLICIT_TEMPORARY_DEFERRAL,
            },
        }
        if not set(self.evidence_codes).issubset(allowed[self.state]):
            raise ValueError("Evidence codes must belong to the selected readiness state")
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("Evidence codes must not be duplicated")
        return self


class NoteReviewBatch(StrictModel):
    case_id: str = Field(..., min_length=1)
    assessments: List[NoteAssessment] = Field(min_length=1)

    @model_validator(mode="after")
    def assessment_ids_are_unique(self) -> "NoteReviewBatch":
        identifiers = [item.recipient_id for item in self.assessments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Assessment recipient IDs must be unique")
        return self


class GuardedDecision(StrictModel):
    donor_id: StrictInt = Field(..., ge=1)
    baseline_primary_recipient_id: StrictInt = Field(..., ge=1)
    baseline_backup_recipient_id: StrictInt = Field(..., ge=1)
    primary_recipient_id: StrictInt = Field(..., ge=1)
    backup_recipient_id: StrictInt = Field(..., ge=1)
    overrode_baseline: StrictBool
    backup_changed: StrictBool
    decision_source: Literal["deterministic_guard"]
    decision_reason: str = Field(..., min_length=1)
    temporary_hold_recipient_ids: List[StrictInt] = Field(default_factory=list)
    review_required_recipient_ids: List[StrictInt] = Field(default_factory=list)
    note_assessments: List[NoteAssessment] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_fields_are_consistent(self) -> "GuardedDecision":
        if self.baseline_primary_recipient_id == self.baseline_backup_recipient_id:
            raise ValueError("Baseline primary and backup IDs must differ")
        if self.primary_recipient_id == self.backup_recipient_id:
            raise ValueError("Guarded primary and backup IDs must differ")
        if self.overrode_baseline != (
            self.primary_recipient_id != self.baseline_primary_recipient_id
        ):
            raise ValueError("Primary override flag does not match the selected IDs")
        if self.backup_changed != (
            self.backup_recipient_id != self.baseline_backup_recipient_id
        ):
            raise ValueError("Backup-change flag does not match the selected IDs")

        held = self.temporary_hold_recipient_ids
        review = self.review_required_recipient_ids
        if any(value < 1 for value in [*held, *review]):
            raise ValueError("Guard classification lists require positive recipient IDs")
        if len(held) != len(set(held)) or len(review) != len(set(review)):
            raise ValueError("Guard classification lists cannot contain duplicate IDs")
        if set(held).intersection(review):
            raise ValueError("Temporary-hold and review-required IDs must be disjoint")
        if self.primary_recipient_id in held or self.backup_recipient_id in held:
            raise ValueError("A temporary-hold candidate cannot be selected")

        assessment_by_id = {item.recipient_id: item for item in self.note_assessments}
        if len(assessment_by_id) != len(self.note_assessments):
            raise ValueError("Decision note assessments cannot contain duplicate IDs")
        required = {
            self.baseline_primary_recipient_id,
            self.baseline_backup_recipient_id,
            self.primary_recipient_id,
            self.backup_recipient_id,
            *held,
            *review,
        }
        if not required.issubset(assessment_by_id):
            raise ValueError("Decision identifiers must have corresponding note assessments")
        if any(
            assessment_by_id[value].state != ReadinessState.TEMPORARY_HOLD
            for value in held
        ):
            raise ValueError("Temporary-hold IDs must reference temporary-hold assessments")
        if any(
            assessment_by_id[value].state != ReadinessState.REVIEW_REQUIRED
            for value in review
        ):
            raise ValueError("Review-required IDs must reference review-required assessments")
        if any(
            assessment_by_id[value].state == ReadinessState.TEMPORARY_HOLD
            for value in (self.primary_recipient_id, self.backup_recipient_id)
        ):
            raise ValueError("A selected candidate cannot have a temporary-hold assessment")
        return self


class ModelRunMetadata(StrictModel):
    model_id: str = Field(..., min_length=1)
    provider: Optional[str] = None
    model_revision: Optional[str] = None
    adapter_id: Optional[str] = None
    adapter_sha256: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    response_id: Optional[str] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    latency_ms: float = Field(..., ge=0)
    input_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    output_tokens: Optional[StrictInt] = Field(default=None, ge=0)


class OnChainRecord(StrictModel):
    tx_hash: str = Field(..., pattern=r"^0x[0-9a-fA-F]{64}$")
    match_id: Optional[StrictInt] = Field(default=None, ge=1)
    gas_used: Optional[StrictInt] = Field(default=None, ge=1)
    contract_address: str = Field(..., pattern=r"^0x[0-9a-fA-F]{40}$")


class MatchResponse(StrictModel):
    donor_id: StrictInt = Field(..., ge=1)
    baseline_top: List[BaselineCandidate]
    guarded_decision: GuardedDecision
    model_run: ModelRunMetadata
    match_cid: str
    onchain: OnChainRecord

    @model_validator(mode="after")
    def response_fields_are_consistent(self) -> "MatchResponse":
        if self.donor_id != self.guarded_decision.donor_id:
            raise ValueError("Response and guarded-decision donor IDs differ")
        identifiers = [item.recipient_id for item in self.baseline_top]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Baseline output cannot contain duplicate recipient IDs")
        decision_identifiers = {
            self.guarded_decision.baseline_primary_recipient_id,
            self.guarded_decision.baseline_backup_recipient_id,
            self.guarded_decision.primary_recipient_id,
            self.guarded_decision.backup_recipient_id,
        }
        if not decision_identifiers.issubset(identifiers):
            raise ValueError("Decision recipients must appear in the baseline output")
        selectable = [item for item in self.baseline_top if item.selectable]
        expected_ranks = list(range(1, len(selectable) + 1))
        if (
            len(selectable) < 2
            or self.baseline_top[: len(selectable)] != selectable
            or [item.rank for item in selectable] != expected_ranks
        ):
            raise ValueError("Selectable baseline candidates must be a consecutive ranked prefix")
        if (
            selectable[0].recipient_id
            != self.guarded_decision.baseline_primary_recipient_id
            or selectable[1].recipient_id
            != self.guarded_decision.baseline_backup_recipient_id
        ):
            raise ValueError("Decision baseline IDs must match baseline ranks one and two")
        assessment_ids = {
            item.recipient_id for item in self.guarded_decision.note_assessments
        }
        if assessment_ids != set(identifiers):
            raise ValueError("Decision assessments must cover the complete baseline output")
        return self
