"""Deterministic guard between note classification and match recording."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Any

from .policy import selectable_order
from .schemas import GuardedDecision, NoteAssessment, NoteReviewBatch, ReadinessState


class GuardError(ValueError):
    pass


class GuardAbstention(GuardError):
    pass


def validate_review_coverage(
    review: NoteReviewBatch,
    expected_recipient_ids: Iterable[int],
) -> dict[int, NoteAssessment]:
    expected_values = [int(value) for value in expected_recipient_ids]
    if any(value <= 0 for value in expected_values) or len(expected_values) != len(
        set(expected_values)
    ):
        raise GuardError("Expected recipient IDs must be unique positive integers")
    expected = set(expected_values)
    observed = [int(item.recipient_id) for item in review.assessments]
    if len(observed) != len(set(observed)):
        raise GuardError("The note review contains duplicate recipient IDs")
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise GuardError(f"The note review has incomplete coverage; missing={missing}, extra={extra}")
    return {int(item.recipient_id): item for item in review.assessments}


def apply_guarded_policy(
    donor_id: int,
    ranked: Sequence[Mapping[str, Any]],
    review: NoteReviewBatch,
    expected_recipient_ids: Iterable[int],
) -> GuardedDecision:
    assessments = validate_review_coverage(review, expected_recipient_ids)
    baseline_order = selectable_order(ranked)
    if len(baseline_order) != len(set(baseline_order)):
        raise GuardError("The baseline order contains duplicate recipient IDs")
    unknown = sorted(set(baseline_order).difference(assessments))
    if unknown:
        raise GuardError(f"The baseline order contains recipients outside the reviewed set: {unknown}")
    if len(baseline_order) < 2:
        raise GuardAbstention("Fewer than two structurally compatible candidates")

    permitted = [
        recipient_id
        for recipient_id in baseline_order
        if assessments[recipient_id].state != ReadinessState.TEMPORARY_HOLD
    ]
    if len(permitted) < 2:
        raise GuardAbstention("Fewer than two candidates remain after temporary holds")

    primary, backup = permitted[:2]
    baseline_primary, baseline_backup = baseline_order[:2]
    held_ids = [
        recipient_id
        for recipient_id in baseline_order
        if assessments[recipient_id].state == ReadinessState.TEMPORARY_HOLD
    ]
    review_ids = [
        recipient_id
        for recipient_id in baseline_order
        if assessments[recipient_id].state == ReadinessState.REVIEW_REQUIRED
    ]
    overrode = primary != baseline_primary
    backup_changed = backup != baseline_backup

    if overrode or backup_changed:
        backup_index = baseline_order.index(backup)
        skipped = [
            recipient_id
            for recipient_id in baseline_order[: backup_index + 1]
            if recipient_id in held_ids
        ]
        reason = (
            "The deterministic guard skipped protocol-defined temporary-hold candidates "
            f"{skipped} encountered before completing the primary-backup pair and selected "
            "the next available candidates in baseline order."
        )
    else:
        reason = "No temporary hold displaced either member of the baseline primary-backup pair."

    return GuardedDecision(
        donor_id=int(donor_id),
        baseline_primary_recipient_id=baseline_primary,
        baseline_backup_recipient_id=baseline_backup,
        primary_recipient_id=primary,
        backup_recipient_id=backup,
        overrode_baseline=overrode,
        backup_changed=backup_changed,
        decision_source="deterministic_guard",
        decision_reason=reason,
        temporary_hold_recipient_ids=held_ids,
        review_required_recipient_ids=review_ids,
        note_assessments=list(review.assessments),
    )
