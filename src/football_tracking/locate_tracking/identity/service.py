"""Commit semantic identity transitions without modifying raw tracker outputs."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.identity.schemas import (
    IdentityStateTransition,
    SemanticIdentitySegment,
    SemanticTarget,
    stable_artifact_id,
)
from football_tracking.locate_tracking.identity.segment_store import save_semantic_target
from football_tracking.locate_tracking.identity.state_machine import validate_transition
from football_tracking.locate_tracking.identity.transition_log import append_transition


class IdentityServiceError(RuntimeError):
    """Raised when an identity transition cannot be committed."""


def _transition(
    *,
    target: SemanticTarget,
    to_state: str,
    frame_index: int,
    event_ids: tuple[str, ...],
    decision_id: str | None,
    new_raw_track_id: int | None,
    reason: str,
    evidence_reference: str | None,
) -> IdentityStateTransition:
    validate_transition(target.state, to_state)  # type: ignore[arg-type]
    return IdentityStateTransition(
        transition_id=stable_artifact_id(
            "transition",
            {
                "semantic_target_id": target.semantic_target_id,
                "from_state": target.state,
                "to_state": to_state,
                "frame_index": frame_index,
                "event_ids": event_ids,
                "decision_id": decision_id,
                "previous_raw_track_id": target.current_raw_track_id,
                "new_raw_track_id": new_raw_track_id,
                "reason": reason,
            },
        ),
        semantic_target_id=target.semantic_target_id,
        from_state=target.state,
        to_state=to_state,  # type: ignore[arg-type]
        frame_index=frame_index,
        event_ids=event_ids,
        decision_id=decision_id,
        previous_raw_track_id=target.current_raw_track_id,
        new_raw_track_id=new_raw_track_id,
        reason=reason,
        evidence_reference=evidence_reference,
    )


def commit_transition(
    *,
    target: SemanticTarget,
    transition: IdentityStateTransition,
    target_path: str | Path,
    transition_log_path: str | Path,
    updated_target: SemanticTarget,
    overwrite_target: bool = True,
) -> SemanticTarget:
    append_transition(transition, transition_log_path)
    save_semantic_target(updated_target, target_path, overwrite=overwrite_target)
    return updated_target


def mark_lost(
    *,
    target: SemanticTarget,
    frame_index: int,
    event_ids: tuple[str, ...],
    target_path: str | Path,
    transition_log_path: str | Path,
    evidence_reference: str | None = None,
) -> SemanticTarget:
    transition = _transition(
        target=target,
        to_state="UNCERTAIN" if target.state == "ACTIVE" else "LOST",
        frame_index=frame_index,
        event_ids=event_ids,
        decision_id=None,
        new_raw_track_id=target.current_raw_track_id,
        reason="uncertainty_or_loss_detected",
        evidence_reference=evidence_reference,
    )
    updated = target.with_updates(state=transition.to_state, last_update_frame=frame_index)
    return commit_transition(
        target=target,
        transition=transition,
        target_path=target_path,
        transition_log_path=transition_log_path,
        updated_target=updated,
    )


def commit_same_raw_resume(
    *,
    target: SemanticTarget,
    resume_frame: int,
    event_ids: tuple[str, ...],
    decision_id: str,
    target_path: str | Path,
    transition_log_path: str | Path,
    evidence_reference: str | None,
) -> SemanticTarget:
    transition = _transition(
        target=target,
        to_state="ACTIVE",
        frame_index=resume_frame,
        event_ids=event_ids,
        decision_id=decision_id,
        new_raw_track_id=target.current_raw_track_id,
        reason="same_raw_id_resumed",
        evidence_reference=evidence_reference,
    )
    updated = target.with_updates(state="ACTIVE", last_update_frame=resume_frame)
    return commit_transition(
        target=target,
        transition=transition,
        target_path=target_path,
        transition_log_path=transition_log_path,
        updated_target=updated,
    )


def commit_provisional_reacquisition(
    *,
    target: SemanticTarget,
    new_raw_track_id: int,
    start_frame: int,
    previous_end_frame: int,
    confidence: float,
    event_ids: tuple[str, ...],
    decision_id: str,
    target_path: str | Path,
    transition_log_path: str | Path,
    evidence_reference: str | None,
) -> SemanticTarget:
    transition = _transition(
        target=target,
        to_state="PROBATION",
        frame_index=start_frame,
        event_ids=event_ids,
        decision_id=decision_id,
        new_raw_track_id=new_raw_track_id,
        reason="provisional_reacquisition",
        evidence_reference=evidence_reference,
    )
    active = target.active_segment
    closed_segments = []
    for segment in target.segments:
        if active is not None and segment.segment_id == active.segment_id:
            closed_segments.append(
                segment.with_updates(end_frame=previous_end_frame, status="closed")
            )
        else:
            closed_segments.append(segment)
    new_segment = SemanticIdentitySegment(
        segment_id=stable_artifact_id(
            "segment",
            {
                "semantic_target_id": target.semantic_target_id,
                "raw_track_id": new_raw_track_id,
                "start_frame": start_frame,
                "decision_id": decision_id,
            },
        ),
        semantic_target_id=target.semantic_target_id,
        raw_track_id=new_raw_track_id,
        start_frame=start_frame,
        end_frame=None,
        source="semantic_reacquisition",
        confidence=confidence,
        status="probation",
        evidence_summary={"decision_id": decision_id},
        transition_id=transition.transition_id,
    )
    updated = target.with_updates(
        state="PROBATION",
        current_raw_track_id=new_raw_track_id,
        segments=tuple(closed_segments) + (new_segment,),
        last_update_frame=start_frame,
    )
    return commit_transition(
        target=target,
        transition=transition,
        target_path=target_path,
        transition_log_path=transition_log_path,
        updated_target=updated,
    )


def confirm_probation(
    *,
    target: SemanticTarget,
    frame_index: int,
    event_ids: tuple[str, ...],
    decision_id: str,
    target_path: str | Path,
    transition_log_path: str | Path,
    evidence_reference: str | None,
) -> SemanticTarget:
    active = target.active_segment
    if active is None or active.status != "probation":
        raise IdentityServiceError("Target has no probation segment to confirm.")
    transition = _transition(
        target=target,
        to_state="ACTIVE",
        frame_index=frame_index,
        event_ids=event_ids,
        decision_id=decision_id,
        new_raw_track_id=active.raw_track_id,
        reason="probation_confirmed",
        evidence_reference=evidence_reference,
    )
    segments = tuple(
        segment.with_updates(status="confirmed", transition_id=transition.transition_id)
        if segment.segment_id == active.segment_id
        else segment
        for segment in target.segments
    )
    updated = target.with_updates(
        state="ACTIVE",
        current_raw_track_id=active.raw_track_id,
        segments=segments,
        last_confirmed_frame=frame_index,
        last_update_frame=frame_index,
    )
    return commit_transition(
        target=target,
        transition=transition,
        target_path=target_path,
        transition_log_path=transition_log_path,
        updated_target=updated,
    )
