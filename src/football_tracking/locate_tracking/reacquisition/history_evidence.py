"""Identity history conflict evidence."""

from __future__ import annotations

from football_tracking.locate_tracking.identity.schemas import SemanticTarget
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    GateResult,
    ReacquisitionCandidate,
)


def history_evidence(
    *,
    candidate: ReacquisitionCandidate,
    target: SemanticTarget,
) -> EvidenceScore:
    conflicts = [
        segment.segment_id
        for segment in target.segments
        if segment.raw_track_id == candidate.raw_track_id
        and segment.status in {"confirmed", "probation"}
        and segment.overlaps(
            type(segment)(
                segment_id="candidate_interval",
                semantic_target_id=target.semantic_target_id,
                raw_track_id=candidate.raw_track_id,
                start_frame=candidate.first_observed_frame,
                end_frame=candidate.last_observed_frame,
                source="candidate",
                confidence=1.0,
                status="confirmed",
            )
        )
    ]
    if conflicts:
        return EvidenceScore(
            name="history",
            score=0.0,
            data_available=True,
            reason="candidate_conflicts_with_existing_segment",
            details={"conflicting_segment_ids": conflicts},
        )
    return EvidenceScore(
        name="history",
        score=1.0,
        data_available=True,
        reason="no_identity_history_conflict",
    )


def identity_conflict_gate(evidence: EvidenceScore) -> GateResult:
    passed = evidence.score is None or evidence.score > 0.0
    return GateResult(
        gate_name="identity_conflict",
        passed=passed,
        score=evidence.score,
        threshold=0.0,
        reason=evidence.reason,
        metadata=dict(evidence.details),
    )
