"""Temporal plausibility gates and evidence."""

from __future__ import annotations

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.reacquisition.candidate_generator import (
    observations_by_track,
)
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    GateResult,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def temporal_gate(
    *,
    candidate: ReacquisitionCandidate,
    previous_raw_track_id: int | None,
    all_observations: tuple[MotTrackObservation, ...],
    config: ReacquisitionConfig,
) -> GateResult:
    if candidate.observation_count < config.min_observations:
        return GateResult(
            gate_name="temporal",
            passed=False,
            score=float(candidate.observation_count),
            threshold=float(config.min_observations),
            reason="below_minimum_observations",
        )
    if candidate.raw_track_id == previous_raw_track_id:
        return GateResult(
            gate_name="temporal",
            passed=True,
            score=1.0,
            threshold=None,
            reason="same_raw_id_candidate",
        )
    grouped = observations_by_track(all_observations)
    candidate_rows = grouped.get(candidate.raw_track_id, ())
    conflict_start = max(
        1,
        candidate.search_window.last_confirmed_frame
        - config.duplicate_overlap_tolerance_frames,
    )
    conflict_end = candidate.search_window.last_confirmed_frame
    conflict_rows = [
        row for row in candidate_rows if conflict_start <= row.frame_index <= conflict_end
    ]
    if conflict_rows:
        return GateResult(
            gate_name="temporal",
            passed=False,
            score=0.0,
            threshold=0.0,
            reason="candidate_present_during_previous_target_context",
            metadata={"conflict_frames": [row.frame_index for row in conflict_rows[:10]]},
        )
    return GateResult(
        gate_name="temporal",
        passed=True,
        score=1.0,
        threshold=None,
        reason="temporal_plausible",
    )


def temporal_evidence(candidate: ReacquisitionCandidate) -> EvidenceScore:
    gap = max(0, candidate.first_observed_frame - candidate.search_window.last_confirmed_frame)
    window = max(
        1,
        candidate.search_window.end_frame - candidate.search_window.last_confirmed_frame,
    )
    score = max(0.0, 1.0 - (gap / window))
    return EvidenceScore(
        name="temporal",
        score=score,
        data_available=True,
        reason="gap_from_last_confirmed_frame",
        details={"gap_frames": gap, "normalization_window": window},
    )
