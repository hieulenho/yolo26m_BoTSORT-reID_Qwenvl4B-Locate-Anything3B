"""Simple read-only motion continuity evidence."""

from __future__ import annotations

import math

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.geometry import bbox_center
from football_tracking.locate_tracking.reacquisition.candidate_generator import (
    observations_by_track,
)
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    GateResult,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def _diag(rows: tuple[MotTrackObservation, ...]) -> float:
    max_x = max((row.bbox_xyxy[2] for row in rows), default=1.0)
    max_y = max((row.bbox_xyxy[3] for row in rows), default=1.0)
    return max(1.0, math.hypot(max_x, max_y))


def motion_evidence(
    *,
    candidate: ReacquisitionCandidate,
    previous_raw_track_id: int | None,
    all_observations: tuple[MotTrackObservation, ...],
    config: ReacquisitionConfig,
) -> EvidenceScore:
    if previous_raw_track_id is None:
        return EvidenceScore(
            name="motion",
            score=None,
            data_available=False,
            reason="previous_raw_track_unavailable",
        )
    grouped = observations_by_track(all_observations)
    previous_rows = [
        row
        for row in grouped.get(previous_raw_track_id, ())
        if row.frame_index <= candidate.search_window.last_confirmed_frame
    ]
    candidate_rows = grouped.get(candidate.raw_track_id, ())
    if not previous_rows or not candidate_rows:
        return EvidenceScore(name="motion", score=None, data_available=False, reason="missing_rows")
    previous = previous_rows[-1]
    first_candidate = next(
        (row for row in candidate_rows if row.frame_index >= candidate.first_observed_frame),
        None,
    )
    if first_candidate is None:
        return EvidenceScore(
            name="motion",
            score=None,
            data_available=False,
            reason="candidate_first_row_missing",
        )
    px, py = bbox_center(previous.bbox_xyxy)
    cx, cy = bbox_center(first_candidate.bbox_xyxy)
    normalized = math.hypot(cx - px, cy - py) / _diag(all_observations)
    score = max(0.0, 1.0 - (normalized / max(config.max_motion_distance_normalized, 1e-6)))
    return EvidenceScore(
        name="motion",
        score=score,
        data_available=True,
        reason="last_confirmed_center_to_candidate_first_center",
        details={
            "distance_normalized": normalized,
            "previous_frame": previous.frame_index,
            "candidate_frame": first_candidate.frame_index,
        },
    )


def motion_gate(evidence: EvidenceScore, config: ReacquisitionConfig) -> GateResult:
    if not evidence.data_available or evidence.score is None:
        return GateResult(
            gate_name="motion",
            passed=True,
            score=None,
            threshold=1.0 - config.max_motion_distance_normalized,
            reason="motion_unavailable_not_hard_rejected",
        )
    distance = evidence.details.get("distance_normalized")
    passed = distance is None or float(distance) <= config.max_motion_distance_normalized
    return GateResult(
        gate_name="motion",
        passed=passed,
        score=evidence.score,
        threshold=1.0 - config.max_motion_distance_normalized,
        reason="motion_plausible" if passed else "outside_motion_gate",
        metadata=dict(evidence.details),
    )
