"""Probation checks for provisional reacquisition."""

from __future__ import annotations

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.reacquisition.candidate_generator import (
    observations_by_track,
)
from football_tracking.locate_tracking.reacquisition.schemas import (
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def evaluate_probation(
    *,
    candidate: ReacquisitionCandidate,
    all_observations: tuple[MotTrackObservation, ...],
    config: ReacquisitionConfig,
) -> tuple[bool, dict[str, object]]:
    end_frame = candidate.first_observed_frame + config.probation_window_frames
    rows = [
        row
        for row in observations_by_track(all_observations).get(candidate.raw_track_id, ())
        if candidate.first_observed_frame <= row.frame_index <= end_frame
    ]
    passed = len(rows) >= config.probation_min_observations
    return passed, {
        "probation_start_frame": candidate.first_observed_frame,
        "probation_end_frame": end_frame,
        "observation_count": len(rows),
        "min_observations": config.probation_min_observations,
        "reason": "probation_observations_sufficient"
        if passed
        else "probation_observations_insufficient",
    }
