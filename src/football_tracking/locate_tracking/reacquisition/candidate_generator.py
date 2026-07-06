"""Generate reacquisition candidates from read-only MOT observations."""

from __future__ import annotations

from collections import defaultdict

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.reacquisition.schemas import (
    CandidateSearchWindow,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def build_candidate_search_window(
    *,
    event: UncertaintyEvent,
    last_confirmed_frame: int,
    total_frames: int | None,
    config: ReacquisitionConfig,
) -> CandidateSearchWindow:
    start = max(
        1,
        min(last_confirmed_frame + 1, event.frame_start - config.pre_event_context_frames),
    )
    end = max(event.frame_end, event.trigger_frame) + config.post_event_context_frames
    if total_frames is not None:
        end = min(end, int(total_frames))
    return CandidateSearchWindow(
        start_frame=start,
        end_frame=end,
        last_confirmed_frame=last_confirmed_frame,
        event_start_frame=event.frame_start,
        event_end_frame=event.frame_end,
        pre_event_context_frames=config.pre_event_context_frames,
        post_event_context_frames=config.post_event_context_frames,
        source_event_ids=(event.event_id,),
    )


def observations_by_track(
    observations: tuple[MotTrackObservation, ...],
) -> dict[int, tuple[MotTrackObservation, ...]]:
    grouped: dict[int, list[MotTrackObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.track_id].append(observation)
    return {
        track_id: tuple(sorted(rows, key=lambda item: item.frame_index))
        for track_id, rows in grouped.items()
    }


def generate_reacquisition_candidates(
    *,
    observations: tuple[MotTrackObservation, ...],
    search_window: CandidateSearchWindow,
) -> tuple[ReacquisitionCandidate, ...]:
    in_window = tuple(
        item
        for item in observations
        if search_window.start_frame <= item.frame_index <= search_window.end_frame
    )
    candidates: list[ReacquisitionCandidate] = []
    for track_id, rows in observations_by_track(in_window).items():
        candidates.append(
            ReacquisitionCandidate(
                raw_track_id=track_id,
                search_window=search_window,
                first_observed_frame=rows[0].frame_index,
                last_observed_frame=rows[-1].frame_index,
                observation_count=len(rows),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.first_observed_frame, item.raw_track_id),
        )
    )


def find_same_raw_id_resume(
    *,
    candidates: tuple[ReacquisitionCandidate, ...],
    observations: tuple[MotTrackObservation, ...] = (),
    previous_raw_track_id: int | None,
    event_end_frame: int,
    min_observations: int,
) -> ReacquisitionCandidate | None:
    if previous_raw_track_id is None:
        return None
    for candidate in candidates:
        if candidate.raw_track_id != previous_raw_track_id:
            continue
        after_event_rows = tuple(
            row
            for row in observations
            if row.track_id == previous_raw_track_id
            and event_end_frame < row.frame_index <= candidate.search_window.end_frame
        )
        if after_event_rows:
            if len(after_event_rows) < min_observations:
                continue
            return candidate.with_updates(
                first_observed_frame=after_event_rows[0].frame_index,
                last_observed_frame=after_event_rows[-1].frame_index,
                observation_count=len(after_event_rows),
            )
        if (
            candidate.last_observed_frame > event_end_frame
            and candidate.observation_count >= min_observations
        ):
            return candidate
    return None
