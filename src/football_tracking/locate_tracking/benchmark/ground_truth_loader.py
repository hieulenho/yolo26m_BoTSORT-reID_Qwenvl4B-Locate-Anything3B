"""Load ground-truth target observations for language-query evaluation."""

from __future__ import annotations

from collections import defaultdict

from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.benchmark.schemas import (
    LanguageQueryAnnotation,
    LanguageTrackingSequence,
)


def ground_truth_observations_for_query(
    sequence: LanguageTrackingSequence,
    query: LanguageQueryAnnotation,
) -> dict[int, tuple[MotTrackObservation, ...]]:
    rows = read_mot_track_file(sequence.mot_ground_truth_path).observations
    by_frame: dict[int, list[MotTrackObservation]] = defaultdict(list)
    for segment in query.identity_segments:
        for row in rows:
            if row.track_id != segment.gt_track_id:
                continue
            if not segment.start_frame <= row.frame_index <= segment.end_frame:
                continue
            if not query.evaluation_start_frame <= row.frame_index <= query.evaluation_end_frame:
                continue
            by_frame[row.frame_index].append(row)
    return {frame: tuple(items) for frame, items in by_frame.items()}
