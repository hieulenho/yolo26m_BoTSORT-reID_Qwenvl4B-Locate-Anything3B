from __future__ import annotations

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.assignment import associate_detections_to_tracks


def test_assignment_handles_empty_inputs() -> None:
    result = associate_detections_to_tracks([], [BoundingBoxXYXY(0, 0, 10, 10)], 0.3)
    assert result.matched_pairs == []
    assert result.unmatched_track_indices == []
    assert result.unmatched_detection_indices == [0]


def test_assignment_is_one_to_one_and_thresholded() -> None:
    tracks = [BoundingBoxXYXY(0, 0, 10, 10), BoundingBoxXYXY(100, 100, 110, 110)]
    detections = [BoundingBoxXYXY(1, 1, 11, 11), BoundingBoxXYXY(200, 200, 210, 210)]
    result = associate_detections_to_tracks(tracks, detections, 0.3)
    assert result.matched_pairs == [(0, 0)]
    assert result.unmatched_track_indices == [1]
    assert result.unmatched_detection_indices == [1]
