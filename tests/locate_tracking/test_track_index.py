from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.locate_tracking.artifacts.mot_schemas import (
    MotArtifactError,
    MotTrackObservation,
)
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex


def _obs(frame: int, track: int) -> MotTrackObservation:
    return MotTrackObservation(
        frame_index=frame,
        track_id=track,
        bbox_ltwh=(0.0, 0.0, 10.0, 10.0),
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        confidence=None,
        source_path=Path("tracks.txt"),
        line_number=track,
    )


def test_track_index_frame_lookup_and_missing_frame() -> None:
    index = FrameTrackIndex.from_observations([_obs(1, 2), _obs(2, 1)])

    assert [row.track_id for row in index.get_frame(1)] == [2]
    assert [row.track_id for row in index.get_frame(2)] == [1]
    assert index.get_frame(3) == ()


def test_track_index_orders_observations_and_reports_metadata() -> None:
    source = [_obs(1, 3), _obs(1, 1), _obs(2, 2)]
    index = FrameTrackIndex.from_observations(source)

    assert [row.track_id for row in index.get_frame(1)] == [1, 3]
    assert index.available_frame_range == (1, 2)
    assert index.unique_track_ids == (1, 2, 3)
    assert index.observation_count == 3
    assert index.frame_count_with_tracks == 2
    assert [row.track_id for row in source] == [3, 1, 2]


def test_track_index_rejects_duplicates() -> None:
    with pytest.raises(MotArtifactError, match="Duplicate"):
        FrameTrackIndex.from_observations([_obs(1, 1), _obs(1, 1)])
