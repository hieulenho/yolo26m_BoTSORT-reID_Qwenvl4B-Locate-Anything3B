from __future__ import annotations

import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackOutput


def test_track_output_serialization_is_stable() -> None:
    track = TrackOutput.from_xyxy(
        frame_index=1,
        sequence_name="seq",
        track_id=7,
        bbox_xyxy=BoundingBoxXYXY(1, 2, 11, 22),
        confidence=None,
    )

    payload = track.to_dict()

    assert payload["track_id"] == 7
    assert payload["confidence"] is None
    assert payload["bbox_ltwh"] == [1, 2, 10, 20]


def test_track_output_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackOutput.from_xyxy(
            frame_index=1,
            sequence_name="seq",
            track_id=-1,
            bbox_xyxy=BoundingBoxXYXY(1, 2, 11, 22),
            confidence=0.5,
        )


def test_track_output_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError, match="Invalid track bbox"):
        TrackOutput.from_xyxy(
            frame_index=1,
            sequence_name="seq",
            track_id=1,
            bbox_xyxy=BoundingBoxXYXY(10, 2, 1, 22),
            confidence=0.5,
        )
