from __future__ import annotations

import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.cache_schema import (
    CachedDetection,
    CachedFrameDetections,
    DetectionCacheSchemaError,
)


def test_cached_frame_roundtrip_and_tracker_filtering() -> None:
    frame = CachedFrameDetections(
        sequence_name="seq",
        frame_index=1,
        image_path="000001.jpg",
        image_width=100,
        image_height=80,
        detections=[
            CachedDetection(BoundingBoxXYXY(1, 2, 11, 22), 0.9),
            CachedDetection(BoundingBoxXYXY(20, 2, 30, 22), 0.05),
        ],
    )

    loaded = CachedFrameDetections.from_dict(frame.to_dict())
    tracker_detections = loaded.to_tracker_detections(confidence_threshold=0.1)

    assert len(tracker_detections) == 1
    assert tracker_detections[0].metadata["detection_source"] == "cache"


def test_cache_schema_rejects_non_player_class() -> None:
    with pytest.raises(DetectionCacheSchemaError):
        CachedDetection(BoundingBoxXYXY(1, 2, 11, 22), 0.9, class_id=1)
