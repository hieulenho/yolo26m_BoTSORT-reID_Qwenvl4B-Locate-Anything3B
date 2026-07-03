from __future__ import annotations

import numpy as np

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackerDetection
from football_tracking.tracking.ultralytics_adapter import (
    UltralyticsTrackerAdapter,
    UltralyticsTrackerRuntimeConfig,
)


class FakeUltralyticsTracker:
    def __init__(self, _args) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, results, _frame):
        assert len(results) == 1
        return np.asarray([[1.0, 2.0, 11.0, 22.0, 7.0, 0.8, 3.0, 0.0]], dtype=np.float32)


def _config() -> UltralyticsTrackerRuntimeConfig:
    return UltralyticsTrackerRuntimeConfig(
        tracker_type="botsort",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=True,
        model="auto",
        output_confirmed_only=True,
        require_recent_update=True,
        max_time_since_update_for_output=0,
    )


def test_ultralytics_tracker_adapter_converts_outputs() -> None:
    adapter = UltralyticsTrackerAdapter(_config(), tracker_factory=FakeUltralyticsTracker)
    detection = TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="seq",
        bbox_xyxy=BoundingBoxXYXY(1, 2, 11, 22),
        confidence=0.8,
        class_id=3,
        class_name="vehicle",
    )

    tracks = adapter.update(1, "seq", [detection], frame=None, image_width=100, image_height=80)

    assert len(tracks) == 1
    assert tracks[0].track_id == 7
    assert tracks[0].class_id == 3
    assert tracks[0].class_name == "vehicle"
    assert tracks[0].metadata["with_reid"] is True
