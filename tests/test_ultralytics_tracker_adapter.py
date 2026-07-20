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
        self.frame_id = 1
        self.tracked_stracks = [FakeUltralyticsTrack()]

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, results, _frame):
        assert len(results) == 1
        return np.asarray([[1.0, 2.0, 11.0, 22.0, 7.0, 0.8, 3.0, 0.0]], dtype=np.float32)


class FakeUltralyticsTrack:
    track_id = 7
    is_activated = True
    frame_id = 1
    tracklet_len = 4


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
        min_hits_for_output=1,
        compact_ids=False,
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
    assert tracks[0].metadata["raw_track_id"] == 7


def test_ultralytics_tracker_adapter_filters_new_tracks_and_compacts_ids() -> None:
    config = _config()
    config = UltralyticsTrackerRuntimeConfig(
        **{**config.to_dict(), "min_hits_for_output": 5, "compact_ids": True}
    )
    adapter = UltralyticsTrackerAdapter(config, tracker_factory=FakeUltralyticsTracker)
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
    assert tracks[0].track_id == 1
    assert tracks[0].metadata["raw_track_id"] == 7

    adapter.tracker.tracked_stracks[0].tracklet_len = 3
    assert adapter.update(2, "seq", [detection], frame=None, image_width=100, image_height=80) == []


class FakeClassAwareTracker:
    next_class_id = 0

    def __init__(self, _args) -> None:
        self.class_id = FakeClassAwareTracker.next_class_id
        FakeClassAwareTracker.next_class_id += 1
        self.frame_id = 1
        self.tracked_stracks = [FakeUltralyticsTrack()]

    def update(self, results, _frame):
        if len(results) == 0:
            return np.empty((0, 8), dtype=np.float32)
        class_id = int(results.cls[0])
        return np.asarray(
            [[1.0, 2.0, 11.0, 22.0, 7.0, 0.8, class_id, 0.0]],
            dtype=np.float32,
        )


def test_class_aware_tracking_keeps_class_names_and_global_ids_distinct() -> None:
    FakeClassAwareTracker.next_class_id = 0
    config = UltralyticsTrackerRuntimeConfig(
        **{**_config().to_dict(), "class_aware": True, "compact_ids": True}
    )
    adapter = UltralyticsTrackerAdapter(config, tracker_factory=FakeClassAwareTracker)
    detections = [
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="seq",
            bbox_xyxy=BoundingBoxXYXY(1, 2, 11, 22),
            confidence=0.8,
            class_id=2,
            class_name="car",
        ),
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="seq",
            bbox_xyxy=BoundingBoxXYXY(20, 2, 30, 22),
            confidence=0.9,
            class_id=5,
            class_name="bus",
        ),
    ]

    tracks = adapter.update(
        1,
        "seq",
        detections,
        frame=None,
        image_width=100,
        image_height=80,
    )

    assert [track.track_id for track in tracks] == [1, 2]
    assert [track.class_name for track in tracks] == ["car", "bus"]
    assert all(track.metadata["raw_track_id"] == 7 for track in tracks)
    assert adapter.initialization_count == 2
