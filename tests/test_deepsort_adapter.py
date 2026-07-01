from __future__ import annotations

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.deepsort_adapter import (
    DeepSortRuntimeConfig,
    DeepSortTrackerAdapter,
)
from football_tracking.tracking.schemas import TrackerDetection


class FakeTrack:
    track_id = 7
    hits = 3

    def __init__(self, state: str = "confirmed", stale: int = 0) -> None:
        self.state = state
        self.time_since_update = stale

    def is_deleted(self) -> bool:
        return self.state == "deleted"

    def is_tentative(self) -> bool:
        return self.state == "tentative"

    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    def to_ltrb(self, orig: bool = False, orig_strict: bool = False) -> list[float] | None:
        if orig and orig_strict:
            return [1, 2, 11, 22]
        return [3, 4, 13, 24]

    def get_det_conf(self) -> float:
        return 0.8

    def get_det_class(self) -> str:
        return "player"


class FakeDeepSort:
    calls: list[list[object]]

    def __init__(self, **_kwargs: object) -> None:
        self.calls = []
        self.next_tracks = [FakeTrack()]

    def update_tracks(self, raw_detections, frame=None):
        self.calls.append(raw_detections)
        return self.next_tracks


def _config() -> DeepSortRuntimeConfig:
    return DeepSortRuntimeConfig(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nn_budget=100,
        embedder="mobilenet",
        half=False,
        bgr=True,
        embedder_gpu=False,
        polygon=False,
        today=None,
        only_position=False,
        use_appearance=True,
        confirmed_only=True,
        require_recent_update=True,
        max_time_since_update_for_output=1,
        use_original_detection_box=True,
    )


def _detection() -> TrackerDetection:
    return TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="seq",
        bbox_xyxy=BoundingBoxXYXY(1, 2, 11, 22),
        confidence=0.9,
    )


def test_adapter_initializes_once_and_prefers_original_box() -> None:
    adapter = DeepSortTrackerAdapter(_config(), tracker_factory=FakeDeepSort)
    adapter.reset()
    tracks = adapter.update(
        1,
        "seq",
        [_detection()],
        frame=object(),
        image_width=100,
        image_height=100,
    )
    adapter.update(2, "seq", [], frame=object(), image_width=100, image_height=100)

    assert adapter.initialization_count == 1
    assert tracks[0].bbox_xyxy == BoundingBoxXYXY(1, 2, 11, 22)
    assert tracks[0].metadata["bbox_source"] == "original_detection"
    assert adapter.tracker.calls[1] == []


def test_adapter_filters_tentative_and_stale_tracks() -> None:
    adapter = DeepSortTrackerAdapter(_config(), tracker_factory=FakeDeepSort)
    adapter.reset()
    adapter.tracker.next_tracks = [FakeTrack("tentative"), FakeTrack("confirmed", stale=2)]

    assert adapter.update(1, "seq", [_detection()], object(), 100, 100) == []
