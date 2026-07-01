from __future__ import annotations

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackerDetection
from football_tracking.tracking.sort_adapter import SortRuntimeConfig, SortTrackerAdapter


def _config() -> SortRuntimeConfig:
    return SortRuntimeConfig(
        max_age=1,
        min_hits=2,
        iou_threshold=0.2,
        confirmed_only=True,
        require_recent_update=True,
        max_time_since_update_for_output=1,
        output_predicted_tracks_without_detection=False,
    )


def _det(frame: int, x1: float = 0.0) -> TrackerDetection:
    return TrackerDetection.from_xyxy(
        frame_index=frame,
        sequence_name="seq",
        bbox_xyxy=BoundingBoxXYXY(x1, 0, x1 + 10, 10),
        confidence=0.8,
    )


def test_sort_lifecycle_confirmation_and_reset() -> None:
    adapter = SortTrackerAdapter(_config())
    adapter.reset()
    assert adapter.update(1, "seq", [_det(1)], image_width=100, image_height=100) == []
    tracks = adapter.update(2, "seq", [_det(2, 1)], image_width=100, image_height=100)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].confidence == 0.8
    assert tracks[0].metadata["bbox_source"] == "matched_detection"

    assert adapter.update(3, "seq", [], image_width=100, image_height=100) == []
    assert adapter.update(4, "seq", [], image_width=100, image_height=100) == []
    assert adapter.tracks == []

    adapter.reset()
    assert adapter.next_track_id == 1


def test_sort_adapter_accepts_no_frame_argument() -> None:
    adapter = SortTrackerAdapter(_config())
    adapter.reset()
    adapter.update(1, "seq", [_det(1)])
    tracks = adapter.update(2, "seq", [_det(2)])
    assert tracks[0].sequence_name == "seq"
