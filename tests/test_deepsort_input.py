from __future__ import annotations

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.deepsort_adapter import detections_to_deepsort_input
from football_tracking.tracking.schemas import TrackerDetection


def test_detections_to_deepsort_input_converts_xyxy_to_ltwh() -> None:
    detection = TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="seq",
        bbox_xyxy=BoundingBoxXYXY(10, 20, 40, 70),
        confidence=0.75,
    )

    raw = detections_to_deepsort_input([detection])

    assert raw == [([10, 20, 30, 50], 0.75, "player")]


def test_detections_to_deepsort_input_accepts_empty_input() -> None:
    assert detections_to_deepsort_input([]) == []
