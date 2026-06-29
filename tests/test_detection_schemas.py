import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.schemas import Detection


def test_detection_validates_and_serializes() -> None:
    detection = Detection(
        frame_index=1,
        sequence_name="sequence_001",
        bbox_xyxy=BoundingBoxXYXY(1, 2, 10, 20),
        confidence=0.9,
        source_class_id=0,
        source_class_name="person",
        target_class_id=0,
        target_class_name="player",
        image_width=64,
        image_height=48,
    )

    payload = detection.to_dict()

    assert payload["confidence"] == 0.9
    assert payload["bbox_xyxy"] == [1, 2, 10, 20]


def test_detection_rejects_invalid_confidence_and_bbox() -> None:
    kwargs = {
        "frame_index": 1,
        "sequence_name": "sequence_001",
        "source_class_id": 0,
        "source_class_name": "person",
        "target_class_id": 0,
        "target_class_name": "player",
        "image_width": 64,
        "image_height": 48,
    }
    with pytest.raises(ValueError, match="confidence"):
        Detection(bbox_xyxy=BoundingBoxXYXY(1, 2, 10, 20), confidence=1.5, **kwargs)
    with pytest.raises(ValueError, match="bbox"):
        Detection(bbox_xyxy=BoundingBoxXYXY(10, 2, 1, 20), confidence=0.5, **kwargs)
