import math

from football_tracking.detection.postprocessing import postprocess_detections


def test_postprocessing_keeps_only_person_filters_confidence_and_clips() -> None:
    raw = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.95, "cls": 0},
        {"xyxy": [0, 0, 5, 5], "conf": 0.99, "cls": 32},
        {"xyxy": [1, 1, 4, 4], "conf": 0.1, "cls": 0},
        {"xyxy": [-5, 5, 70, 40], "conf": 0.8, "cls": 0},
        {"xyxy": [math.nan, 1, 4, 4], "conf": 0.8, "cls": 0},
    ]

    detections = postprocess_detections(
        raw,
        frame_index=1,
        sequence_name="sequence_001",
        image_width=64,
        image_height=48,
        confidence_threshold=0.25,
    )

    assert [detection.confidence for detection in detections] == [0.95, 0.8]
    assert detections[1].bbox_xyxy.x1 == 0
    assert detections[1].bbox_xyxy.x2 == 64
    assert {detection.target_class_name for detection in detections} == {"player"}


def test_postprocessing_sorts_by_frame_confidence_and_coordinates() -> None:
    raw = [
        {"xyxy": [20, 1, 30, 10], "conf": 0.5, "cls": 0},
        {"xyxy": [10, 1, 30, 10], "conf": 0.9, "cls": 0},
        {"xyxy": [5, 1, 30, 10], "conf": 0.9, "cls": 0},
    ]

    detections = postprocess_detections(raw, 2, "sequence_001", 64, 48, 0.25)

    assert [detection.bbox_xyxy.x1 for detection in detections] == [5, 10, 20]
