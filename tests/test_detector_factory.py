from __future__ import annotations

import pytest

from football_tracking.detection.detector import DetectorError
from football_tracking.detection.detector_factory import create_detector, detector_name_from_config


def test_detector_factory_uses_configured_detector_name() -> None:
    detector = create_detector(
        {"name": "yolo26m", "backend": "ultralytics"},
        "yolo26m.pt",
        device="cpu",
    )

    assert detector.model_name == "yolo26m"
    assert detector.metadata()["backend"] == "ultralytics"


def test_detector_name_defaults_to_checkpoint_stem() -> None:
    assert detector_name_from_config({}, "custom_best.pt") == "custom_best"


class FakeYOLOE:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.classes: list[str] = []

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes


def test_detector_factory_configures_yoloe_text_vocabulary() -> None:
    detector = create_detector(
        {
            "name": "yoloe26s",
            "backend": "ultralytics_yoloe",
            "text_classes": ["car", "ambulance", "car"],
        },
        "yoloe-26s-seg.pt",
        device="cpu",
        model_factory=FakeYOLOE,
    )

    model = detector.load_model()

    assert model.classes == ["car", "ambulance"]
    assert detector.metadata()["text_classes"] == ["car", "ambulance"]
    assert detector.metadata()["backend"] == "ultralytics_yoloe"


def test_detector_factory_rejects_empty_yoloe_vocabulary() -> None:
    with pytest.raises(DetectorError, match="at least one"):
        create_detector(
            {"backend": "yoloe", "text_classes": []},
            "yoloe-26s-seg.pt",
            device="cpu",
        )


class FakeRoutedModel:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.calls = 0

    def __call__(self, _frame, **_kwargs):
        self.calls += 1
        class_id = 32 if "yolo26n" in self.weights else 0
        return {"xyxy": [[1, 2, 10, 20]], "conf": [0.9], "cls": [class_id]}


def test_detector_factory_routes_and_remaps_supplemental_classes() -> None:
    detector = create_detector(
        {
            "name": "football",
            "backend": "ultralytics",
            "supplemental_detectors": [
                {
                    "name": "ball",
                    "backend": "ultralytics",
                    "checkpoint": "yolo26n.pt",
                    "input_class_ids": [32],
                    "output_class_ids": [32],
                    "class_names": ["sports ball"],
                    "every_n_frames": 2,
                }
            ],
        },
        "yolo26m.pt",
        device="cpu",
        model_factory=FakeRoutedModel,
    )

    first = detector.predict_frame(object())
    second = detector.predict_frame(object())

    assert [int(row["class_id"]) for row in first] == [0, 32]
    assert first[1]["class_name"] == "sports ball"
    assert [int(row["class_id"]) for row in second] == [0]
    assert detector.metadata()["supplemental"][0]["inference_calls"] == 1
