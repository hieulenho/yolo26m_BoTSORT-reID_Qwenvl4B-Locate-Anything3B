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
