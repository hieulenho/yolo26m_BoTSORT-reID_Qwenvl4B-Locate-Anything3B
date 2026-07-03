from __future__ import annotations

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
