"""Detector registry and factory helpers.

The project originally hard-coded YOLOv8m.  This module keeps that path working
while allowing config-selected Ultralytics detector families such as YOLO26.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.detection.detector import DetectorError, UltralyticsDetector


def detector_name_from_config(model_config: dict[str, Any], checkpoint: str | Path) -> str:
    configured = model_config.get("name") or model_config.get("detector_name")
    if configured:
        return str(configured)
    family = model_config.get("family")
    if family:
        return str(family)
    checkpoint_name = Path(str(checkpoint)).stem
    if checkpoint_name.startswith("yolov8m"):
        return "YOLOv8m"
    if checkpoint_name:
        return checkpoint_name
    return "ultralytics_detector"


def detector_backend_from_config(model_config: dict[str, Any]) -> str:
    return str(model_config.get("backend", "ultralytics")).lower().strip()


def create_detector(
    model_config: dict[str, Any],
    checkpoint: str | Path,
    device: str = "auto",
    half: bool = False,
    model_factory: Any | None = None,
) -> UltralyticsDetector:
    backend = detector_backend_from_config(model_config)
    if backend != "ultralytics":
        raise DetectorError(f"Unsupported detector backend: {backend}")
    return UltralyticsDetector(
        weights=checkpoint,
        device=device,
        half=half,
        detector_name=detector_name_from_config(model_config, checkpoint),
        backend=backend,
        model_factory=model_factory,
    )
