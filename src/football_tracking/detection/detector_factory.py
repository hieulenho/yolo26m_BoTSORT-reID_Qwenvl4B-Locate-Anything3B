"""Detector registry and factory helpers.

The project originally hard-coded YOLOv8m.  This module keeps that path working
while allowing config-selected Ultralytics detector families such as YOLO26.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.detection.detector import (
    DetectorError,
    RoutedCompositeDetector,
    SupplementalDetector,
    UltralyticsDetector,
    UltralyticsOpenVocabularyDetector,
)


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
) -> UltralyticsDetector | RoutedCompositeDetector:
    supplemental = model_config.get("supplemental_detectors", [])
    if supplemental:
        if not isinstance(supplemental, list | tuple):
            raise DetectorError("model.supplemental_detectors must be a list.")
        primary_config = dict(model_config)
        primary_config.pop("supplemental_detectors", None)
        primary = create_detector(
            primary_config,
            checkpoint,
            device=device,
            half=half,
            model_factory=model_factory,
        )
        if not isinstance(primary, UltralyticsDetector):
            raise DetectorError("Nested routed composite detectors are not supported.")
        routed: list[SupplementalDetector] = []
        for index, value in enumerate(supplemental):
            if not isinstance(value, dict):
                raise DetectorError(f"supplemental_detectors[{index}] must be a mapping.")
            supplement_checkpoint = value.get("checkpoint")
            if not isinstance(supplement_checkpoint, str) or not supplement_checkpoint:
                raise DetectorError(
                    f"supplemental_detectors[{index}].checkpoint is required."
                )
            input_ids = value.get("input_class_ids", [])
            output_ids = value.get("output_class_ids", [])
            class_names = value.get("class_names", [])
            if not (
                isinstance(input_ids, list | tuple)
                and isinstance(output_ids, list | tuple)
                and isinstance(class_names, list | tuple)
                and len(input_ids) == len(output_ids) == len(class_names)
                and input_ids
            ):
                raise DetectorError(
                    f"supplemental_detectors[{index}] class mappings must have equal lengths."
                )
            supplement = create_detector(
                value,
                supplement_checkpoint,
                device=device,
                half=bool(value.get("half", half)),
                model_factory=model_factory,
            )
            if not isinstance(supplement, UltralyticsDetector):
                raise DetectorError("Nested routed composite detectors are not supported.")
            output_values = [int(item) for item in output_ids]
            routed.append(
                SupplementalDetector(
                    detector=supplement,
                    class_id_map={
                        int(source): destination
                        for source, destination in zip(
                            input_ids,
                            output_values,
                            strict=True,
                        )
                    },
                    class_names={
                        destination: str(name)
                        for destination, name in zip(
                            output_values,
                            class_names,
                            strict=True,
                        )
                    },
                    every_n_frames=int(value.get("every_n_frames", 1)),
                )
            )
        return RoutedCompositeDetector(primary, routed)
    backend = detector_backend_from_config(model_config)
    if backend in {"ultralytics_yoloe", "yoloe", "open_vocabulary"}:
        text_classes = model_config.get("text_classes", model_config.get("vocabulary", []))
        if not isinstance(text_classes, list | tuple):
            raise DetectorError("model.text_classes must be a list for YOLOE.")
        return UltralyticsOpenVocabularyDetector(
            weights=checkpoint,
            text_classes=[str(item) for item in text_classes],
            device=device,
            half=half,
            detector_name=detector_name_from_config(model_config, checkpoint),
            model_factory=model_factory,
        )
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
