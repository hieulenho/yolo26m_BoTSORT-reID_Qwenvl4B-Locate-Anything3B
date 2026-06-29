"""Detection data models for pretrained detector outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY


@dataclass(frozen=True)
class Detection:
    frame_index: int
    sequence_name: str
    bbox_xyxy: BoundingBoxXYXY
    confidence: float
    source_class_id: int
    source_class_name: str
    target_class_id: int
    target_class_name: str
    image_width: int
    image_height: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise ValueError("frame_index must be >= 1.")
        if not self.sequence_name:
            raise ValueError("sequence_name must not be empty.")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive.")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1].")
        if not is_valid_bbox(self.bbox_xyxy):
            raise ValueError(f"Invalid detection bbox: {self.bbox_xyxy}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "frame_index": self.frame_index,
            "image_path": self.metadata.get("image_path"),
            "bbox_xyxy": [
                self.bbox_xyxy.x1,
                self.bbox_xyxy.y1,
                self.bbox_xyxy.x2,
                self.bbox_xyxy.y2,
            ],
            "confidence": self.confidence,
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DetectionBatchResult:
    detections: list[Detection]
    inference_seconds: float
    preprocessing_seconds: float
    postprocessing_seconds: float
    image_count: int
    model_name: str
    device: str
    image_size: int | tuple[int, int]

    def __post_init__(self) -> None:
        for field_name in ("inference_seconds", "preprocessing_seconds", "postprocessing_seconds"):
            value = getattr(self, field_name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative.")
        if self.image_count < 0:
            raise ValueError("image_count must be non-negative.")
