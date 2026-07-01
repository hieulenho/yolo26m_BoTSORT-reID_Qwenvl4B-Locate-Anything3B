"""Stable JSONL schema for cached detector outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackerDetection

SCHEMA_VERSION = "1.0"


class DetectionCacheSchemaError(ValueError):
    """Raised when a detection cache record is invalid."""


def _finite(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DetectionCacheSchemaError(f"{field_name} must be finite.")
    return number


def _bbox_to_list(box: BoundingBoxXYXY) -> list[float]:
    return [float(box.x1), float(box.y1), float(box.x2), float(box.y2)]


def _bbox_from_value(value: Any) -> BoundingBoxXYXY:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise DetectionCacheSchemaError("bbox_xyxy must be a list of four numbers.")
    x1, y1, x2, y2 = [_finite(float(item), "bbox_xyxy") for item in value]
    box = BoundingBoxXYXY(x1, y1, x2, y2)
    if not is_valid_bbox(box):
        raise DetectionCacheSchemaError(f"Invalid bbox_xyxy: {value}")
    return box


@dataclass(frozen=True)
class CachedDetection:
    bbox_xyxy: BoundingBoxXYXY
    confidence: float
    class_id: int = 0
    class_name: str = "player"
    source_class_id: int | None = None
    source_class_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_valid_bbox(self.bbox_xyxy):
            raise DetectionCacheSchemaError(f"Invalid bbox_xyxy: {self.bbox_xyxy}")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise DetectionCacheSchemaError("confidence must be in [0, 1].")
        if self.class_id != 0 or self.class_name != "player":
            raise DetectionCacheSchemaError("Detection cache only supports class 0/player.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": _bbox_to_list(self.bbox_xyxy),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CachedDetection:
        return cls(
            bbox_xyxy=_bbox_from_value(payload.get("bbox_xyxy")),
            confidence=float(payload.get("confidence")),
            class_id=int(payload.get("class_id", 0)),
            class_name=str(payload.get("class_name", "player")),
            source_class_id=(
                None if payload.get("source_class_id") is None else int(payload["source_class_id"])
            ),
            source_class_name=payload.get("source_class_name"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_tracker_detection(self, frame_index: int, sequence_name: str) -> TrackerDetection:
        return TrackerDetection.from_xyxy(
            frame_index=frame_index,
            sequence_name=sequence_name,
            bbox_xyxy=self.bbox_xyxy,
            confidence=float(self.confidence),
            class_id=self.class_id,
            class_name=self.class_name,
            metadata={
                **dict(self.metadata),
                "detection_source": "cache",
                "source_class_id": self.source_class_id,
                "source_class_name": self.source_class_name,
            },
        )


@dataclass(frozen=True)
class CachedFrameDetections:
    sequence_name: str
    frame_index: int
    image_path: str
    image_width: int
    image_height: int
    detections: list[CachedDetection] = field(default_factory=list)
    detector_seconds: float | None = None
    preprocessing_seconds: float | None = None
    postprocessing_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.sequence_name:
            raise DetectionCacheSchemaError("sequence_name must not be empty.")
        if self.frame_index < 1:
            raise DetectionCacheSchemaError("frame_index must be >= 1.")
        if self.image_width <= 0 or self.image_height <= 0:
            raise DetectionCacheSchemaError("image_width and image_height must be positive.")
        for timing_name in (
            "detector_seconds",
            "preprocessing_seconds",
            "postprocessing_seconds",
        ):
            value = getattr(self, timing_name)
            if value is not None and (_finite(value, timing_name) < 0.0):
                raise DetectionCacheSchemaError(f"{timing_name} must be non-negative.")
        for detection in self.detections:
            if detection.bbox_xyxy.x1 < 0.0 or detection.bbox_xyxy.y1 < 0.0:
                raise DetectionCacheSchemaError("Detection bbox must stay inside the image.")
            if (
                detection.bbox_xyxy.x2 > float(self.image_width)
                or detection.bbox_xyxy.y2 > float(self.image_height)
            ):
                raise DetectionCacheSchemaError("Detection bbox must stay inside the image.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "frame_index": int(self.frame_index),
            "image_path": self.image_path,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "detections": [detection.to_dict() for detection in self.detections],
            "detector_seconds": self.detector_seconds,
            "preprocessing_seconds": self.preprocessing_seconds,
            "postprocessing_seconds": self.postprocessing_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CachedFrameDetections:
        detections = [
            CachedDetection.from_dict(item)
            for item in payload.get("detections", [])
        ]
        return cls(
            sequence_name=str(payload.get("sequence_name")),
            frame_index=int(payload.get("frame_index")),
            image_path=str(payload.get("image_path")),
            image_width=int(payload.get("image_width")),
            image_height=int(payload.get("image_height")),
            detections=detections,
            detector_seconds=payload.get("detector_seconds"),
            preprocessing_seconds=payload.get("preprocessing_seconds"),
            postprocessing_seconds=payload.get("postprocessing_seconds"),
        )

    def to_tracker_detections(self, confidence_threshold: float) -> list[TrackerDetection]:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise DetectionCacheSchemaError("confidence_threshold must be in [0, 1].")
        return [
            detection.to_tracker_detection(self.frame_index, self.sequence_name)
            for detection in self.detections
            if detection.confidence >= confidence_threshold
        ]


@dataclass(frozen=True)
class DetectionCacheMetadata:
    schema_version: str
    dataset: str
    split: str
    sequence_name: str
    expected_frame_count: int | None
    cached_frame_count: int
    checkpoint: str
    checkpoint_type: str
    checkpoint_hash: str | None
    detector_name: str
    image_size: int
    confidence_floor: float
    nms_iou: float
    max_det: int
    class_filter: list[int]
    device: str
    python_version: str | None
    torch_version: str | None
    ultralytics_version: str | None
    created_at: str
    source_sequence_path: str
    complete: bool
    partial: bool
    max_frame: int | None
    file_hash: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DetectionCacheSchemaError(f"Unsupported schema_version: {self.schema_version}")
        if self.cached_frame_count < 0:
            raise DetectionCacheSchemaError("cached_frame_count must be non-negative.")
        if self.expected_frame_count is not None and self.expected_frame_count < 0:
            raise DetectionCacheSchemaError("expected_frame_count must be non-negative.")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise DetectionCacheSchemaError("confidence_floor must be in [0, 1].")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise DetectionCacheSchemaError("nms_iou must be in [0, 1].")
        if self.max_det <= 0:
            raise DetectionCacheSchemaError("max_det must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DetectionCacheMetadata:
        return cls(
            schema_version=str(payload.get("schema_version")),
            dataset=str(payload.get("dataset")),
            split=str(payload.get("split")),
            sequence_name=str(payload.get("sequence_name")),
            expected_frame_count=payload.get("expected_frame_count"),
            cached_frame_count=int(payload.get("cached_frame_count", 0)),
            checkpoint=str(payload.get("checkpoint")),
            checkpoint_type=str(payload.get("checkpoint_type")),
            checkpoint_hash=payload.get("checkpoint_hash"),
            detector_name=str(payload.get("detector_name")),
            image_size=int(payload.get("image_size")),
            confidence_floor=float(payload.get("confidence_floor")),
            nms_iou=float(payload.get("nms_iou")),
            max_det=int(payload.get("max_det")),
            class_filter=[int(value) for value in payload.get("class_filter", [])],
            device=str(payload.get("device")),
            python_version=payload.get("python_version"),
            torch_version=payload.get("torch_version"),
            ultralytics_version=payload.get("ultralytics_version"),
            created_at=str(payload.get("created_at")),
            source_sequence_path=str(payload.get("source_sequence_path")),
            complete=bool(payload.get("complete", False)),
            partial=bool(payload.get("partial", False)),
            max_frame=payload.get("max_frame"),
            file_hash=payload.get("file_hash"),
            warnings=list(payload.get("warnings", [])),
            errors=list(payload.get("errors", [])),
        )
