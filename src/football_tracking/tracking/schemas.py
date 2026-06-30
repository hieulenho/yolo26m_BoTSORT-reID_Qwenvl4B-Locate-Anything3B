"""Typed models for tracking pipeline inputs and outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.data.bbox import is_valid_bbox, xyxy_to_xywh
from football_tracking.data.schemas import BoundingBoxXYWH, BoundingBoxXYXY


def _finite_non_negative(value: float, field_name: str) -> None:
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")


def _validate_confidence(value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1].")


@dataclass(frozen=True)
class TrackerDetection:
    frame_index: int
    sequence_name: str
    bbox_xyxy: BoundingBoxXYXY
    bbox_ltwh: BoundingBoxXYWH
    confidence: float
    class_id: int
    class_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise ValueError("frame_index must be >= 1.")
        if not self.sequence_name:
            raise ValueError("sequence_name must not be empty.")
        if not is_valid_bbox(self.bbox_xyxy):
            raise ValueError(f"Invalid detection bbox: {self.bbox_xyxy}")
        if self.bbox_ltwh.width <= 0 or self.bbox_ltwh.height <= 0:
            raise ValueError("bbox_ltwh width and height must be positive.")
        _validate_confidence(self.confidence)
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative.")
        if not self.class_name:
            raise ValueError("class_name must not be empty.")

    @classmethod
    def from_xyxy(
        cls,
        frame_index: int,
        sequence_name: str,
        bbox_xyxy: BoundingBoxXYXY,
        confidence: float,
        class_id: int = 0,
        class_name: str = "player",
        metadata: dict[str, object] | None = None,
    ) -> TrackerDetection:
        return cls(
            frame_index=frame_index,
            sequence_name=sequence_name,
            bbox_xyxy=bbox_xyxy,
            bbox_ltwh=xyxy_to_xywh(bbox_xyxy),
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "sequence_name": self.sequence_name,
            "bbox_xyxy": [
                self.bbox_xyxy.x1,
                self.bbox_xyxy.y1,
                self.bbox_xyxy.x2,
                self.bbox_xyxy.y2,
            ],
            "bbox_ltwh": [
                self.bbox_ltwh.x,
                self.bbox_ltwh.y,
                self.bbox_ltwh.width,
                self.bbox_ltwh.height,
            ],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrackOutput:
    frame_index: int
    sequence_name: str
    track_id: int
    bbox_xyxy: BoundingBoxXYXY
    bbox_ltwh: BoundingBoxXYWH
    confidence: float | None
    class_id: int
    class_name: str
    confirmed: bool
    time_since_update: int
    hits: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise ValueError("frame_index must be >= 1.")
        if not self.sequence_name:
            raise ValueError("sequence_name must not be empty.")
        if self.track_id <= 0:
            raise ValueError("track_id must be positive.")
        if not is_valid_bbox(self.bbox_xyxy):
            raise ValueError(f"Invalid track bbox: {self.bbox_xyxy}")
        if self.bbox_ltwh.width <= 0 or self.bbox_ltwh.height <= 0:
            raise ValueError("bbox_ltwh width and height must be positive.")
        _validate_confidence(self.confidence)
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative.")
        if self.time_since_update < 0:
            raise ValueError("time_since_update must be non-negative.")

    @classmethod
    def from_xyxy(
        cls,
        frame_index: int,
        sequence_name: str,
        track_id: int,
        bbox_xyxy: BoundingBoxXYXY,
        confidence: float | None,
        class_id: int = 0,
        class_name: str = "player",
        confirmed: bool = True,
        time_since_update: int = 0,
        hits: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrackOutput:
        return cls(
            frame_index=frame_index,
            sequence_name=sequence_name,
            track_id=track_id,
            bbox_xyxy=bbox_xyxy,
            bbox_ltwh=xyxy_to_xywh(bbox_xyxy),
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            confirmed=confirmed,
            time_since_update=time_since_update,
            hits=hits,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "sequence_name": self.sequence_name,
            "track_id": self.track_id,
            "bbox_xyxy": [
                self.bbox_xyxy.x1,
                self.bbox_xyxy.y1,
                self.bbox_xyxy.x2,
                self.bbox_xyxy.y2,
            ],
            "bbox_ltwh": [
                self.bbox_ltwh.x,
                self.bbox_ltwh.y,
                self.bbox_ltwh.width,
                self.bbox_ltwh.height,
            ],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confirmed": self.confirmed,
            "time_since_update": self.time_since_update,
            "hits": self.hits,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrackingFrameResult:
    frame_index: int
    detections: list[TrackerDetection]
    tracks: list[TrackOutput]
    detector_seconds: float
    tracker_seconds: float
    rendering_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise ValueError("frame_index must be >= 1.")
        for field_name in (
            "detector_seconds",
            "tracker_seconds",
            "rendering_seconds",
            "total_seconds",
        ):
            _finite_non_negative(float(getattr(self, field_name)), field_name)


@dataclass(frozen=True)
class TrackingRunSummary:
    sequence_name: str
    frame_count: int
    detection_count: int
    emitted_track_count: int
    unique_track_count: int
    detector_seconds: float
    tracker_seconds: float
    rendering_seconds: float
    total_seconds: float
    detector_fps: float | None
    tracker_fps: float | None
    end_to_end_fps: float | None
    checkpoint: str
    device: str
    output_video: Path | None
    output_mot: Path | None
    smoke_only: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "emitted_track_count": self.emitted_track_count,
            "unique_track_count": self.unique_track_count,
            "detector_seconds": self.detector_seconds,
            "tracker_seconds": self.tracker_seconds,
            "rendering_seconds": self.rendering_seconds,
            "total_seconds": self.total_seconds,
            "detector_fps": self.detector_fps,
            "tracker_fps": self.tracker_fps,
            "end_to_end_fps": self.end_to_end_fps,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "output_video": str(self.output_video) if self.output_video else None,
            "output_mot": str(self.output_mot) if self.output_mot else None,
            "smoke_only": self.smoke_only,
            "warnings": self.warnings,
            "errors": self.errors,
        }
