"""Schemas for target uncertainty monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

SignalType = Literal[
    "TARGET_PRESENCE",
    "TRACK_CONFIDENCE",
    "MOTION_JUMP",
    "SEMANTIC_MARGIN",
    "APPEARANCE_DRIFT",
    "NEIGHBOR_AMBIGUITY",
    "TRACK_GAP",
    "GROUNDING_STALENESS",
]
Severity = Literal["info", "warning", "high", "critical"]
BBox = tuple[float, float, float, float]


class MonitoringSchemaError(ValueError):
    """Raised when monitoring schemas receive invalid values."""


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MonitoringSchemaError(f"{name} must be finite.")
    return numeric


def _unit(value: float, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise MonitoringSchemaError(f"{name} must be in [0, 1].")
    return numeric


def _bbox_or_none(value: BBox | None, name: str) -> BBox | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value)
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise MonitoringSchemaError(f"{name} must contain four finite numbers.")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise MonitoringSchemaError(f"{name} must satisfy x2 > x1 and y2 > y1.")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class MotionMetrics:
    center_x: float
    center_y: float
    displacement_px: float | None = None
    displacement_normalized: float | None = None
    median_baseline: float | None = None
    jump_ratio: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_x", _finite(self.center_x, "center_x"))
        object.__setattr__(self, "center_y", _finite(self.center_y, "center_y"))
        for field_name in (
            "displacement_px",
            "displacement_normalized",
            "median_baseline",
            "jump_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite(value, field_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "displacement_px": self.displacement_px,
            "displacement_normalized": self.displacement_normalized,
            "median_baseline": self.median_baseline,
            "jump_ratio": self.jump_ratio,
        }


@dataclass(frozen=True)
class TargetFrameObservation:
    frame_index: int
    target_present: bool
    raw_track_id: int | None
    bbox_xyxy: BBox | None = None
    tracking_confidence: float | None = None
    semantic_score: float | None = None
    appearance_score: float | None = None
    fused_score: float | None = None
    candidate_count: int | None = None
    runner_up_score: float | None = None
    winner_margin: float | None = None
    neighbor_count: int | None = None
    motion_metrics: MotionMetrics | None = None
    data_availability: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_index) < 1:
            raise MonitoringSchemaError("frame_index must be >= 1.")
        if self.raw_track_id is not None and int(self.raw_track_id) < 1:
            raise MonitoringSchemaError("raw_track_id must be >= 1 when provided.")
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(
            self,
            "raw_track_id",
            None if self.raw_track_id is None else int(self.raw_track_id),
        )
        object.__setattr__(self, "bbox_xyxy", _bbox_or_none(self.bbox_xyxy, "bbox_xyxy"))
        for field_name in (
            "tracking_confidence",
            "semantic_score",
            "appearance_score",
            "fused_score",
            "runner_up_score",
            "winner_margin",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _unit(value, field_name))
        if self.candidate_count is not None:
            object.__setattr__(self, "candidate_count", int(self.candidate_count))
        if self.neighbor_count is not None:
            object.__setattr__(self, "neighbor_count", int(self.neighbor_count))
        object.__setattr__(
            self,
            "data_availability",
            {str(key): str(value) for key, value in self.data_availability.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "target_present": self.target_present,
            "raw_track_id": self.raw_track_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "tracking_confidence": self.tracking_confidence,
            "semantic_score": self.semantic_score,
            "appearance_score": self.appearance_score,
            "fused_score": self.fused_score,
            "candidate_count": self.candidate_count,
            "runner_up_score": self.runner_up_score,
            "winner_margin": self.winner_margin,
            "neighbor_count": self.neighbor_count,
            "motion_metrics": self.motion_metrics.to_dict() if self.motion_metrics else None,
            "data_availability": dict(self.data_availability),
        }


@dataclass(frozen=True)
class TargetObservationTimeline:
    query: str
    semantic_target_hypothesis: str
    current_raw_track_id: int
    start_frame: int
    end_frame: int
    observations: tuple[TargetFrameObservation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.current_raw_track_id) < 1:
            raise MonitoringSchemaError("current_raw_track_id must be >= 1.")
        if int(self.start_frame) < 1 or int(self.end_frame) < int(self.start_frame):
            raise MonitoringSchemaError("Invalid timeline frame range.")
        frames = tuple(self.observations)
        if any(
            item.frame_index < self.start_frame or item.frame_index > self.end_frame
            for item in frames
        ):
            raise MonitoringSchemaError("observation frame is outside timeline range.")
        object.__setattr__(self, "current_raw_track_id", int(self.current_raw_track_id))
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(self, "end_frame", int(self.end_frame))
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(frames, key=lambda item: item.frame_index)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def frame(self, frame_index: int) -> TargetFrameObservation | None:
        for observation in self.observations:
            if observation.frame_index == frame_index:
                return observation
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "semantic_target_hypothesis": self.semantic_target_hypothesis,
            "current_raw_track_id": self.current_raw_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "observations": [item.to_dict() for item in self.observations],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UncertaintySignal:
    signal_id: str
    signal_type: SignalType
    frame_index: int | None
    frame_start: int
    frame_end: int
    raw_track_id: int
    score: float | None
    severity_contribution: Severity
    threshold: float | None
    triggered: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    data_available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if int(self.frame_start) < 1 or int(self.frame_end) < int(self.frame_start):
            raise MonitoringSchemaError("Invalid signal frame range.")
        if int(self.raw_track_id) < 1:
            raise MonitoringSchemaError("raw_track_id must be >= 1.")
        if self.frame_index is not None and int(self.frame_index) < 1:
            raise MonitoringSchemaError("frame_index must be >= 1 when provided.")
        object.__setattr__(self, "frame_start", int(self.frame_start))
        object.__setattr__(self, "frame_end", int(self.frame_end))
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(
            self,
            "frame_index",
            None if self.frame_index is None else int(self.frame_index),
        )
        if self.score is not None:
            object.__setattr__(self, "score", _finite(self.score, "score"))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "frame_index": self.frame_index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "raw_track_id": self.raw_track_id,
            "score": self.score,
            "severity_contribution": self.severity_contribution,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "evidence": dict(self.evidence),
            "data_available": self.data_available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class MonitoringAssessment:
    timeline: TargetObservationTimeline
    signals: tuple[UncertaintySignal, ...]
    aggregate_score: float
    aggregate_severity: Severity
    triggered_signal_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": self.timeline.to_dict(),
            "signals": [item.to_dict() for item in self.signals],
            "aggregate_score": self.aggregate_score,
            "aggregate_severity": self.aggregate_severity,
            "triggered_signal_count": self.triggered_signal_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MonitoringConfig:
    presence_warning_absent_frames: int = 5
    presence_critical_absent_frames: int = 20
    confidence_low_threshold: float = 0.25
    confidence_consecutive_frames: int = 3
    motion_jump_threshold: float = 0.08
    motion_jump_ratio_threshold: float = 4.0
    motion_baseline_window: int = 10
    semantic_margin_threshold: float = 0.08
    appearance_drift_threshold: float = 0.35
    neighbor_distance_threshold: float = 0.045
    neighbor_iou_threshold: float = 0.05
    neighbor_count_threshold: int = 1
    gap_warning_frames: int = 5
    gap_critical_frames: int = 20
    staleness_warning_frames: int = 120
    signal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "presence_warning_absent_frames",
            "presence_critical_absent_frames",
            "confidence_consecutive_frames",
            "motion_baseline_window",
            "neighbor_count_threshold",
            "gap_warning_frames",
            "gap_critical_frames",
            "staleness_warning_frames",
        ):
            if int(getattr(self, field_name)) < 0:
                raise MonitoringSchemaError(f"{field_name} must be >= 0.")
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        for field_name in (
            "confidence_low_threshold",
            "motion_jump_threshold",
            "semantic_margin_threshold",
            "appearance_drift_threshold",
            "neighbor_distance_threshold",
            "neighbor_iou_threshold",
        ):
            object.__setattr__(self, field_name, _unit(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "motion_jump_ratio_threshold",
            max(0.0, _finite(self.motion_jump_ratio_threshold, "motion_jump_ratio_threshold")),
        )
        weights = dict(self.signal_weights) if self.signal_weights else {}
        object.__setattr__(
            self,
            "signal_weights",
            {str(key): float(value) for key, value in weights.items()},
        )

    def weight_for(self, signal_type: str) -> float:
        return self.signal_weights.get(signal_type, 1.0)
