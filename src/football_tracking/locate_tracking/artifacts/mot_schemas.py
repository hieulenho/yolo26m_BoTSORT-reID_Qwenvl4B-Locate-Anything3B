"""Schemas for read-only MOT tracking artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MotArtifactError(ValueError):
    """Raised when a MOT tracking artifact row is invalid."""


def _finite(value: float, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MotArtifactError(f"{field_name} must be finite.")
    return numeric


@dataclass(frozen=True)
class MotTrackObservation:
    frame_index: int
    track_id: int
    bbox_ltwh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float | None
    source_path: Path
    line_number: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_index) < 1:
            raise MotArtifactError("frame_index must be >= 1.")
        if int(self.track_id) < 1:
            raise MotArtifactError("track_id must be >= 1.")
        left, top, width, height = (_finite(value, "bbox_ltwh") for value in self.bbox_ltwh)
        if width <= 0.0:
            raise MotArtifactError("track width must be positive.")
        if height <= 0.0:
            raise MotArtifactError("track height must be positive.")
        x1, y1, x2, y2 = (_finite(value, "bbox_xyxy") for value in self.bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise MotArtifactError("bbox_xyxy must satisfy x2 > x1 and y2 > y1.")
        if self.confidence is not None:
            confidence = _finite(float(self.confidence), "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise MotArtifactError("confidence must be None or in [0, 1].")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "bbox_ltwh", (left, top, width, height))
        object.__setattr__(self, "bbox_xyxy", (x1, y1, x2, y2))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "line_number", int(self.line_number))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "bbox_ltwh": list(self.bbox_ltwh),
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "source_path": str(self.source_path),
            "line_number": self.line_number,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MotTrackFile:
    path: Path
    observations: tuple[MotTrackObservation, ...]

    def __post_init__(self) -> None:
        observations = tuple(
            sorted(self.observations, key=lambda item: (item.frame_index, item.track_id))
        )
        seen: set[tuple[int, int]] = set()
        for observation in observations:
            key = (observation.frame_index, observation.track_id)
            if key in seen:
                raise MotArtifactError(f"Duplicate frame-track pair: {key}")
            seen.add(key)
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "observations", observations)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "observation_count": self.observation_count,
            "observations": [observation.to_dict() for observation in self.observations],
        }
