"""Typed schemas for standalone language grounding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


class GroundingSchemaError(ValueError):
    """Raised when a grounding schema receives invalid data."""


def _tuple4_float(value: Any, field_name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise GroundingSchemaError(f"{field_name} must contain four numbers.") from exc
    if len(values) != 4:
        raise GroundingSchemaError(f"{field_name} must contain four numbers.")
    if not all(math.isfinite(item) for item in values):
        raise GroundingSchemaError(f"{field_name} must contain finite numbers.")
    return values


def _tuple4_int(value: Any, field_name: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item) for item in value)
    except TypeError as exc:
        raise GroundingSchemaError(f"{field_name} must contain four integers.") from exc
    if len(values) != 4:
        raise GroundingSchemaError(f"{field_name} must contain four integers.")
    return values


def _validate_xyxy(box: tuple[float, float, float, float], field_name: str) -> None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise GroundingSchemaError(f"{field_name} must satisfy x2 > x1 and y2 > y1.")


@dataclass(frozen=True)
class GroundingRequest:
    image_path: Path
    query: str
    backend: str
    model_id: str
    inference_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.image_path)
        if not str(path):
            raise GroundingSchemaError("image_path must not be empty.")
        if not str(self.query).strip():
            raise GroundingSchemaError("query must not be empty.")
        if not str(self.backend).strip():
            raise GroundingSchemaError("backend must not be empty.")
        if not str(self.model_id).strip():
            raise GroundingSchemaError("model_id must not be empty.")
        object.__setattr__(self, "image_path", path)
        object.__setattr__(self, "query", str(self.query))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "inference_config", dict(self.inference_config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "query": self.query,
            "backend": self.backend,
            "model_id": self.model_id,
            "inference_config": dict(self.inference_config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingRequest:
        return cls(
            image_path=Path(data["image_path"]),
            query=str(data["query"]),
            backend=str(data["backend"]),
            model_id=str(data["model_id"]),
            inference_config=dict(data.get("inference_config", {})),
        )


@dataclass(frozen=True)
class GroundedBox:
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    normalized_bbox: tuple[int, int, int, int]
    confidence: float | None
    query: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise GroundingSchemaError("label must not be empty.")
        if not str(self.query).strip():
            raise GroundingSchemaError("query must not be empty.")
        bbox = _tuple4_float(self.bbox_xyxy, "bbox_xyxy")
        normalized = _tuple4_int(self.normalized_bbox, "normalized_bbox")
        _validate_xyxy(bbox, "bbox_xyxy")
        _validate_xyxy(tuple(float(item) for item in normalized), "normalized_bbox")
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise GroundingSchemaError("confidence must be None or in [0, 1].")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "query", str(self.query))
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "normalized_bbox", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox_xyxy": list(self.bbox_xyxy),
            "normalized_bbox": list(self.normalized_bbox),
            "confidence": self.confidence,
            "query": self.query,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundedBox:
        return cls(
            label=str(data["label"]),
            bbox_xyxy=tuple(data["bbox_xyxy"]),
            normalized_bbox=tuple(data["normalized_bbox"]),
            confidence=data.get("confidence"),
            query=str(data["query"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class GroundingRuntimeInfo:
    backend: str
    model_id: str
    latency_seconds: float | None = None
    cache_key: str | None = None
    cache_status: str = "disabled"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.backend).strip():
            raise GroundingSchemaError("runtime backend must not be empty.")
        if not str(self.model_id).strip():
            raise GroundingSchemaError("runtime model_id must not be empty.")
        if self.latency_seconds is not None:
            latency = float(self.latency_seconds)
            if latency < 0.0 or not math.isfinite(latency):
                raise GroundingSchemaError("latency_seconds must be non-negative.")
            object.__setattr__(self, "latency_seconds", latency)
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "cache_status", str(self.cache_status))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def with_cache_status(
        self,
        cache_status: str,
        cache_key: str | None = None,
        warning: str | None = None,
    ) -> GroundingRuntimeInfo:
        warnings = self.warnings + ((warning,) if warning else ())
        return replace(
            self,
            cache_status=cache_status,
            cache_key=cache_key if cache_key is not None else self.cache_key,
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "latency_seconds": self.latency_seconds,
            "cache_key": self.cache_key,
            "cache_status": self.cache_status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingRuntimeInfo:
        return cls(
            backend=str(data["backend"]),
            model_id=str(data["model_id"]),
            latency_seconds=data.get("latency_seconds"),
            cache_key=data.get("cache_key"),
            cache_status=str(data.get("cache_status", "disabled")),
            warnings=tuple(data.get("warnings", ())),
            errors=tuple(data.get("errors", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class GroundingResult:
    request: GroundingRequest
    image_width: int
    image_height: int
    boxes: tuple[GroundedBox, ...]
    raw_response: str
    runtime_info: GroundingRuntimeInfo
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise GroundingSchemaError("image dimensions must be positive.")
        boxes = tuple(self.boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.bbox_xyxy
            if x1 < 0.0 or y1 < 0.0 or x2 > self.image_width or y2 > self.image_height:
                raise GroundingSchemaError("bbox_xyxy must lie inside image dimensions.")
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "raw_response", str(self.raw_response))

    @property
    def has_errors(self) -> bool:
        return bool(self.runtime_info.errors)

    def with_cache_hit(self, cache_key: str | None = None) -> GroundingResult:
        return replace(
            self,
            cache_hit=True,
            runtime_info=self.runtime_info.with_cache_status("hit", cache_key=cache_key),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.request.query,
            "image": {
                "path": str(self.request.image_path),
                "width": self.image_width,
                "height": self.image_height,
            },
            "backend": {
                "name": self.request.backend,
                "model_id": self.request.model_id,
            },
            "request": self.request.to_dict(),
            "boxes": [box.to_dict() for box in self.boxes],
            "raw_response": self.raw_response,
            "runtime_info": self.runtime_info.to_dict(),
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingResult:
        request_data = data.get("request")
        if not isinstance(request_data, dict):
            image_data = data.get("image", {})
            backend_data = data.get("backend", {})
            request_data = {
                "image_path": image_data.get("path"),
                "query": data.get("query"),
                "backend": backend_data.get("name"),
                "model_id": backend_data.get("model_id"),
                "inference_config": {},
            }
        image_data = data.get("image", {})
        return cls(
            request=GroundingRequest.from_dict(request_data),
            image_width=int(data.get("image_width", image_data.get("width"))),
            image_height=int(data.get("image_height", image_data.get("height"))),
            boxes=tuple(GroundedBox.from_dict(item) for item in data.get("boxes", [])),
            raw_response=str(data.get("raw_response", "")),
            runtime_info=GroundingRuntimeInfo.from_dict(data["runtime_info"]),
            cache_hit=bool(data.get("cache_hit", False)),
        )

