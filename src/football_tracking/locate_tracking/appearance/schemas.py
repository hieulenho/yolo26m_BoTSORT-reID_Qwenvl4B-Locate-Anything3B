"""Typed schemas for appearance verification artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


class AppearanceSchemaError(ValueError):
    """Raised when an appearance schema receives invalid data."""


PrototypeAggregationStrategy = Literal["mean", "quality_weighted_mean"]
AppearanceEvidenceStatus = Literal[
    "verified",
    "weak",
    "unavailable",
    "insufficient_appearance_evidence",
]


def _bbox4(value: Any, field_name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise AppearanceSchemaError(f"{field_name} must contain four numbers.") from exc
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise AppearanceSchemaError(f"{field_name} must contain four finite numbers.")
    return values


def _finite_or_none(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AppearanceSchemaError(f"{field_name} must be finite when provided.")
    return numeric


def _unit_or_none(value: float | None, field_name: str) -> float | None:
    numeric = _finite_or_none(value, field_name)
    if numeric is not None and not 0.0 <= numeric <= 1.0:
        raise AppearanceSchemaError(f"{field_name} must be in [0, 1].")
    return numeric


def _vector(value: Any, field_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise AppearanceSchemaError(f"{field_name} must be a vector of numbers.") from exc
    if not values:
        raise AppearanceSchemaError(f"{field_name} must not be empty.")
    if not all(math.isfinite(item) for item in values):
        raise AppearanceSchemaError(f"{field_name} must contain finite values.")
    return values


@dataclass(frozen=True)
class CropQualityMetrics:
    width: int
    height: int
    area: float
    aspect_ratio: float
    visible_fraction: float
    sharpness_score: float | None
    brightness_mean: float | None
    passed_quality_gate: bool
    rejection_reasons: tuple[str, ...] = ()
    quality_score: float = 0.0

    def __post_init__(self) -> None:
        if int(self.width) < 0 or int(self.height) < 0:
            raise AppearanceSchemaError("crop quality width and height must be >= 0.")
        area = float(self.area)
        aspect_ratio = float(self.aspect_ratio)
        if area < 0.0 or not math.isfinite(area):
            raise AppearanceSchemaError("crop quality area must be finite and >= 0.")
        if aspect_ratio < 0.0 or not math.isfinite(aspect_ratio):
            raise AppearanceSchemaError("aspect_ratio must be finite and >= 0.")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "aspect_ratio", aspect_ratio)
        object.__setattr__(
            self,
            "visible_fraction",
            _unit_or_none(self.visible_fraction, "visible_fraction"),
        )
        object.__setattr__(
            self,
            "sharpness_score",
            _finite_or_none(self.sharpness_score, "sharpness_score"),
        )
        object.__setattr__(
            self,
            "brightness_mean",
            _finite_or_none(self.brightness_mean, "brightness_mean"),
        )
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        score = _unit_or_none(self.quality_score, "quality_score")
        object.__setattr__(self, "quality_score", 0.0 if score is None else score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
            "visible_fraction": self.visible_fraction,
            "sharpness_score": self.sharpness_score,
            "brightness_mean": self.brightness_mean,
            "passed_quality_gate": self.passed_quality_gate,
            "rejection_reasons": list(self.rejection_reasons),
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropQualityMetrics:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            area=float(data["area"]),
            aspect_ratio=float(data["aspect_ratio"]),
            visible_fraction=float(data["visible_fraction"]),
            sharpness_score=data.get("sharpness_score"),
            brightness_mean=data.get("brightness_mean"),
            passed_quality_gate=bool(data["passed_quality_gate"]),
            rejection_reasons=tuple(data.get("rejection_reasons", ())),
            quality_score=float(data.get("quality_score", 0.0)),
        )


@dataclass(frozen=True)
class CropReference:
    raw_track_id: int
    frame_index: int
    source_video: str
    raw_bbox_xyxy: tuple[float, float, float, float]
    clipped_bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int
    quality_metrics: CropQualityMetrics
    crop_path: str | None = None

    def __post_init__(self) -> None:
        if int(self.raw_track_id) < 1:
            raise AppearanceSchemaError("raw_track_id must be >= 1.")
        if int(self.frame_index) < 1:
            raise AppearanceSchemaError("frame_index must be >= 1.")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "raw_bbox_xyxy", _bbox4(self.raw_bbox_xyxy, "raw_bbox_xyxy"))
        object.__setattr__(
            self,
            "clipped_bbox_xyxy",
            _bbox4(self.clipped_bbox_xyxy, "clipped_bbox_xyxy"),
        )
        object.__setattr__(self, "crop_width", int(self.crop_width))
        object.__setattr__(self, "crop_height", int(self.crop_height))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "frame_index": self.frame_index,
            "source_video": self.source_video,
            "raw_bbox_xyxy": list(self.raw_bbox_xyxy),
            "clipped_bbox_xyxy": list(self.clipped_bbox_xyxy),
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "crop_path": self.crop_path,
            "quality_metrics": self.quality_metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropReference:
        return cls(
            raw_track_id=int(data["raw_track_id"]),
            frame_index=int(data["frame_index"]),
            source_video=str(data["source_video"]),
            raw_bbox_xyxy=tuple(data["raw_bbox_xyxy"]),
            clipped_bbox_xyxy=tuple(data["clipped_bbox_xyxy"]),
            crop_width=int(data["crop_width"]),
            crop_height=int(data["crop_height"]),
            crop_path=data.get("crop_path"),
            quality_metrics=CropQualityMetrics.from_dict(data["quality_metrics"]),
        )


@dataclass(frozen=True)
class AppearanceEmbedding:
    backend: str
    model_id: str
    dimension: int
    vector: tuple[float, ...]
    normalized: bool
    source_track_id: int | None
    source_frame_index: int | None
    vector_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector = _vector(self.vector, "vector")
        dimension = int(self.dimension)
        if dimension != len(vector):
            raise AppearanceSchemaError("embedding dimension must match vector length.")
        if self.source_track_id is not None and int(self.source_track_id) < 1:
            raise AppearanceSchemaError("source_track_id must be >= 1 when provided.")
        if self.source_frame_index is not None and int(self.source_frame_index) < 1:
            raise AppearanceSchemaError("source_frame_index must be >= 1 when provided.")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(
            self,
            "source_track_id",
            None if self.source_track_id is None else int(self.source_track_id),
        )
        object.__setattr__(
            self,
            "source_frame_index",
            None if self.source_frame_index is None else int(self.source_frame_index),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, *, include_vector: bool = True) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "vector": list(self.vector) if include_vector else None,
            "vector_reference": self.vector_reference,
            "normalized": self.normalized,
            "source_track_id": self.source_track_id,
            "source_frame_index": self.source_frame_index,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppearanceEmbedding:
        vector = data.get("vector")
        if vector is None:
            raise AppearanceSchemaError("Serialized embedding vector is required.")
        return cls(
            backend=str(data["backend"]),
            model_id=str(data["model_id"]),
            dimension=int(data["dimension"]),
            vector=tuple(vector),
            vector_reference=data.get("vector_reference"),
            normalized=bool(data["normalized"]),
            source_track_id=data.get("source_track_id"),
            source_frame_index=data.get("source_frame_index"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TrackEmbeddingSample:
    crop_reference: CropReference
    embedding: AppearanceEmbedding
    quality_weight: float

    def __post_init__(self) -> None:
        weight = float(self.quality_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise AppearanceSchemaError("quality_weight must be finite and non-negative.")
        object.__setattr__(self, "quality_weight", weight)

    def to_dict(self, *, include_vector: bool = True) -> dict[str, Any]:
        return {
            "crop_reference": self.crop_reference.to_dict(),
            "embedding": self.embedding.to_dict(include_vector=include_vector),
            "quality_weight": self.quality_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackEmbeddingSample:
        return cls(
            crop_reference=CropReference.from_dict(data["crop_reference"]),
            embedding=AppearanceEmbedding.from_dict(data["embedding"]),
            quality_weight=float(data["quality_weight"]),
        )


@dataclass(frozen=True)
class TrackAppearancePrototype:
    raw_track_id: int
    sample_embeddings: tuple[TrackEmbeddingSample, ...]
    sample_frame_indices: tuple[int, ...]
    prototype_vector: tuple[float, ...]
    sample_count: int
    aggregation_strategy: PrototypeAggregationStrategy
    quality_metadata: dict[str, Any]
    backend: str
    model_id: str
    embedding_dimension: int

    def __post_init__(self) -> None:
        vector = _vector(self.prototype_vector, "prototype_vector")
        if int(self.embedding_dimension) != len(vector):
            raise AppearanceSchemaError("prototype embedding_dimension mismatch.")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "sample_embeddings", tuple(self.sample_embeddings))
        object.__setattr__(
            self,
            "sample_frame_indices",
            tuple(sorted(set(int(item) for item in self.sample_frame_indices))),
        )
        object.__setattr__(self, "prototype_vector", vector)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "quality_metadata", dict(self.quality_metadata))

    def to_dict(self, *, include_vectors: bool = True) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "sample_embeddings": [
                item.to_dict(include_vector=include_vectors) for item in self.sample_embeddings
            ],
            "sample_frame_indices": list(self.sample_frame_indices),
            "prototype_vector": list(self.prototype_vector) if include_vectors else None,
            "sample_count": self.sample_count,
            "aggregation_strategy": self.aggregation_strategy,
            "quality_metadata": dict(self.quality_metadata),
            "backend": self.backend,
            "model_id": self.model_id,
            "embedding_dimension": self.embedding_dimension,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackAppearancePrototype:
        vector = data.get("prototype_vector")
        if vector is None:
            raise AppearanceSchemaError("Serialized prototype vector is required.")
        return cls(
            raw_track_id=int(data["raw_track_id"]),
            sample_embeddings=tuple(
                TrackEmbeddingSample.from_dict(item) for item in data.get("sample_embeddings", [])
            ),
            sample_frame_indices=tuple(data.get("sample_frame_indices", ())),
            prototype_vector=tuple(vector),
            sample_count=int(data["sample_count"]),
            aggregation_strategy=str(data["aggregation_strategy"]),  # type: ignore[arg-type]
            quality_metadata=dict(data.get("quality_metadata", {})),
            backend=str(data["backend"]),
            model_id=str(data["model_id"]),
            embedding_dimension=int(data["embedding_dimension"]),
        )


@dataclass(frozen=True)
class AppearanceCandidateScore:
    raw_track_id: int
    prototype_similarity: float | None
    internal_consistency: float | None
    appearance_score: float | None
    sample_count: int
    evidence_status: AppearanceEvidenceStatus
    decision_reason: str
    score_components: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(
            self,
            "prototype_similarity",
            _unit_or_none(self.prototype_similarity, "prototype_similarity"),
        )
        object.__setattr__(
            self,
            "internal_consistency",
            _unit_or_none(self.internal_consistency, "internal_consistency"),
        )
        object.__setattr__(
            self,
            "appearance_score",
            _unit_or_none(self.appearance_score, "appearance_score"),
        )
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "score_components", dict(self.score_components))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "prototype_similarity": self.prototype_similarity,
            "internal_consistency": self.internal_consistency,
            "appearance_score": self.appearance_score,
            "sample_count": self.sample_count,
            "evidence_status": self.evidence_status,
            "decision_reason": self.decision_reason,
            "score_components": dict(self.score_components),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppearanceCandidateScore:
        return cls(
            raw_track_id=int(data["raw_track_id"]),
            prototype_similarity=data.get("prototype_similarity"),
            internal_consistency=data.get("internal_consistency"),
            appearance_score=data.get("appearance_score"),
            sample_count=int(data.get("sample_count", 0)),
            evidence_status=str(data["evidence_status"]),  # type: ignore[arg-type]
            decision_reason=str(data.get("decision_reason", "")),
            score_components=dict(data.get("score_components", {})),
        )


@dataclass(frozen=True)
class AppearanceRuntimeInfo:
    backend_name: str
    model_id: str
    crop_extraction_seconds: float | None = None
    embedding_seconds: float | None = None
    prototype_build_seconds: float | None = None
    verification_seconds: float | None = None
    crop_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "model_id": self.model_id,
            "crop_extraction_seconds": self.crop_extraction_seconds,
            "embedding_seconds": self.embedding_seconds,
            "prototype_build_seconds": self.prototype_build_seconds,
            "verification_seconds": self.verification_seconds,
            "crop_count": self.crop_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AppearanceVerificationResult:
    query: str
    source_video: str
    tracks_path: str
    semantic_memory_reference: str
    prototypes: tuple[TrackAppearancePrototype, ...]
    candidate_scores: tuple[AppearanceCandidateScore, ...]
    runtime_info: AppearanceRuntimeInfo
    status: str
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_vectors: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "source_video": self.source_video,
            "tracks_path": self.tracks_path,
            "semantic_memory_reference": self.semantic_memory_reference,
            "status": self.status,
            "prototypes": [
                item.to_dict(include_vectors=include_vectors) for item in self.prototypes
            ],
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "runtime_info": self.runtime_info.to_dict(),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppearanceVerificationResult:
        return cls(
            query=str(data["query"]),
            source_video=str(data["source_video"]),
            tracks_path=str(data["tracks_path"]),
            semantic_memory_reference=str(data["semantic_memory_reference"]),
            status=str(data["status"]),
            prototypes=tuple(
                TrackAppearancePrototype.from_dict(item) for item in data.get("prototypes", [])
            ),
            candidate_scores=tuple(
                AppearanceCandidateScore.from_dict(item)
                for item in data.get("candidate_scores", [])
            ),
            runtime_info=AppearanceRuntimeInfo(
                backend_name=str(data["runtime_info"]["backend_name"]),
                model_id=str(data["runtime_info"]["model_id"]),
                crop_extraction_seconds=data["runtime_info"].get("crop_extraction_seconds"),
                embedding_seconds=data["runtime_info"].get("embedding_seconds"),
                prototype_build_seconds=data["runtime_info"].get("prototype_build_seconds"),
                verification_seconds=data["runtime_info"].get("verification_seconds"),
                crop_count=int(data["runtime_info"].get("crop_count", 0)),
                cache_hits=int(data["runtime_info"].get("cache_hits", 0)),
                cache_misses=int(data["runtime_info"].get("cache_misses", 0)),
                metadata=dict(data["runtime_info"].get("metadata", {})),
            ),
            warnings=tuple(data.get("warnings", ())),
        )
