"""Schemas for single-frame grounding-to-track association."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

AssociationStatus = Literal["resolved", "ambiguous", "not_found"]
OverallStatus = Literal["resolved", "partial", "ambiguous", "not_found"]


class AssociationSchemaError(ValueError):
    """Raised when association configuration or results are invalid."""


def _bbox4(value: Any, field_name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise AssociationSchemaError(f"{field_name} must contain four numbers.") from exc
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise AssociationSchemaError(f"{field_name} must contain four finite numbers.")
    return values


def _validate_unit(value: float, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise AssociationSchemaError(f"{field_name} must be in [0, 1].")
    return numeric


@dataclass(frozen=True)
class AssociationConfig:
    min_iou: float = 0.10
    min_track_coverage: float = 0.50
    iou_weight: float = 0.70
    track_coverage_weight: float = 0.20
    center_similarity_weight: float = 0.10
    min_score: float = 0.20
    ambiguity_margin: float = 0.05
    top_k: int = 5
    clip_tracks_to_frame: bool = True
    reject_fully_outside_tracks: bool = True
    save_candidates: bool = True
    save_overlay: bool = False

    def __post_init__(self) -> None:
        for name in ("min_iou", "min_track_coverage", "min_score", "ambiguity_margin"):
            object.__setattr__(self, name, _validate_unit(float(getattr(self, name)), name))
        for name in ("iou_weight", "track_coverage_weight", "center_similarity_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise AssociationSchemaError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.iou_weight + self.track_coverage_weight + self.center_similarity_weight <= 0.0:
            raise AssociationSchemaError("Association score weights must sum to > 0.")
        if int(self.top_k) <= 0:
            raise AssociationSchemaError("top_k must be positive.")
        object.__setattr__(self, "top_k", int(self.top_k))

    @property
    def effective_weights(self) -> dict[str, float]:
        total = self.iou_weight + self.track_coverage_weight + self.center_similarity_weight
        return {
            "iou": self.iou_weight / total,
            "track_coverage": self.track_coverage_weight / total,
            "center_similarity": self.center_similarity_weight / total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_iou": self.min_iou,
            "min_track_coverage": self.min_track_coverage,
            "score": {
                "iou_weight": self.iou_weight,
                "track_coverage_weight": self.track_coverage_weight,
                "center_similarity_weight": self.center_similarity_weight,
                "effective_weights": self.effective_weights,
            },
            "decision": {
                "min_score": self.min_score,
                "ambiguity_margin": self.ambiguity_margin,
                "top_k": self.top_k,
            },
            "geometry": {
                "clip_tracks_to_frame": self.clip_tracks_to_frame,
                "reject_fully_outside_tracks": self.reject_fully_outside_tracks,
            },
            "output": {
                "save_candidates": self.save_candidates,
                "save_overlay": self.save_overlay,
            },
        }


@dataclass(frozen=True)
class AssociationMetrics:
    intersection_area: float
    iou: float
    grounding_coverage: float
    track_coverage: float
    center_distance_px: float
    center_distance_normalized: float
    center_similarity: float
    track_center_inside_grounding: bool
    grounding_center_inside_track: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "intersection_area": self.intersection_area,
            "iou": self.iou,
            "grounding_coverage": self.grounding_coverage,
            "track_coverage": self.track_coverage,
            "center_distance_px": self.center_distance_px,
            "center_distance_normalized": self.center_distance_normalized,
            "center_similarity": self.center_similarity,
            "track_center_inside_grounding": self.track_center_inside_grounding,
            "grounding_center_inside_track": self.grounding_center_inside_track,
        }


@dataclass(frozen=True)
class TrackCandidate:
    grounded_box_index: int
    track_id: int
    frame_index: int
    grounding_bbox: tuple[float, float, float, float]
    raw_track_bbox: tuple[float, float, float, float]
    matching_track_bbox: tuple[float, float, float, float] | None
    was_clipped: bool
    metrics: AssociationMetrics | None
    passed_gate: bool
    gate_reason: str
    final_score: float
    rank: int | None
    effective_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grounding_bbox", _bbox4(self.grounding_bbox, "grounding_bbox"))
        object.__setattr__(self, "raw_track_bbox", _bbox4(self.raw_track_bbox, "raw_track_bbox"))
        if self.matching_track_bbox is not None:
            object.__setattr__(
                self,
                "matching_track_bbox",
                _bbox4(self.matching_track_bbox, "matching_track_bbox"),
            )
        score = float(self.final_score)
        if not math.isfinite(score):
            raise AssociationSchemaError("final_score must be finite.")
        object.__setattr__(self, "final_score", score)

    def with_rank(self, rank: int | None) -> TrackCandidate:
        return replace(self, rank=rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded_box_index": self.grounded_box_index,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "grounding_bbox": list(self.grounding_bbox),
            "raw_track_bbox": list(self.raw_track_bbox),
            "matching_track_bbox": list(self.matching_track_bbox)
            if self.matching_track_bbox is not None
            else None,
            "was_clipped": self.was_clipped,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "passed_gate": self.passed_gate,
            "gate_reason": self.gate_reason,
            "final_score": self.final_score,
            "rank": self.rank,
            "effective_weights": dict(self.effective_weights),
        }


@dataclass(frozen=True)
class GroundedBoxAssociation:
    grounded_box_index: int
    grounded_label: str
    query: str
    status: AssociationStatus
    selected_track_id: int | None
    top_score: float | None
    runner_up_score: float | None
    score_margin: float | None
    candidates: tuple[TrackCandidate, ...]
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded_box_index": self.grounded_box_index,
            "label": self.grounded_label,
            "query": self.query,
            "status": self.status,
            "selected_track_id": self.selected_track_id,
            "top_score": self.top_score,
            "runner_up_score": self.runner_up_score,
            "score_margin": self.score_margin,
            "decision_reason": self.decision_reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class AssociationRuntimeInfo:
    config: AssociationConfig
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class FrameQueryResolution:
    query: str
    frame_index: int
    source_video: str | None
    tracks_path: str
    grounding_result_reference: str | None
    frame: dict[str, object]
    grounding: dict[str, object]
    active_track_ids: tuple[int, ...]
    associations: tuple[GroundedBoxAssociation, ...]
    overall_status: OverallStatus
    runtime_info: AssociationRuntimeInfo

    @property
    def resolved_track_ids(self) -> tuple[int, ...]:
        return tuple(
            association.selected_track_id
            for association in self.associations
            if association.status == "resolved" and association.selected_track_id is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "frame_index": self.frame_index,
            "inputs": {
                "source_video": self.source_video,
                "tracks_path": self.tracks_path,
            },
            "frame": dict(self.frame),
            "grounding": dict(self.grounding),
            "active_tracks": {
                "count": len(self.active_track_ids),
                "track_ids": list(self.active_track_ids),
            },
            "grounding_result_reference": self.grounding_result_reference,
            "grounded_box_count": int(self.grounding.get("box_count", len(self.associations))),
            "active_track_count": len(self.active_track_ids),
            "associations": [association.to_dict() for association in self.associations],
            "resolved_track_ids": list(self.resolved_track_ids),
            "overall_status": self.overall_status,
            "runtime_info": self.runtime_info.to_dict(),
        }
