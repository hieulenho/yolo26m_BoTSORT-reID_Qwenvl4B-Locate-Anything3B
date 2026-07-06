"""Schemas for cross-frame semantic track memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Literal

from football_tracking.locate_tracking.sampling.schemas import FrameSamplingPlan

AggregationStrategy = Literal["weighted", "majority_support"]
DecisionStatus = Literal["resolved", "ambiguous", "not_found", "insufficient_evidence"]
QueryMode = Literal["single_target", "multi_target"]


class SemanticMemoryError(ValueError):
    """Raised when semantic memory input or configuration is invalid."""


def _finite_or_none(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SemanticMemoryError(f"{field_name} must be finite when provided.")
    return numeric


def _validate_unit(value: float, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise SemanticMemoryError(f"{field_name} must be in [0, 1].")
    return numeric


@dataclass(frozen=True)
class SemanticMemoryConfig:
    query_mode: QueryMode = "single_target"
    aggregation_strategy: AggregationStrategy = "weighted"
    support_weight: float = 0.50
    quality_weight: float = 0.30
    consistency_weight: float = 0.20
    quality_score_mode: Literal["mean", "top_k_mean"] = "top_k_mean"
    top_k_quality: int = 3
    resolved_selected_weight: float = 1.0
    ambiguous_candidate_weight: float = 0.25
    weak_candidate_weight: float = 0.10
    min_usable_frames: int = 2
    min_support_frames: int = 2
    min_support_ratio: float = 0.40
    min_aggregate_score: float = 0.35
    winner_margin: float = 0.08

    def __post_init__(self) -> None:
        if self.query_mode not in {"single_target", "multi_target"}:
            raise SemanticMemoryError("query_mode must be single_target or multi_target.")
        if self.aggregation_strategy not in {"weighted", "majority_support"}:
            raise SemanticMemoryError("aggregation_strategy must be weighted or majority_support.")
        if self.quality_score_mode not in {"mean", "top_k_mean"}:
            raise SemanticMemoryError("quality_score_mode must be mean or top_k_mean.")
        weights = (
            float(self.support_weight),
            float(self.quality_weight),
            float(self.consistency_weight),
        )
        if any(not math.isfinite(item) or item < 0.0 for item in weights):
            raise SemanticMemoryError("aggregation weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise SemanticMemoryError("aggregation weights must sum to > 0.")
        object.__setattr__(self, "support_weight", weights[0])
        object.__setattr__(self, "quality_weight", weights[1])
        object.__setattr__(self, "consistency_weight", weights[2])
        object.__setattr__(
            self,
            "resolved_selected_weight",
            _validate_unit(self.resolved_selected_weight, "resolved_selected_weight"),
        )
        object.__setattr__(
            self,
            "ambiguous_candidate_weight",
            _validate_unit(self.ambiguous_candidate_weight, "ambiguous_candidate_weight"),
        )
        object.__setattr__(
            self,
            "weak_candidate_weight",
            _validate_unit(self.weak_candidate_weight, "weak_candidate_weight"),
        )
        if int(self.top_k_quality) < 1:
            raise SemanticMemoryError("top_k_quality must be >= 1.")
        if int(self.min_usable_frames) < 0:
            raise SemanticMemoryError("min_usable_frames must be >= 0.")
        if int(self.min_support_frames) < 0:
            raise SemanticMemoryError("min_support_frames must be >= 0.")
        object.__setattr__(self, "top_k_quality", int(self.top_k_quality))
        object.__setattr__(self, "min_usable_frames", int(self.min_usable_frames))
        object.__setattr__(self, "min_support_frames", int(self.min_support_frames))
        object.__setattr__(
            self, "min_support_ratio", _validate_unit(self.min_support_ratio, "min_support_ratio")
        )
        object.__setattr__(
            self,
            "min_aggregate_score",
            _validate_unit(self.min_aggregate_score, "min_aggregate_score"),
        )
        object.__setattr__(
            self, "winner_margin", _validate_unit(self.winner_margin, "winner_margin")
        )

    @property
    def effective_weights(self) -> dict[str, float]:
        total = self.support_weight + self.quality_weight + self.consistency_weight
        return {
            "support": self.support_weight / total,
            "quality": self.quality_weight / total,
            "consistency": self.consistency_weight / total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_mode": self.query_mode,
            "aggregation": {
                "strategy": self.aggregation_strategy,
                "quality_score_mode": self.quality_score_mode,
                "top_k_quality": self.top_k_quality,
                "weights": {
                    "support_weight": self.support_weight,
                    "quality_weight": self.quality_weight,
                    "consistency_weight": self.consistency_weight,
                    "effective_weights": self.effective_weights,
                },
            },
            "evidence_weights": {
                "resolved_selected_weight": self.resolved_selected_weight,
                "ambiguous_candidate_weight": self.ambiguous_candidate_weight,
                "weak_candidate_weight": self.weak_candidate_weight,
            },
            "decision": {
                "min_usable_frames": self.min_usable_frames,
                "min_support_frames": self.min_support_frames,
                "min_support_ratio": self.min_support_ratio,
                "min_aggregate_score": self.min_aggregate_score,
                "winner_margin": self.winner_margin,
            },
        }


@dataclass(frozen=True)
class SemanticEvidence:
    query: str
    frame_index: int
    grounded_box_index: int
    grounded_label: str
    raw_track_id: int
    single_frame_status: str
    association_score: float | None
    iou: float | None
    track_coverage: float | None
    center_similarity: float | None
    candidate_rank: int | None
    passed_gate: bool
    selected_in_frame: bool
    grounding_cache_hit: bool
    evidence_weight: float
    evidence_reason: str

    def __post_init__(self) -> None:
        if int(self.frame_index) < 1:
            raise SemanticMemoryError("frame_index must be >= 1.")
        if int(self.raw_track_id) < 1:
            raise SemanticMemoryError("raw_track_id must be >= 1.")
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "grounded_box_index", int(self.grounded_box_index))
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(
            self, "association_score", _finite_or_none(self.association_score, "association_score")
        )
        object.__setattr__(self, "iou", _finite_or_none(self.iou, "iou"))
        object.__setattr__(
            self, "track_coverage", _finite_or_none(self.track_coverage, "track_coverage")
        )
        object.__setattr__(
            self, "center_similarity", _finite_or_none(self.center_similarity, "center_similarity")
        )
        if self.candidate_rank is not None:
            object.__setattr__(self, "candidate_rank", int(self.candidate_rank))
        object.__setattr__(
            self, "evidence_weight", _validate_unit(self.evidence_weight, "evidence_weight")
        )

    @property
    def is_positive_support(self) -> bool:
        return self.single_frame_status == "resolved" and self.selected_in_frame

    @property
    def is_ambiguous_support(self) -> bool:
        return self.single_frame_status == "ambiguous" and self.passed_gate

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "frame_index": self.frame_index,
            "grounded_box_index": self.grounded_box_index,
            "grounded_label": self.grounded_label,
            "raw_track_id": self.raw_track_id,
            "single_frame_status": self.single_frame_status,
            "association_score": self.association_score,
            "iou": self.iou,
            "track_coverage": self.track_coverage,
            "center_similarity": self.center_similarity,
            "candidate_rank": self.candidate_rank,
            "passed_gate": self.passed_gate,
            "selected_in_frame": self.selected_in_frame,
            "grounding_cache_hit": self.grounding_cache_hit,
            "evidence_weight": self.evidence_weight,
            "evidence_reason": self.evidence_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticEvidence:
        return cls(
            query=str(data["query"]),
            frame_index=int(data["frame_index"]),
            grounded_box_index=int(data["grounded_box_index"]),
            grounded_label=str(data["grounded_label"]),
            raw_track_id=int(data["raw_track_id"]),
            single_frame_status=str(data["single_frame_status"]),
            association_score=data.get("association_score"),
            iou=data.get("iou"),
            track_coverage=data.get("track_coverage"),
            center_similarity=data.get("center_similarity"),
            candidate_rank=data.get("candidate_rank"),
            passed_gate=bool(data["passed_gate"]),
            selected_in_frame=bool(data["selected_in_frame"]),
            grounding_cache_hit=bool(data.get("grounding_cache_hit", False)),
            evidence_weight=float(data["evidence_weight"]),
            evidence_reason=str(data["evidence_reason"]),
        )


def _valid_scores(evidence: tuple[SemanticEvidence, ...]) -> tuple[float, ...]:
    return tuple(
        item.association_score
        for item in evidence
        if item.association_score is not None and math.isfinite(item.association_score)
    )


def _mean_or_zero(values: tuple[float, ...]) -> float:
    return float(mean(values)) if values else 0.0


@dataclass(frozen=True)
class CandidateSemanticMemory:
    raw_track_id: int
    evidence_history: tuple[SemanticEvidence, ...]
    sampled_frames: tuple[int, ...]
    frames_present: tuple[int, ...]
    frames_with_grounding_match: tuple[int, ...]
    resolved_support_frames: tuple[int, ...]
    ambiguous_support_frames: tuple[int, ...]
    support_count: int
    support_ratio: float
    score_history: tuple[float, ...]
    mean_score: float
    median_score: float
    top_k_mean_score: float
    best_score: float
    worst_score: float
    first_evidence_frame: int | None
    last_evidence_frame: int | None
    cross_frame_consistency: float
    aggregate_score: float
    decision_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.raw_track_id) < 1:
            raise SemanticMemoryError("raw_track_id must be >= 1.")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "evidence_history", tuple(self.evidence_history))
        for name in (
            "sampled_frames",
            "frames_present",
            "frames_with_grounding_match",
            "resolved_support_frames",
            "ambiguous_support_frames",
        ):
            object.__setattr__(
                self,
                name,
                tuple(sorted(set(int(item) for item in getattr(self, name)))),
            )
        object.__setattr__(self, "score_history", tuple(float(item) for item in self.score_history))
        object.__setattr__(
            self, "support_ratio", _validate_unit(self.support_ratio, "support_ratio")
        )
        object.__setattr__(
            self,
            "cross_frame_consistency",
            _validate_unit(self.cross_frame_consistency, "cross_frame_consistency"),
        )
        object.__setattr__(
            self, "aggregate_score", _validate_unit(self.aggregate_score, "aggregate_score")
        )
        object.__setattr__(self, "decision_metadata", dict(self.decision_metadata))

    @classmethod
    def from_evidence(
        cls,
        *,
        raw_track_id: int,
        evidence_history: tuple[SemanticEvidence, ...],
        sampled_frames: tuple[int, ...],
        usable_grounding_frame_count: int,
        config: SemanticMemoryConfig,
        decision_metadata: dict[str, Any] | None = None,
    ) -> CandidateSemanticMemory:
        evidence = tuple(
            sorted(
                evidence_history,
                key=lambda item: (
                    item.frame_index,
                    item.grounded_box_index,
                    item.candidate_rank or 9999,
                ),
            )
        )
        frames_present = tuple(sorted({item.frame_index for item in evidence}))
        frames_with_match = tuple(
            sorted({item.frame_index for item in evidence if item.passed_gate})
        )
        resolved_support_frames = tuple(
            sorted({item.frame_index for item in evidence if item.is_positive_support})
        )
        ambiguous_support_frames = tuple(
            sorted({item.frame_index for item in evidence if item.is_ambiguous_support})
        )
        support_count = len(resolved_support_frames)
        denominator = max(1, int(usable_grounding_frame_count))
        support_ratio = support_count / denominator
        scores = _valid_scores(evidence)
        top_k_scores = tuple(sorted(scores, reverse=True)[: config.top_k_quality])
        quality_score = (
            _mean_or_zero(top_k_scores)
            if config.quality_score_mode == "top_k_mean"
            else _mean_or_zero(scores)
        )
        consistency_denominator = max(1, len(frames_with_match))
        consistency = support_count / consistency_denominator
        effective = config.effective_weights
        aggregate = (
            effective["support"] * support_ratio
            + effective["quality"] * quality_score
            + effective["consistency"] * consistency
        )
        metadata = dict(decision_metadata or {})
        metadata.setdefault(
            "aggregate_components",
            {
                "support_score": support_ratio,
                "quality_score": quality_score,
                "consistency_score": consistency,
                "effective_weights": effective,
                "strategy": config.aggregation_strategy,
            },
        )
        return cls(
            raw_track_id=raw_track_id,
            evidence_history=evidence,
            sampled_frames=sampled_frames,
            frames_present=frames_present,
            frames_with_grounding_match=frames_with_match,
            resolved_support_frames=resolved_support_frames,
            ambiguous_support_frames=ambiguous_support_frames,
            support_count=support_count,
            support_ratio=support_ratio,
            score_history=scores,
            mean_score=_mean_or_zero(scores),
            median_score=float(median(scores)) if scores else 0.0,
            top_k_mean_score=_mean_or_zero(top_k_scores),
            best_score=max(scores) if scores else 0.0,
            worst_score=min(scores) if scores else 0.0,
            first_evidence_frame=frames_present[0] if frames_present else None,
            last_evidence_frame=frames_present[-1] if frames_present else None,
            cross_frame_consistency=consistency,
            aggregate_score=min(1.0, max(0.0, aggregate)),
            decision_metadata=metadata,
        )

    def with_decision_metadata(self, metadata: dict[str, Any]) -> CandidateSemanticMemory:
        merged = dict(self.decision_metadata)
        merged.update(metadata)
        return CandidateSemanticMemory(
            raw_track_id=self.raw_track_id,
            evidence_history=self.evidence_history,
            sampled_frames=self.sampled_frames,
            frames_present=self.frames_present,
            frames_with_grounding_match=self.frames_with_grounding_match,
            resolved_support_frames=self.resolved_support_frames,
            ambiguous_support_frames=self.ambiguous_support_frames,
            support_count=self.support_count,
            support_ratio=self.support_ratio,
            score_history=self.score_history,
            mean_score=self.mean_score,
            median_score=self.median_score,
            top_k_mean_score=self.top_k_mean_score,
            best_score=self.best_score,
            worst_score=self.worst_score,
            first_evidence_frame=self.first_evidence_frame,
            last_evidence_frame=self.last_evidence_frame,
            cross_frame_consistency=self.cross_frame_consistency,
            aggregate_score=self.aggregate_score,
            decision_metadata=merged,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "evidence_history": [item.to_dict() for item in self.evidence_history],
            "sampled_frames": list(self.sampled_frames),
            "frames_present": list(self.frames_present),
            "frames_with_grounding_match": list(self.frames_with_grounding_match),
            "resolved_support_frames": list(self.resolved_support_frames),
            "ambiguous_support_frames": list(self.ambiguous_support_frames),
            "support_count": self.support_count,
            "support_ratio": self.support_ratio,
            "score_history": list(self.score_history),
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "top_k_mean_score": self.top_k_mean_score,
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "first_evidence_frame": self.first_evidence_frame,
            "last_evidence_frame": self.last_evidence_frame,
            "cross_frame_consistency": self.cross_frame_consistency,
            "aggregate_score": self.aggregate_score,
            "decision_metadata": dict(self.decision_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSemanticMemory:
        return cls(
            raw_track_id=int(data["raw_track_id"]),
            evidence_history=tuple(
                SemanticEvidence.from_dict(item) for item in data.get("evidence_history", [])
            ),
            sampled_frames=tuple(data.get("sampled_frames", ())),
            frames_present=tuple(data.get("frames_present", ())),
            frames_with_grounding_match=tuple(data.get("frames_with_grounding_match", ())),
            resolved_support_frames=tuple(data.get("resolved_support_frames", ())),
            ambiguous_support_frames=tuple(data.get("ambiguous_support_frames", ())),
            support_count=int(data.get("support_count", 0)),
            support_ratio=float(data.get("support_ratio", 0.0)),
            score_history=tuple(data.get("score_history", ())),
            mean_score=float(data.get("mean_score", 0.0)),
            median_score=float(data.get("median_score", 0.0)),
            top_k_mean_score=float(data.get("top_k_mean_score", 0.0)),
            best_score=float(data.get("best_score", 0.0)),
            worst_score=float(data.get("worst_score", 0.0)),
            first_evidence_frame=data.get("first_evidence_frame"),
            last_evidence_frame=data.get("last_evidence_frame"),
            cross_frame_consistency=float(data.get("cross_frame_consistency", 0.0)),
            aggregate_score=float(data.get("aggregate_score", 0.0)),
            decision_metadata=dict(data.get("decision_metadata", {})),
        )


@dataclass(frozen=True)
class SemanticMemory:
    query: str
    query_mode: QueryMode
    planned_frame_count: int
    processed_frame_count: int
    usable_grounding_frame_count: int
    resolved_frame_count: int
    ambiguous_frame_count: int
    not_found_frame_count: int
    sampled_frames: tuple[int, ...]
    candidate_memories: tuple[CandidateSemanticMemory, ...]
    aggregation_config: dict[str, Any]
    runtime_info: dict[str, Any] = field(default_factory=dict)

    def sorted_candidates(self) -> tuple[CandidateSemanticMemory, ...]:
        if all(item.decision_metadata.get("rank") is not None for item in self.candidate_memories):
            return tuple(
                sorted(
                    self.candidate_memories,
                    key=lambda item: (int(item.decision_metadata["rank"]), item.raw_track_id),
                )
            )
        return tuple(
            sorted(
                self.candidate_memories,
                key=lambda item: (
                    -item.aggregate_score,
                    -item.support_count,
                    -item.mean_score,
                    item.raw_track_id,
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_mode": self.query_mode,
            "counts": {
                "planned_frame_count": self.planned_frame_count,
                "processed_frame_count": self.processed_frame_count,
                "usable_grounding_frame_count": self.usable_grounding_frame_count,
                "resolved_frame_count": self.resolved_frame_count,
                "ambiguous_frame_count": self.ambiguous_frame_count,
                "not_found_frame_count": self.not_found_frame_count,
            },
            "sampled_frames": list(self.sampled_frames),
            "candidate_memories": [item.to_dict() for item in self.candidate_memories],
            "aggregation_config": dict(self.aggregation_config),
            "runtime_info": dict(self.runtime_info),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticMemory:
        counts = data.get("counts", {})
        return cls(
            query=str(data["query"]),
            query_mode=str(data.get("query_mode", "single_target")),  # type: ignore[arg-type]
            planned_frame_count=int(counts.get("planned_frame_count", 0)),
            processed_frame_count=int(counts.get("processed_frame_count", 0)),
            usable_grounding_frame_count=int(counts.get("usable_grounding_frame_count", 0)),
            resolved_frame_count=int(counts.get("resolved_frame_count", 0)),
            ambiguous_frame_count=int(counts.get("ambiguous_frame_count", 0)),
            not_found_frame_count=int(counts.get("not_found_frame_count", 0)),
            sampled_frames=tuple(data.get("sampled_frames", ())),
            candidate_memories=tuple(
                CandidateSemanticMemory.from_dict(item)
                for item in data.get("candidate_memories", [])
            ),
            aggregation_config=dict(data.get("aggregation_config", {})),
            runtime_info=dict(data.get("runtime_info", {})),
        )


@dataclass(frozen=True)
class FinalLanguageTrackResolution:
    query: str
    query_mode: QueryMode
    status: DecisionStatus
    selected_track_id: int | None
    selected_track_ids: tuple[int, ...]
    candidate_count: int
    decision_reason: str
    best_raw_track_id: int | None
    runner_up_raw_track_id: int | None
    score_margin: float | None
    thresholds: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    semantic_memory_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_mode": self.query_mode,
            "status": self.status,
            "selected_track_id": self.selected_track_id,
            "selected_track_ids": list(self.selected_track_ids),
            "candidate_count": self.candidate_count,
            "decision_reason": self.decision_reason,
            "best_raw_track_id": self.best_raw_track_id,
            "runner_up_raw_track_id": self.runner_up_raw_track_id,
            "score_margin": self.score_margin,
            "thresholds": dict(self.thresholds),
            "candidates": [dict(item) for item in self.candidates],
            "semantic_memory_reference": self.semantic_memory_reference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalLanguageTrackResolution:
        return cls(
            query=str(data["query"]),
            query_mode=str(data.get("query_mode", "single_target")),  # type: ignore[arg-type]
            status=str(data["status"]),  # type: ignore[arg-type]
            selected_track_id=data.get("selected_track_id"),
            selected_track_ids=tuple(data.get("selected_track_ids", ())),
            candidate_count=int(data.get("candidate_count", 0)),
            decision_reason=str(data.get("decision_reason", "")),
            best_raw_track_id=data.get("best_raw_track_id"),
            runner_up_raw_track_id=data.get("runner_up_raw_track_id"),
            score_margin=data.get("score_margin"),
            thresholds=dict(data.get("thresholds", {})),
            candidates=tuple(dict(item) for item in data.get("candidates", [])),
            semantic_memory_reference=data.get("semantic_memory_reference"),
        )


@dataclass(frozen=True)
class LanguageTrackQuerySession:
    session_id: str
    query: str
    source_video: str | None
    tracks_path: str | None
    sampling_plan: FrameSamplingPlan | None
    frame_resolutions: tuple[dict[str, Any], ...]
    semantic_memory: SemanticMemory
    final_resolution: FinalLanguageTrackResolution
    runtime_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "source_video": self.source_video,
            "tracks_path": self.tracks_path,
            "sampling_plan": self.sampling_plan.to_dict() if self.sampling_plan else None,
            "frame_resolutions": [dict(item) for item in self.frame_resolutions],
            "semantic_memory": self.semantic_memory.to_dict(),
            "final_resolution": self.final_resolution.to_dict(),
            "runtime_info": dict(self.runtime_info),
        }
