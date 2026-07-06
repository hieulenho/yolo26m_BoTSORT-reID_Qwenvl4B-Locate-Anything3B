"""Schemas for semantic + appearance score fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

FusionStatus = Literal["resolved", "ambiguous", "not_found", "insufficient_evidence"]
MissingAppearancePolicy = Literal["semantic_only", "penalize"]


class FusionSchemaError(ValueError):
    """Raised when fusion schemas or configs are invalid."""


def _unit(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise FusionSchemaError(f"{name} must be in [0, 1].")
    return numeric


@dataclass(frozen=True)
class FusionConfig:
    semantic_weight: float = 0.65
    appearance_weight: float = 0.35
    missing_appearance_policy: MissingAppearancePolicy = "semantic_only"
    missing_appearance_penalty: float = 0.15
    min_fused_score: float = 0.35
    winner_margin: float = 0.05
    query_mode: Literal["single_target", "multi_target"] = "single_target"

    def __post_init__(self) -> None:
        if self.missing_appearance_policy not in {"semantic_only", "penalize"}:
            raise FusionSchemaError("missing_appearance_policy must be semantic_only or penalize.")
        weights = (float(self.semantic_weight), float(self.appearance_weight))
        if any(not math.isfinite(item) or item < 0.0 for item in weights):
            raise FusionSchemaError("fusion weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise FusionSchemaError("fusion weights must sum to > 0.")
        object.__setattr__(self, "semantic_weight", weights[0])
        object.__setattr__(self, "appearance_weight", weights[1])
        object.__setattr__(
            self,
            "missing_appearance_penalty",
            _unit(self.missing_appearance_penalty, "missing_appearance_penalty"),
        )
        object.__setattr__(self, "min_fused_score", _unit(self.min_fused_score, "min_fused_score"))
        object.__setattr__(self, "winner_margin", _unit(self.winner_margin, "winner_margin"))

    @property
    def effective_weights(self) -> dict[str, float]:
        total = self.semantic_weight + self.appearance_weight
        return {
            "semantic": self.semantic_weight / total,
            "appearance": self.appearance_weight / total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_weight": self.semantic_weight,
            "appearance_weight": self.appearance_weight,
            "effective_weights": self.effective_weights,
            "missing_appearance_policy": self.missing_appearance_policy,
            "missing_appearance_penalty": self.missing_appearance_penalty,
            "min_fused_score": self.min_fused_score,
            "winner_margin": self.winner_margin,
            "query_mode": self.query_mode,
        }


@dataclass(frozen=True)
class FusedCandidateScore:
    raw_track_id: int
    semantic_score: float
    appearance_score: float | None
    fused_score: float
    appearance_status: str
    components: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "semantic_score", _unit(self.semantic_score, "semantic_score"))
        if self.appearance_score is not None:
            object.__setattr__(
                self,
                "appearance_score",
                _unit(self.appearance_score, "appearance_score"),
            )
        object.__setattr__(self, "fused_score", _unit(self.fused_score, "fused_score"))
        object.__setattr__(self, "components", dict(self.components))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "semantic_score": self.semantic_score,
            "appearance_score": self.appearance_score,
            "fused_score": self.fused_score,
            "appearance_status": self.appearance_status,
            "components": dict(self.components),
        }


@dataclass(frozen=True)
class FusionResult:
    query: str
    status: FusionStatus
    selected_track_id: int | None
    selected_track_ids: tuple[int, ...]
    candidate_scores: tuple[FusedCandidateScore, ...]
    decision_reason: str
    semantic_memory_reference: str
    appearance_scores_reference: str
    config: FusionConfig
    score_margin: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "selected_track_id": self.selected_track_id,
            "selected_track_ids": list(self.selected_track_ids),
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "decision_reason": self.decision_reason,
            "score_margin": self.score_margin,
            "semantic_memory_reference": self.semantic_memory_reference,
            "appearance_scores_reference": self.appearance_scores_reference,
            "config": self.config.to_dict(),
            "warnings": list(self.warnings),
        }
