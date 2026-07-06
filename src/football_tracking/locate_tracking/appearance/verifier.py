"""Appearance evidence scoring for semantic track candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.locate_tracking.appearance.prototype_bank import (
    internal_consistency_score,
)
from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceCandidateScore,
    TrackAppearancePrototype,
)


@dataclass(frozen=True)
class AppearanceVerifierConfig:
    min_samples_for_consistency: int = 2
    min_verified_score: float = 0.70
    weak_score_threshold: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples_for_consistency": self.min_samples_for_consistency,
            "min_verified_score": self.min_verified_score,
            "weak_score_threshold": self.weak_score_threshold,
        }


def score_track_appearance(
    prototype: TrackAppearancePrototype,
    config: AppearanceVerifierConfig | None = None,
) -> AppearanceCandidateScore:
    cfg = config or AppearanceVerifierConfig()
    if prototype.sample_count < cfg.min_samples_for_consistency:
        return AppearanceCandidateScore(
            raw_track_id=prototype.raw_track_id,
            prototype_similarity=None,
            internal_consistency=None,
            appearance_score=None,
            sample_count=prototype.sample_count,
            evidence_status="insufficient_appearance_evidence",
            decision_reason="not enough samples for leave-one-out appearance consistency",
            score_components={"config": cfg.to_dict()},
        )
    consistency = internal_consistency_score(
        prototype.sample_embeddings,
        strategy=prototype.aggregation_strategy,
    )
    if consistency is None:
        return AppearanceCandidateScore(
            raw_track_id=prototype.raw_track_id,
            prototype_similarity=None,
            internal_consistency=None,
            appearance_score=None,
            sample_count=prototype.sample_count,
            evidence_status="unavailable",
            decision_reason="appearance consistency unavailable",
            score_components={"config": cfg.to_dict()},
        )
    status = "verified" if consistency >= cfg.min_verified_score else "weak"
    if consistency < cfg.weak_score_threshold:
        status = "weak"
    return AppearanceCandidateScore(
        raw_track_id=prototype.raw_track_id,
        prototype_similarity=consistency,
        internal_consistency=consistency,
        appearance_score=consistency,
        sample_count=prototype.sample_count,
        evidence_status=status,
        decision_reason="leave-one-out prototype consistency computed",
        score_components={
            "prototype_similarity": consistency,
            "internal_consistency": consistency,
            "config": cfg.to_dict(),
        },
    )


def score_appearance_prototypes(
    prototypes: tuple[TrackAppearancePrototype, ...],
    config: AppearanceVerifierConfig | None = None,
) -> tuple[AppearanceCandidateScore, ...]:
    return tuple(
        sorted(
            (score_track_appearance(prototype, config) for prototype in prototypes),
            key=lambda item: (
                -(item.appearance_score if item.appearance_score is not None else -1.0),
                item.raw_track_id,
            ),
        )
    )
