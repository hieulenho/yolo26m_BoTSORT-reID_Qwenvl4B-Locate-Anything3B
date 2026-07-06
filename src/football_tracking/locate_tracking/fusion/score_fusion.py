"""Score fusion between M3 semantic memory and M4 appearance evidence."""

from __future__ import annotations

from football_tracking.locate_tracking.appearance.schemas import AppearanceCandidateScore
from football_tracking.locate_tracking.fusion.schemas import (
    FusedCandidateScore,
    FusionConfig,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemory


def _fuse_one(
    *,
    raw_track_id: int,
    semantic_score: float,
    appearance: AppearanceCandidateScore | None,
    config: FusionConfig,
) -> FusedCandidateScore:
    appearance_score = appearance.appearance_score if appearance is not None else None
    appearance_status = appearance.evidence_status if appearance is not None else "missing"
    weights = config.effective_weights
    if appearance_score is None:
        if config.missing_appearance_policy == "penalize":
            fused = max(0.0, semantic_score - config.missing_appearance_penalty)
            reason = "appearance unavailable, semantic score penalized"
        else:
            fused = semantic_score
            reason = "appearance unavailable, semantic score kept"
    else:
        fused = weights["semantic"] * semantic_score + weights["appearance"] * appearance_score
        reason = "semantic and appearance weighted"
    return FusedCandidateScore(
        raw_track_id=raw_track_id,
        semantic_score=semantic_score,
        appearance_score=appearance_score,
        fused_score=max(0.0, min(1.0, fused)),
        appearance_status=appearance_status,
        components={
            "effective_weights": weights,
            "missing_appearance_policy": config.missing_appearance_policy,
            "reason": reason,
        },
    )


def fuse_scores(
    *,
    semantic_memory: SemanticMemory,
    appearance_scores: tuple[AppearanceCandidateScore, ...],
    config: FusionConfig | None = None,
) -> tuple[FusedCandidateScore, ...]:
    cfg = config or FusionConfig(query_mode=semantic_memory.query_mode)
    appearance_by_track = {item.raw_track_id: item for item in appearance_scores}
    fused = [
        _fuse_one(
            raw_track_id=candidate.raw_track_id,
            semantic_score=candidate.aggregate_score,
            appearance=appearance_by_track.get(candidate.raw_track_id),
            config=cfg,
        )
        for candidate in semantic_memory.candidate_memories
    ]
    return tuple(
        sorted(
            fused,
            key=lambda item: (-item.fused_score, -item.semantic_score, item.raw_track_id),
        )
    )
