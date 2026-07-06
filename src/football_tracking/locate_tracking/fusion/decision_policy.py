"""Decision policy over fused semantic + appearance scores."""

from __future__ import annotations

from football_tracking.locate_tracking.fusion.schemas import (
    FusedCandidateScore,
    FusionConfig,
    FusionResult,
)


def decide_fused_resolution(
    *,
    query: str,
    fused_scores: tuple[FusedCandidateScore, ...],
    semantic_memory_reference: str,
    appearance_scores_reference: str,
    config: FusionConfig | None = None,
    warnings: tuple[str, ...] = (),
) -> FusionResult:
    cfg = config or FusionConfig()
    if not fused_scores:
        return FusionResult(
            query=query,
            status="not_found",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_scores=(),
            decision_reason="no fused candidate scores",
            semantic_memory_reference=semantic_memory_reference,
            appearance_scores_reference=appearance_scores_reference,
            config=cfg,
            warnings=warnings,
        )
    ranked = tuple(
        sorted(
            fused_scores,
            key=lambda item: (-item.fused_score, -item.semantic_score, item.raw_track_id),
        )
    )
    if cfg.query_mode == "multi_target":
        selected = tuple(
            item.raw_track_id for item in ranked if item.fused_score >= cfg.min_fused_score
        )
        return FusionResult(
            query=query,
            status="resolved" if selected else "not_found",
            selected_track_id=selected[0] if len(selected) == 1 else None,
            selected_track_ids=selected,
            candidate_scores=ranked,
            decision_reason=(
                "all candidates above min_fused_score selected"
                if selected
                else "no candidate exceeded min_fused_score"
            ),
            semantic_memory_reference=semantic_memory_reference,
            appearance_scores_reference=appearance_scores_reference,
            config=cfg,
            warnings=warnings,
        )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = best.fused_score - runner_up.fused_score if runner_up else None
    if best.fused_score < cfg.min_fused_score:
        return FusionResult(
            query=query,
            status="not_found",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_scores=ranked,
            decision_reason="best candidate below min_fused_score",
            score_margin=margin,
            semantic_memory_reference=semantic_memory_reference,
            appearance_scores_reference=appearance_scores_reference,
            config=cfg,
            warnings=warnings,
        )
    if runner_up is not None and margin is not None and margin < cfg.winner_margin:
        return FusionResult(
            query=query,
            status="ambiguous",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_scores=ranked,
            decision_reason="top fused candidates are within winner_margin",
            score_margin=margin,
            semantic_memory_reference=semantic_memory_reference,
            appearance_scores_reference=appearance_scores_reference,
            config=cfg,
            warnings=warnings,
        )
    return FusionResult(
        query=query,
        status="resolved",
        selected_track_id=best.raw_track_id,
        selected_track_ids=(best.raw_track_id,),
        candidate_scores=ranked,
        decision_reason="best fused candidate passed threshold and margin",
        score_margin=margin,
        semantic_memory_reference=semantic_memory_reference,
        appearance_scores_reference=appearance_scores_reference,
        config=cfg,
        warnings=warnings,
    )
