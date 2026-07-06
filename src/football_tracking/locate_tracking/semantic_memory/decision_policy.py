"""Decision policy for final language-track resolution."""

from __future__ import annotations

from football_tracking.locate_tracking.semantic_memory.schemas import (
    CandidateSemanticMemory,
    FinalLanguageTrackResolution,
    SemanticMemory,
    SemanticMemoryConfig,
)


def _candidate_summary(memory: CandidateSemanticMemory) -> dict[str, object]:
    return {
        "raw_track_id": memory.raw_track_id,
        "aggregate_score": memory.aggregate_score,
        "support_count": memory.support_count,
        "support_ratio": memory.support_ratio,
        "mean_score": memory.mean_score,
        "best_score": memory.best_score,
        "cross_frame_consistency": memory.cross_frame_consistency,
        "rank": memory.decision_metadata.get("rank"),
    }


def _thresholds(config: SemanticMemoryConfig) -> dict[str, object]:
    return {
        "min_usable_frames": config.min_usable_frames,
        "min_support_frames": config.min_support_frames,
        "min_support_ratio": config.min_support_ratio,
        "min_aggregate_score": config.min_aggregate_score,
        "winner_margin": config.winner_margin,
        "aggregation_strategy": config.aggregation_strategy,
    }


def _passes_candidate(memory: CandidateSemanticMemory, config: SemanticMemoryConfig) -> bool:
    return (
        memory.support_count >= config.min_support_frames
        and memory.support_ratio >= config.min_support_ratio
        and memory.aggregate_score >= config.min_aggregate_score
    )


def decide_final_resolution(
    semantic_memory: SemanticMemory,
    config: SemanticMemoryConfig | None = None,
    *,
    semantic_memory_reference: str | None = None,
) -> FinalLanguageTrackResolution:
    cfg = config or SemanticMemoryConfig(query_mode=semantic_memory.query_mode)
    candidates = semantic_memory.sorted_candidates()
    summaries = tuple(_candidate_summary(item) for item in candidates)
    if semantic_memory.usable_grounding_frame_count < cfg.min_usable_frames:
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="insufficient_evidence",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_count=len(candidates),
            decision_reason="usable grounding frames below min_usable_frames",
            best_raw_track_id=candidates[0].raw_track_id if candidates else None,
            runner_up_raw_track_id=candidates[1].raw_track_id if len(candidates) > 1 else None,
            score_margin=None,
            thresholds=_thresholds(cfg),
            candidates=summaries,
            semantic_memory_reference=semantic_memory_reference,
        )
    if not candidates:
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="not_found",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_count=0,
            decision_reason="no candidate memories were produced",
            best_raw_track_id=None,
            runner_up_raw_track_id=None,
            score_margin=None,
            thresholds=_thresholds(cfg),
            candidates=(),
            semantic_memory_reference=semantic_memory_reference,
        )
    if cfg.query_mode == "multi_target":
        selected = tuple(item.raw_track_id for item in candidates if _passes_candidate(item, cfg))
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="resolved" if selected else "not_found",
            selected_track_id=selected[0] if len(selected) == 1 else None,
            selected_track_ids=selected,
            candidate_count=len(candidates),
            decision_reason=(
                "all candidates passing multi-target thresholds selected"
                if selected
                else "no candidates passed multi-target thresholds"
            ),
            best_raw_track_id=candidates[0].raw_track_id,
            runner_up_raw_track_id=candidates[1].raw_track_id if len(candidates) > 1 else None,
            score_margin=None,
            thresholds=_thresholds(cfg),
            candidates=summaries,
            semantic_memory_reference=semantic_memory_reference,
        )
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = best.aggregate_score - runner_up.aggregate_score if runner_up else None
    if best.support_count < cfg.min_support_frames:
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="insufficient_evidence",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_count=len(candidates),
            decision_reason="best candidate support_count below min_support_frames",
            best_raw_track_id=best.raw_track_id,
            runner_up_raw_track_id=runner_up.raw_track_id if runner_up else None,
            score_margin=margin,
            thresholds=_thresholds(cfg),
            candidates=summaries,
            semantic_memory_reference=semantic_memory_reference,
        )
    if best.support_ratio < cfg.min_support_ratio or best.aggregate_score < cfg.min_aggregate_score:
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="not_found",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_count=len(candidates),
            decision_reason="best candidate below support ratio or aggregate score threshold",
            best_raw_track_id=best.raw_track_id,
            runner_up_raw_track_id=runner_up.raw_track_id if runner_up else None,
            score_margin=margin,
            thresholds=_thresholds(cfg),
            candidates=summaries,
            semantic_memory_reference=semantic_memory_reference,
        )
    if runner_up is not None and margin is not None and margin < cfg.winner_margin:
        return FinalLanguageTrackResolution(
            query=semantic_memory.query,
            query_mode=cfg.query_mode,
            status="ambiguous",
            selected_track_id=None,
            selected_track_ids=(),
            candidate_count=len(candidates),
            decision_reason="top two candidates are within winner_margin",
            best_raw_track_id=best.raw_track_id,
            runner_up_raw_track_id=runner_up.raw_track_id,
            score_margin=margin,
            thresholds=_thresholds(cfg),
            candidates=summaries,
            semantic_memory_reference=semantic_memory_reference,
        )
    return FinalLanguageTrackResolution(
        query=semantic_memory.query,
        query_mode=cfg.query_mode,
        status="resolved",
        selected_track_id=best.raw_track_id,
        selected_track_ids=(best.raw_track_id,),
        candidate_count=len(candidates),
        decision_reason="best candidate passed cross-frame thresholds and margin",
        best_raw_track_id=best.raw_track_id,
        runner_up_raw_track_id=runner_up.raw_track_id if runner_up else None,
        score_margin=margin,
        thresholds=_thresholds(cfg),
        candidates=summaries,
        semantic_memory_reference=semantic_memory_reference,
    )
