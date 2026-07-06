"""Decision policy for ranked reacquisition candidates."""

from __future__ import annotations

from football_tracking.locate_tracking.identity.schemas import SemanticTarget, stable_artifact_id
from football_tracking.locate_tracking.reacquisition.schemas import (
    ReacquisitionCandidate,
    ReacquisitionConfig,
    ReacquisitionDecision,
)


def decide_reacquisition(
    *,
    target: SemanticTarget,
    ranked_candidates: tuple[ReacquisitionCandidate, ...],
    all_candidates: tuple[ReacquisitionCandidate, ...],
    config: ReacquisitionConfig,
    event_ids: tuple[str, ...],
) -> ReacquisitionDecision:
    if not ranked_candidates:
        return ReacquisitionDecision(
            decision_id=stable_artifact_id(
                "decision",
                {
                    "target": target.semantic_target_id,
                    "event_ids": event_ids,
                    "status": "not_found",
                },
            ),
            status="not_found",
            semantic_target_id=target.semantic_target_id,
            previous_raw_track_id=target.current_raw_track_id,
            selected_raw_track_id=None,
            selected_start_frame=None,
            final_score=None,
            score_margin=None,
            reason="no_candidate_passed_hard_gates",
            event_ids=event_ids,
            candidate_count=len(all_candidates),
            ranked_candidates=ranked_candidates,
        )
    top = ranked_candidates[0]
    second = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    top_score = top.final_score
    if top_score is None or top_score < config.min_final_score:
        return ReacquisitionDecision(
            decision_id=stable_artifact_id(
                "decision",
                {"target": target.semantic_target_id, "event_ids": event_ids, "status": "rejected"},
            ),
            status="rejected",
            semantic_target_id=target.semantic_target_id,
            previous_raw_track_id=target.current_raw_track_id,
            selected_raw_track_id=None,
            selected_start_frame=None,
            final_score=top_score,
            score_margin=None,
            reason="best_candidate_below_min_final_score",
            event_ids=event_ids,
            candidate_count=len(all_candidates),
            ranked_candidates=ranked_candidates,
        )
    margin = (
        top_score - second.final_score
        if second is not None and second.final_score is not None
        else None
    )
    if margin is not None and margin < config.ambiguity_margin:
        return ReacquisitionDecision(
            decision_id=stable_artifact_id(
                "decision",
                {
                    "target": target.semantic_target_id,
                    "event_ids": event_ids,
                    "status": "ambiguous",
                },
            ),
            status="ambiguous",
            semantic_target_id=target.semantic_target_id,
            previous_raw_track_id=target.current_raw_track_id,
            selected_raw_track_id=None,
            selected_start_frame=None,
            final_score=top_score,
            score_margin=margin,
            reason="top_candidate_margin_below_threshold",
            event_ids=event_ids,
            candidate_count=len(all_candidates),
            ranked_candidates=ranked_candidates,
        )
    return ReacquisitionDecision(
        decision_id=stable_artifact_id(
            "decision",
            {
                "target": target.semantic_target_id,
                "event_ids": event_ids,
                "status": "provisional",
                "raw_track_id": top.raw_track_id,
            },
        ),
        status="provisional",
        semantic_target_id=target.semantic_target_id,
        previous_raw_track_id=target.current_raw_track_id,
        selected_raw_track_id=top.raw_track_id,
        selected_start_frame=top.first_observed_frame,
        final_score=top_score,
        score_margin=margin,
        reason="best_candidate_passed_score_and_margin",
        event_ids=event_ids,
        candidate_count=len(all_candidates),
        ranked_candidates=ranked_candidates,
    )
