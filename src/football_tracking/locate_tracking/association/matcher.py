"""Resolve grounded boxes against active track observations in one frame."""

from __future__ import annotations

from dataclasses import replace

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.candidate_generator import generate_candidates
from football_tracking.locate_tracking.association.schemas import (
    AssociationConfig,
    AssociationRuntimeInfo,
    FrameQueryResolution,
    GroundedBoxAssociation,
    OverallStatus,
    TrackCandidate,
)
from football_tracking.locate_tracking.grounding.schemas import GroundingResult


class GroundingTrackMatcherError(RuntimeError):
    """Raised when grounding-to-track matching cannot be evaluated."""


def _top_passed(candidates: tuple[TrackCandidate, ...]) -> list[TrackCandidate]:
    return [candidate for candidate in candidates if candidate.passed_gate]


def _candidate_payload(
    candidates: tuple[TrackCandidate, ...],
    config: AssociationConfig,
) -> tuple[TrackCandidate, ...]:
    if config.save_candidates:
        return candidates
    return tuple(candidate for candidate in candidates if candidate.passed_gate)[: config.top_k]


def _resolve_one(
    *,
    grounded_box_index: int,
    grounded_label: str,
    query: str,
    candidates: tuple[TrackCandidate, ...],
    config: AssociationConfig,
) -> GroundedBoxAssociation:
    passed = _top_passed(candidates)
    payload = _candidate_payload(candidates, config)
    if not passed:
        return GroundedBoxAssociation(
            grounded_box_index=grounded_box_index,
            grounded_label=grounded_label,
            query=query,
            status="not_found",
            selected_track_id=None,
            top_score=None,
            runner_up_score=None,
            score_margin=None,
            candidates=payload,
            decision_reason="no candidate passed geometric gate",
        )
    top = passed[0]
    runner_up = passed[1] if len(passed) > 1 else None
    runner_up_score = runner_up.final_score if runner_up else None
    margin = top.final_score - runner_up.final_score if runner_up else None
    if top.final_score < config.min_score:
        return GroundedBoxAssociation(
            grounded_box_index=grounded_box_index,
            grounded_label=grounded_label,
            query=query,
            status="not_found",
            selected_track_id=None,
            top_score=top.final_score,
            runner_up_score=runner_up_score,
            score_margin=margin,
            candidates=payload,
            decision_reason="top candidate score is below min_score",
        )
    if runner_up is not None and margin is not None and margin < config.ambiguity_margin:
        return GroundedBoxAssociation(
            grounded_box_index=grounded_box_index,
            grounded_label=grounded_label,
            query=query,
            status="ambiguous",
            selected_track_id=None,
            top_score=top.final_score,
            runner_up_score=runner_up_score,
            score_margin=margin,
            candidates=payload,
            decision_reason="top candidates are geometrically too close",
        )
    return GroundedBoxAssociation(
        grounded_box_index=grounded_box_index,
        grounded_label=grounded_label,
        query=query,
        status="resolved",
        selected_track_id=top.track_id,
        top_score=top.final_score,
        runner_up_score=runner_up_score,
        score_margin=margin,
        candidates=payload,
        decision_reason="top candidate passed score threshold and ambiguity margin",
    )


def _apply_duplicate_conflict_policy(
    associations: tuple[GroundedBoxAssociation, ...],
) -> tuple[GroundedBoxAssociation, ...]:
    owners: dict[int, GroundedBoxAssociation] = {}
    for association in associations:
        if association.status != "resolved" or association.selected_track_id is None:
            continue
        current = owners.get(association.selected_track_id)
        if current is None:
            owners[association.selected_track_id] = association
            continue
        current_score = current.top_score if current.top_score is not None else -1.0
        new_score = association.top_score if association.top_score is not None else -1.0
        if (new_score, -association.grounded_box_index) > (
            current_score,
            -current.grounded_box_index,
        ):
            owners[association.selected_track_id] = association
    resolved_owner_keys = {
        (owner.grounded_box_index, owner.selected_track_id) for owner in owners.values()
    }
    updated: list[GroundedBoxAssociation] = []
    for association in associations:
        key = (association.grounded_box_index, association.selected_track_id)
        if (
            association.status == "resolved"
            and association.selected_track_id is not None
            and key not in resolved_owner_keys
        ):
            updated.append(
                replace(
                    association,
                    status="ambiguous",
                    selected_track_id=None,
                    decision_reason="duplicate assignment conflict with another grounded box",
                )
            )
        else:
            updated.append(association)
    return tuple(updated)


def _overall_status(associations: tuple[GroundedBoxAssociation, ...]) -> OverallStatus:
    if not associations:
        return "not_found"
    statuses = [association.status for association in associations]
    if all(status == "resolved" for status in statuses):
        return "resolved"
    if any(status == "resolved" for status in statuses):
        return "partial"
    if any(status == "ambiguous" for status in statuses):
        return "ambiguous"
    return "not_found"


class GroundingTrackMatcher:
    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()

    def match(
        self,
        *,
        grounding_result: GroundingResult,
        track_observations: tuple[MotTrackObservation, ...],
        frame_width: int,
        frame_height: int,
        source_video: str | None,
        tracks_path: str,
        frame_index: int,
        grounding_result_reference: str | None = None,
        frame_info: dict[str, object] | None = None,
    ) -> FrameQueryResolution:
        if frame_width <= 0 or frame_height <= 0:
            raise GroundingTrackMatcherError("frame dimensions must be positive.")
        if frame_index < 1:
            raise GroundingTrackMatcherError("frame_index must be >= 1.")
        frame_tracks = tuple(sorted(track_observations, key=lambda item: item.track_id))
        associations: list[GroundedBoxAssociation] = []
        for index, grounded_box in enumerate(grounding_result.boxes):
            candidates = generate_candidates(
                grounded_box=grounded_box,
                grounded_box_index=index,
                track_observations=frame_tracks,
                frame_width=frame_width,
                frame_height=frame_height,
                config=self.config,
            )
            associations.append(
                _resolve_one(
                    grounded_box_index=index,
                    grounded_label=grounded_box.label,
                    query=grounding_result.request.query,
                    candidates=candidates,
                    config=self.config,
                )
            )
        associations_tuple = _apply_duplicate_conflict_policy(tuple(associations))
        return FrameQueryResolution(
            query=grounding_result.request.query,
            frame_index=frame_index,
            source_video=source_video,
            tracks_path=tracks_path,
            grounding_result_reference=grounding_result_reference,
            frame=dict(frame_info or {"width": frame_width, "height": frame_height}),
            grounding={
                "backend": grounding_result.request.backend,
                "model_id": grounding_result.request.model_id,
                "cache_hit": grounding_result.cache_hit,
                "box_count": len(grounding_result.boxes),
            },
            active_track_ids=tuple(track.track_id for track in frame_tracks),
            associations=associations_tuple,
            overall_status=_overall_status(associations_tuple),
            runtime_info=AssociationRuntimeInfo(config=self.config),
        )
