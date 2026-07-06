"""Candidate generation for grounding-to-track association."""

from __future__ import annotations

from dataclasses import replace

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.geometry import (
    GeometryError,
    association_metrics,
    sanitize_track_bbox,
)
from football_tracking.locate_tracking.association.schemas import (
    AssociationConfig,
    TrackCandidate,
)
from football_tracking.locate_tracking.grounding.schemas import GroundedBox


def _score_candidate(metrics, config: AssociationConfig) -> float:
    weights = config.effective_weights
    return (
        weights["iou"] * metrics.iou
        + weights["track_coverage"] * metrics.track_coverage
        + weights["center_similarity"] * metrics.center_similarity
    )


def _gate_reason(metrics, config: AssociationConfig) -> tuple[bool, str]:
    if metrics.iou >= config.min_iou:
        return True, "iou threshold satisfied"
    if (
        metrics.track_center_inside_grounding
        and metrics.track_coverage >= config.min_track_coverage
    ):
        return True, "track center and coverage threshold satisfied"
    if not metrics.track_center_inside_grounding:
        return False, "center_incompatible"
    if metrics.track_coverage < config.min_track_coverage:
        return False, "insufficient_track_coverage"
    return False, "below_iou_threshold"


def generate_candidates(
    *,
    grounded_box: GroundedBox,
    grounded_box_index: int,
    track_observations: tuple[MotTrackObservation, ...],
    frame_width: int,
    frame_height: int,
    config: AssociationConfig,
) -> tuple[TrackCandidate, ...]:
    candidates: list[TrackCandidate] = []
    for observation in track_observations:
        try:
            sanitized = sanitize_track_bbox(
                observation.bbox_xyxy,
                frame_width,
                frame_height,
                clip_to_frame=config.clip_tracks_to_frame,
            )
            if not sanitized.valid or sanitized.matching_bbox is None:
                candidates.append(
                    TrackCandidate(
                        grounded_box_index=grounded_box_index,
                        track_id=observation.track_id,
                        frame_index=observation.frame_index,
                        grounding_bbox=grounded_box.bbox_xyxy,
                        raw_track_bbox=sanitized.raw_bbox,
                        matching_track_bbox=None,
                        was_clipped=sanitized.was_clipped,
                        metrics=None,
                        passed_gate=False,
                        gate_reason=sanitized.reason,
                        final_score=0.0,
                        rank=None,
                        effective_weights=config.effective_weights,
                    )
                )
                continue
            metrics = association_metrics(
                grounded_box.bbox_xyxy,
                sanitized.matching_bbox,
                frame_width,
                frame_height,
            )
            passed, reason = _gate_reason(metrics, config)
            candidates.append(
                TrackCandidate(
                    grounded_box_index=grounded_box_index,
                    track_id=observation.track_id,
                    frame_index=observation.frame_index,
                    grounding_bbox=grounded_box.bbox_xyxy,
                    raw_track_bbox=sanitized.raw_bbox,
                    matching_track_bbox=sanitized.matching_bbox,
                    was_clipped=sanitized.was_clipped,
                    metrics=metrics,
                    passed_gate=passed,
                    gate_reason=reason,
                    final_score=_score_candidate(metrics, config),
                    rank=None,
                    effective_weights=config.effective_weights,
                )
            )
        except GeometryError as exc:
            candidates.append(
                TrackCandidate(
                    grounded_box_index=grounded_box_index,
                    track_id=observation.track_id,
                    frame_index=observation.frame_index,
                    grounding_bbox=grounded_box.bbox_xyxy,
                    raw_track_bbox=observation.bbox_xyxy,
                    matching_track_bbox=None,
                    was_clipped=False,
                    metrics=None,
                    passed_gate=False,
                    gate_reason=f"invalid_track_geometry: {exc}",
                    final_score=0.0,
                    rank=None,
                    effective_weights=config.effective_weights,
                )
            )
    passed = sorted(
        (candidate for candidate in candidates if candidate.passed_gate),
        key=lambda item: (
            -item.final_score,
            -(item.metrics.iou if item.metrics else 0.0),
            -(item.metrics.track_coverage if item.metrics else 0.0),
            item.track_id,
        ),
    )
    ranked_passed = [replace(candidate, rank=index) for index, candidate in enumerate(passed, 1)]
    rejected = sorted(
        (candidate for candidate in candidates if not candidate.passed_gate),
        key=lambda item: item.track_id,
    )
    return tuple(ranked_passed + rejected)
