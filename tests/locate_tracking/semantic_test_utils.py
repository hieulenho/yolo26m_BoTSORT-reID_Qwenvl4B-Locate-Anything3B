from __future__ import annotations

from football_tracking.locate_tracking.association.schemas import (
    AssociationConfig,
    AssociationMetrics,
    AssociationRuntimeInfo,
    FrameQueryResolution,
    GroundedBoxAssociation,
    TrackCandidate,
)


def candidate(
    track_id: int,
    frame_index: int,
    score: float,
    *,
    rank: int = 1,
    passed: bool = True,
) -> TrackCandidate:
    return TrackCandidate(
        grounded_box_index=0,
        track_id=track_id,
        frame_index=frame_index,
        grounding_bbox=(10, 10, 30, 30),
        raw_track_bbox=(10, 10, 30, 30),
        matching_track_bbox=(10, 10, 30, 30),
        was_clipped=False,
        metrics=AssociationMetrics(
            intersection_area=400,
            iou=score,
            grounding_coverage=score,
            track_coverage=score,
            center_distance_px=0,
            center_distance_normalized=0,
            center_similarity=score,
            track_center_inside_grounding=True,
            grounding_center_inside_track=True,
        ),
        passed_gate=passed,
        gate_reason="passed" if passed else "below threshold",
        final_score=score,
        rank=rank,
        effective_weights={"iou": 0.7, "track_coverage": 0.2, "center_similarity": 0.1},
    )


def association(
    *,
    frame_index: int,
    status: str,
    selected_track_id: int | None,
    candidates: tuple[TrackCandidate, ...],
) -> GroundedBoxAssociation:
    top = candidates[0].final_score if candidates else None
    runner = candidates[1].final_score if len(candidates) > 1 else None
    return GroundedBoxAssociation(
        grounded_box_index=0,
        grounded_label="player",
        query="player",
        status=status,  # type: ignore[arg-type]
        selected_track_id=selected_track_id,
        top_score=top,
        runner_up_score=runner,
        score_margin=top - runner if top is not None and runner is not None else None,
        candidates=candidates,
        decision_reason=f"{status} at {frame_index}",
    )


def frame_resolution(
    frame_index: int,
    associations: tuple[GroundedBoxAssociation, ...],
    *,
    cache_hit: bool = False,
) -> FrameQueryResolution:
    if not associations:
        overall_status = "not_found"
    elif all(item.status == "resolved" for item in associations):
        overall_status = "resolved"
    elif any(item.status == "resolved" for item in associations):
        overall_status = "partial"
    elif any(item.status == "ambiguous" for item in associations):
        overall_status = "ambiguous"
    else:
        overall_status = "not_found"
    return FrameQueryResolution(
        query="player",
        frame_index=frame_index,
        source_video="video.mp4",
        tracks_path="tracks.txt",
        grounding_result_reference=f"frame_{frame_index}.grounding.json",
        frame={"width": 100, "height": 100},
        grounding={
            "backend": "mock",
            "model_id": "mock-grounding",
            "cache_hit": cache_hit,
            "box_count": len(associations),
        },
        active_track_ids=tuple(
            sorted({candidate.track_id for assoc in associations for candidate in assoc.candidates})
        ),
        associations=associations,
        overall_status=overall_status,  # type: ignore[arg-type]
        runtime_info=AssociationRuntimeInfo(config=AssociationConfig()),
    )


def resolved_frame(frame_index: int, track_id: int, score: float = 0.8) -> FrameQueryResolution:
    cand = candidate(track_id, frame_index, score)
    return frame_resolution(
        frame_index,
        (
            association(
                frame_index=frame_index,
                status="resolved",
                selected_track_id=track_id,
                candidates=(cand,),
            ),
        ),
    )


def ambiguous_frame(
    frame_index: int,
    left_track: int = 3,
    right_track: int = 7,
) -> FrameQueryResolution:
    left = candidate(left_track, frame_index, 0.71, rank=1)
    right = candidate(right_track, frame_index, 0.69, rank=2)
    return frame_resolution(
        frame_index,
        (
            association(
                frame_index=frame_index,
                status="ambiguous",
                selected_track_id=None,
                candidates=(left, right),
            ),
        ),
    )
