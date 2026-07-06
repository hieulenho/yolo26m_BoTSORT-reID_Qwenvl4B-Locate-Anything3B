"""Build read-only target observation timelines from M2-M4 artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.appearance.schemas import AppearanceVerificationResult
from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.association.geometry import bbox_center
from football_tracking.locate_tracking.monitoring.schemas import (
    MotionMetrics,
    TargetFrameObservation,
    TargetObservationTimeline,
)
from football_tracking.locate_tracking.semantic_memory.schemas import (
    CandidateSemanticMemory,
    SemanticMemory,
)
from football_tracking.locate_tracking.semantic_memory.serialization import load_semantic_memory


class TargetObserverError(RuntimeError):
    """Raised when a target observation timeline cannot be built."""


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise TargetObserverError(f"Artifact does not exist: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TargetObserverError(f"Invalid JSON artifact: {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise TargetObserverError(f"JSON artifact must be an object: {resolved}")
    return data


def load_appearance_result(path: str | Path | None) -> AppearanceVerificationResult | None:
    if path is None:
        return None
    return AppearanceVerificationResult.from_dict(_load_json(path))


def load_fusion_result_dict(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_json(path)


def _candidate(memory: SemanticMemory, raw_track_id: int) -> CandidateSemanticMemory | None:
    for item in memory.candidate_memories:
        if item.raw_track_id == raw_track_id:
            return item
    return None


def _appearance_score(
    appearance_result: AppearanceVerificationResult | None,
    raw_track_id: int,
) -> tuple[float | None, str]:
    if appearance_result is None:
        return None, "missing_appearance_artifact"
    for item in appearance_result.candidate_scores:
        if item.raw_track_id == raw_track_id:
            return item.appearance_score, item.evidence_status
    return None, "track_not_in_appearance_artifact"


def _fusion_scores(
    fusion_result: dict[str, Any] | None,
    raw_track_id: int,
) -> tuple[float | None, float | None, float | None, int | None, str]:
    if fusion_result is None:
        return None, None, None, None, "missing_fusion_artifact"
    raw_candidates = fusion_result.get("candidate_scores", ())
    if not isinstance(raw_candidates, list):
        return None, None, None, None, "invalid_fusion_candidates"
    sorted_candidates = sorted(
        (
            item
            for item in raw_candidates
            if isinstance(item, dict) and item.get("fused_score") is not None
        ),
        key=lambda item: (-float(item["fused_score"]), int(item.get("raw_track_id", 0))),
    )
    target = next(
        (item for item in sorted_candidates if int(item.get("raw_track_id", -1)) == raw_track_id),
        None,
    )
    if target is None:
        return None, None, None, len(sorted_candidates), "track_not_in_fusion_artifact"
    target_score = float(target["fused_score"])
    runner = next(
        (float(item["fused_score"]) for item in sorted_candidates if item is not target),
        None,
    )
    margin = target_score - runner if runner is not None else None
    return target_score, runner, margin, len(sorted_candidates), "session_level"


def select_current_track_id(
    semantic_memory: SemanticMemory,
    fusion_result: dict[str, Any] | None = None,
    explicit_track_id: int | None = None,
) -> int:
    if explicit_track_id is not None:
        if int(explicit_track_id) < 1:
            raise TargetObserverError("current track id must be >= 1.")
        return int(explicit_track_id)
    if fusion_result is not None:
        selected = fusion_result.get("selected_track_id")
        if selected is not None:
            return int(selected)
        selected_ids = fusion_result.get("selected_track_ids", ())
        if selected_ids:
            return int(selected_ids[0])
    candidates = semantic_memory.sorted_candidates()
    if not candidates:
        raise TargetObserverError("Semantic memory has no candidate tracks.")
    return int(candidates[0].raw_track_id)


def _observations_by_track(
    frame_rows: tuple[MotTrackObservation, ...],
) -> dict[int, MotTrackObservation]:
    return {row.track_id: row for row in frame_rows}


def _motion_metrics(row: MotTrackObservation | None) -> MotionMetrics | None:
    if row is None:
        return None
    center_x, center_y = bbox_center(row.bbox_xyxy)
    return MotionMetrics(center_x=center_x, center_y=center_y)


def _timeline_range(
    index: FrameTrackIndex,
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int]:
    available = index.available_frame_range
    if available is None:
        raise TargetObserverError("MOT file has no observations.")
    start = available[0] if start_frame is None else int(start_frame)
    end = available[1] if end_frame is None else int(end_frame)
    if start < 1 or end < start:
        raise TargetObserverError("Invalid timeline frame range.")
    return start, end


def build_target_observation_timeline(
    *,
    tracks_path: str | Path,
    semantic_memory: SemanticMemory,
    appearance_result: AppearanceVerificationResult | None = None,
    fusion_result: dict[str, Any] | None = None,
    current_raw_track_id: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    source_video: str | Path | None = None,
) -> TargetObservationTimeline:
    mot_file = read_mot_track_file(tracks_path)
    index = FrameTrackIndex.from_observations(mot_file.observations)
    selected_track_id = select_current_track_id(
        semantic_memory,
        fusion_result=fusion_result,
        explicit_track_id=current_raw_track_id,
    )
    start, end = _timeline_range(index, start_frame, end_frame)
    semantic_candidate = _candidate(semantic_memory, selected_track_id)
    semantic_score = semantic_candidate.aggregate_score if semantic_candidate else None
    semantic_frames = (
        set(semantic_candidate.frames_present)
        | set(semantic_candidate.resolved_support_frames)
        | set(semantic_candidate.ambiguous_support_frames)
        if semantic_candidate
        else set()
    )
    appearance_score, appearance_status = _appearance_score(appearance_result, selected_track_id)
    fused_score, runner_up, margin, candidate_count, fusion_availability = _fusion_scores(
        fusion_result,
        selected_track_id,
    )
    observations: list[TargetFrameObservation] = []
    for frame_index in range(start, end + 1):
        frame_rows = index.get_frame(frame_index)
        by_track = _observations_by_track(frame_rows)
        row = by_track.get(selected_track_id)
        availability: dict[str, str] = {
            "tracking": "present" if row else "absent",
            "confidence": "mot_confidence" if row and row.confidence is not None else "unavailable",
            "semantic_score": (
                "sampled_frame"
                if frame_index in semantic_frames
                else ("session_level_only" if semantic_candidate else "unavailable")
            ),
            "appearance_score": appearance_status,
            "fused_score": fusion_availability,
        }
        observations.append(
            TargetFrameObservation(
                frame_index=frame_index,
                target_present=row is not None,
                raw_track_id=selected_track_id if row is not None else None,
                bbox_xyxy=row.bbox_xyxy if row is not None else None,
                tracking_confidence=row.confidence if row is not None else None,
                semantic_score=semantic_score if frame_index in semantic_frames else None,
                appearance_score=appearance_score,
                fused_score=fused_score,
                candidate_count=candidate_count,
                runner_up_score=runner_up,
                winner_margin=margin,
                motion_metrics=_motion_metrics(row),
                data_availability=availability,
            )
        )
    hypothesis = f"query={semantic_memory.query!r}, raw_track_id={selected_track_id}"
    metadata = {
        "tracks_path": str(tracks_path),
        "source_video": str(source_video) if source_video is not None else None,
        "mot_observation_count": mot_file.observation_count,
        "available_track_ids": list(index.unique_track_ids),
        "score_granularity": {
            "semantic": "sampled_frames_only",
            "appearance": "session_level",
            "fusion": "session_level",
        },
    }
    if semantic_score is not None and not math.isfinite(float(semantic_score)):
        raise TargetObserverError("Semantic score is not finite.")
    return TargetObservationTimeline(
        query=semantic_memory.query,
        semantic_target_hypothesis=hypothesis,
        current_raw_track_id=selected_track_id,
        start_frame=start,
        end_frame=end,
        observations=tuple(observations),
        metadata=metadata,
    )


def build_target_observation_timeline_from_paths(
    *,
    tracks_path: str | Path,
    semantic_memory_path: str | Path,
    appearance_result_path: str | Path | None = None,
    fusion_result_path: str | Path | None = None,
    current_raw_track_id: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    source_video: str | Path | None = None,
) -> TargetObservationTimeline:
    return build_target_observation_timeline(
        tracks_path=tracks_path,
        semantic_memory=load_semantic_memory(semantic_memory_path),
        appearance_result=load_appearance_result(appearance_result_path),
        fusion_result=load_fusion_result_dict(fusion_result_path),
        current_raw_track_id=current_raw_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        source_video=source_video,
    )
