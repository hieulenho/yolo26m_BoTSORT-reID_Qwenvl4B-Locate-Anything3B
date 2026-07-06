"""Services for multi-frame language-track resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.association.service import (
    FrameTrackQueryService,
    FrameTrackQueryServiceError,
)
from football_tracking.locate_tracking.sampling.planner import build_frame_sampling_plan
from football_tracking.locate_tracking.sampling.schemas import (
    FrameSamplingPlan,
    FrameSamplingRequest,
)
from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.decision_policy import (
    decide_final_resolution,
)
from football_tracking.locate_tracking.semantic_memory.schemas import (
    LanguageTrackQuerySession,
    SemanticMemoryConfig,
)
from football_tracking.locate_tracking.semantic_memory.serialization import (
    load_frame_resolutions,
    save_final_resolution,
    save_language_track_session,
    save_semantic_memory,
)


class SemanticMemoryServiceError(RuntimeError):
    """Raised when semantic memory orchestration cannot complete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_frame_count(path: Path) -> int:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SemanticMemoryServiceError(f"Could not open video: {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 1:
            raise SemanticMemoryServiceError(f"Could not determine frame count: {path}")
        return total
    finally:
        capture.release()


def deterministic_session_id(
    *,
    query: str,
    source_video: str | None,
    tracks_path: str | None,
    sampling_plan: FrameSamplingPlan | None,
    config: SemanticMemoryConfig,
) -> str:
    payload = {
        "query": query,
        "source_video": source_video,
        "tracks_path": tracks_path,
        "sampling_plan": sampling_plan.to_dict() if sampling_plan else None,
        "semantic_memory_config": config.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class SemanticMemoryService:
    def __init__(
        self,
        *,
        config: SemanticMemoryConfig | None = None,
        frame_query_service: FrameTrackQueryService | None = None,
        overwrite: bool = False,
    ) -> None:
        self.config = config or SemanticMemoryConfig()
        self.frame_query_service = frame_query_service
        self.overwrite = bool(overwrite)

    def aggregate_frame_resolutions(
        self,
        *,
        query: str,
        frame_resolution_paths: tuple[str | Path, ...],
        output_dir: str | Path,
        sampled_frames: tuple[int, ...] | None = None,
        source_video: str | None = None,
        tracks_path: str | None = None,
    ) -> LanguageTrackQuerySession:
        frame_resolutions = load_frame_resolutions(frame_resolution_paths)
        memory = build_semantic_memory(
            query=query,
            frame_resolutions=frame_resolutions,
            config=self.config,
            sampled_frames=sampled_frames,
            runtime_info={"mode": "aggregate_existing_frame_resolutions"},
        )
        output_root = Path(output_dir)
        semantic_path = output_root / "semantic_memory.json"
        final = decide_final_resolution(
            memory,
            self.config,
            semantic_memory_reference=str(semantic_path),
        )
        session = LanguageTrackQuerySession(
            session_id=deterministic_session_id(
                query=query,
                source_video=source_video,
                tracks_path=tracks_path,
                sampling_plan=None,
                config=self.config,
            ),
            query=query,
            source_video=source_video,
            tracks_path=tracks_path,
            sampling_plan=None,
            frame_resolutions=frame_resolutions,
            semantic_memory=memory,
            final_resolution=final,
            runtime_info={
                "mode": "aggregate_existing_frame_resolutions",
                "frame_resolution_paths": [str(path) for path in frame_resolution_paths],
            },
        )
        save_semantic_memory(memory, semantic_path, overwrite=self.overwrite)
        save_final_resolution(
            final, output_root / "final_resolution.json", overwrite=self.overwrite
        )
        save_language_track_session(session, output_root / "session.json", overwrite=self.overwrite)
        return session

    def resolve_language_track(
        self,
        *,
        source_video: str | Path,
        tracks_path: str | Path,
        query: str,
        sampling_request: FrameSamplingRequest,
        output_dir: str | Path,
        save_overlay: bool = False,
    ) -> LanguageTrackQuerySession:
        if self.frame_query_service is None:
            raise SemanticMemoryServiceError(
                "frame_query_service is required for end-to-end video resolution."
            )
        video_path = Path(source_video)
        mot_path = Path(tracks_path)
        if not video_path.is_file():
            raise SemanticMemoryServiceError(f"Video does not exist: {video_path}")
        if not mot_path.is_file():
            raise SemanticMemoryServiceError(f"MOT track file does not exist: {mot_path}")
        tracks_hash_before = _sha256_file(mot_path)
        plan = build_frame_sampling_plan(sampling_request)
        output_root = Path(output_dir)
        frame_resolutions: list[dict[str, Any]] = []
        for selected in plan.selected_frames:
            frame_dir = output_root / "frames" / f"frame_{selected.frame_index:06d}"
            try:
                result = self.frame_query_service.query_track_frame(
                    source_video=video_path,
                    tracks_path=mot_path,
                    frame_index=selected.frame_index,
                    query=query,
                    output_dir=frame_dir,
                    save_overlay=save_overlay,
                )
            except FrameTrackQueryServiceError as exc:
                raise SemanticMemoryServiceError(str(exc)) from exc
            frame_resolutions.append(result.to_dict())
        tracks_hash_after = _sha256_file(mot_path)
        if tracks_hash_before != tracks_hash_after:
            raise SemanticMemoryServiceError("MOT track file changed during semantic run.")
        memory = build_semantic_memory(
            query=query,
            frame_resolutions=tuple(frame_resolutions),
            config=self.config,
            sampled_frames=plan.frame_indices,
            runtime_info={
                "mode": "end_to_end_video_resolution",
                "tracks_sha256_before": tracks_hash_before,
                "tracks_sha256_after": tracks_hash_after,
            },
        )
        semantic_path = output_root / "semantic_memory.json"
        final = decide_final_resolution(
            memory,
            self.config,
            semantic_memory_reference=str(semantic_path),
        )
        session = LanguageTrackQuerySession(
            session_id=deterministic_session_id(
                query=query,
                source_video=str(video_path),
                tracks_path=str(mot_path),
                sampling_plan=plan,
                config=self.config,
            ),
            query=query,
            source_video=str(video_path),
            tracks_path=str(mot_path),
            sampling_plan=plan,
            frame_resolutions=tuple(frame_resolutions),
            semantic_memory=memory,
            final_resolution=final,
            runtime_info={
                "mode": "end_to_end_video_resolution",
                "tracks_sha256_before": tracks_hash_before,
                "tracks_sha256_after": tracks_hash_after,
            },
        )
        save_semantic_memory(memory, semantic_path, overwrite=self.overwrite)
        save_final_resolution(
            final, output_root / "final_resolution.json", overwrite=self.overwrite
        )
        save_language_track_session(session, output_root / "session.json", overwrite=self.overwrite)
        return session


def build_sampling_request_for_video(
    *,
    source_video: str | Path,
    sampling_config: dict[str, Any],
) -> FrameSamplingRequest:
    total_frames = _video_frame_count(Path(source_video))
    return FrameSamplingRequest(
        total_frames=total_frames,
        max_frames=int(sampling_config.get("max_frames", 5)),
        start_frame=int(sampling_config.get("start_frame", 1)),
        end_frame=(
            int(sampling_config["end_frame"])
            if sampling_config.get("end_frame") is not None
            else None
        ),
        explicit_frames=tuple(sampling_config.get("explicit_frames", ())),
        mode=str(sampling_config.get("mode", "uniform")),  # type: ignore[arg-type]
    )
