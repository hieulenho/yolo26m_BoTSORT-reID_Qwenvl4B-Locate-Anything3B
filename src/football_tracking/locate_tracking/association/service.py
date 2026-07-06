"""Orchestration for single-frame language query to track association."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_reader import (
    MotReaderError,
    read_mot_track_file,
)
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.association.matcher import GroundingTrackMatcher
from football_tracking.locate_tracking.association.schemas import FrameQueryResolution
from football_tracking.locate_tracking.grounding.schemas import GroundingResult
from football_tracking.locate_tracking.grounding.service import GroundingService
from football_tracking.locate_tracking.video.frame_extractor import (
    FrameExtractionError,
    extract_video_frame,
    save_extracted_frame,
)
from football_tracking.locate_tracking.visualization.association_overlay import (
    render_association_overlay,
)


class FrameTrackQueryServiceError(RuntimeError):
    """Raised when a frame query cannot be resolved against tracking artifacts."""


def save_association_result(
    result: FrameQueryResolution,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FrameTrackQueryServiceError(f"Association output exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_grounding_result(path: str | Path) -> GroundingResult:
    resolved = Path(path)
    if not resolved.is_file():
        raise FrameTrackQueryServiceError(f"Grounding result does not exist: {resolved}")
    try:
        return GroundingResult.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        raise FrameTrackQueryServiceError(
            f"Could not load grounding result JSON: {resolved}: {exc}"
        ) from exc


class FrameTrackQueryService:
    def __init__(
        self,
        *,
        matcher: GroundingTrackMatcher,
        grounding_service: GroundingService | None = None,
        overwrite: bool = False,
    ) -> None:
        self.matcher = matcher
        self.grounding_service = grounding_service
        self.overwrite = bool(overwrite)

    def match_existing_grounding(
        self,
        *,
        grounding_result_path: str | Path,
        tracks_path: str | Path,
        frame_index: int,
        frame_width: int,
        frame_height: int,
        output_path: str | Path | None = None,
    ) -> FrameQueryResolution:
        grounding_result = load_grounding_result(grounding_result_path)
        mot_file = read_mot_track_file(tracks_path)
        frame_tracks = FrameTrackIndex.from_observations(mot_file.observations).get_frame(
            frame_index
        )
        result = self.matcher.match(
            grounding_result=grounding_result,
            track_observations=frame_tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            source_video=None,
            tracks_path=str(tracks_path),
            frame_index=frame_index,
            grounding_result_reference=str(grounding_result_path),
            frame_info={"width": frame_width, "height": frame_height},
        )
        if output_path is not None:
            save_association_result(result, output_path, overwrite=self.overwrite)
        return result

    def query_track_frame(
        self,
        *,
        source_video: str | Path,
        tracks_path: str | Path,
        frame_index: int,
        query: str,
        output_dir: str | Path,
        save_overlay: bool = False,
    ) -> FrameQueryResolution:
        if self.grounding_service is None:
            raise FrameTrackQueryServiceError("grounding_service is required for query mode.")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            frame = extract_video_frame(source_video, frame_index)
        except FrameExtractionError as exc:
            raise FrameTrackQueryServiceError(str(exc)) from exc
        frame_path = save_extracted_frame(frame, output_root / f"frame_{frame_index:06d}.jpg")
        grounding_path = output_root / "grounding.json"
        grounding_result = self.grounding_service.ground_image(
            image_path=frame_path,
            query=query,
            output_path=grounding_path,
            overwrite=self.overwrite,
        )
        try:
            mot_file = read_mot_track_file(tracks_path)
        except MotReaderError as exc:
            raise FrameTrackQueryServiceError(str(exc)) from exc
        frame_tracks = FrameTrackIndex.from_observations(mot_file.observations).get_frame(
            frame_index
        )
        result = self.matcher.match(
            grounding_result=grounding_result,
            track_observations=frame_tracks,
            frame_width=frame.width,
            frame_height=frame.height,
            source_video=str(source_video),
            tracks_path=str(tracks_path),
            frame_index=frame_index,
            grounding_result_reference=str(grounding_path),
            frame_info={
                "width": frame.width,
                "height": frame.height,
                "fps": frame.fps,
                "timestamp_seconds": frame.timestamp_seconds,
                "total_frames": frame.total_frames,
            },
        )
        association_path = output_root / "association.json"
        save_association_result(result, association_path, overwrite=self.overwrite)
        if save_overlay:
            try:
                render_association_overlay(frame.image, result, output_root / "overlay.jpg")
            except Exception:  # noqa: BLE001
                pass
        return result
