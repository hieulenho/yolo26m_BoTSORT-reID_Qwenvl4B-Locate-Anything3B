"""Render MOT tracking outputs as annotated videos."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.bbox import xywh_to_xyxy
from football_tracking.data.schemas import BoundingBoxXYWH
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.schemas import TrackOutput
from football_tracking.tracking.sequence_runner import (
    SequenceSource,
    discover_mot_sequences,
    iter_source_frames,
)
from football_tracking.tracking.trajectory import TrajectoryStore
from football_tracking.visualization.draw_tracks import draw_tracks
from football_tracking.visualization.tracking_video import create_tracking_video_writer


class RenderVideoError(RuntimeError):
    """Raised when an annotated tracking video cannot be rendered."""


@dataclass(frozen=True)
class RenderVideoConfig:
    project_root: Path
    config_path: Path
    mot_root: Path
    split: str
    seqmap: Path | None
    tracker_name: str
    tracks_root: Path
    output_root: Path
    show_confidence: bool
    show_class: bool
    show_track_id: bool
    show_trajectory: bool
    show_fps: bool
    show_frame_number: bool
    show_sequence_name: bool
    show_tracker_name: bool
    trajectory_length: int
    line_thickness: int
    font_scale: float
    max_sequences: int | None
    max_frames_per_sequence: int | None
    overwrite: bool
    log_level: str


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RenderVideoError(f"{section} must be a mapping.")
    return value


def _resolve_path(
    value: Any,
    project_root: Path,
    section: str,
    required: bool = True,
) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RenderVideoError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_render_video_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> RenderVideoConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise RenderVideoError(f"Render config does not exist: {resolved}")
    raw = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "render config root")
    dataset = _mapping(raw.get("dataset"), "dataset")
    tracking = _mapping(raw.get("tracking"), "tracking")
    output = _mapping(raw.get("output"), "output")
    overlay = _mapping(raw.get("overlay", {}), "overlay")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    config = RenderVideoConfig(
        project_root=project_root,
        config_path=resolved,
        mot_root=_resolve_path(dataset.get("mot_root"), project_root, "dataset.mot_root"),
        split=str(dataset.get("split", "val")),
        seqmap=_resolve_path(dataset.get("seqmap"), project_root, "dataset.seqmap", required=False),
        tracker_name=str(tracking.get("tracker_name", "deepsort")),
        tracks_root=_resolve_path(
            tracking.get("tracks_root"),
            project_root,
            "tracking.tracks_root",
        ),
        output_root=_resolve_path(output.get("videos_root"), project_root, "output.videos_root"),
        show_confidence=bool(overlay.get("show_confidence", True)),
        show_class=bool(overlay.get("show_class", True)),
        show_track_id=bool(overlay.get("show_track_id", True)),
        show_trajectory=bool(overlay.get("show_trajectory", True)),
        show_fps=bool(overlay.get("show_fps", True)),
        show_frame_number=bool(overlay.get("show_frame_number", True)),
        show_sequence_name=bool(overlay.get("show_sequence_name", True)),
        show_tracker_name=bool(overlay.get("show_tracker_name", True)),
        trajectory_length=int(overlay.get("trajectory_length", 30)),
        line_thickness=int(overlay.get("line_thickness", 2)),
        font_scale=float(overlay.get("font_scale", 0.6)),
        max_sequences=runtime.get("max_sequences"),
        max_frames_per_sequence=runtime.get(
            "max_frames_per_sequence",
            runtime.get("max_frames"),
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if config.max_sequences is not None:
        config = replace(config, max_sequences=int(config.max_sequences))
    if config.max_frames_per_sequence is not None:
        config = replace(config, max_frames_per_sequence=int(config.max_frames_per_sequence))
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_config(config)
    return config


def _apply_overrides(config: RenderVideoConfig, overrides: dict[str, Any]) -> RenderVideoConfig:
    changes: dict[str, Any] = {}
    if overrides.get("tracker") is not None:
        changes["tracker_name"] = str(overrides["tracker"])
    if overrides.get("max_sequences") is not None:
        changes["max_sequences"] = int(overrides["max_sequences"])
    if overrides.get("max_frames") is not None:
        changes["max_frames_per_sequence"] = int(overrides["max_frames"])
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config


def _validate_config(config: RenderVideoConfig) -> None:
    if not config.mot_root.is_dir():
        raise RenderVideoError(f"dataset.mot_root does not exist: {config.mot_root}")
    if config.seqmap is not None and not config.seqmap.is_file():
        raise RenderVideoError(f"dataset.seqmap does not exist: {config.seqmap}")
    if config.trajectory_length < 0:
        raise RenderVideoError("overlay.trajectory_length must be non-negative.")
    if config.max_sequences is not None and config.max_sequences <= 0:
        raise RenderVideoError("runtime.max_sequences must be positive when set.")
    if config.max_frames_per_sequence is not None and config.max_frames_per_sequence <= 0:
        raise RenderVideoError("runtime.max_frames_per_sequence must be positive when set.")


def _tracks_path(config: RenderVideoConfig, source: SequenceSource) -> Path:
    return config.tracks_root / config.tracker_name / config.split / f"{source.name}.txt"


def _output_path(config: RenderVideoConfig, source: SequenceSource) -> Path:
    return config.output_root / config.tracker_name / config.split / f"{source.name}.mp4"


def _read_mot_tracks(path: Path, sequence_name: str) -> dict[int, list[TrackOutput]]:
    if not path.is_file():
        raise RenderVideoError(f"Missing MOT prediction file: {path}")
    by_frame: dict[int, list[TrackOutput]] = defaultdict(list)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 9:
            raise RenderVideoError(f"MOT row must have 9 fields at {path}:{line_number}")
        try:
            frame_index = int(float(fields[0]))
            track_id = int(float(fields[1]))
            left, top, width, height, confidence = [float(value) for value in fields[2:7]]
        except ValueError as exc:
            raise RenderVideoError(f"Invalid MOT row at {path}:{line_number}") from exc
        confidence_value = None if confidence < 0.0 else confidence
        bbox_xywh = BoundingBoxXYWH(left, top, width, height)
        by_frame[frame_index].append(
            TrackOutput.from_xyxy(
                frame_index=frame_index,
                sequence_name=sequence_name,
                track_id=track_id,
                bbox_xyxy=xywh_to_xyxy(bbox_xywh),
                confidence=confidence_value,
                class_id=0,
                class_name="player",
                confirmed=True,
                time_since_update=0,
                metadata={"bbox_source": "mot_prediction"},
            )
        )
    return by_frame


def render_videos(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_render_video_config(config_path, overrides=overrides)
    sources = discover_mot_sequences(
        config.mot_root,
        config.split,
        config.seqmap,
        max_sequences=config.max_sequences,
    )
    plan = [
        {
            "sequence": source.name,
            "frames": source.frame_count,
            "tracks": str(_tracks_path(config, source)),
            "video": str(_output_path(config, source)),
        }
        for source in sources
    ]
    if dry_run:
        return {
            "dry_run": True,
            "tracker": config.tracker_name,
            "sequence_count": len(sources),
            "sequences": plan,
        }
    rendered = [_render_sequence(config, source) for source in sources]
    return {
        "dry_run": False,
        "tracker": config.tracker_name,
        "sequence_count": len(rendered),
        "videos": rendered,
    }


def _render_sequence(config: RenderVideoConfig, source: SequenceSource) -> dict[str, Any]:
    tracks_by_frame = _read_mot_tracks(_tracks_path(config, source), source.name)
    output_path = _output_path(config, source)
    video_writer = create_tracking_video_writer(
        output_path,
        fps=source.fps,
        width=source.width,
        height=source.height,
        overwrite=config.overwrite,
    ).open()
    trajectory = TrajectoryStore(
        trajectory_length=config.trajectory_length,
        enabled=config.show_trajectory,
    )
    frame_count = 0
    started = time.perf_counter()
    try:
        for frame_item in iter_source_frames(source, max_frames=config.max_frames_per_sequence):
            tracks = tracks_by_frame.get(frame_item.frame_index, [])
            trajectory.update(tracks)
            rendered = draw_tracks(
                frame_item.image,
                tracks,
                trajectory_store=trajectory,
                show_confidence=config.show_confidence,
                show_class=config.show_class,
                show_track_id=config.show_track_id,
                show_trajectory=config.show_trajectory,
                show_fps=config.show_fps,
                fps=source.fps,
                frame_index=frame_item.frame_index if config.show_frame_number else None,
                sequence_name=source.name if config.show_sequence_name else None,
                tracker_name=config.tracker_name if config.show_tracker_name else None,
                line_thickness=config.line_thickness,
                font_scale=config.font_scale,
            )
            video_writer.write(rendered)
            frame_count += 1
    finally:
        video_writer.close()
    metadata = {
        "sequence": source.name,
        "tracker": config.tracker_name,
        "video": str(output_path),
        "fps": source.fps,
        "width": source.width,
        "height": source.height,
        "frame_count": frame_count,
        "seconds": time.perf_counter() - started,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {**metadata, "metadata": str(metadata_path)}
