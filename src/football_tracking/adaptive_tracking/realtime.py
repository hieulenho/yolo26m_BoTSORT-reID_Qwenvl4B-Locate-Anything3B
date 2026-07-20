"""Low-latency stream runner using a precomputed adaptive detector plan."""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from football_tracking.detection.detector import resolve_device
from football_tracking.detection.detector_factory import create_detector
from football_tracking.detection.serialization import runtime_versions
from football_tracking.tracking.checkpoint_resolver import resolve_detector_checkpoint
from football_tracking.tracking.mot_writer import format_mot_row
from football_tracking.tracking.pipeline import (
    all_detections_from_prediction,
    load_tracking_config,
    partition_tracking_detections,
    predict_tracking_frame,
)
from football_tracking.tracking.tracker_factory import create_tracker
from football_tracking.tracking.trajectory import TrajectoryStore
from football_tracking.visualization.draw_tracks import (
    draw_detection_overlays,
    draw_tracks,
)


class RealtimeTrackingError(RuntimeError):
    """Raised when an adaptive stream cannot be started or processed."""


def run_realtime_tracking(
    *,
    config_path: str | Path,
    stream_source: str | int,
    output_video: str | Path | None = None,
    output_mot: str | Path | None = None,
    metadata_path: str | Path | None = None,
    show_window: bool = True,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    metadata = _metadata_output_path(metadata_path, output_video)
    _validate_output_paths(output_video, output_mot, metadata, overwrite)
    config = load_tracking_config(config_path)
    config = _with_resolved_device(config)
    checkpoint = resolve_detector_checkpoint(config.model, config.project_root)
    detector_load_started = time.perf_counter()
    detector = create_detector(
        config.model,
        checkpoint.checkpoint,
        device=config.device,
        half=config.half,
    )
    detector.load_model()
    detector_load_seconds = time.perf_counter() - detector_load_started
    tracker_load_started = time.perf_counter()
    tracker = create_tracker(config.tracker_name, config.tracker_config, device=config.device)
    tracker.reset()
    tracker_load_seconds = time.perf_counter() - tracker_load_started

    source_value = _capture_source(stream_source)
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RealtimeTrackingError(f"Could not open stream source: {stream_source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = source_fps if source_fps > 1.0 else 30.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RealtimeTrackingError("Stream did not report valid frame dimensions.")
    writer = _open_writer(output_video, output_fps, width, height)
    mot_handle = _open_mot(output_mot)
    trajectory = TrajectoryStore(config.trajectory_length, enabled=config.show_trajectory)
    frame_times: list[float] = []
    rolling_times: deque[float] = deque(maxlen=30)
    unique_tracks: set[int] = set()
    detection_count = 0
    tracker_detection_count = 0
    detection_only_count = 0
    track_box_count = 0
    started = time.perf_counter()
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            frame_started = time.perf_counter()
            raw = predict_tracking_frame(detector, frame, config)
            all_detections = all_detections_from_prediction(
                raw,
                frame_index,
                "realtime",
                width,
                height,
                config,
                checkpoint.checkpoint_type,
            )
            detections, detection_only = partition_tracking_detections(
                all_detections,
                config.tracker_class_ids,
            )
            tracks = tracker.update(
                frame_index,
                "realtime",
                detections,
                frame,
                width,
                height,
            )
            trajectory.update(tracks)
            elapsed = time.perf_counter() - frame_started
            frame_times.append(elapsed)
            rolling_times.append(elapsed)
            display_fps = len(rolling_times) / max(sum(rolling_times), 1e-9)
            rendered = draw_tracks(
                frame,
                tracks,
                trajectory_store=trajectory,
                show_confidence=config.show_confidence,
                show_class=config.show_class,
                show_track_id=config.show_track_id,
                show_trajectory=config.show_trajectory,
                show_fps=True,
                fps=display_fps,
                frame_index=frame_index,
                sequence_name="live",
                tracker_name=config.tracker_name,
                line_thickness=config.line_thickness,
                font_scale=config.font_scale,
            )
            rendered = draw_detection_overlays(
                rendered,
                detection_only,
                show_confidence=config.show_confidence,
                line_thickness=config.line_thickness,
                font_scale=config.font_scale,
            )
            if writer is not None:
                writer.write(rendered)
            if mot_handle is not None:
                for track in tracks:
                    mot_handle.write(format_mot_row(track) + "\n")
                mot_handle.flush()
            if show_window:
                cv2.imshow("adaptive tracking", rendered)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
            detection_count += len(all_detections)
            tracker_detection_count += len(detections)
            detection_only_count += len(detection_only)
            track_box_count += len(tracks)
            unique_tracks.update(track.track_id for track in tracks)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if mot_handle is not None:
            mot_handle.close()
        tracker.close()
        if show_window:
            cv2.destroyAllWindows()
    total_seconds = time.perf_counter() - started
    result = {
        "mode": "realtime",
        "stream_source": str(stream_source),
        "config": str(Path(config_path).resolve()),
        "route": {
            "checkpoint": str(checkpoint.checkpoint),
            "checkpoint_type": checkpoint.checkpoint_type,
            "classes": config.source_class_names,
        },
        "detector": detector.metadata() if hasattr(detector, "metadata") else None,
        "tracker": config.tracker_name,
        "device": config.device,
        "hardware": runtime_versions(),
        "frames": frame_index,
        "detections": detection_count,
        "tracker_detections": tracker_detection_count,
        "detection_only_boxes": detection_only_count,
        "track_boxes": track_box_count,
        "unique_tracks": len(unique_tracks),
        "timing": {
            **_timing_summary(frame_times, total_seconds),
            "detector_load_seconds": detector_load_seconds,
            "tracker_load_seconds": tracker_load_seconds,
            "startup_seconds": detector_load_seconds + tracker_load_seconds,
        },
        "outputs": {
            "video": str(Path(output_video).resolve()) if output_video else None,
            "mot": str(Path(output_mot).resolve()) if output_mot else None,
        },
    }
    if metadata is not None:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        result["outputs"]["metadata"] = str(metadata.resolve())
    return result


def _metadata_output_path(
    metadata_path: str | Path | None,
    output_video: str | Path | None,
) -> Path | None:
    if metadata_path is not None:
        return Path(metadata_path)
    if output_video is not None:
        return Path(output_video).with_suffix(".realtime.json")
    return None


def _validate_output_paths(
    output_video: str | Path | None,
    output_mot: str | Path | None,
    metadata_path: Path | None,
    overwrite: bool,
) -> None:
    requested = [
        Path(path)
        for path in (output_video, output_mot, metadata_path)
        if path is not None
    ]
    existing = [path for path in requested if path.exists()]
    if existing and not overwrite:
        raise RealtimeTrackingError(
            "Realtime output exists and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )


def _with_resolved_device(config: Any) -> Any:
    from dataclasses import replace

    return replace(config, device=resolve_device(config.device))


def _capture_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def _open_writer(
    output_video: str | Path | None,
    fps: float,
    width: int,
    height: int,
) -> Any | None:
    if output_video is None:
        return None
    path = Path(output_video)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RealtimeTrackingError(f"Could not open output video: {path}")
    return writer


def _open_mot(output_mot: str | Path | None) -> Any | None:
    if output_mot is None:
        return None
    path = Path(output_mot)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", buffering=1)


def _timing_summary(frame_times: list[float], total_seconds: float) -> dict[str, Any]:
    if not frame_times:
        return {
            "total_seconds": total_seconds,
            "end_to_end_fps": 0.0,
            "processing_fps": 0.0,
            "steady_state_processing_fps": 0.0,
            "warmup_frame_count": 0,
            "frame_latency_ms_mean": None,
            "frame_latency_ms_p95": None,
        }
    ordered = sorted(frame_times)
    p95_index = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
    warmup_count = min(5, max(len(frame_times) - 1, 0))
    steady_times = frame_times[warmup_count:]
    return {
        "total_seconds": total_seconds,
        "end_to_end_fps": len(frame_times) / total_seconds if total_seconds > 0 else 0.0,
        "processing_fps": len(frame_times) / max(sum(frame_times), 1e-9),
        "steady_state_processing_fps": (
            len(steady_times) / max(sum(steady_times), 1e-9) if steady_times else 0.0
        ),
        "warmup_frame_count": warmup_count,
        "frame_latency_ms_mean": statistics.fmean(frame_times) * 1000.0,
        "frame_latency_ms_median": statistics.median(frame_times) * 1000.0,
        "frame_latency_ms_p95": ordered[p95_index] * 1000.0,
    }
