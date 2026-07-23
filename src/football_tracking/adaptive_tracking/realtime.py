"""Low-latency stream runner using a precomputed adaptive detector plan."""

from __future__ import annotations

import json
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticCacheView,
    SemanticEventQueue,
)
from football_tracking.adaptive_tracking.shot_sampling import OnlineShotChangeDetector
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
    semantic_queue_dir: str | Path | None = None,
    semantic_cache_path: str | Path | None = None,
    semantic_event_interval_frames: int = 90,
    semantic_cache_reload_frames: int = 15,
    semantic_events_per_frame: int = 2,
    semantic_max_pending_events: int = 256,
    asynchronous_video_write: bool = True,
    video_write_queue_size: int = 128,
    reset_on_scene_cut: bool = True,
    scene_cut_threshold: float = 0.65,
    scene_cut_min_gap_frames: int = 15,
    scene_cut_check_interval_frames: int = 5,
    prewarm_detector: bool = True,
    drop_late_frames: bool = True,
    max_catchup_frames: int = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    if semantic_event_interval_frames < 1:
        raise RealtimeTrackingError("semantic_event_interval_frames must be positive.")
    if semantic_cache_reload_frames < 1:
        raise RealtimeTrackingError("semantic_cache_reload_frames must be positive.")
    if semantic_events_per_frame < 0:
        raise RealtimeTrackingError("semantic_events_per_frame must be non-negative.")
    if semantic_max_pending_events < 1:
        raise RealtimeTrackingError("semantic_max_pending_events must be positive.")
    if video_write_queue_size < 1:
        raise RealtimeTrackingError("video_write_queue_size must be positive.")
    if max_catchup_frames < 1:
        raise RealtimeTrackingError("max_catchup_frames must be positive.")
    metadata = _metadata_output_path(metadata_path, output_video)
    _validate_output_paths(output_video, output_mot, metadata, overwrite)
    _reset_peak_cuda_memory()
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
    latest_reader = (
        _LatestFrameReader(capture)
        if drop_late_frames and _is_live_source(source_value)
        else None
    )
    if latest_reader is not None:
        latest_reader.start()
    prefetched_frame = None
    prefetched_live_sequence: int | None = None
    last_live_sequence = 0
    detector_warmup_seconds = 0.0
    if prewarm_detector:
        if latest_reader is not None:
            ok, warmup_sequence, prefetched_frame = latest_reader.read_after(
                0,
                timeout_seconds=5.0,
            )
        else:
            ok, prefetched_frame = capture.read()
            warmup_sequence = 0
        if not ok:
            _release_capture(capture, latest_reader)
            raise RealtimeTrackingError("Could not read a frame for detector warm-up.")
        warmup_started = time.perf_counter()
        predict_tracking_frame(detector, prefetched_frame, config)
        detector_warmup_seconds = time.perf_counter() - warmup_started
        if latest_reader is not None:
            prefetched_live_sequence, prefetched_frame = latest_reader.latest()
            if prefetched_frame is None:
                prefetched_live_sequence = warmup_sequence
            last_live_sequence = max(int(prefetched_live_sequence or 1) - 1, 0)
    writer = _open_writer(
        output_video,
        output_fps,
        width,
        height,
        asynchronous=asynchronous_video_write,
        queue_size=video_write_queue_size,
    )
    mot_handle = _open_mot(output_mot)
    trajectory = TrajectoryStore(config.trajectory_length, enabled=config.show_trajectory)
    frame_times: list[float] = []
    detector_times: list[float] = []
    tracker_times: list[float] = []
    semantic_times: list[float] = []
    render_times: list[float] = []
    rolling_times: deque[float] = deque(maxlen=30)
    unique_tracks: set[int] = set()
    detection_count = 0
    tracker_detection_count = 0
    detection_only_count = 0
    track_box_count = 0
    semantic_events_enqueued = 0
    semantic_cache_refreshes = 0
    semantic_cache = SemanticCacheView(semantic_cache_path)
    semantic_queue = (
        SemanticEventQueue(
            semantic_queue_dir,
            context_id=f"{Path(config_path).resolve()}::{stream_source}",
            max_pending_events=semantic_max_pending_events,
        )
        if semantic_queue_dir is not None
        else None
    )
    started = time.perf_counter()
    process_monitor = _ProcessResourceMonitor()
    frame_index = 0
    processed_frame_count = 0
    dropped_late_frames = 0
    shot_index = 0
    scene_cuts: list[dict[str, Any]] = []
    shot_detector = (
        OnlineShotChangeDetector(
            threshold=scene_cut_threshold,
            min_gap_frames=scene_cut_min_gap_frames,
            check_interval_frames=scene_cut_check_interval_frames,
        )
        if reset_on_scene_cut
        else None
    )
    tracker_diagnostics: dict[str, Any] | None = None
    stream_finished = started
    shutdown_seconds = 0.0
    try:
        while max_frames is None or processed_frame_count < max_frames:
            if prefetched_frame is not None:
                frame = prefetched_frame
                prefetched_frame = None
                live_sequence = prefetched_live_sequence
                prefetched_live_sequence = None
            elif latest_reader is not None:
                ok, live_sequence, frame = latest_reader.read_after(
                    last_live_sequence,
                    timeout_seconds=5.0,
                )
                if not ok:
                    break
            else:
                live_sequence = None
                if drop_late_frames and source_fps > 0:
                    expected_index = int(
                        (time.perf_counter() - started) * source_fps
                    ) + 1
                    catchup = min(
                        max(expected_index - (frame_index + 1), 0),
                        max_catchup_frames,
                    )
                    for _ in range(catchup):
                        if not capture.grab():
                            break
                        frame_index += 1
                        dropped_late_frames += 1
                ok, frame = capture.read()
                if not ok:
                    break
            if live_sequence is not None:
                skipped = _dropped_between(last_live_sequence, live_sequence)
                dropped_late_frames += skipped
                frame_index += skipped
                last_live_sequence = live_sequence
            frame_index += 1
            processed_frame_count += 1
            frame_started = time.perf_counter()
            if shot_detector is not None:
                scene_cut, transition_score = shot_detector.update(frame_index, frame)
                if scene_cut:
                    if hasattr(tracker, "reset_scene"):
                        tracker.reset_scene()
                    else:
                        tracker.reset()
                    trajectory = TrajectoryStore(
                        config.trajectory_length,
                        enabled=config.show_trajectory,
                    )
                    shot_index += 1
                    scene_cuts.append(
                        {
                            "frame_index": frame_index,
                            "transition_score": round(transition_score, 6),
                        }
                    )
            detector_started = time.perf_counter()
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
            detector_times.append(time.perf_counter() - detector_started)
            tracker_started = time.perf_counter()
            tracks = tracker.update(
                frame_index,
                "realtime",
                detections,
                frame,
                width,
                height,
            )
            tracks = [
                replace(
                    track,
                    metadata={**track.metadata, "shot_index": shot_index},
                )
                for track in tracks
            ]
            tracker_times.append(time.perf_counter() - tracker_started)
            semantic_started = time.perf_counter()
            if frame_index == 1 or frame_index % semantic_cache_reload_frames == 0:
                semantic_cache_refreshes += int(semantic_cache.refresh())
            if semantic_queue is not None and semantic_events_per_frame:
                ranked_tracks = sorted(
                    tracks,
                    key=lambda track: (
                        semantic_cache.accepted(track.track_id) is not None,
                        -(track.confidence or 0.0),
                        track.track_id,
                    ),
                )
                emitted_this_frame = 0
                for track in ranked_tracks:
                    if emitted_this_frame >= semantic_events_per_frame:
                        break
                    event = semantic_queue.enqueue(
                        frame=frame,
                        frame_index=frame_index,
                        track=track,
                        reason=(
                            "periodic_refresh"
                            if semantic_cache.accepted(track.track_id)
                            else "unknown_track"
                        ),
                        minimum_frame_gap=semantic_event_interval_frames,
                    )
                    if event is not None:
                        semantic_events_enqueued += 1
                        emitted_this_frame += 1
            semantic_times.append(time.perf_counter() - semantic_started)
            trajectory.update(tracks)
            partial_elapsed = time.perf_counter() - frame_started
            display_fps = (len(rolling_times) + 1) / max(
                sum(rolling_times) + partial_elapsed,
                1e-9,
            )
            render_started = time.perf_counter()
            rendered = draw_tracks(
                frame,
                semantic_cache.decorate(tracks),
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
            render_times.append(time.perf_counter() - render_started)
            if writer is not None:
                writer.write(rendered)
            if mot_handle is not None:
                for track in tracks:
                    mot_handle.write(format_mot_row(track) + "\n")
                mot_handle.flush()
            elapsed = time.perf_counter() - frame_started
            frame_times.append(elapsed)
            rolling_times.append(elapsed)
            detection_count += len(all_detections)
            tracker_detection_count += len(detections)
            detection_only_count += len(detection_only)
            track_box_count += len(tracks)
            unique_tracks.update(track.track_id for track in tracks)
            process_monitor.sample()
            if show_window:
                cv2.imshow("adaptive tracking", rendered)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
    finally:
        stream_finished = time.perf_counter()
        _release_capture(capture, latest_reader)
        if writer is not None:
            writer.release()
        if mot_handle is not None:
            mot_handle.close()
        tracker_diagnostics = (
            tracker.get_diagnostics() if hasattr(tracker, "get_diagnostics") else None
        )
        tracker.close()
        if show_window:
            cv2.destroyAllWindows()
        shutdown_seconds = time.perf_counter() - stream_finished
    total_seconds = stream_finished - started
    result = {
        "mode": "realtime",
        "stream_source": str(stream_source),
        "capture_mode": "latest_frame" if latest_reader is not None else "sequential",
        "config": str(Path(config_path).resolve()),
        "route": {
            "checkpoint": str(checkpoint.checkpoint),
            "checkpoint_type": checkpoint.checkpoint_type,
            "classes": config.source_class_names,
        },
        "detector": detector.metadata() if hasattr(detector, "metadata") else None,
        "tracker": config.tracker_name,
        "tracker_diagnostics": tracker_diagnostics,
        "device": config.device,
        "hardware": runtime_versions(),
        "resources": process_monitor.summary(total_seconds),
        "cuda_memory": _peak_cuda_memory(),
        "frames": processed_frame_count,
        "source_frames_consumed": processed_frame_count + dropped_late_frames,
        "dropped_late_frames": dropped_late_frames,
        "late_frame_drop_rate": (
            dropped_late_frames / max(processed_frame_count + dropped_late_frames, 1)
        ),
        "detections": detection_count,
        "tracker_detections": tracker_detection_count,
        "detection_only_boxes": detection_only_count,
        "track_boxes": track_box_count,
        "unique_tracks": len(unique_tracks),
        "semantic": {
            "queue_dir": (
                str(Path(semantic_queue_dir).resolve())
                if semantic_queue_dir is not None
                else None
            ),
            "cache_path": (
                str(Path(semantic_cache_path).resolve())
                if semantic_cache_path is not None
                else None
            ),
            "events_enqueued": semantic_events_enqueued,
            "events_dropped_queue_full": (
                semantic_queue.dropped_full if semantic_queue is not None else 0
            ),
            "pending_events": (
                semantic_queue.pending_count if semantic_queue is not None else 0
            ),
            "max_pending_events": semantic_max_pending_events,
            "cache_refreshes": semantic_cache_refreshes,
            "accepted_cached_tracks": sum(
                semantic_cache.accepted(track_id) is not None
                for track_id in unique_tracks
            ),
            "non_blocking": True,
        },
        "scene_cuts": {
            "enabled": reset_on_scene_cut,
            "threshold": scene_cut_threshold,
            "minimum_gap_frames": scene_cut_min_gap_frames,
            "check_interval_frames": scene_cut_check_interval_frames,
            "count": len(scene_cuts),
            "events": scene_cuts,
        },
        "timing": {
            **_timing_summary(
                frame_times,
                total_seconds,
                source_fps=source_fps,
                source_frames_consumed=processed_frame_count + dropped_late_frames,
            ),
            "detector_seconds": sum(detector_times),
            "tracker_seconds": sum(tracker_times),
            "semantic_queue_seconds": sum(semantic_times),
            "render_seconds": sum(render_times),
            "detector_fps": _stage_fps(processed_frame_count, detector_times),
            "tracker_fps": _stage_fps(processed_frame_count, tracker_times),
            "semantic_queue_fps": _stage_fps(processed_frame_count, semantic_times),
            "render_fps": _stage_fps(processed_frame_count, render_times),
            "detector_load_seconds": detector_load_seconds,
            "detector_warmup_seconds": detector_warmup_seconds,
            "tracker_load_seconds": tracker_load_seconds,
            "startup_seconds": (
                detector_load_seconds + tracker_load_seconds + detector_warmup_seconds
            ),
            "shutdown_seconds": shutdown_seconds,
            "prewarm_detector": prewarm_detector,
            "drop_late_frames": drop_late_frames,
            "max_catchup_frames": max_catchup_frames,
        },
        "outputs": {
            "video": str(Path(output_video).resolve()) if output_video else None,
            "mot": str(Path(output_mot).resolve()) if output_mot else None,
            "video_writer": (
                writer.summary()
                if writer is not None and hasattr(writer, "summary")
                else {"mode": "synchronous"} if writer is not None else None
            ),
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


def _is_live_source(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    return str(source).strip().lower().startswith(("rtsp://", "rtsps://"))


def _dropped_between(previous_sequence: int, current_sequence: int) -> int:
    return max(int(current_sequence) - int(previous_sequence) - 1, 0)


def _release_capture(capture: Any, latest_reader: _LatestFrameReader | None) -> None:
    if latest_reader is not None:
        latest_reader.release()
    else:
        capture.release()


class _LatestFrameReader:
    """Continuously retain the newest physical-stream frame and sequence number."""

    def __init__(self, capture: Any) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._sequence = 0
        self._frame: Any | None = None
        self._ended = False
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                with self._condition:
                    if not ok:
                        self._ended = True
                        self._condition.notify_all()
                        return
                    self._sequence += 1
                    self._frame = frame
                    self._condition.notify_all()
        except BaseException as exc:  # pragma: no cover - OpenCV backend failure
            with self._condition:
                self._error = exc
                self._ended = True
                self._condition.notify_all()

    def read_after(
        self,
        sequence: int,
        *,
        timeout_seconds: float,
    ) -> tuple[bool, int, Any | None]:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._sequence <= sequence and not self._ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise RealtimeTrackingError(
                    f"Physical stream reader failed: {self._error}"
                )
            if self._sequence <= sequence or self._frame is None:
                return False, self._sequence, None
            return True, self._sequence, self._frame

    def latest(self) -> tuple[int, Any | None]:
        with self._condition:
            return self._sequence, self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


def _open_writer(
    output_video: str | Path | None,
    fps: float,
    width: int,
    height: int,
    *,
    asynchronous: bool,
    queue_size: int,
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
    return _AsyncVideoWriter(writer, queue_size=queue_size) if asynchronous else writer


class _AsyncVideoWriter:
    """Move CPU video encoding off the latency-sensitive tracking loop."""

    def __init__(self, writer: Any, *, queue_size: int) -> None:
        self._writer = writer
        self._queue: queue.Queue[Any | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._error: BaseException | None = None
        self.written_frames = 0
        self.dropped_frames = 0
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                try:
                    if frame is None:
                        return
                    self._writer.write(frame)
                    self.written_frames += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # pragma: no cover - OpenCV backend failure
            self._error = exc

    def write(self, frame: Any) -> None:
        if self._error is not None:
            raise RealtimeTrackingError(f"Video writer failed: {self._error}")
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def release(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._writer.release()
        if self._error is not None:
            raise RealtimeTrackingError(f"Video writer failed: {self._error}")

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "asynchronous",
            "written_frames": self.written_frames,
            "dropped_frames": self.dropped_frames,
            "queue_capacity": self._queue.maxsize,
        }


def _open_mot(output_mot: str | Path | None) -> Any | None:
    if output_mot is None:
        return None
    path = Path(output_mot)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", buffering=1)


def _timing_summary(
    frame_times: list[float],
    total_seconds: float,
    *,
    source_fps: float = 0.0,
    source_frames_consumed: int = 0,
) -> dict[str, Any]:
    if not frame_times:
        return {
            "total_seconds": total_seconds,
            "end_to_end_fps": 0.0,
            "processing_fps": 0.0,
            "steady_state_processing_fps": 0.0,
            "warmup_frame_count": 0,
            "frame_latency_ms_mean": None,
            "frame_latency_ms_p95": None,
            "source_fps": source_fps,
            "realtime_factor": None,
            "keeps_up_with_source": None,
            "source_progress_fps": 0.0,
        }
    ordered = sorted(frame_times)
    warmup_count = min(5, max(len(frame_times) - 1, 0))
    steady_times = frame_times[warmup_count:]
    processing_fps = len(frame_times) / max(sum(frame_times), 1e-9)
    return {
        "total_seconds": total_seconds,
        "end_to_end_fps": len(frame_times) / total_seconds if total_seconds > 0 else 0.0,
        "processing_fps": processing_fps,
        "steady_state_processing_fps": (
            len(steady_times) / max(sum(steady_times), 1e-9) if steady_times else 0.0
        ),
        "warmup_frame_count": warmup_count,
        "frame_latency_ms_mean": statistics.fmean(frame_times) * 1000.0,
        "frame_latency_ms_median": statistics.median(frame_times) * 1000.0,
        "frame_latency_ms_p50": _percentile(ordered, 0.50) * 1000.0,
        "frame_latency_ms_p90": _percentile(ordered, 0.90) * 1000.0,
        "frame_latency_ms_p95": _percentile(ordered, 0.95) * 1000.0,
        "frame_latency_ms_p99": _percentile(ordered, 0.99) * 1000.0,
        "source_fps": source_fps,
        "realtime_factor": (
            processing_fps / source_fps if source_fps > 0 else None
        ),
        "keeps_up_with_source": (
            processing_fps >= source_fps if source_fps > 0 else None
        ),
        "source_progress_fps": source_frames_consumed / max(total_seconds, 1e-9),
        "latency_bounded_by_frame_dropping": source_frames_consumed > len(frame_times),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    index = min(int(round(quantile * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


class _ProcessResourceMonitor:
    def __init__(self) -> None:
        self._rss_peak = 0
        self._cpu_start: float | None = None
        self._logical_cpus: int | None = None
        self._process: Any | None = None
        try:
            import psutil  # type: ignore[import-not-found]

            self._process = psutil.Process()
            cpu_times = self._process.cpu_times()
            self._cpu_start = float(cpu_times.user + cpu_times.system)
            self._logical_cpus = psutil.cpu_count(logical=True)
        except (ImportError, OSError):
            pass

    def sample(self) -> None:
        if self._process is None:
            return
        try:
            self._rss_peak = max(
                self._rss_peak,
                int(self._process.memory_info().rss),
            )
        except OSError:
            return

    def summary(self, wall_seconds: float) -> dict[str, Any]:
        process_cpu_percent = None
        if self._process is not None and self._cpu_start is not None:
            try:
                cpu_times = self._process.cpu_times()
                used = float(cpu_times.user + cpu_times.system) - self._cpu_start
                logical_cpus = max(int(self._logical_cpus or 1), 1)
                process_cpu_percent = 100.0 * used / max(wall_seconds * logical_cpus, 1e-9)
            except OSError:
                pass
        return {
            "peak_process_rss_bytes": self._rss_peak or None,
            "process_cpu_percent_of_system": process_cpu_percent,
            "sampling_scope": "tracking_process",
        }


def _stage_fps(frame_count: int, times: list[float]) -> float:
    return frame_count / max(sum(times), 1e-9) if frame_count else 0.0


def _reset_peak_cuda_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _peak_cuda_memory() -> dict[str, int | None]:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except (ImportError, RuntimeError):
        pass
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
