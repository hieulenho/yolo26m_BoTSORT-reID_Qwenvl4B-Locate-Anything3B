"""Detector plus multi-object tracker orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.detector import resolve_device
from football_tracking.detection.detector_factory import create_detector
from football_tracking.detection.postprocessing import postprocess_detections
from football_tracking.detection.serialization import runtime_versions
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.reporting.tracking_run_report import write_tracking_run_report
from football_tracking.tracking.checkpoint_resolver import (
    ResolvedCheckpoint,
    resolve_detector_checkpoint,
)
from football_tracking.tracking.mot_writer import MotPredictionWriter
from football_tracking.tracking.schemas import TrackerDetection, TrackingRunSummary
from football_tracking.tracking.sequence_runner import (
    SequenceSource,
    discover_mot_sequences,
    iter_source_frames,
    video_source,
)
from football_tracking.tracking.timing import TrackingTiming, timed_section
from football_tracking.tracking.tracker_factory import (
    create_tracker,
    load_tracker_config_object,
)
from football_tracking.tracking.trajectory import TrajectoryStore
from football_tracking.tracking.validation import (
    TrackValidationReport,
    validate_mot_prediction_file,
    write_track_validation_report,
)
from football_tracking.visualization.draw_tracks import draw_tracks
from football_tracking.visualization.tracking_video import create_tracking_video_writer


class TrackingPipelineError(RuntimeError):
    """Raised when tracking cannot run safely."""


@dataclass(frozen=True)
class TrackingConfig:
    project_root: Path
    config_path: Path
    model: dict[str, Any]
    tracker_name: str
    tracker_config: Path
    source_type: str
    source_path: Path | None
    mot_root: Path | None
    split: str | None
    seqmap: Path | None
    output_video: Path | None
    output_mot: Path | None
    output_metadata: Path | None
    tracks_dir: Path
    videos_dir: Path
    metrics_dir: Path
    render_video: bool
    save_mot: bool
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str
    half: bool
    class_ids: tuple[int, ...] | None
    target_class_id: int
    target_class_name: str
    preserve_source_classes: bool
    source_class_names: dict[int, str]
    show_confidence: bool
    show_class: bool
    show_track_id: bool
    show_trajectory: bool
    trajectory_length: int
    line_thickness: int
    font_scale: float
    show_fps: bool
    max_sequences: int | None
    max_frames_per_sequence: int | None
    start_frame: int
    overwrite: bool
    show_window: bool
    fail_fast: bool
    smoke_only: bool


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrackingPipelineError(f"{section} must be a mapping.")
    return value


def _resolve_path(
    value: Any,
    project_root: Path,
    section: str,
    required: bool = True,
) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str | Path) or not str(value).strip():
        raise TrackingPipelineError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_tracking_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> TrackingConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = (
        path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    )
    if not resolved_config.is_file():
        raise TrackingPipelineError(f"Tracking config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    root = _mapping(raw, "tracking config root")
    model = dict(_mapping(root.get("model"), "model"))
    detector = _mapping(root.get("detector"), "detector")
    tracker = _mapping(root.get("tracker"), "tracker")
    output = _mapping(root.get("output"), "output")
    render = _mapping(root.get("render", {}), "render")
    runtime = _mapping(root.get("runtime", {}), "runtime")

    dataset = root.get("dataset")
    source = root.get("source")
    if dataset is not None:
        dataset_map = _mapping(dataset, "dataset")
        source_type = "sportsmot"
        source_path = None
        mot_root = _resolve_path(dataset_map.get("mot_root"), project_root, "dataset.mot_root")
        split = str(dataset_map.get("split", "val"))
        seqmap = _resolve_path(dataset_map.get("seqmap"), project_root, "dataset.seqmap")
    elif source is not None:
        source_map = _mapping(source, "source")
        source_type = str(source_map.get("type", "video"))
        source_path = _resolve_path(source_map.get("path"), project_root, "source.path")
        mot_root = None
        split = None
        seqmap = None
    else:
        raise TrackingPipelineError("Tracking config must define dataset or source.")

    render_video = bool(output.get("render_video", render.get("enabled", False)))
    class_values = detector.get("class_ids", [0])
    class_ids = None if class_values is None else tuple(int(value) for value in class_values)
    config = TrackingConfig(
        project_root=project_root,
        config_path=resolved_config,
        model=model,
        tracker_name=str(tracker.get("name", "deepsort")),
        tracker_config=_resolve_path(tracker.get("config"), project_root, "tracker.config"),
        source_type=source_type,
        source_path=source_path,
        mot_root=mot_root,
        split=split,
        seqmap=seqmap,
        output_video=_resolve_path(
            output.get("video"),
            project_root,
            "output.video",
            required=False,
        ),
        output_mot=_resolve_path(output.get("mot"), project_root, "output.mot", required=False),
        output_metadata=_resolve_path(
            output.get("metadata"),
            project_root,
            "output.metadata",
            required=False,
        ),
        tracks_dir=_resolve_path(
            output.get("tracks_dir", "outputs/tracks/deepsort"),
            project_root,
            "output.tracks_dir",
        ),
        videos_dir=_resolve_path(
            output.get("videos_dir", "outputs/videos/deepsort"),
            project_root,
            "output.videos_dir",
        ),
        metrics_dir=_resolve_path(
            output.get("metrics_dir", "outputs/metrics"),
            project_root,
            "output.metrics_dir",
        ),
        render_video=render_video,
        save_mot=bool(output.get("save_mot", runtime.get("save_mot", True))),
        imgsz=int(detector.get("imgsz", 960)),
        conf=float(detector.get("conf", 0.15)),
        iou=float(detector.get("iou", 0.70)),
        max_det=int(detector.get("max_det", 300)),
        device=str(detector.get("device", "auto")),
        half=bool(detector.get("half", False)),
        class_ids=class_ids,
        target_class_id=int(detector.get("target_class_id", 0)),
        target_class_name=str(detector.get("target_class_name", "player")),
        preserve_source_classes=bool(detector.get("preserve_source_classes", False)),
        source_class_names={
            int(key): str(value)
            for key, value in (detector.get("source_class_names", {}) or {}).items()
        },
        show_confidence=bool(render.get("show_confidence", True)),
        show_class=bool(render.get("show_class", True)),
        show_track_id=bool(render.get("show_track_id", True)),
        show_trajectory=bool(render.get("show_trajectory", True)),
        trajectory_length=int(render.get("trajectory_length", 30)),
        line_thickness=int(render.get("line_thickness", 2)),
        font_scale=float(render.get("font_scale", 0.6)),
        show_fps=bool(render.get("show_fps", True)),
        max_sequences=runtime.get("max_sequences"),
        max_frames_per_sequence=runtime.get("max_frames_per_sequence", runtime.get("max_frames")),
        start_frame=int(runtime.get("start_frame", 1)),
        overwrite=bool(runtime.get("overwrite", False)),
        show_window=bool(runtime.get("show_window", False)),
        fail_fast=bool(runtime.get("fail_fast", True)),
        smoke_only=bool(runtime.get("smoke_only", False)),
    )
    if config.max_sequences is not None:
        config = replace(config, max_sequences=int(config.max_sequences))
    if config.max_frames_per_sequence is not None:
        config = replace(config, max_frames_per_sequence=int(config.max_frames_per_sequence))
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_tracking_config(config)
    return config


def _apply_overrides(config: TrackingConfig, overrides: dict[str, Any]) -> TrackingConfig:
    changes: dict[str, Any] = {}
    if overrides.get("source") is not None:
        source_path = _resolve_path(overrides["source"], config.project_root, "--source")
        changes["source_path"] = source_path
        changes["source_type"] = "video"
        output_mot, output_metadata, output_video = _default_video_output_paths(source_path)
        changes["output_mot"] = output_mot
        changes["output_metadata"] = output_metadata
        changes["output_video"] = output_video
    if overrides.get("output_video") is not None:
        output_video = _resolve_path(
            overrides["output_video"],
            config.project_root,
            "--output-video",
        )
        output_mot, output_metadata, output_video = _video_sidecar_output_paths(output_video)
        changes["output_mot"] = output_mot
        changes["output_metadata"] = output_metadata
        changes["output_video"] = output_video
    if overrides.get("checkpoint") is not None:
        model = dict(config.model)
        model["checkpoint"] = str(overrides["checkpoint"])
        changes["model"] = model
    for key in ("device", "conf", "imgsz", "max_sequences", "overwrite", "show_window"):
        if overrides.get(key) is not None:
            changes[key] = overrides[key]
    if overrides.get("max_frames") is not None:
        changes["max_frames_per_sequence"] = overrides["max_frames"]
    if overrides.get("render") is True:
        changes["render_video"] = True
    if overrides.get("no_render") is True:
        changes["render_video"] = False
    if overrides.get("save_mot") is not None:
        changes["save_mot"] = bool(overrides["save_mot"])
    return replace(config, **changes) if changes else config


def _default_video_output_paths(source_path: Path) -> tuple[Path, Path, Path]:
    output_stem = f"{source_path.stem}_tracked"
    return _video_sidecar_output_paths(source_path.with_name(f"{output_stem}.mp4"))


def _video_sidecar_output_paths(output_video: Path) -> tuple[Path, Path, Path]:
    return (
        output_video.with_suffix(".txt"),
        output_video.with_name(f"{output_video.stem}.metadata.json"),
        output_video,
    )


def _validate_tracking_config(config: TrackingConfig) -> None:
    if config.source_type not in {"sportsmot", "video"}:
        raise TrackingPipelineError(f"Unsupported source type: {config.source_type}")
    if config.source_type == "video" and config.source_path is None:
        raise TrackingPipelineError("source.path is required for video tracking.")
    if config.source_type == "sportsmot" and (config.mot_root is None or config.split is None):
        raise TrackingPipelineError("dataset.mot_root and dataset.split are required.")
    if not 0.0 <= config.conf <= 1.0:
        raise TrackingPipelineError("detector.conf must be in [0, 1].")
    if not 0.0 <= config.iou <= 1.0:
        raise TrackingPipelineError("detector.iou must be in [0, 1].")
    if config.imgsz <= 0 or config.max_det <= 0:
        raise TrackingPipelineError("detector.imgsz and detector.max_det must be positive.")
    if config.max_sequences is not None and config.max_sequences <= 0:
        raise TrackingPipelineError("runtime.max_sequences must be positive when set.")
    if config.max_frames_per_sequence is not None and config.max_frames_per_sequence <= 0:
        raise TrackingPipelineError("runtime.max_frames_per_sequence must be positive when set.")


def _discover_sources(config: TrackingConfig) -> list[SequenceSource]:
    if config.source_type == "video":
        return [video_source(config.source_path)]
    return discover_mot_sequences(
        config.mot_root,
        config.split,
        config.seqmap,
        max_sequences=config.max_sequences,
    )


def _mot_output_paths(
    config: TrackingConfig,
    source: SequenceSource,
) -> tuple[Path | None, Path | None, Path | None]:
    if config.source_type == "video":
        return config.output_mot, config.output_metadata, config.output_video
    split = config.split or "unknown"
    mot_path = config.tracks_dir / split / f"{source.name}.txt"
    metadata_path = config.tracks_dir / split / f"{source.name}.metadata.json"
    video_path = config.videos_dir / split / f"{source.name}.mp4"
    return mot_path, metadata_path, video_path


def _dry_run_plan(
    config: TrackingConfig,
    checkpoint: ResolvedCheckpoint,
    sources: list[SequenceSource],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "checkpoint": checkpoint.to_dict(),
        "source_type": config.source_type,
        "split": config.split,
        "sequence_count": len(sources),
        "sequences": [
            {
                "name": source.name,
                "frame_count": source.frame_count,
                "fps": source.fps,
                "width": source.width,
                "height": source.height,
                "output_mot": str(_mot_output_paths(config, source)[0])
                if _mot_output_paths(config, source)[0]
                else None,
                "output_video": str(_mot_output_paths(config, source)[2])
                if _mot_output_paths(config, source)[2]
                else None,
            }
            for source in sources
        ],
        "render_video": config.render_video,
        "save_mot": config.save_mot,
        "action": "validated config, checkpoint, and sequence list; inference was not run",
    }


def _tracker_detections_from_raw(
    raw_prediction: Any,
    frame_index: int,
    sequence_name: str,
    image_width: int,
    image_height: int,
    config: TrackingConfig,
    checkpoint_type: str,
    image_path: Path | None,
) -> list[TrackerDetection]:
    keep_only_person = checkpoint_type == "pretrained_coco"
    detections = postprocess_detections(
        raw_prediction,
        frame_index=frame_index,
        sequence_name=sequence_name,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=config.conf,
        coco_person_class_id=0,
        target_class_id=config.target_class_id,
        target_class_name=config.target_class_name,
        keep_only_person=keep_only_person,
        allowed_class_ids=config.class_ids,
        source_class_names=config.source_class_names,
        preserve_source_class=config.preserve_source_classes,
        image_path=image_path,
    )
    return [
        TrackerDetection.from_xyxy(
            frame_index=detection.frame_index,
            sequence_name=detection.sequence_name,
            bbox_xyxy=detection.bbox_xyxy,
            confidence=detection.confidence,
            class_id=detection.target_class_id,
            class_name=detection.target_class_name,
            metadata=detection.metadata,
        )
        for detection in detections
    ]


def _predict_frame(detector: Any, frame: Any, config: TrackingConfig) -> Any:
    if hasattr(detector, "predict_frame"):
        return detector.predict_frame(
            frame,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            max_det=config.max_det,
            device=config.device,
        )
    model = detector.load_model()
    predict_kwargs = {
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "max_det": config.max_det,
        "device": getattr(detector, "device", config.device),
        "verbose": False,
    }
    if getattr(detector, "half", config.half):
        predict_kwargs["half"] = True
    raw = model(frame, **predict_kwargs)
    if isinstance(raw, list | tuple):
        return raw[0] if raw else None
    return raw


def _load_detector_once(
    config: TrackingConfig,
    checkpoint: ResolvedCheckpoint,
    detector: Any | None,
) -> Any:
    if detector is not None:
        return detector
    resolved_device = resolve_device(config.device)
    return create_detector(
        config.model,
        checkpoint.checkpoint,
        device=resolved_device,
        half=config.half,
    )


def run_tracking(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    detector: Any | None = None,
    tracker_adapter_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    config = load_tracking_config(config_path, overrides=overrides)
    config = replace(config, device=resolve_device(config.device))
    checkpoint = resolve_detector_checkpoint(config.model, config.project_root)
    sources = _discover_sources(config)
    if dry_run:
        return _dry_run_plan(config, checkpoint, sources)

    if not sources:
        raise TrackingPipelineError("No tracking sources were found.")
    tracker_runtime_config = load_tracker_config_object(
        config.tracker_name,
        config.tracker_config,
        device=config.device,
    )
    started_at = datetime.now(UTC).isoformat()
    global_timing = TrackingTiming()
    load_started = time.perf_counter()
    detector = _load_detector_once(config, checkpoint, detector)
    if hasattr(detector, "load_model"):
        detector.load_model()
    global_timing.model_load_seconds = time.perf_counter() - load_started

    sequence_summaries: list[dict[str, Any]] = []
    all_warnings = list(checkpoint.warnings)
    all_errors: list[str] = []
    validation_report = TrackValidationReport()
    for source in sources:
        try:
            summary, sequence_validation = _run_sequence(
                config,
                source,
                checkpoint,
                tracker_runtime_config,
                detector,
                tracker_adapter_factory,
            )
            sequence_summaries.append(summary.to_dict())
            validation_report.extend(sequence_validation)
            all_warnings.extend(summary.warnings)
            all_errors.extend(summary.errors)
            global_timing.processed_frames += summary.frame_count
            global_timing.detector_seconds += summary.detector_seconds
            global_timing.tracker_seconds += summary.tracker_seconds
            global_timing.rendering_seconds += summary.rendering_seconds
            global_timing.video_write_seconds += summary.video_write_seconds
            global_timing.total_pipeline_seconds += summary.total_seconds
        except Exception as exc:  # noqa: BLE001
            message = f"{source.name}: {exc}"
            all_errors.append(message)
            if config.fail_fast:
                raise TrackingPipelineError(message) from exc
    runtime = runtime_versions()
    try:
        import deep_sort_realtime  # type: ignore[import-not-found]

        runtime["deep_sort_realtime"] = getattr(deep_sort_realtime, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        runtime["deep_sort_realtime_error"] = str(exc)
    payload = {
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint.to_dict(),
        "split": config.split,
        "sequence_count": len(sequence_summaries),
        "processed_frame_count": sum(row["frame_count"] for row in sequence_summaries),
        "detection_count": sum(row["detection_count"] for row in sequence_summaries),
        "emitted_track_count": sum(row["emitted_track_count"] for row in sequence_summaries),
        "unique_predicted_track_count": sum(
            row["unique_track_count"] for row in sequence_summaries
        ),
        "timing": global_timing.to_dict(),
        "rendering_enabled": config.render_video,
        "device": config.device,
        "runtime": runtime,
        "cuda": runtime.get("cuda_available"),
        "gpu": runtime.get("gpu_name"),
        "smoke_only": config.smoke_only or checkpoint.smoke_only,
        "warnings": all_warnings,
        "errors": all_errors,
        "sequences": sequence_summaries,
        "validation": validation_report.to_dict(),
    }
    report_paths = write_tracking_run_report(payload, config.metrics_dir)
    tracker_slug = config.tracker_name.lower().replace("-", "_")
    validation_path = write_track_validation_report(
        validation_report,
        config.metrics_dir / f"{tracker_slug}_output_validation.json",
    )
    return {
        "dry_run": False,
        "summary": payload,
        "paths": {
            "run_json": str(report_paths["json"]),
            "per_sequence_csv": str(report_paths["csv"]),
            "validation_json": str(validation_path),
        },
    }


def _run_sequence(
    config: TrackingConfig,
    source: SequenceSource,
    checkpoint: ResolvedCheckpoint,
    tracker_runtime_config: Any,
    detector: Any,
    tracker_adapter_factory: Callable[[Any], Any] | None,
) -> tuple[TrackingRunSummary, TrackValidationReport]:
    mot_path, metadata_path, video_path = _mot_output_paths(config, source)
    if config.save_mot and mot_path is not None and mot_path.exists() and not config.overwrite:
        raise TrackingPipelineError(f"MOT output exists and overwrite=false: {mot_path}")
    if (
        config.render_video
        and video_path is not None
        and video_path.exists()
        and not config.overwrite
    ):
        raise TrackingPipelineError(f"Video output exists and overwrite=false: {video_path}")

    adapter = (
        tracker_adapter_factory(tracker_runtime_config)
        if tracker_adapter_factory is not None
        else create_tracker(config.tracker_name, config.tracker_config, device=config.device)
    )
    adapter.reset()
    trajectory = TrajectoryStore(
        trajectory_length=config.trajectory_length,
        enabled=config.show_trajectory,
    )
    writer = MotPredictionWriter(mot_path, metadata_path) if config.save_mot and mot_path else None
    video_writer = None
    if config.render_video and video_path is not None:
        video_writer = create_tracking_video_writer(
            video_path,
            fps=source.fps,
            width=source.width,
            height=source.height,
            overwrite=config.overwrite,
        ).open()
    timing = TrackingTiming()
    detection_count = 0
    emitted_track_count = 0
    unique_tracks: set[int] = set()
    frame_count = 0
    warnings = list(source.warnings)
    errors: list[str] = []
    sequence_started = time.perf_counter()
    try:
        for frame_item in iter_source_frames(
            source,
            start_frame=config.start_frame,
            max_frames=config.max_frames_per_sequence,
        ):
            frame_started = time.perf_counter()
            frame = frame_item.image
            height, width = frame.shape[:2]
            with timed_section(timing, "detector_seconds", config.device, synchronize_cuda=True):
                raw_prediction = _predict_frame(detector, frame, config)
            with timed_section(timing, "detector_postprocess_seconds"):
                detections = _tracker_detections_from_raw(
                    raw_prediction,
                    frame_item.frame_index,
                    source.name,
                    width,
                    height,
                    config,
                    checkpoint.checkpoint_type,
                    frame_item.image_path,
                )
            with timed_section(timing, "tracker_seconds"):
                tracks = adapter.update(
                    frame_item.frame_index,
                    source.name,
                    detections,
                    frame,
                    width,
                    height,
                )
            detection_count += len(detections)
            emitted_track_count += len(tracks)
            unique_tracks.update(track.track_id for track in tracks)
            if writer is not None:
                writer.add_tracks(tracks)
            trajectory.update(tracks)
            if video_writer is not None:
                render_started = time.perf_counter()
                fps = 1.0 / max(time.perf_counter() - frame_started, 1e-9)
                rendered = draw_tracks(
                    frame,
                    tracks,
                    trajectory_store=trajectory,
                    show_confidence=config.show_confidence,
                    show_class=config.show_class,
                    show_track_id=config.show_track_id,
                    show_trajectory=config.show_trajectory,
                    show_fps=config.show_fps,
                    fps=fps,
                    frame_index=frame_item.frame_index,
                    sequence_name=source.name,
                    line_thickness=config.line_thickness,
                    font_scale=config.font_scale,
                )
                timing.rendering_seconds += time.perf_counter() - render_started
                write_started = time.perf_counter()
                video_writer.write(rendered)
                timing.video_write_seconds += time.perf_counter() - write_started
            if config.show_window:
                import cv2  # type: ignore[import-not-found]

                cv2.imshow("football_tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    warnings.append("Stopped early by interactive keypress.")
                    break
            frame_count += 1
            timing.processed_frames += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        if config.show_window:
            try:
                import cv2  # type: ignore[import-not-found]

                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
    timing.total_pipeline_seconds = time.perf_counter() - sequence_started
    if writer is not None:
        writer.write()
    metadata = {
        "sequence": source.name,
        "source_type": source.source_type,
        "source_path": str(source.source_path),
        "output_mot": str(mot_path) if mot_path is not None else None,
        "output_video": str(video_path) if video_path is not None else None,
        "detector_checkpoint": str(checkpoint.checkpoint),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint_type": checkpoint.checkpoint_type,
        "detector_config": {
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "max_det": config.max_det,
            "class_ids": list(config.class_ids) if config.class_ids is not None else None,
            "target_class_id": config.target_class_id,
            "target_class_name": config.target_class_name,
            "preserve_source_classes": config.preserve_source_classes,
        },
        "tracker": config.tracker_name,
        "tracker_config": (
            tracker_runtime_config.to_dict()
            if hasattr(tracker_runtime_config, "to_dict")
            else dict(tracker_runtime_config)
            if isinstance(tracker_runtime_config, dict)
            else str(tracker_runtime_config)
        ),
        "deepsort_config": (
            tracker_runtime_config.to_dict()
            if config.tracker_name.lower() == "deepsort"
            and hasattr(tracker_runtime_config, "to_dict")
            else None
        ),
        "frame_count": frame_count,
        "detection_count": detection_count,
        "unique_track_count": len(unique_tracks),
        "device": config.device,
        "timing": timing.to_dict(),
        "smoke_only": config.smoke_only or checkpoint.smoke_only,
        "warnings": warnings,
        "errors": errors,
    }
    if writer is not None:
        writer.write_metadata(metadata)
    sequence_validation = (
        validate_mot_prediction_file(mot_path, source.seqinfo_path, metadata_path)
        if mot_path is not None
        else TrackValidationReport()
    )
    summary = TrackingRunSummary(
        sequence_name=source.name,
        frame_count=frame_count,
        detection_count=detection_count,
        emitted_track_count=emitted_track_count,
        unique_track_count=len(unique_tracks),
        detector_seconds=timing.detector_seconds,
        tracker_seconds=timing.tracker_seconds,
        rendering_seconds=timing.rendering_seconds,
        video_write_seconds=timing.video_write_seconds,
        total_seconds=timing.total_pipeline_seconds,
        detector_fps=timing.fps(timing.detector_seconds),
        tracker_fps=timing.fps(timing.tracker_seconds),
        end_to_end_fps=timing.fps(timing.total_pipeline_seconds),
        checkpoint=str(checkpoint.checkpoint),
        device=config.device,
        output_video=video_path if config.render_video else None,
        output_mot=mot_path if config.save_mot else None,
        smoke_only=config.smoke_only or checkpoint.smoke_only,
        warnings=warnings,
        errors=errors,
    )
    return summary, sequence_validation


def validate_tracking_outputs(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_tracking_config(config_path, overrides=overrides)
    sources = _discover_sources(config)
    report = TrackValidationReport()
    for source in sources:
        mot_path, metadata_path, _video_path = _mot_output_paths(config, source)
        if mot_path is None:
            continue
        report.extend(validate_mot_prediction_file(mot_path, source.seqinfo_path, metadata_path))
    path = write_track_validation_report(
        report,
        config.metrics_dir
        / f"{config.tracker_name.lower().replace('-', '_')}_output_validation.json",
    )
    return {"report": report.to_dict(), "path": str(path)}


def tracker_detection_from_box(
    frame_index: int,
    sequence_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> TrackerDetection:
    return TrackerDetection.from_xyxy(
        frame_index=frame_index,
        sequence_name=sequence_name,
        bbox_xyxy=BoundingBoxXYXY(x1, y1, x2, y2),
        confidence=confidence,
    )
