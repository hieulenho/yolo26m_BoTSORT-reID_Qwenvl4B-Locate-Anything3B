"""Build a standard tracking config from one explicit task specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from football_tracking.task_pipeline.config import TaskPipelineConfig


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _supplemental_class_names(
    supplemental_detectors: tuple[dict[str, Any], ...],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for detector in supplemental_detectors:
        output_ids = detector.get("output_class_ids", [])
        class_names = detector.get("class_names", [])
        if not isinstance(output_ids, list | tuple) or not isinstance(
            class_names, list | tuple
        ):
            continue
        for class_id, class_name in zip(output_ids, class_names, strict=False):
            names[str(int(class_id))] = str(class_name)
    return names


def _tracker_runtime_name(
    *,
    default_name: str,
    config_path: Path,
    is_override: bool,
) -> str:
    """Return the factory name represented by an explicit tracker override."""
    if not is_override:
        return default_name
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Tracker config root must be a mapping: {config_path}")
    tracker = raw.get("tracker")
    if not isinstance(tracker, dict):
        raise ValueError(
            f"Tracker override must contain a tracker mapping: {config_path}"
        )
    tracker_type = str(tracker.get("tracker_type", "")).strip().casefold()
    if not tracker_type:
        raise ValueError(
            f"Tracker override must define tracker.tracker_type: {config_path}"
        )
    if bool(tracker.get("with_reid", False)) and tracker_type in {
        "botsort",
        "deepocsort",
        "tracktrack",
    }:
        return f"{tracker_type}_reid"
    return tracker_type


def build_tracking_payload(
    config: TaskPipelineConfig,
    *,
    device: str,
    output_video: str | Path,
    overwrite: bool,
    preprocessing_mode: str | None = None,
    tracker_config_path: str | Path | None = None,
) -> dict[str, Any]:
    detector = config.detector
    model: dict[str, Any] = {
        "name": detector.name,
        "backend": detector.backend,
        "checkpoint": detector.checkpoint,
        "alternative_checkpoints": [],
        "fallback_checkpoint": detector.checkpoint,
        "allow_pretrained_fallback": True,
        "allow_smoke_checkpoint": False,
    }
    if detector.text_classes:
        model["text_classes"] = list(detector.text_classes)
    if detector.supplemental_detectors:
        model["supplemental_detectors"] = [
            dict(item) for item in detector.supplemental_detectors
        ]
    supplemental_class_names = _supplemental_class_names(
        detector.supplemental_detectors
    )
    output = Path(output_video).resolve()
    resolved_preprocessing_mode = (
        str(preprocessing_mode).strip().lower()
        if preprocessing_mode is not None
        else detector.preprocessing_mode
    )
    if resolved_preprocessing_mode not in {"none", "auto_low_light", "clahe"}:
        raise ValueError(
            "preprocessing_mode must be one of none, auto_low_light, or clahe."
        )
    tracker_candidate = (
        Path(tracker_config_path)
        if tracker_config_path is not None
        else config.tracker.config_path
    )
    resolved_tracker_config = (
        tracker_candidate.resolve()
        if tracker_candidate.is_absolute()
        else (config.project_root / tracker_candidate).resolve()
    )
    if not resolved_tracker_config.is_file():
        raise ValueError(f"Tracker config does not exist: {resolved_tracker_config}")
    tracker_runtime_name = _tracker_runtime_name(
        default_name=config.tracker.name,
        config_path=resolved_tracker_config,
        is_override=tracker_config_path is not None,
    )
    return {
        "model": model,
        "detector": {
            "imgsz": detector.imgsz,
            "conf": detector.confidence,
            "iou": detector.iou,
            "max_det": detector.max_det,
            "device": device,
            "half": detector.half and str(device).casefold() != "cpu",
            "class_ids": (
                None if detector.class_ids is None else list(detector.class_ids)
            ),
            "tracker_class_ids": (
                None
                if detector.tracker_class_ids is None
                else list(detector.tracker_class_ids)
            ),
            "target_class_id": detector.target_class_id,
            "target_class_name": detector.target_class_name,
            "preserve_source_classes": detector.preserve_source_classes,
            "source_class_names": supplemental_class_names,
            "preprocessing": {
                "mode": resolved_preprocessing_mode,
                "clahe_clip_limit": detector.clahe_clip_limit,
                "clahe_grid_size": detector.clahe_grid_size,
                "low_light_threshold": detector.low_light_threshold,
            },
        },
        "tracker": {
            "name": tracker_runtime_name,
            "config": str(resolved_tracker_config),
            "stabilize_classes": config.tracker.stabilize_classes,
            "class_history_frames": config.tracker.class_history_frames,
            "class_switch_margin": config.tracker.class_switch_margin,
        },
        "source": {
            "type": "stream",
            "shot_starts": [],
            "reset_tracker_on_shot_change": True,
        },
        "output": {
            "video": str(output),
            "mot": str(output.with_suffix(".txt")),
            "metadata": str(output.with_name(f"{output.stem}.metadata.json")),
            "render_video": True,
            "save_mot": True,
        },
        "render": {
            "enabled": True,
            "show_confidence": bool(config.render.get("show_confidence", True)),
            "show_class": True,
            "show_track_id": True,
            "show_trajectory": bool(config.render.get("show_trajectory", True)),
            "trajectory_length": int(config.render.get("trajectory_length", 20)),
            "line_thickness": int(config.render.get("line_thickness", 2)),
            "font_scale": float(config.render.get("font_scale", 0.55)),
            "show_fps": True,
        },
        "runtime": {
            "start_frame": 1,
            "overwrite": overwrite,
            "show_window": False,
            "save_mot": True,
            "fail_fast": True,
            "log_level": "INFO",
        },
    }


def write_task_runtime(
    *,
    config: TaskPipelineConfig,
    output_dir: str | Path,
    output_video: str | Path,
    device: str,
    overwrite: bool,
    preprocessing_mode: str | None = None,
    tracker_config_path: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tracking_path = root / "tracking.generated.yaml"
    task_path = root / "task.resolved.json"
    if not overwrite:
        existing = [path for path in (tracking_path, task_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Task runtime output exists and overwrite=false: "
                + ", ".join(str(path) for path in existing)
            )
    tracking = build_tracking_payload(
        config,
        device=device,
        output_video=output_video,
        overwrite=overwrite,
        preprocessing_mode=preprocessing_mode,
        tracker_config_path=tracker_config_path,
    )
    resolved = {
        "schema_version": "1.0",
        "task_id": config.task_id,
        "task_name": config.task_name,
        "description": config.description,
        "source_config": str(config.config_path),
        "detector": tracking["detector"],
        "model": tracking["model"],
        "tracker": tracking["tracker"],
        "semantic": {
            "enabled": config.semantic.enabled,
            **config.semantic_event_payload(),
            "event_interval_frames": config.semantic.event_interval_frames,
            "events_per_frame": config.semantic.events_per_frame,
            "max_pending_events": config.semantic.max_pending_events,
            "minimum_track_age_frames": config.semantic.minimum_track_age_frames,
        },
    }
    _atomic_text(
        tracking_path,
        yaml.safe_dump(tracking, sort_keys=False, allow_unicode=False),
    )
    _atomic_text(
        task_path,
        json.dumps(resolved, indent=2, ensure_ascii=False),
    )
    return {"tracking_config": tracking_path, "resolved_task": task_path}
