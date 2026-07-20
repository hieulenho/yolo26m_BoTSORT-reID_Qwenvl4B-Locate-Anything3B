"""Generate a runnable tracking config from an adaptive detector route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from football_tracking.adaptive_tracking.router import DetectorRoute
from football_tracking.adaptive_tracking.schemas import SceneDiscovery


def build_tracking_payload(
    *,
    source_video: str | Path,
    output_video: str | Path,
    route: DetectorRoute,
    tracker_config: str = "configs/trackers/deepocsort_reid_realtime.yaml",
    device: str = "auto",
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    source = Path(source_video).resolve()
    output = Path(output_video).resolve()
    model: dict[str, Any] = {
        "name": route.route_name,
        "backend": route.backend,
        "checkpoint": route.checkpoint,
        "alternative_checkpoints": [],
        "fallback_checkpoint": route.checkpoint,
        "allow_pretrained_fallback": True,
        "allow_smoke_checkpoint": False,
    }
    preserve_classes = route.route_name != "football_finetuned"
    if route.backend == "ultralytics_yoloe":
        model["text_classes"] = list(route.class_names)
    source_names = {
        str(class_id): class_name
        for class_id, class_name in zip(route.class_ids, route.class_names, strict=True)
    }
    return {
        "model": model,
        "detector": {
            "imgsz": 736 if route.profile == "realtime" else 960,
            "conf": 0.20 if route.route_name != "football_finetuned" else 0.10,
            "iou": 0.65,
            "max_det": 300,
            "device": device,
            "half": False,
            "class_ids": list(route.class_ids),
            "target_class_id": 0,
            "target_class_name": "player" if not preserve_classes else "object",
            "preserve_source_classes": preserve_classes,
            "source_class_names": source_names,
        },
        "tracker": {
            "name": "deepocsort_reid",
            "config": tracker_config,
        },
        "source": {"path": str(source), "type": "video"},
        "output": {
            "video": str(output),
            "mot": str(output.with_suffix(".txt")),
            "metadata": str(output.with_name(f"{output.stem}.metadata.json")),
            "render_video": True,
            "save_mot": True,
        },
        "render": {
            "enabled": True,
            "show_confidence": False,
            "show_class": True,
            "show_track_id": True,
            "show_trajectory": True,
            "trajectory_length": 20,
            "line_thickness": 2,
            "font_scale": 0.55,
            "show_fps": True,
        },
        "runtime": {
            "max_frames": max_frames,
            "start_frame": 1,
            "overwrite": overwrite,
            "show_window": False,
            "save_mot": True,
            "fail_fast": True,
            "log_level": "INFO",
        },
    }


def write_adaptive_plan(
    *,
    output_dir: str | Path,
    discovery: SceneDiscovery,
    route: DetectorRoute,
    tracking_payload: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    discovery_path = root / "scene_discovery.json"
    route_path = root / "detector_route.json"
    config_path = root / "tracking.generated.yaml"
    plan_path = root / "adaptive_plan.json"
    paths = (discovery_path, route_path, config_path, plan_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Adaptive output exists and overwrite=false: {existing[0]}")
    discovery_path.write_text(
        json.dumps(discovery.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    route_path.write_text(json.dumps(route.to_dict(), indent=2), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(tracking_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    plan = {
        "schema_version": "1.0",
        "source_video": discovery.source_video,
        "domain": discovery.domain,
        "route": route.to_dict(),
        "stages": [
            "shot_keyframe_sampling",
            "qwen_scene_discovery",
            "vocabulary_normalization",
            "detector_routing",
            "frame_detection",
            "deepocsort_reid_tracking",
            "track_crop_sampling",
            "qwen_track_semantics",
            "temporal_fusion_unknown_rejection",
            "render_and_metrics",
        ],
        "locateanything_policy": {
            "mode": "event_triggered",
            "triggers": [
                "open_vocabulary_class",
                "low_semantic_confidence",
                "identity_reacquisition",
            ],
            "note": "LocateAnything grounds selected keyframes; it is not run on every frame.",
        },
        "paths": {
            "discovery": str(discovery_path),
            "route": str(route_path),
            "tracking_config": str(config_path),
        },
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {
        "discovery": discovery_path,
        "route": route_path,
        "tracking_config": config_path,
        "plan": plan_path,
    }
