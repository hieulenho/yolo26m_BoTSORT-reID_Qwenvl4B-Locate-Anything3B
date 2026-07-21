"""Generate a runnable tracking config from an adaptive detector route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from football_tracking.adaptive_tracking.router import DetectorRoute
from football_tracking.adaptive_tracking.schemas import SceneDiscovery

TRACKER_CONFIGS = {
    "ocsort": "configs/trackers/ocsort_realtime.yaml",
    "deepocsort_reid": "configs/trackers/deepocsort_reid_realtime.yaml",
    "tracktrack": "configs/trackers/tracktrack_realtime.yaml",
    "botsort_reid": "configs/trackers/botsort_reid_identity_stable.yaml",
    "fasttrack": "configs/trackers/fasttrack_realtime.yaml",
    "ocsort_open_vocab": "configs/trackers/ocsort_open_vocab_realtime.yaml",
}

TRACKER_PROFILES = {
    "realtime": ("ocsort", "configs/trackers/ocsort_realtime.yaml"),
    "realtime_stable": (
        "tracktrack",
        "configs/trackers/tracktrack_realtime.yaml",
    ),
    "balanced": (
        "tracktrack",
        "configs/trackers/tracktrack_realtime.yaml",
    ),
    "accuracy": (
        "botsort_reid",
        "configs/trackers/botsort_reid_identity_stable.yaml",
    ),
}

SMALL_FAST_CLASSES = {
    "ball",
    "sports ball",
    "football",
    "frisbee",
    "drone",
}


def _class_route_tracker(
    profile: str,
    class_name: str,
    *,
    open_vocabulary: bool = False,
) -> tuple[str, str, str]:
    normalized = class_name.strip().lower()
    if profile in {"realtime", "realtime_stable"} and open_vocabulary:
        return (
            "ocsort",
            "configs/trackers/ocsort_open_vocab_realtime.yaml",
            "low-confidence open-vocabulary detections with observation-centric association",
        )
    if normalized in SMALL_FAST_CLASSES:
        return (
            "ocsort",
            "configs/trackers/ocsort_small_fast.yaml",
            "motion model tuned for small, fast and non-linear targets",
        )
    tracker_name, tracker_config = TRACKER_PROFILES[profile]
    rationale = {
        "realtime": "low-latency observation-centric association",
        "realtime_stable": "identity-stable association with a realtime detector route",
        "balanced": "stronger association while retaining practical throughput",
        "accuracy": "appearance-aware identity association",
    }[profile]
    return tracker_name, tracker_config, rationale


def build_tracker_routing_payload(route: DetectorRoute) -> dict[str, Any]:
    """Group classes by motion regime instead of isolating every class.

    Class-isolated trackers turn a temporary detector label change into a new
    identity.  Regular classes therefore share the default tracker.  Only
    genuinely different motion regimes use dedicated delegates.
    """
    default_name, default_config = TRACKER_PROFILES[route.profile]
    class_names = dict(zip(route.class_ids, route.class_names, strict=True))
    routes: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {"small_fast": [], "open_vocabulary": []}
    for class_id in route.tracker_class_ids:
        class_name = class_names.get(class_id, f"class_{class_id}")
        if class_name.strip().lower() in SMALL_FAST_CLASSES:
            groups["small_fast"].append(class_id)
        elif route.backend == "ultralytics_yoloe" or class_id >= 1000:
            groups["open_vocabulary"].append(class_id)

    for group_name, class_ids in groups.items():
        if not class_ids:
            continue
        representative = class_names.get(class_ids[0], f"class_{class_ids[0]}")
        tracker_name, tracker_config, rationale = _class_route_tracker(
            route.profile,
            representative,
            open_vocabulary=group_name == "open_vocabulary",
        )
        routes.append(
            {
                "route_name": group_name,
                "class_ids": class_ids,
                "class_names": [
                    class_names.get(value, f"class_{value}") for value in class_ids
                ],
                "name": tracker_name,
                "config": tracker_config,
                "rationale": rationale,
            }
        )
    return {
        "tracker": {
            "name": "adaptive_routed",
            "profile": route.profile,
            "default": {
                "route_name": "default",
                "name": default_name,
                "config": default_config,
                "class_ids": [
                    class_id
                    for class_id in route.tracker_class_ids
                    if class_id not in groups["small_fast"]
                    and class_id not in groups["open_vocabulary"]
                ],
            },
            "routes": routes,
        }
    }


def build_tracking_payload(
    *,
    source_video: str | Path,
    output_video: str | Path,
    route: DetectorRoute,
    tracker_name: str | None = None,
    tracker_config: str | None = None,
    device: str = "auto",
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    source = Path(source_video).resolve()
    output = Path(output_video).resolve()
    if tracker_name is None:
        selected_tracker = "adaptive_routed"
        selected_tracker_config = "__GENERATED_TRACKER_ROUTING__"
    else:
        selected_tracker = tracker_name
        selected_tracker_config = tracker_config or TRACKER_CONFIGS[tracker_name]
    model: dict[str, Any] = {
        "name": route.route_name,
        "backend": route.backend,
        "checkpoint": route.checkpoint,
        "alternative_checkpoints": [],
        "fallback_checkpoint": route.checkpoint,
        "allow_pretrained_fallback": True,
        "allow_smoke_checkpoint": False,
    }
    preserve_classes = (
        route.route_name != "football_finetuned" or bool(route.supplemental_detectors)
    )
    if route.backend == "ultralytics_yoloe":
        model["text_classes"] = list(route.class_names)
    if route.supplemental_detectors:
        model["supplemental_detectors"] = [
            dict(item) for item in route.supplemental_detectors
        ]
    source_names = {
        str(class_id): class_name
        for class_id, class_name in zip(route.class_ids, route.class_names, strict=True)
    }
    return {
        "model": model,
        "detector": {
            "imgsz": (
                576
                if route.profile == "realtime"
                else 640
                if route.profile == "realtime_stable"
                else 960
            ),
            "conf": (
                0.05
                if route.route_name == "open_vocabulary"
                else 0.10
                if route.route_name in {"football_finetuned", "coco_open_composite"}
                else 0.20
            ),
            "iou": 0.65,
            "max_det": 200 if route.profile == "realtime" else 300,
            "device": device,
            # The primary YOLO path supports FP16. YOLOE supplements explicitly
            # remain FP32 because their text projection currently requires it.
            "half": (
                route.profile in {"realtime", "realtime_stable"}
                and str(device).lower() != "cpu"
            ),
            "class_ids": list(route.primary_class_ids or route.class_ids),
            "tracker_class_ids": list(route.tracker_class_ids),
            "target_class_id": 0,
            "target_class_name": "player" if not preserve_classes else "object",
            "preserve_source_classes": preserve_classes,
            "source_class_names": source_names,
        },
        "tracker": {
            "name": selected_tracker,
            "config": selected_tracker_config,
        },
        "source": {
            "path": str(source),
            "type": "video",
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
    tracker_routing_path = root / "tracker_routing.generated.yaml"
    plan_path = root / "adaptive_plan.json"
    uses_routing = tracking_payload["tracker"]["name"] == "adaptive_routed"
    tracking_payload.setdefault("source", {})["shot_starts"] = list(
        discovery.shot_starts
    )
    paths = (
        discovery_path,
        route_path,
        config_path,
        plan_path,
        *((tracker_routing_path,) if uses_routing else ()),
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Adaptive output exists and overwrite=false: {existing[0]}")
    discovery_path.write_text(
        json.dumps(discovery.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    route_path.write_text(json.dumps(route.to_dict(), indent=2), encoding="utf-8")
    if uses_routing:
        tracker_routing_path.write_text(
            yaml.safe_dump(
                build_tracker_routing_payload(route),
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        tracking_payload["tracker"]["config"] = str(tracker_routing_path.resolve())
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
            f"{tracking_payload['tracker']['name']}_tracking",
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
            "tracker_routing": str(tracker_routing_path) if uses_routing else None,
        },
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    result = {
        "discovery": discovery_path,
        "route": route_path,
        "tracking_config": config_path,
        "plan": plan_path,
    }
    if uses_routing:
        result["tracker_routing"] = tracker_routing_path
    return result
