"""Validated task configuration for the single-VLM realtime pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class TaskPipelineConfigError(RuntimeError):
    """Raised when a task pipeline configuration is incomplete or unsafe."""


_TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_DETECTOR_BACKENDS = {"ultralytics", "ultralytics_yoloe"}
_TRACKERS = {
    "tracktrack",
    "tracktrack_reid",
    "ocsort",
    "deepocsort_reid",
    "botsort_reid",
    "bytetrack",
}
_PREPROCESSING_MODES = {"none", "auto_low_light", "clahe"}
_REALTIME_QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskPipelineConfigError(f"{name} must be a mapping.")
    return dict(value)


def _non_empty_string(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TaskPipelineConfigError(f"{name} must be a non-empty string.")
    return text


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise TaskPipelineConfigError(f"{name} must be a list of strings.")
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _non_empty_string(item, f"{name}[{index}]")
        key = text.casefold()
        if key not in seen:
            output.append(text)
            seen.add(key)
    return tuple(output)


def _class_ids(value: Any, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise TaskPipelineConfigError(f"{name} must be null or a list of integers.")
    output = tuple(dict.fromkeys(int(item) for item in value))
    if any(item < 0 for item in output):
        raise TaskPipelineConfigError(f"{name} cannot contain negative IDs.")
    return output


def _unit_interval(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise TaskPipelineConfigError(f"{name} must be in [0, 1].")
    return number


@dataclass(frozen=True)
class DetectorTaskConfig:
    backend: str
    checkpoint: str
    name: str
    text_classes: tuple[str, ...]
    class_ids: tuple[int, ...] | None
    tracker_class_ids: tuple[int, ...] | None
    preserve_source_classes: bool
    target_class_id: int
    target_class_name: str
    imgsz: int
    confidence: float
    iou: float
    max_det: int
    half: bool
    preprocessing_mode: str
    clahe_clip_limit: float
    clahe_grid_size: int
    low_light_threshold: float


@dataclass(frozen=True)
class TrackerTaskConfig:
    name: str
    config_path: Path
    stabilize_classes: bool
    class_history_frames: int
    class_switch_margin: float


@dataclass(frozen=True)
class SemanticTaskConfig:
    enabled: bool
    provider: str
    model_id: str
    quantization: str
    label_mode: str
    enable_fine_labels: bool
    allowed_labels: tuple[str, ...]
    attributes: tuple[str, ...]
    instruction: str
    unknown_threshold: float
    event_interval_frames: int
    events_per_frame: int
    max_pending_events: int
    minimum_track_age_frames: int
    max_evidence_images: int
    evidence_layout: str
    evidence_panel_width: int
    evidence_panel_height: int
    minimum_crop_quality: float
    replacement_quality_margin: float

    def event_payload(self, *, task_id: str, task_name: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "task_name": task_name,
            "provider": self.provider,
            "model_id": self.model_id,
            "quantization": self.quantization,
            "label_mode": self.label_mode,
            "enable_fine_labels": self.enable_fine_labels,
            "allowed_labels": list(self.allowed_labels),
            "attributes": list(self.attributes),
            "instruction": self.instruction,
            "unknown_threshold": self.unknown_threshold,
            "max_evidence_images": self.max_evidence_images,
            "evidence_layout": self.evidence_layout,
            "evidence_panel_width": self.evidence_panel_width,
            "evidence_panel_height": self.evidence_panel_height,
            "minimum_crop_quality": self.minimum_crop_quality,
            "replacement_quality_margin": self.replacement_quality_margin,
        }


@dataclass(frozen=True)
class TaskPipelineConfig:
    config_path: Path
    project_root: Path
    task_id: str
    task_name: str
    description: str
    detector: DetectorTaskConfig
    tracker: TrackerTaskConfig
    semantic: SemanticTaskConfig
    render: dict[str, Any]

    def semantic_event_payload(self) -> dict[str, Any]:
        return self.semantic.event_payload(task_id=self.task_id, task_name=self.task_name)


def load_task_pipeline_config(path: str | Path) -> TaskPipelineConfig:
    project_root = get_project_root()
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else resolve_project_path(candidate, project_root)
    )
    if not resolved.is_file():
        raise TaskPipelineConfigError(f"Task config does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "task config root")
    schema_version = str(root.get("schema_version", "1.0"))
    if schema_version != "1.0":
        raise TaskPipelineConfigError(
            f"Unsupported task config schema_version: {schema_version}"
        )

    task = _mapping(root.get("task"), "task")
    task_id = _non_empty_string(task.get("id"), "task.id").casefold()
    if not _TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskPipelineConfigError(
            "task.id must contain 2-64 lowercase letters, digits, '_' or '-'."
        )
    task_name = _non_empty_string(task.get("name", task_id), "task.name")
    description = _non_empty_string(task.get("description"), "task.description")

    detector_raw = _mapping(root.get("detector"), "detector")
    backend = str(detector_raw.get("backend", "ultralytics")).strip().lower()
    if backend not in _DETECTOR_BACKENDS:
        raise TaskPipelineConfigError(
            f"detector.backend must be one of {sorted(_DETECTOR_BACKENDS)}."
        )
    text_classes = _string_tuple(detector_raw.get("text_classes"), "detector.text_classes")
    if backend == "ultralytics_yoloe" and not text_classes:
        raise TaskPipelineConfigError(
            "detector.text_classes is required for the YOLOE backend."
        )
    class_ids = _class_ids(detector_raw.get("class_ids"), "detector.class_ids")
    tracker_class_ids = _class_ids(
        detector_raw.get("tracker_class_ids", detector_raw.get("class_ids")),
        "detector.tracker_class_ids",
    )
    preprocessing = _mapping(
        detector_raw.get("preprocessing", {}),
        "detector.preprocessing",
    )
    preprocessing_mode = str(preprocessing.get("mode", "none")).strip().lower()
    if preprocessing_mode not in _PREPROCESSING_MODES:
        raise TaskPipelineConfigError(
            f"detector.preprocessing.mode must be one of {sorted(_PREPROCESSING_MODES)}."
        )
    detector = DetectorTaskConfig(
        backend=backend,
        checkpoint=_non_empty_string(detector_raw.get("checkpoint"), "detector.checkpoint"),
        name=_non_empty_string(detector_raw.get("name", "task_detector"), "detector.name"),
        text_classes=text_classes,
        class_ids=class_ids,
        tracker_class_ids=tracker_class_ids,
        preserve_source_classes=bool(detector_raw.get("preserve_source_classes", True)),
        target_class_id=int(detector_raw.get("target_class_id", 0)),
        target_class_name=_non_empty_string(
            detector_raw.get("target_class_name", "object"),
            "detector.target_class_name",
        ),
        imgsz=int(detector_raw.get("imgsz", 640)),
        confidence=_unit_interval(detector_raw.get("conf", 0.20), "detector.conf"),
        iou=_unit_interval(detector_raw.get("iou", 0.65), "detector.iou"),
        max_det=int(detector_raw.get("max_det", 300)),
        half=bool(detector_raw.get("half", True)),
        preprocessing_mode=preprocessing_mode,
        clahe_clip_limit=float(preprocessing.get("clahe_clip_limit", 2.0)),
        clahe_grid_size=int(preprocessing.get("clahe_grid_size", 8)),
        low_light_threshold=float(preprocessing.get("low_light_threshold", 70.0)),
    )
    if detector.imgsz < 256 or detector.max_det < 1:
        raise TaskPipelineConfigError(
            "detector.imgsz must be >= 256 and detector.max_det must be positive."
        )
    if detector.clahe_clip_limit <= 0 or detector.clahe_grid_size < 2:
        raise TaskPipelineConfigError("Invalid CLAHE preprocessing parameters.")
    if not 0.0 <= detector.low_light_threshold <= 255.0:
        raise TaskPipelineConfigError(
            "detector.preprocessing.low_light_threshold must be in [0, 255]."
        )

    tracker_raw = _mapping(root.get("tracker"), "tracker")
    tracker_name = str(tracker_raw.get("name", "tracktrack")).strip().lower()
    if tracker_name not in _TRACKERS:
        raise TaskPipelineConfigError(
            f"tracker.name must be one of {sorted(_TRACKERS)}."
        )
    tracker_path_raw = _non_empty_string(tracker_raw.get("config"), "tracker.config")
    tracker_path = Path(tracker_path_raw)
    tracker_path = (
        tracker_path.resolve()
        if tracker_path.is_absolute()
        else resolve_project_path(tracker_path, project_root)
    )
    if not tracker_path.is_file():
        raise TaskPipelineConfigError(f"Tracker config does not exist: {tracker_path}")
    tracker = TrackerTaskConfig(
        name=tracker_name,
        config_path=tracker_path,
        stabilize_classes=bool(tracker_raw.get("stabilize_classes", True)),
        class_history_frames=int(tracker_raw.get("class_history_frames", 30)),
        class_switch_margin=_unit_interval(
            tracker_raw.get("class_switch_margin", 0.20),
            "tracker.class_switch_margin",
        ),
    )
    if tracker.class_history_frames < 2:
        raise TaskPipelineConfigError("tracker.class_history_frames must be >= 2.")

    semantic_raw = _mapping(root.get("semantic", {}), "semantic")
    semantic_enabled = bool(semantic_raw.get("enabled", True))
    label_mode = str(semantic_raw.get("label_mode", "open")).strip().lower()
    if label_mode not in {"open", "closed"}:
        raise TaskPipelineConfigError("semantic.label_mode must be 'open' or 'closed'.")
    allowed_labels = _string_tuple(
        semantic_raw.get("allowed_labels"),
        "semantic.allowed_labels",
    )
    if semantic_enabled and label_mode == "closed" and not allowed_labels:
        raise TaskPipelineConfigError(
            "semantic.allowed_labels is required when label_mode=closed."
        )
    provider = str(semantic_raw.get("provider", "qwen")).strip().lower()
    quantization = str(semantic_raw.get("quantization", "8bit")).strip().lower()
    if semantic_enabled and provider != "qwen":
        raise TaskPipelineConfigError(
            "The main realtime pipeline supports one semantic provider: qwen."
        )
    if semantic_enabled and quantization != "8bit":
        raise TaskPipelineConfigError(
            "The supported realtime Qwen profile is fixed to 8bit quantization."
        )
    model_id = _non_empty_string(
        semantic_raw.get("model_id", _REALTIME_QWEN_MODEL_ID),
        "semantic.model_id",
    )
    if semantic_enabled and model_id != _REALTIME_QWEN_MODEL_ID:
        raise TaskPipelineConfigError(
            "The supported realtime semantic model is fixed to "
            f"{_REALTIME_QWEN_MODEL_ID}."
        )
    semantic = SemanticTaskConfig(
        enabled=semantic_enabled,
        provider=provider,
        model_id=model_id,
        quantization=quantization,
        label_mode=label_mode,
        enable_fine_labels=bool(
            semantic_raw.get("enable_fine_labels", label_mode == "open")
        ),
        allowed_labels=allowed_labels,
        attributes=_string_tuple(semantic_raw.get("attributes"), "semantic.attributes"),
        instruction=_non_empty_string(
            semantic_raw.get(
                "instruction",
                "Assign the most specific visually supported stable label.",
            ),
            "semantic.instruction",
        ),
        unknown_threshold=_unit_interval(
            semantic_raw.get("unknown_threshold", 0.70),
            "semantic.unknown_threshold",
        ),
        event_interval_frames=int(semantic_raw.get("event_interval_frames", 45)),
        events_per_frame=int(semantic_raw.get("events_per_frame", 2)),
        max_pending_events=int(semantic_raw.get("max_pending_events", 128)),
        minimum_track_age_frames=int(
            semantic_raw.get("minimum_track_age_frames", 8)
        ),
        max_evidence_images=int(semantic_raw.get("max_evidence_images", 2)),
        evidence_layout=str(
            semantic_raw.get("evidence_layout", "panel")
        ).strip().lower(),
        evidence_panel_width=int(semantic_raw.get("evidence_panel_width", 512)),
        evidence_panel_height=int(semantic_raw.get("evidence_panel_height", 384)),
        minimum_crop_quality=_unit_interval(
            semantic_raw.get("minimum_crop_quality", 0.25),
            "semantic.minimum_crop_quality",
        ),
        replacement_quality_margin=_unit_interval(
            semantic_raw.get("replacement_quality_margin", 0.08),
            "semantic.replacement_quality_margin",
        ),
    )
    if semantic.event_interval_frames < 1:
        raise TaskPipelineConfigError("semantic.event_interval_frames must be positive.")
    if not 0 <= semantic.events_per_frame <= 16:
        raise TaskPipelineConfigError("semantic.events_per_frame must be in [0, 16].")
    if semantic.max_pending_events < 1:
        raise TaskPipelineConfigError("semantic.max_pending_events must be positive.")
    if semantic.minimum_track_age_frames < 1:
        raise TaskPipelineConfigError(
            "semantic.minimum_track_age_frames must be positive."
        )
    if not 1 <= semantic.max_evidence_images <= 8:
        raise TaskPipelineConfigError("semantic.max_evidence_images must be in [1, 8].")
    if semantic.evidence_layout not in {"panel", "crops"}:
        raise TaskPipelineConfigError(
            "semantic.evidence_layout must be 'panel' or 'crops'."
        )
    if not 256 <= semantic.evidence_panel_width <= 1024:
        raise TaskPipelineConfigError(
            "semantic.evidence_panel_width must be in [256, 1024]."
        )
    if not 256 <= semantic.evidence_panel_height <= 1024:
        raise TaskPipelineConfigError(
            "semantic.evidence_panel_height must be in [256, 1024]."
        )
    if semantic.evidence_panel_width * semantic.evidence_panel_height > 524_288:
        raise TaskPipelineConfigError(
            "semantic evidence panel is too large for the 8 GiB realtime profile."
        )

    render = _mapping(root.get("render", {}), "render")
    return TaskPipelineConfig(
        config_path=resolved,
        project_root=project_root,
        task_id=task_id,
        task_name=task_name,
        description=description,
        detector=detector,
        tracker=tracker,
        semantic=semantic,
        render=render,
    )
