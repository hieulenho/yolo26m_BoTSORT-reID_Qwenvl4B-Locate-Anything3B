"""Configuration loader for uncertainty monitoring and event grounding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.grounding_scheduler.schemas import SchedulerConfig
from football_tracking.locate_tracking.monitoring.schemas import MonitoringConfig
from football_tracking.paths import get_project_root, resolve_project_path


class UncertaintyConfigError(RuntimeError):
    """Raised when uncertainty monitoring config cannot be loaded."""


@dataclass(frozen=True)
class UncertaintyPipelineConfig:
    project_root: Path
    config_path: Path
    monitoring: MonitoringConfig
    scheduler: SchedulerConfig
    output_dir: Path
    overwrite: bool = False


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise UncertaintyConfigError(f"{section} must be a mapping.")
    return dict(value)


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise UncertaintyConfigError(f"{section} must be a non-empty path.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _monitoring_config(raw: dict[str, Any]) -> MonitoringConfig:
    thresholds = _mapping(raw.get("thresholds"), "monitoring.thresholds", default={})
    weights = _mapping(raw.get("signal_weights"), "monitoring.signal_weights", default={})
    return MonitoringConfig(
        presence_warning_absent_frames=int(thresholds.get("presence_warning_absent_frames", 5)),
        presence_critical_absent_frames=int(thresholds.get("presence_critical_absent_frames", 20)),
        confidence_low_threshold=float(thresholds.get("confidence_low_threshold", 0.25)),
        confidence_consecutive_frames=int(thresholds.get("confidence_consecutive_frames", 3)),
        motion_jump_threshold=float(thresholds.get("motion_jump_threshold", 0.08)),
        motion_jump_ratio_threshold=float(thresholds.get("motion_jump_ratio_threshold", 4.0)),
        motion_baseline_window=int(thresholds.get("motion_baseline_window", 10)),
        semantic_margin_threshold=float(thresholds.get("semantic_margin_threshold", 0.08)),
        appearance_drift_threshold=float(thresholds.get("appearance_drift_threshold", 0.35)),
        neighbor_distance_threshold=float(thresholds.get("neighbor_distance_threshold", 0.045)),
        neighbor_iou_threshold=float(thresholds.get("neighbor_iou_threshold", 0.05)),
        neighbor_count_threshold=int(thresholds.get("neighbor_count_threshold", 1)),
        gap_warning_frames=int(thresholds.get("gap_warning_frames", 5)),
        gap_critical_frames=int(thresholds.get("gap_critical_frames", 20)),
        staleness_warning_frames=int(thresholds.get("staleness_warning_frames", 120)),
        signal_weights={str(key): float(value) for key, value in weights.items()},
    )


def _scheduler_config(raw: dict[str, Any]) -> SchedulerConfig:
    cooldown = _mapping(raw.get("cooldown"), "scheduler.cooldown", default={})
    budget = _mapping(raw.get("budget"), "scheduler.budget", default={})
    frame_selection = _mapping(raw.get("frame_selection"), "scheduler.frame_selection", default={})
    priority = _mapping(raw.get("priority"), "scheduler.priority", default={})
    return SchedulerConfig(
        min_severity=str(raw.get("min_severity", "warning")),
        cooldown_frames=int(cooldown.get("frames", 50)),
        critical_overrides_cooldown=bool(cooldown.get("critical_overrides_cooldown", True)),
        max_requests_per_session=int(budget.get("max_requests_per_session", 20)),
        max_frames_per_request=int(budget.get("max_frames_per_request", 3)),
        frame_strategy=str(frame_selection.get("strategy", "window_representative")),
        event_type_priority=tuple(priority.get("event_type_order", ())),
    )


def load_uncertainty_pipeline_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> UncertaintyPipelineConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved_config.is_file():
        raise UncertaintyConfigError(f"Uncertainty config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "uncertainty config root")
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = UncertaintyPipelineConfig(
        project_root=project_root,
        config_path=resolved_config,
        monitoring=_monitoring_config(_mapping(root.get("monitoring"), "monitoring", default={})),
        scheduler=_scheduler_config(_mapping(root.get("scheduler"), "scheduler", default={})),
        output_dir=_resolve_path(
            output.get("directory", "outputs/locate_tracking/uncertainty"),
            project_root,
            "output.directory",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
    )
    if overrides:
        changes: dict[str, Any] = {}
        if overrides.get("output_dir") is not None:
            changes["output_dir"] = _resolve_path(
                overrides["output_dir"],
                project_root,
                "output_dir",
            )
        if overrides.get("overwrite") is not None:
            changes["overwrite"] = bool(overrides["overwrite"])
        config = replace(config, **changes) if changes else config
    return config
