"""Load comparison experiment configs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.experiments.schemas import TrackerSpec
from football_tracking.paths import get_project_root, resolve_project_path


class ExperimentConfigError(RuntimeError):
    """Raised when an experiment config is invalid."""


@dataclass(frozen=True)
class CompareTrackersConfig:
    project_root: Path
    config_path: Path
    experiment_name: str
    seed: int
    split: str
    mot_root: Path
    seqmap: Path | None
    detection_cache_config: Path
    detection_cache_root: Path
    confidence_threshold: float
    trackers: tuple[TrackerSpec, ...]
    trackeval_config: Path | None
    metrics: tuple[str, ...]
    allow_partial_sequences: bool
    render_video: bool
    measure_tracker_only: bool
    measure_end_to_end_from_cache: bool
    warmup_sequences: int
    output_root: Path
    tracks_root: Path
    metrics_root: Path
    figures_root: Path
    videos_root: Path
    max_sequences: int | None
    max_frames_per_sequence: int | None
    overwrite: bool
    fail_fast: bool
    log_level: str
    smoke_only: bool


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{section} must be a mapping.")
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
        raise ExperimentConfigError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_compare_trackers_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> CompareTrackersConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise ExperimentConfigError(f"Experiment config does not exist: {resolved}")
    raw = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "experiment config root")
    experiment = _mapping(raw.get("experiment"), "experiment")
    dataset = _mapping(raw.get("dataset"), "dataset")
    detections = _mapping(raw.get("detections"), "detections")
    evaluation = _mapping(raw.get("evaluation", {}), "evaluation")
    benchmark = _mapping(raw.get("benchmark", {}), "benchmark")
    output = _mapping(raw.get("output"), "output")
    runtime = _mapping(raw.get("runtime", {}), "runtime")

    trackers = []
    for item in raw.get("trackers", []):
        tracker = _mapping(item, "trackers[]")
        trackers.append(
            TrackerSpec(
                name=str(tracker.get("name")).lower(),
                config=_resolve_path(tracker.get("config"), project_root, "trackers[].config"),
            )
        )
    config = CompareTrackersConfig(
        project_root=project_root,
        config_path=resolved,
        experiment_name=str(experiment.get("name", "sort_vs_deepsort")),
        seed=int(experiment.get("seed", 42)),
        split=str(experiment.get("split", dataset.get("split", "val"))),
        mot_root=_resolve_path(dataset.get("mot_root"), project_root, "dataset.mot_root"),
        seqmap=_resolve_path(dataset.get("seqmap"), project_root, "dataset.seqmap", required=False),
        detection_cache_config=_resolve_path(
            detections.get("cache_config"),
            project_root,
            "detections.cache_config",
        ),
        detection_cache_root=_resolve_path(
            detections.get("cache_root"),
            project_root,
            "detections.cache_root",
        ),
        confidence_threshold=float(detections.get("confidence_threshold", 0.15)),
        trackers=tuple(trackers),
        trackeval_config=_resolve_path(
            evaluation.get("trackeval_config"),
            project_root,
            "evaluation.trackeval_config",
            required=False,
        ),
        metrics=tuple(
            str(value)
            for value in evaluation.get("metrics", ["HOTA", "CLEAR", "Identity"])
        ),
        allow_partial_sequences=bool(evaluation.get("allow_partial_sequences", False)),
        render_video=bool(benchmark.get("render_video", False)),
        measure_tracker_only=bool(benchmark.get("measure_tracker_only", True)),
        measure_end_to_end_from_cache=bool(benchmark.get("measure_end_to_end_from_cache", True)),
        warmup_sequences=int(benchmark.get("warmup_sequences", 0)),
        output_root=_resolve_path(output.get("root"), project_root, "output.root"),
        tracks_root=_resolve_path(output.get("tracks_root"), project_root, "output.tracks_root"),
        metrics_root=_resolve_path(output.get("metrics_root"), project_root, "output.metrics_root"),
        figures_root=_resolve_path(output.get("figures_root"), project_root, "output.figures_root"),
        videos_root=_resolve_path(
            output.get("videos_root", "outputs/videos/comparison"),
            project_root,
            "output.videos_root",
        ),
        max_sequences=runtime.get("max_sequences"),
        max_frames_per_sequence=runtime.get(
            "max_frames_per_sequence",
            runtime.get("max_frames"),
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        fail_fast=bool(runtime.get("fail_fast", True)),
        log_level=str(runtime.get("log_level", "INFO")),
        smoke_only=bool(runtime.get("smoke_only", False)),
    )
    if config.max_sequences is not None:
        config = replace(config, max_sequences=int(config.max_sequences))
    if config.max_frames_per_sequence is not None:
        config = replace(config, max_frames_per_sequence=int(config.max_frames_per_sequence))
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_config(config)
    return config


def _apply_overrides(
    config: CompareTrackersConfig,
    overrides: dict[str, Any],
) -> CompareTrackersConfig:
    changes: dict[str, Any] = {}
    if overrides.get("split") is not None:
        changes["split"] = str(overrides["split"])
    if overrides.get("confidence") is not None:
        changes["confidence_threshold"] = float(overrides["confidence"])
    if overrides.get("max_sequences") is not None:
        changes["max_sequences"] = int(overrides["max_sequences"])
    if overrides.get("max_frames") is not None:
        changes["max_frames_per_sequence"] = int(overrides["max_frames"])
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config


def _validate_config(config: CompareTrackersConfig) -> None:
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ExperimentConfigError("detections.confidence_threshold must be in [0, 1].")
    if not config.trackers:
        raise ExperimentConfigError("At least one tracker must be configured.")
    if config.max_sequences is not None and config.max_sequences <= 0:
        raise ExperimentConfigError("runtime.max_sequences must be positive when set.")
    if config.max_frames_per_sequence is not None and config.max_frames_per_sequence <= 0:
        raise ExperimentConfigError("runtime.max_frames_per_sequence must be positive when set.")
    if not config.mot_root.is_dir():
        raise ExperimentConfigError(f"dataset.mot_root does not exist: {config.mot_root}")
    if config.seqmap is not None and not config.seqmap.is_file():
        raise ExperimentConfigError(f"dataset.seqmap does not exist: {config.seqmap}")
