"""Typed experiment records for tracker comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACKING_METRIC_NAMES = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "IDSW",
    "FP",
    "FN",
    "Frag",
)


@dataclass(frozen=True)
class TrackerSpec:
    name: str
    config: Path

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "config": str(self.config)}


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    experiment_name: str
    tracker_name: str
    tracker_config: Path
    tracker_config_hash: str | None
    detector_cache_root: Path
    detection_cache_hashes: dict[str, str | None]
    confidence_threshold: float
    split: str
    sequences: list[str]
    partial: bool
    smoke_only: bool
    output_directory: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "tracker_name": self.tracker_name,
            "tracker_config": str(self.tracker_config),
            "tracker_config_hash": self.tracker_config_hash,
            "detector_cache_root": str(self.detector_cache_root),
            "detection_cache_hashes": self.detection_cache_hashes,
            "confidence_threshold": self.confidence_threshold,
            "split": self.split,
            "sequences": self.sequences,
            "partial": self.partial,
            "smoke_only": self.smoke_only,
            "output_directory": str(self.output_directory),
        }


@dataclass
class ExperimentResult:
    experiment_id: str
    tracker_name: str
    status: str
    sequence_count: int
    frame_count: int
    detection_count: int
    emitted_track_count: int
    unique_track_count: int
    tracker_seconds: float
    frame_read_seconds: float
    cache_read_seconds: float
    mot_write_seconds: float
    total_seconds: float
    tracker_fps: float | None
    cached_pipeline_fps: float | None
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        metric_values = {name: self.metrics.get(name) for name in TRACKING_METRIC_NAMES}
        return {
            "experiment_id": self.experiment_id,
            "tracker_name": self.tracker_name,
            "status": self.status,
            "sequence_count": self.sequence_count,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "emitted_track_count": self.emitted_track_count,
            "unique_track_count": self.unique_track_count,
            "tracker_seconds": self.tracker_seconds,
            "frame_read_seconds": self.frame_read_seconds,
            "cache_read_seconds": self.cache_read_seconds,
            "mot_write_seconds": self.mot_write_seconds,
            "total_seconds": self.total_seconds,
            "tracker_fps": self.tracker_fps,
            "cached_pipeline_fps": self.cached_pipeline_fps,
            **metric_values,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }
