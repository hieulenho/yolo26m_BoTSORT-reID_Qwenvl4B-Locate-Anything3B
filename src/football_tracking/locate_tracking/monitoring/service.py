"""End-to-end uncertainty monitoring service for a selected semantic target."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from football_tracking.locate_tracking.events.deduplicator import deduplicate_events
from football_tracking.locate_tracking.events.event_detector import detect_uncertainty_events
from football_tracking.locate_tracking.events.event_store import write_events_jsonl
from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.grounding_scheduler.planner import (
    build_grounding_plan,
    save_grounding_plan,
)
from football_tracking.locate_tracking.grounding_scheduler.schemas import GroundingPlan
from football_tracking.locate_tracking.monitoring.aggregator import aggregate_uncertainty
from football_tracking.locate_tracking.monitoring.appearance_signals import (
    detect_appearance_drift_signals,
)
from football_tracking.locate_tracking.monitoring.confidence_signals import (
    detect_confidence_signals,
)
from football_tracking.locate_tracking.monitoring.gap_signals import detect_track_gap_signals
from football_tracking.locate_tracking.monitoring.motion_signals import detect_motion_signals
from football_tracking.locate_tracking.monitoring.neighborhood_signals import (
    detect_neighbor_ambiguity_signals,
)
from football_tracking.locate_tracking.monitoring.presence_signals import detect_presence_signals
from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringAssessment,
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.semantic_signals import (
    detect_semantic_margin_signals,
)
from football_tracking.locate_tracking.monitoring.staleness_signals import (
    detect_grounding_staleness_signals,
)
from football_tracking.locate_tracking.monitoring.target_observer import (
    build_target_observation_timeline_from_paths,
)
from football_tracking.locate_tracking.visualization.uncertainty_timeline import (
    write_uncertainty_summary,
)


class UncertaintyMonitoringServiceError(RuntimeError):
    """Raised when uncertainty monitoring cannot be completed."""


@dataclass(frozen=True)
class UncertaintyMonitoringRun:
    assessment: MonitoringAssessment
    events: tuple[UncertaintyEvent, ...]
    grounding_plan: GroundingPlan
    paths: dict[str, Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "assessment": self.assessment.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "grounding_plan": self.grounding_plan.to_dict(),
            "paths": {key: str(value) for key, value in self.paths.items()},
        }


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_all_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    signal_groups = (
        detect_presence_signals(timeline, config),
        detect_confidence_signals(timeline, config),
        detect_motion_signals(timeline, config),
        detect_semantic_margin_signals(timeline, config),
        detect_appearance_drift_signals(timeline, config),
        detect_neighbor_ambiguity_signals(timeline, config),
        detect_track_gap_signals(timeline, config),
        detect_grounding_staleness_signals(timeline, config),
    )
    return tuple(signal for group in signal_groups for signal in group)


def save_assessment(
    assessment: MonitoringAssessment,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise UncertaintyMonitoringServiceError(
            f"Assessment output exists and overwrite=false: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment.to_dict(), indent=2, default=str), encoding="utf-8")
    return output


def save_timeline(
    timeline: TargetObservationTimeline,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise UncertaintyMonitoringServiceError(
            f"Timeline output exists and overwrite=false: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timeline.to_dict(), indent=2, default=str), encoding="utf-8")
    return output


def analyze_and_plan_event_grounding(
    *,
    source_video: str | Path,
    tracks_path: str | Path,
    semantic_memory_path: str | Path,
    appearance_result_path: str | Path | None,
    fusion_result_path: str | Path | None,
    output_dir: str | Path,
    monitoring_config: MonitoringConfig,
    scheduler_config,
    current_raw_track_id: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    overwrite: bool = False,
) -> UncertaintyMonitoringRun:
    timeline = build_target_observation_timeline_from_paths(
        tracks_path=tracks_path,
        semantic_memory_path=semantic_memory_path,
        appearance_result_path=appearance_result_path,
        fusion_result_path=fusion_result_path,
        current_raw_track_id=current_raw_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        source_video=source_video,
    )
    artifact_hashes = {
        "tracks_sha256": _sha256(tracks_path),
        "semantic_memory_sha256": _sha256(semantic_memory_path),
        "appearance_result_sha256": _sha256(appearance_result_path),
        "fusion_result_sha256": _sha256(fusion_result_path),
    }
    timeline = TargetObservationTimeline(
        query=timeline.query,
        semantic_target_hypothesis=timeline.semantic_target_hypothesis,
        current_raw_track_id=timeline.current_raw_track_id,
        start_frame=timeline.start_frame,
        end_frame=timeline.end_frame,
        observations=timeline.observations,
        metadata={**timeline.metadata, "input_hashes": artifact_hashes},
    )
    signals = detect_all_signals(timeline, monitoring_config)
    assessment = aggregate_uncertainty(timeline=timeline, signals=signals, config=monitoring_config)
    events = deduplicate_events(detect_uncertainty_events(signals))
    plan = build_grounding_plan(
        events=events,
        query=timeline.query,
        source_video=source_video,
        config=scheduler_config,
    )
    output = Path(output_dir)
    paths = {
        "timeline_json": save_timeline(
            timeline,
            output / "target_observation_timeline.json",
            overwrite=overwrite,
        ),
        "assessment_json": save_assessment(
            assessment,
            output / "uncertainty_assessment.json",
            overwrite=overwrite,
        ),
        "events_jsonl": write_events_jsonl(
            events,
            output / "uncertainty_events.jsonl",
            overwrite=overwrite,
        ),
        "grounding_plan_json": save_grounding_plan(
            plan,
            output / "grounding_plan.json",
            overwrite=overwrite,
        ),
    }
    paths["summary_md"] = write_uncertainty_summary(
        assessment=assessment,
        events=events,
        grounding_plan=plan,
        output_path=output / "uncertainty_summary.md",
        overwrite=overwrite,
    )
    return UncertaintyMonitoringRun(
        assessment=assessment,
        events=events,
        grounding_plan=plan,
        paths=paths,
    )
