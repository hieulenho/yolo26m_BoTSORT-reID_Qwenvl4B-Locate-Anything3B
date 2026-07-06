"""CLI helper for Milestone 5 uncertainty analysis and grounding planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.monitoring.config import load_uncertainty_pipeline_config
from football_tracking.locate_tracking.monitoring.service import analyze_and_plan_event_grounding


def run_analyze_target_uncertainty(
    *,
    config_path: str | Path,
    source_video: str | Path,
    tracks: str | Path,
    semantic_memory: str | Path,
    appearance_result: str | Path | None,
    fusion_result: str | Path | None,
    output_dir: str | Path | None,
    current_track_id: int | None,
    start_frame: int | None,
    end_frame: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_uncertainty_pipeline_config(
        config_path,
        overrides={"output_dir": output_dir, "overwrite": overwrite},
    )
    run = analyze_and_plan_event_grounding(
        source_video=source_video,
        tracks_path=tracks,
        semantic_memory_path=semantic_memory,
        appearance_result_path=appearance_result,
        fusion_result_path=fusion_result,
        output_dir=config.output_dir,
        monitoring_config=config.monitoring,
        scheduler_config=config.scheduler,
        current_raw_track_id=current_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        overwrite=config.overwrite,
    )
    return {
        "status": "ok",
        "query": run.assessment.timeline.query,
        "current_raw_track_id": run.assessment.timeline.current_raw_track_id,
        "frame_range": {
            "start": run.assessment.timeline.start_frame,
            "end": run.assessment.timeline.end_frame,
        },
        "aggregate": {
            "severity": run.assessment.aggregate_severity,
            "score": run.assessment.aggregate_score,
            "triggered_signal_count": run.assessment.triggered_signal_count,
            "event_count": len(run.events),
            "planned_grounding_request_count": len(run.grounding_plan.items),
        },
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity,
                "frame_start": event.frame_start,
                "frame_end": event.frame_end,
                "trigger_frame": event.trigger_frame,
                "raw_track_id": event.raw_track_id,
            }
            for event in run.events[:20]
        ],
        "truncated_events": max(0, len(run.events) - 20),
        "paths": {key: str(value) for key, value in run.paths.items()},
    }
