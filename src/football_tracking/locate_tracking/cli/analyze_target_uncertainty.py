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
    return run.to_dict()
