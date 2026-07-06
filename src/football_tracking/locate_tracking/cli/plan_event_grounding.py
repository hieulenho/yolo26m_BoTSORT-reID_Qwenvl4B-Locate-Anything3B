"""CLI helper for planning grounding requests from existing event JSONL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.events.event_store import read_events_jsonl
from football_tracking.locate_tracking.grounding_scheduler.planner import (
    build_grounding_plan,
    save_grounding_plan,
)
from football_tracking.locate_tracking.monitoring.config import load_uncertainty_pipeline_config


def run_plan_event_grounding(
    *,
    config_path: str | Path,
    events_jsonl: str | Path,
    query: str,
    source_video: str | Path | None,
    output: str | Path,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_uncertainty_pipeline_config(config_path, overrides={"overwrite": overwrite})
    events = read_events_jsonl(events_jsonl)
    plan = build_grounding_plan(
        events=events,
        query=query,
        source_video=source_video,
        config=config.scheduler,
    )
    output_path = save_grounding_plan(plan, output, overwrite=overwrite or config.overwrite)
    return {"plan": plan.to_dict(), "paths": {"grounding_plan_json": str(output_path)}}
