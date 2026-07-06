"""CLI helper for M6 candidate search and no-commit ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.reacquisition.config import load_reacquisition_config
from football_tracking.locate_tracking.reacquisition.service import run_reacquisition


def run_search_reacquisition_candidates(
    *,
    config_path: str | Path,
    semantic_target: str | Path,
    tracks: str | Path,
    events: str | Path,
    output_dir: str | Path | None,
    grounding_plan: str | Path | None,
    grounding_manifest: str | Path | None,
    appearance_result: str | Path | None,
    event_id: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_reacquisition_config(
        config_path,
        overrides={"output_dir": output_dir, "overwrite": overwrite},
    )
    run = run_reacquisition(
        semantic_target_path=semantic_target,
        tracks_path=tracks,
        events_path=events,
        output_dir=config.output_dir,
        config=config.reacquisition,
        grounding_plan_path=grounding_plan,
        grounding_manifest_path=grounding_manifest,
        appearance_result_path=appearance_result,
        event_id=event_id,
        commit=False,
        overwrite=config.overwrite,
    )
    return {
        "status": "ok",
        "decision": run.decision.to_dict(),
        "candidate_count": len(run.candidates),
        "paths": {key: str(value) for key, value in run.paths.items()},
    }
