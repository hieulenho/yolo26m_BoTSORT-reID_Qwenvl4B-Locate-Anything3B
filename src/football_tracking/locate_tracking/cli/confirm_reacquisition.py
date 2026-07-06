"""CLI helper for confirming a probationary reacquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.reacquisition.config import load_reacquisition_config
from football_tracking.locate_tracking.reacquisition.service import confirm_reacquisition_probation


def run_confirm_reacquisition(
    *,
    config_path: str | Path,
    semantic_target: str | Path,
    tracks: str | Path,
    decision: str | Path,
    output_dir: str | Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_reacquisition_config(
        config_path,
        overrides={"output_dir": output_dir, "overwrite": overwrite},
    )
    target = confirm_reacquisition_probation(
        semantic_target_path=semantic_target,
        tracks_path=tracks,
        decision_path=decision,
        output_dir=config.output_dir,
        config=config.reacquisition,
        overwrite=config.overwrite,
    )
    return {"status": "ok", "semantic_target": target.to_dict()}
