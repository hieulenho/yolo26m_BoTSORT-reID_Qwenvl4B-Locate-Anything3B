"""CLI helper for creating a semantic target identity artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.identity.segment_store import save_semantic_target
from football_tracking.locate_tracking.identity.semantic_target import (
    create_initial_semantic_target,
)


def run_init_semantic_target(
    *,
    query: str,
    raw_track_id: int,
    start_frame: int,
    output: str | Path,
    end_frame: int | None = None,
    last_confirmed_frame: int | None = None,
    semantic_memory: str | Path | None = None,
    appearance_reference: str | Path | None = None,
    semantic_target_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = create_initial_semantic_target(
        query=query,
        raw_track_id=raw_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        semantic_memory_path=semantic_memory,
        appearance_reference_path=appearance_reference,
        semantic_target_id=semantic_target_id,
    )
    if last_confirmed_frame is not None:
        target = target.with_updates(
            last_confirmed_frame=last_confirmed_frame,
            last_update_frame=last_confirmed_frame,
        )
    path = save_semantic_target(target, output, overwrite=overwrite)
    return {
        "status": "ok",
        "semantic_target": target.to_dict(),
        "paths": {"semantic_target": str(path)},
    }
