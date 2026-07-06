"""CLI helper for rendering semantic target overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.visualization.semantic_target_renderer import (
    render_semantic_target_video,
)


def run_render_semantic_target(
    *,
    source_video: str | Path,
    tracks: str | Path,
    semantic_target: str | Path,
    output: str | Path,
    debug_raw_id: bool,
) -> dict[str, Any]:
    path = render_semantic_target_video(
        source_video=source_video,
        tracks_path=tracks,
        semantic_target_path=semantic_target,
        output_video=output,
        debug_raw_id=debug_raw_id,
    )
    return {"status": "ok", "paths": {"output_video": str(path)}}
