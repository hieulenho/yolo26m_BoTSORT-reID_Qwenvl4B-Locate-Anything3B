"""Video writing helpers for tracking visualization."""

from __future__ import annotations

from pathlib import Path

from football_tracking.utils.video import ManagedVideoWriter


def create_tracking_video_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
    overwrite: bool = False,
) -> ManagedVideoWriter:
    return ManagedVideoWriter(path, fps=fps, width=width, height=height, overwrite=overwrite)
