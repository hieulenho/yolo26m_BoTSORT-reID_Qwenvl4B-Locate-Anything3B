"""Lifecycle mapping for DeepSORT tracks."""

from __future__ import annotations

from typing import Any, Literal

TrackState = Literal["tentative", "confirmed", "lost", "deleted"]


def track_lifecycle_state(track: Any) -> TrackState:
    if hasattr(track, "is_deleted") and track.is_deleted():
        return "deleted"
    if hasattr(track, "is_tentative") and track.is_tentative():
        return "tentative"
    if hasattr(track, "is_confirmed") and track.is_confirmed():
        time_since_update = int(getattr(track, "time_since_update", 0))
        return "confirmed" if time_since_update <= 0 else "lost"
    return "lost"


def should_emit_track(
    state: TrackState,
    confirmed_only: bool,
    require_recent_update: bool,
    time_since_update: int,
    max_time_since_update: int,
) -> bool:
    if state == "deleted":
        return False
    if confirmed_only and state != "confirmed":
        return False
    if require_recent_update and time_since_update > max_time_since_update:
        return False
    return True
