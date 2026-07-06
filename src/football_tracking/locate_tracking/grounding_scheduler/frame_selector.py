"""Select video frames for event-triggered grounding."""

from __future__ import annotations

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent


def select_event_frames(
    event: UncertaintyEvent,
    *,
    strategy: str,
    max_frames: int,
) -> tuple[int, ...]:
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1.")
    if strategy == "trigger":
        return (event.trigger_frame,)
    if strategy == "strongest_frame":
        return (event.trigger_frame,)
    if strategy != "window_representative":
        raise ValueError(f"Unknown frame selection strategy: {strategy}")
    if max_frames == 1 or event.frame_start == event.frame_end:
        return (event.trigger_frame,)
    if max_frames == 2:
        return tuple(sorted({event.frame_start, event.frame_end}))
    if max_frames >= 3:
        middle = (event.frame_start + event.frame_end) // 2
        return tuple(sorted({event.frame_start, middle, event.frame_end}))[:max_frames]
    return (event.trigger_frame,)
