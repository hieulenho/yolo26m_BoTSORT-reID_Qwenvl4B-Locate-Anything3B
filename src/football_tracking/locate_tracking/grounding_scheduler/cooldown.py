"""Cooldown logic for grounding requests."""

from __future__ import annotations

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent


def is_allowed_by_cooldown(
    event: UncertaintyEvent,
    accepted_trigger_frames: tuple[int, ...],
    *,
    cooldown_frames: int,
    critical_overrides: bool,
) -> bool:
    if critical_overrides and event.severity == "critical":
        return True
    return all(
        abs(event.trigger_frame - frame) >= cooldown_frames
        for frame in accepted_trigger_frames
    )
