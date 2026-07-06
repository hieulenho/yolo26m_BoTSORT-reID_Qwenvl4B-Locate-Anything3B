"""Priority ordering for uncertainty events."""

from __future__ import annotations

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.monitoring.signal_utils import SEVERITY_RANK


def sort_events_by_priority(
    events: tuple[UncertaintyEvent, ...],
    *,
    event_type_priority: tuple[str, ...] = (),
) -> tuple[UncertaintyEvent, ...]:
    type_rank = {event_type: index for index, event_type in enumerate(event_type_priority)}
    fallback = len(type_rank)
    return tuple(
        sorted(
            events,
            key=lambda event: (
                -SEVERITY_RANK[event.severity],
                type_rank.get(event.event_type, fallback),
                event.frame_start,
                event.event_id,
            ),
        )
    )
