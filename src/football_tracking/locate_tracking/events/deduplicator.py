"""Deterministic overlap deduplication for uncertainty events."""

from __future__ import annotations

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.events.severity import max_severity
from football_tracking.locate_tracking.monitoring.signal_utils import stable_id


def _overlaps(left: UncertaintyEvent, right: UncertaintyEvent) -> bool:
    return left.frame_start <= right.frame_end and right.frame_start <= left.frame_end


def _merge(events: list[UncertaintyEvent]) -> UncertaintyEvent:
    first = events[0]
    frame_start = min(item.frame_start for item in events)
    frame_end = max(item.frame_end for item in events)
    trigger_frame = min(item.trigger_frame for item in events)
    severity = first.severity
    for item in events[1:]:
        severity = max_severity(severity, item.severity)
    signal_ids = tuple(sorted({signal_id for item in events for signal_id in item.signal_ids}))
    payload = {
        "event_type": first.event_type,
        "raw_track_id": first.raw_track_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "signal_ids": signal_ids,
    }
    return UncertaintyEvent(
        event_id=stable_id("event", payload),
        event_type=first.event_type,
        severity=severity,
        frame_start=frame_start,
        frame_end=frame_end,
        trigger_frame=trigger_frame,
        raw_track_id=first.raw_track_id,
        signal_ids=signal_ids,
        score=max((item.score or 0.0 for item in events), default=None),
        evidence={"merged_event_ids": [item.event_id for item in events]},
    )


def deduplicate_events(events: tuple[UncertaintyEvent, ...]) -> tuple[UncertaintyEvent, ...]:
    sorted_events = sorted(
        events,
        key=lambda item: (item.raw_track_id, item.event_type, item.frame_start, item.frame_end),
    )
    merged: list[UncertaintyEvent] = []
    bucket: list[UncertaintyEvent] = []
    for event in sorted_events:
        if (
            bucket
            and event.raw_track_id == bucket[-1].raw_track_id
            and event.event_type == bucket[-1].event_type
            and _overlaps(bucket[-1], event)
        ):
            bucket.append(event)
            continue
        if bucket:
            merged.append(_merge(bucket))
        bucket = [event]
    if bucket:
        merged.append(_merge(bucket))
    return tuple(
        sorted(merged, key=lambda item: (item.frame_start, item.event_type, item.event_id))
    )
