"""Convert triggered uncertainty signals into structured events."""

from __future__ import annotations

from football_tracking.locate_tracking.events.schemas import (
    SIGNAL_TO_EVENT_TYPE,
    UncertaintyEvent,
)
from football_tracking.locate_tracking.monitoring.schemas import UncertaintySignal
from football_tracking.locate_tracking.monitoring.signal_utils import stable_id


def event_from_signal(signal: UncertaintySignal) -> UncertaintyEvent | None:
    if not signal.triggered:
        return None
    event_type = SIGNAL_TO_EVENT_TYPE[signal.signal_type]
    trigger_frame = signal.frame_index if signal.frame_index is not None else signal.frame_end
    payload = {
        "event_type": event_type,
        "frame_start": signal.frame_start,
        "frame_end": signal.frame_end,
        "trigger_frame": trigger_frame,
        "raw_track_id": signal.raw_track_id,
        "signal_ids": [signal.signal_id],
    }
    return UncertaintyEvent(
        event_id=stable_id("event", payload),
        event_type=event_type,
        severity=signal.severity_contribution,
        frame_start=signal.frame_start,
        frame_end=signal.frame_end,
        trigger_frame=trigger_frame,
        raw_track_id=signal.raw_track_id,
        signal_ids=(signal.signal_id,),
        score=signal.score,
        evidence={
            "signal_type": signal.signal_type,
            "threshold": signal.threshold,
            "signal_evidence": dict(signal.evidence),
        },
    )


def detect_uncertainty_events(
    signals: tuple[UncertaintySignal, ...],
) -> tuple[UncertaintyEvent, ...]:
    events = [event for signal in signals if (event := event_from_signal(signal)) is not None]
    return tuple(
        sorted(events, key=lambda item: (item.frame_start, item.event_type, item.event_id))
    )
