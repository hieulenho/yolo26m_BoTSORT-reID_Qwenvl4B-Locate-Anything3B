"""Signals for missing intervals in the current raw track timeline."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import group_consecutive, make_signal


def detect_track_gap_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    present_frames = {item.frame_index for item in timeline.observations if item.target_present}
    if not present_frames:
        return ()
    first_present = min(present_frames)
    last_present = max(present_frames)
    internal_missing = [
        frame
        for frame in range(first_present, last_present + 1)
        if frame not in present_frames
    ]
    signals: list[UncertaintySignal] = []
    for start, end in group_consecutive(internal_missing):
        length = end - start + 1
        triggered = length >= config.gap_warning_frames > 0
        severity = "critical" if length >= config.gap_critical_frames > 0 else "warning"
        signals.append(
            make_signal(
                signal_type="TRACK_GAP",
                frame_start=start,
                frame_end=end,
                frame_index=end,
                raw_track_id=timeline.current_raw_track_id,
                score=float(length),
                severity=severity,
                threshold=float(config.gap_warning_frames),
                triggered=triggered,
                evidence={
                    "gap_frame_count": length,
                    "first_present_frame": first_present,
                    "last_present_frame": last_present,
                },
            )
        )
    return tuple(signals)
