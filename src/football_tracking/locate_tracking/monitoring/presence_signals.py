"""Target presence and absence signals."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import group_consecutive, make_signal


def detect_presence_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    missing_frames = [
        item.frame_index for item in timeline.observations if not item.target_present
    ]
    signals: list[UncertaintySignal] = []
    for start, end in group_consecutive(missing_frames):
        length = end - start + 1
        triggered = length >= config.presence_warning_absent_frames > 0
        severity = "critical" if length >= config.presence_critical_absent_frames > 0 else "warning"
        signals.append(
            make_signal(
                signal_type="TARGET_PRESENCE",
                frame_start=start,
                frame_end=end,
                frame_index=end,
                raw_track_id=timeline.current_raw_track_id,
                score=float(length),
                severity=severity,
                threshold=float(config.presence_warning_absent_frames),
                triggered=triggered,
                evidence={
                    "absent_frame_count": length,
                    "warning_absent_frames": config.presence_warning_absent_frames,
                    "critical_absent_frames": config.presence_critical_absent_frames,
                },
            )
        )
    return tuple(signals)
