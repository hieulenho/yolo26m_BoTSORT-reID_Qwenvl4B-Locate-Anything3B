"""Signals derived from MOT detector confidence when available."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import group_consecutive, make_signal


def detect_confidence_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    confidences = [
        item
        for item in timeline.observations
        if item.target_present and item.tracking_confidence is not None
    ]
    if not confidences:
        return (
            make_signal(
                signal_type="TRACK_CONFIDENCE",
                frame_start=timeline.start_frame,
                frame_end=timeline.end_frame,
                raw_track_id=timeline.current_raw_track_id,
                score=None,
                severity="info",
                threshold=config.confidence_low_threshold,
                triggered=False,
                data_available=False,
                unavailable_reason="mot_confidence_unavailable",
                evidence={
                    "present_frame_count": sum(
                        item.target_present for item in timeline.observations
                    )
                },
            ),
        )
    low_frames = [
        item.frame_index
        for item in confidences
        if item.tracking_confidence is not None
        and item.tracking_confidence < config.confidence_low_threshold
    ]
    signals: list[UncertaintySignal] = []
    for start, end in group_consecutive(low_frames):
        length = end - start + 1
        triggered = length >= config.confidence_consecutive_frames > 0
        minimum = min(
            item.tracking_confidence
            for item in confidences
            if start <= item.frame_index <= end and item.tracking_confidence is not None
        )
        severity = "high" if minimum < config.confidence_low_threshold * 0.5 else "warning"
        signals.append(
            make_signal(
                signal_type="TRACK_CONFIDENCE",
                frame_start=start,
                frame_end=end,
                frame_index=end,
                raw_track_id=timeline.current_raw_track_id,
                score=float(minimum),
                severity=severity,
                threshold=config.confidence_low_threshold,
                triggered=triggered,
                evidence={"consecutive_low_confidence_frames": length},
            )
        )
    return tuple(signals)
