"""Signals for stale semantic grounding evidence."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def detect_grounding_staleness_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    semantic_frames = [
        item.frame_index
        for item in timeline.observations
        if item.data_availability.get("semantic_score") == "sampled_frame"
    ]
    if not semantic_frames:
        return (
            make_signal(
                signal_type="GROUNDING_STALENESS",
                frame_start=timeline.start_frame,
                frame_end=timeline.end_frame,
                raw_track_id=timeline.current_raw_track_id,
                score=None,
                severity="info",
                threshold=float(config.staleness_warning_frames),
                triggered=False,
                data_available=False,
                unavailable_reason="no_strong_semantic_grounding_frame",
            ),
        )
    last_frame = max(semantic_frames)
    stale_frames = max(0, timeline.end_frame - last_frame)
    return (
        make_signal(
            signal_type="GROUNDING_STALENESS",
            frame_start=last_frame,
            frame_end=timeline.end_frame,
            frame_index=timeline.end_frame,
            raw_track_id=timeline.current_raw_track_id,
            score=float(stale_frames),
            severity="warning",
            threshold=float(config.staleness_warning_frames),
            triggered=stale_frames >= config.staleness_warning_frames > 0,
            evidence={"last_strong_semantic_grounding_frame": last_frame},
        ),
    )
