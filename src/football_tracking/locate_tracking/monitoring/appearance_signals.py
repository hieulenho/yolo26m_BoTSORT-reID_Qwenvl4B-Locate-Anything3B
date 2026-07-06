"""Appearance drift signals from immutable M4 verification artifacts."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def detect_appearance_drift_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    scores = [
        item.appearance_score
        for item in timeline.observations
        if item.appearance_score is not None
    ]
    if not scores:
        return (
            make_signal(
                signal_type="APPEARANCE_DRIFT",
                frame_start=timeline.start_frame,
                frame_end=timeline.end_frame,
                raw_track_id=timeline.current_raw_track_id,
                score=None,
                severity="info",
                threshold=config.appearance_drift_threshold,
                triggered=False,
                data_available=False,
                unavailable_reason="appearance_score_unavailable",
            ),
        )
    score = float(scores[0])
    triggered = score < config.appearance_drift_threshold
    return (
        make_signal(
            signal_type="APPEARANCE_DRIFT",
            frame_start=timeline.start_frame,
            frame_end=timeline.end_frame,
            frame_index=timeline.end_frame,
            raw_track_id=timeline.current_raw_track_id,
            score=score,
            severity="high" if score < config.appearance_drift_threshold * 0.5 else "warning",
            threshold=config.appearance_drift_threshold,
            triggered=triggered,
            evidence={"score_granularity": "session_level", "prototype_updated": False},
        ),
    )
