"""Semantic margin signals from session-level M3/M4 evidence."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def detect_semantic_margin_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    margins = [
        item.winner_margin
        for item in timeline.observations
        if item.winner_margin is not None
    ]
    if not margins:
        return (
            make_signal(
                signal_type="SEMANTIC_MARGIN",
                frame_start=timeline.start_frame,
                frame_end=timeline.end_frame,
                raw_track_id=timeline.current_raw_track_id,
                score=None,
                severity="info",
                threshold=config.semantic_margin_threshold,
                triggered=False,
                data_available=False,
                unavailable_reason="runner_up_margin_unavailable",
            ),
        )
    margin = float(margins[0])
    triggered = margin < config.semantic_margin_threshold
    return (
        make_signal(
            signal_type="SEMANTIC_MARGIN",
            frame_start=timeline.start_frame,
            frame_end=timeline.end_frame,
            frame_index=timeline.end_frame,
            raw_track_id=timeline.current_raw_track_id,
            score=margin,
            severity="high" if margin < config.semantic_margin_threshold * 0.5 else "warning",
            threshold=config.semantic_margin_threshold,
            triggered=triggered,
            evidence={
                "score_granularity": "session_level",
                "candidate_count": timeline.observations[0].candidate_count
                if timeline.observations
                else None,
            },
        ),
    )
