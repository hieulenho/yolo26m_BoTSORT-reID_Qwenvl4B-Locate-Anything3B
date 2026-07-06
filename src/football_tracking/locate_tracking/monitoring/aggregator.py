"""Aggregate independent uncertainty signals without taking tracking actions."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringAssessment,
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import (
    SEVERITY_RANK,
    highest_severity,
)


def aggregate_uncertainty(
    *,
    timeline: TargetObservationTimeline,
    signals: tuple[UncertaintySignal, ...],
    config: MonitoringConfig,
) -> MonitoringAssessment:
    triggered = tuple(item for item in signals if item.triggered)
    weighted_score = 0.0
    total_weight = 0.0
    for signal in triggered:
        weight = config.weight_for(signal.signal_type)
        contribution = SEVERITY_RANK[signal.severity_contribution] / 3.0
        weighted_score += weight * contribution
        total_weight += weight
    aggregate_score = weighted_score / total_weight if total_weight > 0.0 else 0.0
    return MonitoringAssessment(
        timeline=timeline,
        signals=signals,
        aggregate_score=aggregate_score,
        aggregate_severity=highest_severity(triggered),
        triggered_signal_count=len(triggered),
        metadata={
            "triggered_signal_ids": [item.signal_id for item in triggered],
            "note": "diagnostic uncertainty only; no raw-ID aliasing or reacquisition performed",
        },
    )
