"""Helpers shared by uncertainty signal detectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from football_tracking.locate_tracking.monitoring.schemas import (
    Severity,
    SignalType,
    TargetFrameObservation,
    UncertaintySignal,
)

SEVERITY_RANK: dict[Severity, int] = {
    "info": 0,
    "warning": 1,
    "high": 2,
    "critical": 3,
}


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def make_signal(
    *,
    signal_type: SignalType,
    frame_start: int,
    frame_end: int,
    raw_track_id: int,
    score: float | None,
    severity: Severity,
    threshold: float | None,
    triggered: bool,
    evidence: dict[str, Any] | None = None,
    frame_index: int | None = None,
    data_available: bool = True,
    unavailable_reason: str | None = None,
) -> UncertaintySignal:
    payload = {
        "signal_type": signal_type,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "raw_track_id": raw_track_id,
        "score": score,
        "threshold": threshold,
        "triggered": triggered,
        "evidence": evidence or {},
        "data_available": data_available,
        "unavailable_reason": unavailable_reason,
    }
    return UncertaintySignal(
        signal_id=stable_id("signal", payload),
        signal_type=signal_type,
        frame_index=frame_index,
        frame_start=frame_start,
        frame_end=frame_end,
        raw_track_id=raw_track_id,
        score=score,
        severity_contribution=severity,
        threshold=threshold,
        triggered=triggered,
        evidence=evidence or {},
        data_available=data_available,
        unavailable_reason=unavailable_reason,
    )


def group_consecutive(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return tuple(ranges)


def highest_severity(signals: Iterable[UncertaintySignal]) -> Severity:
    severity: Severity = "info"
    for signal in signals:
        if (
            signal.triggered
            and SEVERITY_RANK[signal.severity_contribution] > SEVERITY_RANK[severity]
        ):
            severity = signal.severity_contribution
    return severity


def present_observations(
    observations: Iterable[TargetFrameObservation],
) -> tuple[TargetFrameObservation, ...]:
    return tuple(
        item for item in observations if item.target_present and item.bbox_xyxy is not None
    )
