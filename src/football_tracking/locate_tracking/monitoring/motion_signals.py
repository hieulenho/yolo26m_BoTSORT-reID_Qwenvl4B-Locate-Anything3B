"""Motion-jump signals computed from observed boxes only."""

from __future__ import annotations

import math
from statistics import median

from football_tracking.locate_tracking.association.geometry import bbox_center
from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def _frame_diagonal(timeline: TargetObservationTimeline) -> float:
    width = timeline.metadata.get("frame_width")
    height = timeline.metadata.get("frame_height")
    if width and height:
        return math.hypot(float(width), float(height))
    max_x = max(
        (item.bbox_xyxy[2] for item in timeline.observations if item.bbox_xyxy),
        default=1.0,
    )
    max_y = max(
        (item.bbox_xyxy[3] for item in timeline.observations if item.bbox_xyxy),
        default=1.0,
    )
    return max(1.0, math.hypot(max_x, max_y))


def detect_motion_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    present = [item for item in timeline.observations if item.target_present and item.bbox_xyxy]
    if len(present) < 2:
        return ()
    diagonal = _frame_diagonal(timeline)
    previous = present[0]
    normalized_history: list[float] = []
    signals: list[UncertaintySignal] = []
    for current in present[1:]:
        previous_center = bbox_center(previous.bbox_xyxy)  # type: ignore[arg-type]
        current_center = bbox_center(current.bbox_xyxy)  # type: ignore[arg-type]
        displacement_px = math.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        normalized = displacement_px / diagonal
        baseline_values = normalized_history[-config.motion_baseline_window :]
        baseline = median(baseline_values) if baseline_values else None
        ratio = normalized / baseline if baseline and baseline > 0.0 else None
        absolute_trigger = normalized >= config.motion_jump_threshold
        relative_trigger = (
            ratio is not None and ratio >= config.motion_jump_ratio_threshold and normalized > 0.0
        )
        if absolute_trigger or relative_trigger:
            signals.append(
                make_signal(
                    signal_type="MOTION_JUMP",
                    frame_start=previous.frame_index,
                    frame_end=current.frame_index,
                    frame_index=current.frame_index,
                    raw_track_id=timeline.current_raw_track_id,
                    score=normalized,
                    severity=(
                        "high"
                        if normalized >= config.motion_jump_threshold * 2
                        else "warning"
                    ),
                    threshold=config.motion_jump_threshold,
                    triggered=True,
                    evidence={
                        "previous_frame": previous.frame_index,
                        "current_frame": current.frame_index,
                        "displacement_px": displacement_px,
                        "displacement_normalized": normalized,
                        "median_baseline": baseline,
                        "jump_ratio": ratio,
                    },
                )
            )
        normalized_history.append(normalized)
        previous = current
    return tuple(signals)
