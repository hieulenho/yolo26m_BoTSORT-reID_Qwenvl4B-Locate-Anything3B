"""Reacquisition opportunity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.locate_tracking.benchmark.prediction_loader import (
    load_reacquisition_payload,
)
from football_tracking.locate_tracking.benchmark.query_metrics import QueryFrameMetrics
from football_tracking.locate_tracking.benchmark.schemas import (
    GroundTruthReacquisitionEvent,
    LanguagePrediction,
)


@dataclass(frozen=True)
class ReacquisitionMetricResult:
    opportunity_count: int
    confirmed_success_count: int
    false_reacquisition_count: int
    committed_count: int
    frames_to_reacquire: tuple[int, ...]

    @property
    def success_rate(self) -> float | None:
        return _ratio(self.confirmed_success_count, self.opportunity_count)

    @property
    def false_reacquisition_rate(self) -> float | None:
        return _ratio(self.false_reacquisition_count, self.committed_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_count": self.opportunity_count,
            "confirmed_success_count": self.confirmed_success_count,
            "false_reacquisition_count": self.false_reacquisition_count,
            "committed_count": self.committed_count,
            "reacquisition_success_rate": self.success_rate,
            "false_reacquisition_rate": self.false_reacquisition_rate,
            "frames_to_reacquire": list(self.frames_to_reacquire),
            "mean_frames_to_reacquire": _mean(self.frames_to_reacquire),
        }


def evaluate_reacquisition(
    *,
    opportunities: tuple[GroundTruthReacquisitionEvent, ...],
    prediction: LanguagePrediction | None,
    frame_metrics: QueryFrameMetrics,
) -> ReacquisitionMetricResult:
    if not opportunities:
        return ReacquisitionMetricResult(0, 0, 0, 0, ())
    payload = load_reacquisition_payload(prediction)
    decision = (payload or {}).get("decision", {})
    status = str(decision.get("status", ""))
    committed = status in {"same_raw_id_resumed", "provisional", "confirmed"}
    confirmed = bool(
        (prediction.metadata if prediction else {}).get("reacquisition_confirmed", True)
    )
    frames_to_reacquire: list[int] = []
    success = 0
    false = 0
    committed_count = 0
    correct_frames = {
        item.frame_index
        for item in frame_metrics.frame_results
        if item.correct_count > 0
    }
    for opportunity in opportunities:
        selected_start = decision.get("selected_start_frame")
        if selected_start is not None:
            frames_to_reacquire.append(
                max(0, int(selected_start) - opportunity.gt_reappearance_frame)
            )
        if not committed:
            continue
        committed_count += 1
        window_correct = any(
            frame in correct_frames
            for frame in range(
                opportunity.evaluation_start_frame,
                opportunity.evaluation_end_frame + 1,
            )
        )
        if confirmed and window_correct:
            success += 1
        elif confirmed:
            false += 1
    return ReacquisitionMetricResult(
        opportunity_count=len(opportunities),
        confirmed_success_count=success,
        false_reacquisition_count=false,
        committed_count=committed_count,
        frames_to_reacquire=tuple(frames_to_reacquire),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _mean(values: tuple[int, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
