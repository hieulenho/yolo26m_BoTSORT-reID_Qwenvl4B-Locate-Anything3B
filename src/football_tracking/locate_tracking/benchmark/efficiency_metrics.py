"""Efficiency and model-call metrics for language tracking benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.locate_tracking.benchmark.schemas import LanguagePrediction


@dataclass(frozen=True)
class EfficiencyMetrics:
    grounding_call_count: int
    eval_frame_count: int
    runtime_seconds: float | None

    @property
    def grounding_calls_per_1000_frames(self) -> float | None:
        if self.eval_frame_count <= 0:
            return None
        return 1000.0 * self.grounding_call_count / self.eval_frame_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding_call_count": self.grounding_call_count,
            "eval_frame_count": self.eval_frame_count,
            "runtime_seconds": self.runtime_seconds,
            "grounding_calls_per_1000_frames": self.grounding_calls_per_1000_frames,
        }


def efficiency_for_prediction(
    prediction: LanguagePrediction | None,
    *,
    eval_frame_count: int,
) -> EfficiencyMetrics:
    return EfficiencyMetrics(
        grounding_call_count=prediction.grounding_call_count if prediction else 0,
        eval_frame_count=eval_frame_count,
        runtime_seconds=prediction.runtime_seconds if prediction else None,
    )
