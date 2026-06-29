"""Timing helpers for detector baseline runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def safe_fps(image_count: int, seconds: float) -> float | None:
    if image_count <= 0 or seconds <= 0.0:
        return None
    return image_count / seconds


def maybe_synchronize_cuda(device: str) -> None:
    if not str(device).lower().startswith(("cuda", "0", "1", "2", "3")):
        return
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return


@dataclass
class TimingStats:
    model_load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocessing_seconds: float = 0.0
    serialization_seconds: float = 0.0
    total_pipeline_seconds: float = 0.0

    def detector_fps(self, image_count: int) -> float | None:
        return safe_fps(image_count, self.inference_seconds)

    def end_to_end_fps(self, image_count: int) -> float | None:
        return safe_fps(image_count, self.total_pipeline_seconds)

    def latency_per_image_seconds(self, image_count: int) -> float | None:
        if image_count <= 0:
            return None
        return self.inference_seconds / image_count

    def to_dict(self, image_count: int) -> dict[str, Any]:
        return {
            "model_load_seconds": self.model_load_seconds,
            "warmup_seconds": self.warmup_seconds,
            "preprocessing_seconds": self.preprocessing_seconds,
            "inference_seconds": self.inference_seconds,
            "postprocessing_seconds": self.postprocessing_seconds,
            "serialization_seconds": self.serialization_seconds,
            "total_pipeline_seconds": self.total_pipeline_seconds,
            "latency_per_image_seconds": self.latency_per_image_seconds(image_count),
            "detector_fps": self.detector_fps(image_count),
            "end_to_end_fps": self.end_to_end_fps(image_count),
        }


class Timer:
    def __enter__(self) -> Timer:
        self.started = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self.started
