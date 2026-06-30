"""Timing helpers for tracking runs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


def maybe_synchronize_cuda(device: str) -> None:
    normalized = str(device).lower()
    if normalized == "cpu":
        return
    try:
        import torch  # type: ignore[import-not-found]

        wants_cuda = normalized == "cuda" or normalized.startswith("cuda") or normalized.isdigit()
        if torch.cuda.is_available() and wants_cuda:
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return


@dataclass
class TrackingTiming:
    model_load_seconds: float = 0.0
    detector_warmup_seconds: float = 0.0
    frame_read_seconds: float = 0.0
    detector_seconds: float = 0.0
    detector_postprocess_seconds: float = 0.0
    tracker_seconds: float = 0.0
    rendering_seconds: float = 0.0
    video_write_seconds: float = 0.0
    total_pipeline_seconds: float = 0.0
    processed_frames: int = 0

    def fps(self, seconds: float) -> float | None:
        if self.processed_frames <= 0 or seconds <= 0.0:
            return None
        return self.processed_frames / seconds

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "model_load_seconds": self.model_load_seconds,
            "detector_warmup_seconds": self.detector_warmup_seconds,
            "frame_read_seconds": self.frame_read_seconds,
            "detector_seconds": self.detector_seconds,
            "detector_postprocess_seconds": self.detector_postprocess_seconds,
            "tracker_seconds": self.tracker_seconds,
            "rendering_seconds": self.rendering_seconds,
            "video_write_seconds": self.video_write_seconds,
            "total_pipeline_seconds": self.total_pipeline_seconds,
            "processed_frames": self.processed_frames,
            "detector_fps": self.fps(self.detector_seconds),
            "tracker_fps": self.fps(self.tracker_seconds),
            "end_to_end_fps": self.fps(self.total_pipeline_seconds),
        }


@contextmanager
def timed_section(
    timing: TrackingTiming,
    field_name: str,
    device: str = "cpu",
    synchronize_cuda: bool = False,
) -> Iterator[None]:
    if synchronize_cuda:
        maybe_synchronize_cuda(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if synchronize_cuda:
            maybe_synchronize_cuda(device)
        setattr(timing, field_name, getattr(timing, field_name) + time.perf_counter() - started)
