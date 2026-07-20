"""Shot-aware keyframe sampling for offline discovery and stream bootstrap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ShotSamplingError(RuntimeError):
    """Raised when representative frames cannot be sampled."""


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class SampledKeyframe:
    frame_index: int
    time_seconds: float
    shot_index: int
    quality_score: float
    transition_score: float
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_shot_starts(
    frame_indices: list[int],
    transition_scores: list[float],
    *,
    threshold: float,
    min_gap_frames: int,
) -> list[int]:
    """Return sampled-frame positions that begin shots."""
    if len(frame_indices) != len(transition_scores):
        raise ValueError("frame_indices and transition_scores must have equal length.")
    if not frame_indices:
        return []
    starts = [frame_indices[0]]
    for frame_index, score in zip(frame_indices[1:], transition_scores[1:], strict=True):
        if score >= threshold and frame_index - starts[-1] >= min_gap_frames:
            starts.append(frame_index)
    return starts


def sample_shot_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    max_keyframes: int = 8,
    sample_fps: float = 2.0,
    transition_threshold: float = 0.45,
    min_shot_seconds: float = 0.75,
    jpeg_quality: int = 92,
) -> tuple[VideoProbe, tuple[SampledKeyframe, ...]]:
    """Sample one high-quality representative frame from each detected shot."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ShotSamplingError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ShotSamplingError(f"Invalid video metadata: {path}")
        probe = VideoProbe(width=width, height=height, fps=fps, frame_count=frame_count)
        stride = max(int(round(fps / max(sample_fps, 0.1))), 1)
        candidates: list[dict[str, Any]] = []
        previous_hist: np.ndarray | None = None
        for frame_index in range(1, frame_count + 1, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            ok, frame = capture.read()
            if not ok:
                continue
            hist = _frame_histogram(frame)
            transition = (
                0.0
                if previous_hist is None
                else float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            )
            candidates.append(
                {
                    "frame_index": frame_index,
                    "frame": frame,
                    "transition": transition,
                    "quality": _quality_score(frame),
                }
            )
            previous_hist = hist
    finally:
        capture.release()
    if not candidates:
        raise ShotSamplingError(f"No frames could be sampled from: {path}")

    starts = detect_shot_starts(
        [int(item["frame_index"]) for item in candidates],
        [float(item["transition"]) for item in candidates],
        threshold=transition_threshold,
        min_gap_frames=max(int(round(min_shot_seconds * probe.fps)), 1),
    )
    grouped: list[list[dict[str, Any]]] = [[] for _ in starts]
    shot_index = 0
    for candidate in candidates:
        while (
            shot_index + 1 < len(starts)
            and int(candidate["frame_index"]) >= starts[shot_index + 1]
        ):
            shot_index += 1
        grouped[shot_index].append(candidate)
    representatives = [max(group, key=lambda item: item["quality"]) for group in grouped if group]
    if len(representatives) > max_keyframes:
        positions = np.linspace(0, len(representatives) - 1, max_keyframes, dtype=int)
        representatives = [representatives[int(index)] for index in positions]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[SampledKeyframe] = []
    for selected_index, candidate in enumerate(representatives):
        frame_index = int(candidate["frame_index"])
        frame_path = output / f"shot_{selected_index:03d}_frame_{frame_index:07d}.jpg"
        if not cv2.imwrite(
            str(frame_path),
            candidate["frame"],
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        ):
            raise ShotSamplingError(f"Could not write keyframe: {frame_path}")
        records.append(
            SampledKeyframe(
                frame_index=frame_index,
                time_seconds=(frame_index - 1) / probe.fps,
                shot_index=selected_index,
                quality_score=round(float(candidate["quality"]), 6),
                transition_score=round(float(candidate["transition"]), 6),
                path=str(frame_path.resolve()),
            )
        )
    return probe, tuple(records)


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _quality_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    return float(np.log1p(sharpness) + exposure)
