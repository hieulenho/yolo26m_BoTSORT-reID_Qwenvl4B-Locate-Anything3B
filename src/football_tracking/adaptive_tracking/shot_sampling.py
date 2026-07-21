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
    shot_starts: tuple[int, ...] = ()

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
    probe = VideoProbe(
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frame_count=probe.frame_count,
        shot_starts=tuple(starts),
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
    for group_index, group in enumerate(grouped):
        for candidate in group:
            candidate["shot_index"] = group_index
    representatives = _select_representative_candidates(
        candidates,
        grouped,
        max_keyframes=max_keyframes,
        frame_count=probe.frame_count,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[SampledKeyframe] = []
    selected_capture = cv2.VideoCapture(str(path))
    if not selected_capture.isOpened():
        raise ShotSamplingError(f"Could not reopen video for keyframe export: {path}")
    try:
        for candidate in representatives:
            frame_index = int(candidate["frame_index"])
            shot_index = int(candidate["shot_index"])
            selected_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            ok, frame = selected_capture.read()
            if not ok:
                raise ShotSamplingError(
                    f"Could not read selected keyframe {frame_index} from: {path}"
                )
            frame_path = output / f"shot_{shot_index:03d}_frame_{frame_index:07d}.jpg"
            if not cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise ShotSamplingError(f"Could not write keyframe: {frame_path}")
            records.append(
                SampledKeyframe(
                    frame_index=frame_index,
                    time_seconds=(frame_index - 1) / probe.fps,
                    shot_index=shot_index,
                    quality_score=round(float(candidate["quality"]), 6),
                    transition_score=round(float(candidate["transition"]), 6),
                    path=str(frame_path.resolve()),
                )
            )
    finally:
        selected_capture.release()
    return probe, tuple(records)


def _select_representative_candidates(
    candidates: list[dict[str, Any]],
    grouped: list[list[dict[str, Any]]],
    *,
    max_keyframes: int,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Keep shot coverage, then fill spare slots using farthest temporal sampling."""
    if max_keyframes <= 0:
        raise ValueError("max_keyframes must be positive.")
    if not candidates:
        return []
    shot_representatives = [
        max(group, key=lambda item: (float(item["quality"]), -int(item["frame_index"])))
        for group in grouped
        if group
    ]
    target_count = min(max_keyframes, len(candidates))
    if len(shot_representatives) >= target_count:
        positions = np.linspace(0, len(shot_representatives) - 1, target_count, dtype=int)
        return sorted(
            (shot_representatives[int(index)] for index in positions),
            key=lambda item: int(item["frame_index"]),
        )

    selected = list(shot_representatives)
    selected_frames = {int(item["frame_index"]) for item in selected}
    quality_max = max(float(item["quality"]) for item in candidates) or 1.0
    temporal_scale = max(frame_count - 1, 1)
    while len(selected) < target_count:
        remaining = [
            item for item in candidates if int(item["frame_index"]) not in selected_frames
        ]
        if not remaining:
            break

        def coverage_score(item: dict[str, Any]) -> tuple[float, float, int]:
            frame_index = int(item["frame_index"])
            distance = min(
                abs(frame_index - int(existing["frame_index"])) for existing in selected
            )
            return (
                distance / temporal_scale,
                float(item["quality"]) / quality_max,
                -frame_index,
            )

        chosen = max(remaining, key=coverage_score)
        selected.append(chosen)
        selected_frames.add(int(chosen["frame_index"]))
    return sorted(selected, key=lambda item: int(item["frame_index"]))


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class OnlineShotChangeDetector:
    """Detect hard cuts online without buffering the video."""

    def __init__(
        self,
        *,
        threshold: float = 0.65,
        min_gap_frames: int = 15,
        check_interval_frames: int = 5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if min_gap_frames < 1:
            raise ValueError("min_gap_frames must be positive.")
        if check_interval_frames < 1:
            raise ValueError("check_interval_frames must be positive.")
        self.threshold = float(threshold)
        self.min_gap_frames = int(min_gap_frames)
        self.check_interval_frames = int(check_interval_frames)
        self._previous_hist: np.ndarray | None = None
        self._last_cut_frame = 1

    def update(self, frame_index: int, frame: np.ndarray) -> tuple[bool, float]:
        if (
            self._previous_hist is not None
            and (frame_index - 1) % self.check_interval_frames != 0
        ):
            return False, 0.0
        histogram = _frame_histogram(frame)
        score = (
            0.0
            if self._previous_hist is None
            else float(
                cv2.compareHist(
                    self._previous_hist,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
        )
        self._previous_hist = histogram
        is_cut = (
            frame_index > 1
            and score >= self.threshold
            and frame_index - self._last_cut_frame >= self.min_gap_frames
        )
        if is_cut:
            self._last_cut_frame = frame_index
        return is_cut, score


def _quality_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    return float(np.log1p(sharpness) + exposure)
