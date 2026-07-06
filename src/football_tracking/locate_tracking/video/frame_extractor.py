"""Deterministic 1-based frame extraction for association queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FrameExtractionError(RuntimeError):
    """Raised when an exact requested video frame cannot be extracted."""


@dataclass(frozen=True)
class ExtractedFrame:
    requested_frame_index: int
    video_position_zero_based: int
    image: Any
    width: int
    height: int
    fps: float | None
    total_frames: int | None
    timestamp_seconds: float | None
    source_video: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_frame_index": self.requested_frame_index,
            "video_position_zero_based": self.video_position_zero_based,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "timestamp_seconds": self.timestamp_seconds,
            "source_video": str(self.source_video),
        }


def _one_based_to_zero_based(frame_index: int) -> int:
    if int(frame_index) < 1:
        raise FrameExtractionError("frame_index must be >= 1.")
    return int(frame_index) - 1


def extract_video_frame(source_video: str | Path, frame_index: int) -> ExtractedFrame:
    import cv2  # type: ignore[import-not-found]

    path = Path(source_video)
    if not path.is_file():
        raise FrameExtractionError(f"Video file does not exist: {path}")
    zero_based = _one_based_to_zero_based(frame_index)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FrameExtractionError(f"Could not open video: {path}")
        total_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = total_raw if total_raw > 0 else None
        if total_frames is not None and frame_index > total_frames:
            raise FrameExtractionError(
                f"frame_index {frame_index} exceeds video frame count {total_frames}."
            )
        fps_raw = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_raw if fps_raw > 0.0 else None
        capture.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise FrameExtractionError(f"Could not read exact frame {frame_index} from {path}")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise FrameExtractionError(
                f"Invalid frame dimensions at frame {frame_index}: {frame.shape}"
            )
        timestamp = (zero_based / fps) if fps else None
        return ExtractedFrame(
            requested_frame_index=frame_index,
            video_position_zero_based=zero_based,
            image=frame,
            width=int(width),
            height=int(height),
            fps=fps,
            total_frames=total_frames,
            timestamp_seconds=timestamp,
            source_video=path,
        )
    finally:
        capture.release()


def save_extracted_frame(frame: ExtractedFrame, output_path: str | Path) -> Path:
    import cv2  # type: ignore[import-not-found]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame.image):
        raise FrameExtractionError(f"Could not write extracted frame: {path}")
    return path
