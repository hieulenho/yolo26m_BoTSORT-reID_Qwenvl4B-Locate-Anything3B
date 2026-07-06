from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from football_tracking.locate_tracking.video.frame_extractor import (
    FrameExtractionError,
    extract_video_frame,
)


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 16),
    )
    assert writer.isOpened()
    for value in (30, 120, 220):
        frame = np.full((16, 16, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _mean(frame) -> float:
    return float(np.mean(frame.image))


def test_frame_extractor_one_based_indexing(tmp_path: Path) -> None:
    video = _video(tmp_path / "tiny.avi")

    first = extract_video_frame(video, 1)
    second = extract_video_frame(video, 2)
    third = extract_video_frame(video, 3)

    assert first.video_position_zero_based == 0
    assert second.video_position_zero_based == 1
    assert third.video_position_zero_based == 2
    assert _mean(first) < _mean(second) < _mean(third)


def test_frame_extractor_reports_dimensions(tmp_path: Path) -> None:
    frame = extract_video_frame(_video(tmp_path / "tiny.avi"), 1)

    assert frame.width == 16
    assert frame.height == 16
    assert frame.total_frames == 3
    assert frame.fps == 5.0


@pytest.mark.parametrize("frame_index", [0, -1, 4])
def test_frame_extractor_rejects_bad_frame_indices(tmp_path: Path, frame_index: int) -> None:
    with pytest.raises(FrameExtractionError):
        extract_video_frame(_video(tmp_path / "tiny.avi"), frame_index)


def test_frame_extractor_rejects_unreadable_video(tmp_path: Path) -> None:
    with pytest.raises(FrameExtractionError):
        extract_video_frame(tmp_path / "missing.mp4", 1)
