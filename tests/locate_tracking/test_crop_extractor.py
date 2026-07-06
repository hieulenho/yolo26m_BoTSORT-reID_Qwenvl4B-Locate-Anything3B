from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_tracking.locate_tracking.appearance.crop_extractor import (
    CropExtractionError,
    TrackCropExtractor,
)
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation


def _observation(bbox: tuple[float, float, float, float]) -> MotTrackObservation:
    x1, y1, x2, y2 = bbox
    return MotTrackObservation(
        frame_index=1,
        track_id=7,
        bbox_ltwh=(x1, y1, x2 - x1, y2 - y1),
        bbox_xyxy=bbox,
        confidence=None,
        source_path=Path("tracks.txt"),
        line_number=1,
    )


def test_crop_extractor_uses_source_frame_and_clips_bbox() -> None:
    frame = np.ones((32, 32, 3), dtype=np.uint8) * 9
    extractor = TrackCropExtractor()

    crop = extractor.extract(
        frame=frame,
        observation=_observation((-5, -5, 10, 10)),
        source_video="source.mp4",
    )

    assert crop.image.shape[:2] == (10, 10)
    assert crop.reference.source_video == "source.mp4"
    assert crop.reference.clipped_bbox_xyxy == (0.0, 0.0, 10.0, 10.0)


def test_crop_extractor_rejects_fully_outside_bbox() -> None:
    with pytest.raises(CropExtractionError):
        TrackCropExtractor().extract(
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            observation=_observation((40, 40, 50, 50)),
            source_video="source.mp4",
        )
