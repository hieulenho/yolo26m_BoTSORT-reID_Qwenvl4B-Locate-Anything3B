"""Extract clean track crops from original source video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_tracking.locate_tracking.appearance.crop_quality import (
    CropQualityConfig,
    evaluate_crop_quality,
)
from football_tracking.locate_tracking.appearance.schemas import CropReference
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation


class CropExtractionError(RuntimeError):
    """Raised when a track crop cannot be extracted."""


@dataclass(frozen=True)
class TrackCrop:
    reference: CropReference
    image: np.ndarray


def _clip_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox_xyxy
    clipped = (
        max(0.0, min(float(frame_width), x1)),
        max(0.0, min(float(frame_height), y1)),
        max(0.0, min(float(frame_width), x2)),
        max(0.0, min(float(frame_height), y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


class TrackCropExtractor:
    def __init__(self, quality_config: CropQualityConfig | None = None) -> None:
        self.quality_config = quality_config or CropQualityConfig()

    def extract(
        self,
        *,
        frame: np.ndarray,
        observation: MotTrackObservation,
        source_video: str | Path,
    ) -> TrackCrop:
        if frame.ndim < 2:
            raise CropExtractionError("source frame must have image dimensions.")
        frame_height, frame_width = frame.shape[:2]
        raw_bbox = observation.bbox_xyxy
        clipped_bbox = _clip_bbox(raw_bbox, frame_width, frame_height)
        if clipped_bbox is None:
            raise CropExtractionError(
                f"Track {observation.track_id} at frame {observation.frame_index} is outside frame."
            )
        x1, y1, x2, y2 = clipped_bbox
        raw_area = max(0.0, (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1]))
        clipped_area = max(0.0, (x2 - x1) * (y2 - y1))
        visible_fraction = clipped_area / raw_area if raw_area > 0.0 else 0.0
        crop = frame[int(y1) : int(y2), int(x1) : int(x2)].copy()
        quality = evaluate_crop_quality(
            crop,
            raw_area=raw_area,
            visible_fraction=visible_fraction,
            config=self.quality_config,
        )
        if crop.size == 0:
            raise CropExtractionError("extracted crop is empty.")
        reference = CropReference(
            raw_track_id=observation.track_id,
            frame_index=observation.frame_index,
            source_video=str(source_video),
            raw_bbox_xyxy=raw_bbox,
            clipped_bbox_xyxy=clipped_bbox,
            crop_width=int(crop.shape[1]),
            crop_height=int(crop.shape[0]),
            quality_metrics=quality,
            crop_path=None,
        )
        return TrackCrop(reference=reference, image=crop)
