from __future__ import annotations

import numpy as np

from football_tracking.locate_tracking.appearance.crop_extractor import TrackCrop
from football_tracking.locate_tracking.appearance.crop_selection import (
    RepresentativeCropSelectionConfig,
    select_representative_crops,
)
from tests.locate_tracking.appearance_test_utils import crop_reference


def _crop(frame_index: int, quality: float) -> TrackCrop:
    reference = crop_reference(7, frame_index)
    metrics = reference.quality_metrics
    updated_metrics = type(metrics)(
        width=metrics.width,
        height=metrics.height,
        area=metrics.area,
        aspect_ratio=metrics.aspect_ratio,
        visible_fraction=metrics.visible_fraction,
        sharpness_score=metrics.sharpness_score,
        brightness_mean=metrics.brightness_mean,
        passed_quality_gate=True,
        rejection_reasons=(),
        quality_score=quality,
    )
    updated_reference = type(reference)(
        raw_track_id=reference.raw_track_id,
        frame_index=reference.frame_index,
        source_video=reference.source_video,
        raw_bbox_xyxy=reference.raw_bbox_xyxy,
        clipped_bbox_xyxy=reference.clipped_bbox_xyxy,
        crop_width=reference.crop_width,
        crop_height=reference.crop_height,
        quality_metrics=updated_metrics,
    )
    return TrackCrop(reference=updated_reference, image=np.zeros((20, 20, 3), dtype=np.uint8))


def test_representative_crop_selection_respects_max_samples_and_temporal_gap() -> None:
    crops = (_crop(100, 0.9), _crop(101, 1.0), _crop(140, 0.8), _crop(180, 0.7))

    selected = select_representative_crops(
        crops,
        RepresentativeCropSelectionConfig(max_samples_per_track=3, min_frame_gap=10),
    )

    assert len(selected) == 3
    assert [crop.reference.frame_index for crop in selected] == [101, 140, 180]
