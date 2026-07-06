from __future__ import annotations

import numpy as np

from football_tracking.locate_tracking.appearance.crop_quality import (
    CropQualityConfig,
    evaluate_crop_quality,
)


def test_crop_quality_passes_good_crop() -> None:
    crop = np.ones((20, 20, 3), dtype=np.uint8) * 100

    quality = evaluate_crop_quality(crop, raw_area=400, visible_fraction=1.0)

    assert quality.passed_quality_gate is True
    assert quality.quality_score > 0


def test_crop_quality_rejects_too_small_and_clipped_crop() -> None:
    crop = np.ones((4, 4, 3), dtype=np.uint8)

    quality = evaluate_crop_quality(
        crop,
        raw_area=400,
        visible_fraction=0.1,
        config=CropQualityConfig(min_width=8, min_height=8, min_area=64),
    )

    assert quality.passed_quality_gate is False
    assert {"too_narrow", "too_short", "too_small", "severely_clipped"}.issubset(
        set(quality.rejection_reasons)
    )
