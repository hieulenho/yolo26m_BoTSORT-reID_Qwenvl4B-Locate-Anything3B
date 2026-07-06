"""Deterministic quality metrics for track crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from football_tracking.locate_tracking.appearance.schemas import CropQualityMetrics


@dataclass(frozen=True)
class CropQualityConfig:
    min_width: int = 8
    min_height: int = 8
    min_area: float = 64.0
    min_visible_fraction: float = 0.30
    min_sharpness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_area": self.min_area,
            "min_visible_fraction": self.min_visible_fraction,
            "min_sharpness": self.min_sharpness,
        }


def _sharpness_score(crop: np.ndarray) -> float | None:
    try:
        import cv2  # type: ignore[import-not-found]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:  # noqa: BLE001
        return None


def evaluate_crop_quality(
    crop: np.ndarray,
    *,
    raw_area: float,
    visible_fraction: float,
    config: CropQualityConfig | None = None,
) -> CropQualityMetrics:
    cfg = config or CropQualityConfig()
    if crop.ndim < 2:
        width = 0
        height = 0
    else:
        height, width = crop.shape[:2]
    area = float(width * height)
    aspect_ratio = float(width / height) if height > 0 else 0.0
    rejection_reasons: list[str] = []
    if width < cfg.min_width:
        rejection_reasons.append("too_narrow")
    if height < cfg.min_height:
        rejection_reasons.append("too_short")
    if area < cfg.min_area:
        rejection_reasons.append("too_small")
    if visible_fraction < cfg.min_visible_fraction:
        rejection_reasons.append("severely_clipped")
    if area <= 0.0 or raw_area <= 0.0:
        rejection_reasons.append("invalid_crop")
    sharpness = _sharpness_score(crop) if area > 0.0 else None
    if cfg.min_sharpness is not None and sharpness is not None and sharpness < cfg.min_sharpness:
        rejection_reasons.append("low_sharpness")
    brightness = float(np.asarray(crop).mean()) if np.asarray(crop).size else None
    area_score = min(1.0, area / max(cfg.min_area * 4.0, 1.0))
    quality_score = max(0.0, min(1.0, visible_fraction * area_score))
    if rejection_reasons:
        quality_score = min(quality_score, 0.25)
    return CropQualityMetrics(
        width=width,
        height=height,
        area=area,
        aspect_ratio=aspect_ratio,
        visible_fraction=max(0.0, min(1.0, visible_fraction)),
        sharpness_score=sharpness,
        brightness_mean=brightness,
        passed_quality_gate=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        quality_score=quality_score,
    )
