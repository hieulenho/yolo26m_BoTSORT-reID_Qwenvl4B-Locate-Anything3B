"""Deterministic representative crop selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.locate_tracking.appearance.crop_extractor import TrackCrop


@dataclass(frozen=True)
class RepresentativeCropSelectionConfig:
    max_samples_per_track: int = 4
    min_frame_gap: int = 5
    require_quality_gate: bool = True

    def __post_init__(self) -> None:
        if int(self.max_samples_per_track) < 1:
            raise ValueError("max_samples_per_track must be >= 1.")
        if int(self.min_frame_gap) < 0:
            raise ValueError("min_frame_gap must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_samples_per_track": self.max_samples_per_track,
            "min_frame_gap": self.min_frame_gap,
            "require_quality_gate": self.require_quality_gate,
        }


def select_representative_crops(
    crops: tuple[TrackCrop, ...],
    config: RepresentativeCropSelectionConfig | None = None,
) -> tuple[TrackCrop, ...]:
    cfg = config or RepresentativeCropSelectionConfig()
    valid = [
        crop
        for crop in crops
        if crop.reference.quality_metrics.passed_quality_gate or not cfg.require_quality_gate
    ]
    ranked = sorted(
        valid,
        key=lambda crop: (
            -crop.reference.quality_metrics.quality_score,
            crop.reference.frame_index,
            crop.reference.raw_track_id,
        ),
    )
    selected: list[TrackCrop] = []
    for crop in ranked:
        if all(
            abs(crop.reference.frame_index - chosen.reference.frame_index) >= cfg.min_frame_gap
            for chosen in selected
        ):
            selected.append(crop)
        if len(selected) >= cfg.max_samples_per_track:
            break
    if len(selected) < min(cfg.max_samples_per_track, len(ranked)):
        selected_frames = {crop.reference.frame_index for crop in selected}
        for crop in ranked:
            if crop.reference.frame_index not in selected_frames:
                selected.append(crop)
                selected_frames.add(crop.reference.frame_index)
            if len(selected) >= cfg.max_samples_per_track:
                break
    return tuple(sorted(selected, key=lambda crop: crop.reference.frame_index))
