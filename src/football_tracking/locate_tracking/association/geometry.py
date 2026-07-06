"""Pure geometry helpers for grounding-to-track matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

from football_tracking.locate_tracking.association.schemas import AssociationMetrics

BBox = tuple[float, float, float, float]


class GeometryError(ValueError):
    """Raised when geometry is invalid for association."""


@dataclass(frozen=True)
class SanitizedTrackBox:
    raw_bbox: BBox
    matching_bbox: BBox | None
    was_clipped: bool
    valid: bool
    reason: str


def validate_bbox(box: BBox, field_name: str = "bbox") -> BBox:
    values = tuple(float(item) for item in box)
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise GeometryError(f"{field_name} must contain four finite numbers.")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise GeometryError(f"{field_name} must satisfy x2 > x1 and y2 > y1.")
    return values


def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = validate_bbox(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(left: BBox, right: BBox) -> float:
    left = validate_bbox(left, "left")
    right = validate_bbox(right, "right")
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection = intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def coverage_of_left(left: BBox, right: BBox) -> float:
    area = bbox_area(left)
    return intersection_area(left, right) / area if area > 0.0 else 0.0


def bbox_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = validate_bbox(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside_bbox(point: tuple[float, float], box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = validate_bbox(box)
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance_px(left: BBox, right: BBox) -> float:
    lx, ly = bbox_center(left)
    rx, ry = bbox_center(right)
    return math.hypot(lx - rx, ly - ry)


def normalized_center_distance(
    left: BBox,
    right: BBox,
    frame_width: int,
    frame_height: int,
) -> float:
    if frame_width <= 0 or frame_height <= 0:
        raise GeometryError("frame dimensions must be positive.")
    diagonal = math.hypot(float(frame_width), float(frame_height))
    return min(center_distance_px(left, right) / diagonal, 1.0)


def center_similarity(left: BBox, right: BBox, frame_width: int, frame_height: int) -> float:
    return max(0.0, 1.0 - normalized_center_distance(left, right, frame_width, frame_height))


def sanitize_track_bbox(
    raw_bbox: BBox,
    frame_width: int,
    frame_height: int,
    *,
    clip_to_frame: bool = True,
) -> SanitizedTrackBox:
    raw = validate_bbox(raw_bbox, "raw_track_bbox")
    if frame_width <= 0 or frame_height <= 0:
        raise GeometryError("frame dimensions must be positive.")
    if not clip_to_frame:
        return SanitizedTrackBox(
            raw_bbox=raw,
            matching_bbox=raw,
            was_clipped=False,
            valid=True,
            reason="not_clipped",
        )
    x1 = max(0.0, min(float(frame_width), raw[0]))
    y1 = max(0.0, min(float(frame_height), raw[1]))
    x2 = max(0.0, min(float(frame_width), raw[2]))
    y2 = max(0.0, min(float(frame_height), raw[3]))
    was_clipped = (x1, y1, x2, y2) != raw
    if x2 <= x1 or y2 <= y1:
        return SanitizedTrackBox(
            raw_bbox=raw,
            matching_bbox=None,
            was_clipped=was_clipped,
            valid=False,
            reason="fully_outside_or_degenerate_after_clipping",
        )
    return SanitizedTrackBox(
        raw_bbox=raw,
        matching_bbox=(x1, y1, x2, y2),
        was_clipped=was_clipped,
        valid=True,
        reason="clipped" if was_clipped else "inside_frame",
    )


def association_metrics(
    grounding_bbox: BBox,
    track_bbox: BBox,
    frame_width: int,
    frame_height: int,
) -> AssociationMetrics:
    grounding_bbox = validate_bbox(grounding_bbox, "grounding_bbox")
    track_bbox = validate_bbox(track_bbox, "track_bbox")
    intersection = intersection_area(grounding_bbox, track_bbox)
    return AssociationMetrics(
        intersection_area=intersection,
        iou=bbox_iou(grounding_bbox, track_bbox),
        grounding_coverage=coverage_of_left(grounding_bbox, track_bbox),
        track_coverage=coverage_of_left(track_bbox, grounding_bbox),
        center_distance_px=center_distance_px(grounding_bbox, track_bbox),
        center_distance_normalized=normalized_center_distance(
            grounding_bbox,
            track_bbox,
            frame_width,
            frame_height,
        ),
        center_similarity=center_similarity(grounding_bbox, track_bbox, frame_width, frame_height),
        track_center_inside_grounding=point_inside_bbox(bbox_center(track_bbox), grounding_bbox),
        grounding_center_inside_track=point_inside_bbox(bbox_center(grounding_bbox), track_bbox),
    )
