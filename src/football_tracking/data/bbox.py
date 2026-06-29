"""Bounding box conversion and validation helpers."""

from __future__ import annotations

import math

from football_tracking.data.schemas import BoundingBoxXYWH, BoundingBoxXYXY

FLOAT_TOLERANCE = 1e-9


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def xyxy_to_xywh(box: BoundingBoxXYXY) -> BoundingBoxXYWH:
    return BoundingBoxXYWH(
        x=box.x1,
        y=box.y1,
        width=box.x2 - box.x1,
        height=box.y2 - box.y1,
    )


def xywh_to_xyxy(box: BoundingBoxXYWH) -> BoundingBoxXYXY:
    return BoundingBoxXYXY(
        x1=box.x,
        y1=box.y,
        x2=box.x + box.width,
        y2=box.y + box.height,
    )


def clip_xyxy_to_image(
    box: BoundingBoxXYXY,
    image_width: int,
    image_height: int,
) -> BoundingBoxXYXY:
    return BoundingBoxXYXY(
        x1=max(0.0, min(float(image_width), box.x1)),
        y1=max(0.0, min(float(image_height), box.y1)),
        x2=max(0.0, min(float(image_width), box.x2)),
        y2=max(0.0, min(float(image_height), box.y2)),
    )


def bbox_area(box: BoundingBoxXYXY) -> float:
    if not is_valid_bbox(box):
        return 0.0
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def is_valid_bbox(box: BoundingBoxXYXY) -> bool:
    values = (box.x1, box.y1, box.x2, box.y2)
    if not all(_is_finite(float(value)) for value in values):
        return False
    return box.x2 > box.x1 and box.y2 > box.y1


def xyxy_to_yolo_normalized(
    box: BoundingBoxXYXY,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not is_valid_bbox(box):
        raise ValueError(f"Invalid bounding box: {box}")

    x_center = ((box.x1 + box.x2) / 2.0) / image_width
    y_center = ((box.y1 + box.y2) / 2.0) / image_height
    width = (box.x2 - box.x1) / image_width
    height = (box.y2 - box.y1) / image_height
    values = (x_center, y_center, width, height)
    if any(value < -FLOAT_TOLERANCE or value > 1.0 + FLOAT_TOLERANCE for value in values):
        raise ValueError(f"YOLO normalized coordinates are outside [0, 1]: {values}")
    return tuple(max(0.0, min(1.0, value)) for value in values)


def yolo_normalized_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> BoundingBoxXYXY:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    values = (x_center, y_center, width, height)
    if any(not _is_finite(float(value)) for value in values):
        raise ValueError(f"YOLO coordinates must be finite: {values}")
    if any(value < -FLOAT_TOLERANCE or value > 1.0 + FLOAT_TOLERANCE for value in values):
        raise ValueError(f"YOLO coordinates must be in [0, 1]: {values}")
    if width <= 0 or height <= 0:
        raise ValueError("YOLO width and height must be positive.")

    pixel_width = width * image_width
    pixel_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return BoundingBoxXYXY(
        x1=center_x - pixel_width / 2.0,
        y1=center_y - pixel_height / 2.0,
        x2=center_x + pixel_width / 2.0,
        y2=center_y + pixel_height / 2.0,
    )
