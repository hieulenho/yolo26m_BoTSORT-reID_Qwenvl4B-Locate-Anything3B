"""Coordinate conversion helpers for LocateAnything grounding outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence


class CoordinateError(ValueError):
    """Raised when normalized grounding coordinates are invalid."""


def validate_image_size(image_width: int, image_height: int) -> None:
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise CoordinateError("image_width and image_height must be positive.")


def validate_normalized_xyxy(
    normalized_bbox: Sequence[int | float],
) -> tuple[int, int, int, int]:
    if len(normalized_bbox) != 4:
        raise CoordinateError("normalized bbox must contain four coordinates.")
    values: list[int] = []
    for value in normalized_bbox:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise CoordinateError("normalized bbox coordinates must be finite.")
        if numeric < 0.0 or numeric > 1000.0:
            raise CoordinateError("normalized bbox coordinates must be in [0, 1000].")
        values.append(int(round(numeric)))
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise CoordinateError("normalized bbox must satisfy x2 > x1 and y2 > y1.")
    return x1, y1, x2, y2


def normalized_to_pixel_xyxy(
    normalized_bbox: Sequence[int | float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert 0-1000 normalized XYXY coordinates into image pixels."""

    validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = validate_normalized_xyxy(normalized_bbox)
    return (
        x1 / 1000.0 * image_width,
        y1 / 1000.0 * image_height,
        x2 / 1000.0 * image_width,
        y2 / 1000.0 * image_height,
    )

