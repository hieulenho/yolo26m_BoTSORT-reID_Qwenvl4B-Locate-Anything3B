from __future__ import annotations

import pytest

from football_tracking.locate_tracking.grounding.coordinates import (
    CoordinateError,
    normalized_to_pixel_xyxy,
)


def test_normalized_to_pixel_for_hd_image() -> None:
    assert normalized_to_pixel_xyxy((100, 200, 500, 800), 1920, 1080) == (
        192.0,
        216.0,
        960.0,
        864.0,
    )


def test_normalized_to_pixel_boundaries() -> None:
    assert normalized_to_pixel_xyxy((0, 0, 1000, 1000), 640, 480) == (
        0.0,
        0.0,
        640.0,
        480.0,
    )


def test_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(CoordinateError):
        normalized_to_pixel_xyxy((-1, 0, 1000, 1000), 640, 480)


def test_rejects_inverted_coordinates() -> None:
    with pytest.raises(CoordinateError):
        normalized_to_pixel_xyxy((500, 0, 100, 1000), 640, 480)


def test_rejects_invalid_image_size() -> None:
    with pytest.raises(CoordinateError):
        normalized_to_pixel_xyxy((0, 0, 1000, 1000), 0, 480)

