from __future__ import annotations

import pytest

from football_tracking.locate_tracking.association.geometry import (
    bbox_iou,
    center_distance_px,
    coverage_of_left,
    intersection_area,
    normalized_center_distance,
    sanitize_track_bbox,
)


def test_geometry_perfect_overlap() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou(box, box) == 1.0
    assert intersection_area(box, box) == 100.0


def test_geometry_partial_and_no_overlap() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert bbox_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_geometry_containment_coverages() -> None:
    outer = (0.0, 0.0, 100.0, 100.0)
    inner = (25.0, 25.0, 75.0, 75.0)

    assert coverage_of_left(inner, outer) == 1.0
    assert coverage_of_left(outer, inner) == 0.25


def test_geometry_center_distance() -> None:
    left = (0.0, 0.0, 10.0, 10.0)
    right = (10.0, 0.0, 20.0, 10.0)

    assert center_distance_px(left, right) == 10.0
    assert normalized_center_distance(left, right, 100, 100) == pytest.approx(
        10 / 141.421356,
        rel=1e-5,
    )


def test_sanitize_partially_out_of_frame_track() -> None:
    sanitized = sanitize_track_bbox((-5.0, 5.0, 10.0, 15.0), 100, 100)

    assert sanitized.valid
    assert sanitized.was_clipped
    assert sanitized.matching_bbox == (0.0, 5.0, 10.0, 15.0)


def test_sanitize_fully_outside_or_degenerate_rejected() -> None:
    outside = sanitize_track_bbox((-20.0, 5.0, -10.0, 15.0), 100, 100)
    degenerate = sanitize_track_bbox((100.0, 0.0, 110.0, 10.0), 100, 100)

    assert not outside.valid
    assert outside.matching_bbox is None
    assert not degenerate.valid
