from __future__ import annotations

import numpy as np
import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.iou import IoUError, bbox_iou, pairwise_iou_matrix


def test_bbox_iou_exact_and_disjoint() -> None:
    box = BoundingBoxXYXY(0, 0, 10, 10)
    assert bbox_iou(box, box) == 1.0
    assert bbox_iou(box, BoundingBoxXYXY(20, 20, 30, 30)) == 0.0


def test_bbox_iou_partial_overlap() -> None:
    value = bbox_iou(BoundingBoxXYXY(0, 0, 10, 10), BoundingBoxXYXY(5, 5, 15, 15))
    assert value == pytest.approx(25 / 175)


def test_pairwise_iou_matrix_shape_and_invalid_box() -> None:
    matrix = pairwise_iou_matrix(
        [BoundingBoxXYXY(0, 0, 10, 10)],
        [BoundingBoxXYXY(0, 0, 10, 10), BoundingBoxXYXY(20, 20, 30, 30)],
    )
    assert matrix.shape == (1, 2)
    assert np.allclose(matrix, [[1.0, 0.0]])
    with pytest.raises(IoUError):
        pairwise_iou_matrix([BoundingBoxXYXY(10, 0, 0, 10)], [])
