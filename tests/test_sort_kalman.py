from __future__ import annotations

import numpy as np

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.sort_kalman import SortKalmanFilter, xyxy_to_z


def test_kalman_state_shapes_and_box_conversion() -> None:
    kalman = SortKalmanFilter.from_bbox(BoundingBoxXYXY(0, 0, 10, 20))
    assert kalman.state.shape == (7, 1)
    assert kalman.covariance.shape == (7, 7)
    assert np.allclose(xyxy_to_z(BoundingBoxXYXY(0, 0, 10, 20)).ravel(), [5, 10, 200, 0.5])


def test_predict_moves_by_velocity_and_update_reduces_residual() -> None:
    kalman = SortKalmanFilter.from_bbox(BoundingBoxXYXY(0, 0, 10, 10))
    kalman.state[4, 0] = 2.0
    predicted = kalman.predict()
    assert predicted.x1 > 1.0
    before = abs(kalman.state[0, 0] - 20.0)
    kalman.update(BoundingBoxXYXY(15, 0, 25, 10))
    after = abs(kalman.state[0, 0] - 20.0)
    assert after < before
    assert np.isfinite(kalman.state).all()
