"""Constant-velocity Kalman filter used by the SORT baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY


class SortKalmanError(ValueError):
    """Raised when SORT Kalman inputs or state are invalid."""


def xyxy_to_z(box: BoundingBoxXYXY) -> np.ndarray:
    if not is_valid_bbox(box):
        raise SortKalmanError(f"Invalid xyxy box: {box}")
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    center_x = box.x1 + width / 2.0
    center_y = box.y1 + height / 2.0
    scale = width * height
    ratio = width / height
    return np.array([[center_x], [center_y], [scale], [ratio]], dtype=float)


def x_to_xyxy(state: np.ndarray) -> BoundingBoxXYXY:
    center_x = float(state[0, 0])
    center_y = float(state[1, 0])
    scale = max(float(state[2, 0]), 1e-9)
    ratio = max(float(state[3, 0]), 1e-9)
    width = math.sqrt(scale * ratio)
    height = scale / width
    box = BoundingBoxXYXY(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )
    if not is_valid_bbox(box):
        raise SortKalmanError(f"Kalman state produced invalid box: {box}")
    return box


@dataclass
class SortKalmanFilter:
    """Seven-dimensional SORT Kalman state: [u, v, s, r, du, dv, ds]."""

    state: np.ndarray
    covariance: np.ndarray
    motion_matrix: np.ndarray
    update_matrix: np.ndarray
    motion_noise: np.ndarray
    measurement_noise: np.ndarray

    @classmethod
    def from_bbox(cls, box: BoundingBoxXYXY) -> SortKalmanFilter:
        state = np.zeros((7, 1), dtype=float)
        state[:4] = xyxy_to_z(box)
        covariance = np.eye(7, dtype=float)
        covariance[4:, 4:] *= 1000.0
        covariance *= 10.0
        motion_matrix = np.eye(7, dtype=float)
        motion_matrix[0, 4] = 1.0
        motion_matrix[1, 5] = 1.0
        motion_matrix[2, 6] = 1.0
        update_matrix = np.zeros((4, 7), dtype=float)
        update_matrix[:4, :4] = np.eye(4, dtype=float)
        motion_noise = np.eye(7, dtype=float)
        motion_noise[4:, 4:] *= 0.01
        motion_noise[-1, -1] *= 0.01
        measurement_noise = np.eye(4, dtype=float)
        measurement_noise[2:, 2:] *= 10.0
        return cls(
            state=state,
            covariance=covariance,
            motion_matrix=motion_matrix,
            update_matrix=update_matrix,
            motion_noise=motion_noise,
            measurement_noise=measurement_noise,
        )

    def predict(self) -> BoundingBoxXYXY:
        if self.state[2, 0] + self.state[6, 0] <= 0.0:
            self.state[6, 0] = 0.0
        self.state = self.motion_matrix @ self.state
        self.covariance = (
            self.motion_matrix @ self.covariance @ self.motion_matrix.T + self.motion_noise
        )
        self._validate_state()
        return self.bbox()

    def update(self, box: BoundingBoxXYXY) -> BoundingBoxXYXY:
        measurement = xyxy_to_z(box)
        residual = measurement - (self.update_matrix @ self.state)
        residual_covariance = (
            self.update_matrix @ self.covariance @ self.update_matrix.T
            + self.measurement_noise
        )
        gain = self.covariance @ self.update_matrix.T @ np.linalg.inv(residual_covariance)
        self.state = self.state + gain @ residual
        identity = np.eye(self.covariance.shape[0], dtype=float)
        self.covariance = (identity - gain @ self.update_matrix) @ self.covariance
        self._validate_state()
        return self.bbox()

    def bbox(self) -> BoundingBoxXYXY:
        return x_to_xyxy(self.state)

    def _validate_state(self) -> None:
        if self.state.shape != (7, 1) or self.covariance.shape != (7, 7):
            raise SortKalmanError("Invalid Kalman state or covariance shape.")
        if not np.isfinite(self.state).all() or not np.isfinite(self.covariance).all():
            raise SortKalmanError("Kalman state contains NaN or infinity.")
