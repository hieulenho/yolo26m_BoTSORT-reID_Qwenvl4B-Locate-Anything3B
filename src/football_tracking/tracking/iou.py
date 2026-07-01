"""IoU helpers for SORT association."""

from __future__ import annotations

import numpy as np

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY


class IoUError(ValueError):
    """Raised when IoU inputs are invalid."""


def bbox_iou(left: BoundingBoxXYXY, right: BoundingBoxXYXY) -> float:
    """Return IoU for two xyxy boxes."""
    if not is_valid_bbox(left) or not is_valid_bbox(right):
        raise IoUError("bbox_iou expects valid xyxy boxes.")
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def pairwise_iou_matrix(
    track_boxes: list[BoundingBoxXYXY],
    detection_boxes: list[BoundingBoxXYXY],
) -> np.ndarray:
    matrix = np.zeros((len(track_boxes), len(detection_boxes)), dtype=float)
    for track_index, track_box in enumerate(track_boxes):
        if not is_valid_bbox(track_box):
            raise IoUError(f"Invalid track box at index {track_index}: {track_box}")
        for detection_index, detection_box in enumerate(detection_boxes):
            if not is_valid_bbox(detection_box):
                raise IoUError(f"Invalid detection box at index {detection_index}: {detection_box}")
            matrix[track_index, detection_index] = bbox_iou(track_box, detection_box)
    return matrix


def iou_cost_matrix(
    track_boxes: list[BoundingBoxXYXY],
    detection_boxes: list[BoundingBoxXYXY],
) -> np.ndarray:
    return 1.0 - pairwise_iou_matrix(track_boxes, detection_boxes)
