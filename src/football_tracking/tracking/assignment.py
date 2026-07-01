"""Hungarian assignment for SORT using IoU costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.iou import iou_cost_matrix, pairwise_iou_matrix


@dataclass(frozen=True)
class AssociationResult:
    matched_pairs: list[tuple[int, int]]
    unmatched_track_indices: list[int]
    unmatched_detection_indices: list[int]
    iou_matrix: np.ndarray
    cost_matrix: np.ndarray


def associate_detections_to_tracks(
    predicted_track_boxes: list[BoundingBoxXYXY],
    detection_boxes: list[BoundingBoxXYXY],
    iou_threshold: float,
) -> AssociationResult:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")
    track_count = len(predicted_track_boxes)
    detection_count = len(detection_boxes)
    if track_count == 0 or detection_count == 0:
        iou_matrix = np.zeros((track_count, detection_count), dtype=float)
        cost_matrix = np.ones((track_count, detection_count), dtype=float)
        return AssociationResult(
            matched_pairs=[],
            unmatched_track_indices=list(range(track_count)),
            unmatched_detection_indices=list(range(detection_count)),
            iou_matrix=iou_matrix,
            cost_matrix=cost_matrix,
        )

    iou_matrix = pairwise_iou_matrix(predicted_track_boxes, detection_boxes)
    cost_matrix = iou_cost_matrix(predicted_track_boxes, detection_boxes)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_pairs: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for track_index, detection_index in zip(
        row_indices.tolist(),
        col_indices.tolist(),
        strict=True,
    ):
        if iou_matrix[track_index, detection_index] < iou_threshold:
            continue
        matched_pairs.append((int(track_index), int(detection_index)))
        matched_tracks.add(int(track_index))
        matched_detections.add(int(detection_index))

    return AssociationResult(
        matched_pairs=matched_pairs,
        unmatched_track_indices=[
            index for index in range(track_count) if index not in matched_tracks
        ],
        unmatched_detection_indices=[
            index for index in range(detection_count) if index not in matched_detections
        ],
        iou_matrix=iou_matrix,
        cost_matrix=cost_matrix,
    )
