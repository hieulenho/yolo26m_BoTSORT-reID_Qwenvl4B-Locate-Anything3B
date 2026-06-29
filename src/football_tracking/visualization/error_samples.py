"""Simple error sample helpers for visual analysis only."""

from __future__ import annotations

from football_tracking.data.bbox import bbox_area
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.schemas import Detection


def iou_xyxy(left: BoundingBoxXYXY, right: BoundingBoxXYXY) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def categorize_predictions_for_visualization(
    detections: list[Detection],
    ground_truth: list[BoundingBoxXYXY],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0
    localization_error = 0
    for detection in detections:
        best_iou = 0.0
        best_index = None
        for index, gt_box in enumerate(ground_truth):
            if index in matched_gt:
                continue
            current_iou = iou_xyxy(detection.bbox_xyxy, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            true_positive += 1
        elif best_iou > 0.0:
            localization_error += 1
        else:
            false_positive += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": len(ground_truth) - len(matched_gt),
        "localization_error": localization_error,
        "crowded_frame": int(len(ground_truth) >= 10),
        "small_player": 0,
        "partially_occluded_player": 0,
    }
