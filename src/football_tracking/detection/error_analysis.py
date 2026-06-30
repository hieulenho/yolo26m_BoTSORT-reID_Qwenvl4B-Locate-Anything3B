"""Detection error analysis based on IoU matching for visualization only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.schemas import Detection
from football_tracking.visualization.error_samples import iou_xyxy


@dataclass(frozen=True)
class ErrorAnalysisResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    localization_errors: int
    duplicate_detections: int
    crowded_frames: int
    small_player_failures: int
    occlusion_failures: int
    border_object_failures: int
    note: str = "IoU matching is for error analysis only and does not replace mAP."

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def analyze_detection_errors(
    detections: list[Detection],
    ground_truth: list[BoundingBoxXYXY],
    tp_iou_threshold: float = 0.50,
    localization_iou_threshold: float = 0.10,
) -> ErrorAnalysisResult:
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    true_positives = 0
    localization_errors = 0
    duplicate_detections = 0
    for pred_index, detection in enumerate(sorted(detections, key=lambda item: -item.confidence)):
        best_iou = 0.0
        best_gt = None
        for gt_index, gt_box in enumerate(ground_truth):
            current = iou_xyxy(detection.bbox_xyxy, gt_box)
            if current > best_iou:
                best_iou = current
                best_gt = gt_index
        if best_gt is not None and best_iou >= tp_iou_threshold:
            if best_gt in matched_gt:
                duplicate_detections += 1
            else:
                matched_gt.add(best_gt)
                matched_predictions.add(pred_index)
                true_positives += 1
        elif best_iou >= localization_iou_threshold:
            localization_errors += 1
            matched_predictions.add(pred_index)
    false_positives = len(detections) - len(matched_predictions) - duplicate_detections
    false_negatives = len(ground_truth) - len(matched_gt)
    return ErrorAnalysisResult(
        true_positives=true_positives,
        false_positives=max(0, false_positives),
        false_negatives=max(0, false_negatives),
        localization_errors=localization_errors,
        duplicate_detections=duplicate_detections,
        crowded_frames=int(len(ground_truth) >= 10),
        small_player_failures=0,
        occlusion_failures=0,
        border_object_failures=0,
    )
