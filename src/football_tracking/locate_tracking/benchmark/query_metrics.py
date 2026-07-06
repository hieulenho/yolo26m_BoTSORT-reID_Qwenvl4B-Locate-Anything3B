"""Frame-level query metrics for semantic target predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.geometry import bbox_iou


@dataclass(frozen=True)
class FrameMatchResult:
    frame_index: int
    gt_count: int
    predicted_count: int
    correct_count: int
    best_iou: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "gt_count": self.gt_count,
            "predicted_count": self.predicted_count,
            "correct_count": self.correct_count,
            "best_iou": self.best_iou,
        }


@dataclass(frozen=True)
class QueryFrameMetrics:
    query_id: str
    status: str
    gt_frame_count: int
    predicted_frame_count: int
    correct_frame_count: int
    gt_box_count: int
    predicted_box_count: int
    correct_box_count: int
    target_precision: float | None
    target_recall: float | None
    target_f1: float | None
    initial_selection_correct: bool | None
    iou_threshold: float
    iou_values: tuple[float, ...] = ()
    frame_results: tuple[FrameMatchResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "status": self.status,
            "gt_frame_count": self.gt_frame_count,
            "predicted_frame_count": self.predicted_frame_count,
            "correct_frame_count": self.correct_frame_count,
            "gt_box_count": self.gt_box_count,
            "predicted_box_count": self.predicted_box_count,
            "correct_box_count": self.correct_box_count,
            "target_precision": self.target_precision,
            "target_recall": self.target_recall,
            "target_f1": self.target_f1,
            "initial_selection_correct": self.initial_selection_correct,
            "iou_threshold": self.iou_threshold,
            "iou_values": list(self.iou_values),
            "frame_results": [item.to_dict() for item in self.frame_results],
        }


def evaluate_query_frames(
    *,
    query_id: str,
    status: str,
    gt_by_frame: dict[int, tuple[MotTrackObservation, ...]],
    pred_by_frame: dict[int, tuple[MotTrackObservation, ...]],
    start_frame: int,
    end_frame: int,
    iou_threshold: float,
) -> QueryFrameMetrics:
    frame_results: list[FrameMatchResult] = []
    iou_values: list[float] = []
    correct_frames = 0
    gt_box_count = 0
    pred_box_count = 0
    correct_box_count = 0
    predicted_frames = 0
    initial_selection_correct: bool | None = None
    for frame_index in range(start_frame, end_frame + 1):
        gt_rows = gt_by_frame.get(frame_index, ())
        pred_rows = pred_by_frame.get(frame_index, ())
        if pred_rows:
            predicted_frames += 1
        match_count, best_iou = match_frame_boxes(gt_rows, pred_rows, iou_threshold)
        gt_box_count += len(gt_rows)
        pred_box_count += len(pred_rows)
        correct_box_count += match_count
        if match_count > 0:
            correct_frames += 1
            if best_iou is not None:
                iou_values.append(best_iou)
        if initial_selection_correct is None and pred_rows:
            initial_selection_correct = match_count > 0
        frame_results.append(
            FrameMatchResult(
                frame_index=frame_index,
                gt_count=len(gt_rows),
                predicted_count=len(pred_rows),
                correct_count=match_count,
                best_iou=best_iou,
            )
        )
    precision = _ratio(correct_box_count, pred_box_count)
    recall = _ratio(correct_box_count, gt_box_count)
    return QueryFrameMetrics(
        query_id=query_id,
        status=status,
        gt_frame_count=sum(1 for rows in gt_by_frame.values() if rows),
        predicted_frame_count=predicted_frames,
        correct_frame_count=correct_frames,
        gt_box_count=gt_box_count,
        predicted_box_count=pred_box_count,
        correct_box_count=correct_box_count,
        target_precision=precision,
        target_recall=recall,
        target_f1=_f1(precision, recall),
        initial_selection_correct=initial_selection_correct,
        iou_threshold=iou_threshold,
        iou_values=tuple(iou_values),
        frame_results=tuple(frame_results),
    )


def match_frame_boxes(
    gt_rows: tuple[MotTrackObservation, ...],
    pred_rows: tuple[MotTrackObservation, ...],
    iou_threshold: float,
) -> tuple[int, float | None]:
    pairs = [
        (bbox_iou(gt.bbox_xyxy, pred.bbox_xyxy), gt_index, pred_index)
        for gt_index, gt in enumerate(gt_rows)
        for pred_index, pred in enumerate(pred_rows)
    ]
    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = 0
    best_iou = pairs[0][0] if pairs else None
    for iou, gt_index, pred_index in pairs:
        if iou < iou_threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches += 1
    return matches, best_iou


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)
