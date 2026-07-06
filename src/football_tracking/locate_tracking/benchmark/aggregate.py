"""Aggregate per-query language tracking benchmark metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def aggregate_query_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_count = len(rows)
    status_counts = Counter(str(row.get("status", "missing_prediction")) for row in rows)
    resolved_count = status_counts.get("resolved", 0)
    initial_correct = sum(1 for row in rows if row.get("initial_selection_correct") is True)
    initial_resolved = sum(1 for row in rows if row.get("initial_selection_correct") is not None)
    correct_boxes = sum(int(row.get("correct_box_count") or 0) for row in rows)
    predicted_boxes = sum(int(row.get("predicted_box_count") or 0) for row in rows)
    gt_boxes = sum(int(row.get("gt_box_count") or 0) for row in rows)
    opportunities = sum(int(row.get("reacquisition_opportunity_count") or 0) for row in rows)
    reaq_success = sum(int(row.get("reacquisition_success_count") or 0) for row in rows)
    false_reaq = sum(int(row.get("false_reacquisition_count") or 0) for row in rows)
    committed_reaq = sum(int(row.get("committed_reacquisition_count") or 0) for row in rows)
    eval_frames = sum(int(row.get("eval_frame_count") or 0) for row in rows)
    grounding_calls = sum(int(row.get("grounding_call_count") or 0) for row in rows)
    precision = _ratio(correct_boxes, predicted_boxes)
    recall = _ratio(correct_boxes, gt_boxes)
    aggregate = {
        "query_count": query_count,
        "status_counts": dict(sorted(status_counts.items())),
        "query_resolution_rate": _ratio(resolved_count, query_count),
        "initial_selection_accuracy_strict": _ratio(initial_correct, query_count),
        "initial_selection_accuracy_resolved": _ratio(initial_correct, initial_resolved),
        "micro_target_precision": precision,
        "micro_target_recall": recall,
        "micro_target_f1": _f1(precision, recall),
        "macro_target_precision": _mean_metric(rows, "target_precision"),
        "macro_target_recall": _mean_metric(rows, "target_recall"),
        "macro_target_f1": _mean_metric(rows, "target_f1"),
        "macro_continuity_ratio": _mean_metric(rows, "target_continuity_ratio"),
        "semantic_target_switches": sum(
            int(row.get("semantic_target_switches") or 0) for row in rows
        ),
        "raw_id_transitions_along_semantic_target": sum(
            int(row.get("raw_id_transitions_along_semantic_target") or 0) for row in rows
        ),
        "reacquisition_opportunity_count": opportunities,
        "reacquisition_success_rate": _ratio(reaq_success, opportunities),
        "false_reacquisition_rate": _ratio(false_reaq, committed_reaq),
        "mean_frames_to_reacquire": _mean_values(
            value
            for row in rows
            for value in row.get("frames_to_reacquire", [])
        ),
        "grounding_call_count": grounding_calls,
        "eval_frame_count": eval_frames,
        "grounding_calls_per_1000_frames": (
            1000.0 * grounding_calls / eval_frames if eval_frames else None
        ),
        "runtime_seconds_total": _sum_optional(rows, "runtime_seconds"),
        "by_category": _group_average(rows, "query_category"),
        "by_difficulty": _group_average(rows, "difficulty"),
    }
    return aggregate


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    return _mean_values(row.get(key) for row in rows)


def _mean_values(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _sum_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values)


def _group_average(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        name: {
            "query_count": len(items),
            "macro_target_f1": _mean_metric(items, "target_f1"),
            "macro_continuity_ratio": _mean_metric(items, "target_continuity_ratio"),
            "resolution_rate": _ratio(
                sum(1 for item in items if item.get("status") == "resolved"),
                len(items),
            ),
        }
        for name, items in sorted(grouped.items())
    }
