"""Semantic identity continuity metrics."""

from __future__ import annotations

from football_tracking.locate_tracking.benchmark.query_metrics import FrameMatchResult


def longest_correct_run(frame_results: tuple[FrameMatchResult, ...]) -> int:
    best = 0
    current = 0
    for result in frame_results:
        if result.correct_count > 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def continuity_ratio(
    frame_results: tuple[FrameMatchResult, ...],
    gt_frame_count: int,
) -> float | None:
    if gt_frame_count <= 0:
        return None
    return longest_correct_run(frame_results) / gt_frame_count


def semantic_target_switches(
    frame_results: tuple[FrameMatchResult, ...],
    *,
    persistence_frames: int = 2,
) -> int:
    switches = 0
    seen_correct = False
    incorrect_run = 0
    for result in frame_results:
        has_prediction = result.predicted_count > 0
        is_correct = result.correct_count > 0
        if is_correct:
            seen_correct = True
            incorrect_run = 0
            continue
        if seen_correct and has_prediction:
            incorrect_run += 1
            if incorrect_run == persistence_frames:
                switches += 1
        else:
            incorrect_run = 0
    return switches
