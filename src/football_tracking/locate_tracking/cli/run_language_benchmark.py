"""CLI helper for evaluating saved language benchmark predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.service import run_benchmark_evaluation


def run_language_benchmark(
    *,
    manifest: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    iou_threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    return run_benchmark_evaluation(
        manifest=manifest,
        predictions=predictions,
        output_dir=output_dir,
        iou_threshold=iou_threshold,
        overwrite=overwrite,
    )
