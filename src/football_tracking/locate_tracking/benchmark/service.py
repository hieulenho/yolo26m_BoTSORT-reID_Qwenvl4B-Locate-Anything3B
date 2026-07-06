"""Service helpers for language benchmark CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.evaluator import evaluate_language_benchmark
from football_tracking.locate_tracking.benchmark.validation import validate_benchmark_manifest


def run_validation(
    *,
    manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    report = validate_benchmark_manifest(manifest)
    payload = report.to_dict()
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["paths"] = {"validation": str(output_path)}
    return payload


def run_benchmark_evaluation(
    *,
    manifest: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    evaluation = evaluate_language_benchmark(
        manifest_path=manifest,
        prediction_manifest_path=predictions,
        output_dir=output_dir,
        iou_threshold=iou_threshold,
        overwrite=overwrite,
    )
    return evaluation.to_dict()
