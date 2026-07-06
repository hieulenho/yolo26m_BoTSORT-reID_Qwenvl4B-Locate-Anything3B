"""CLI helper for language benchmark report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.reporting.report_builder import generate_language_report


def run_generate_language_report(
    *,
    evaluation: str | Path,
    output: str | Path,
    ablation: str | Path | None,
    failures: str | Path | None,
    mot_metrics: str | Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    return generate_language_report(
        evaluation=evaluation,
        output=output,
        ablation=ablation,
        failures=failures,
        mot_metrics=mot_metrics,
        overwrite=overwrite,
    )
