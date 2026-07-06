"""CLI helper for language benchmark failure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.failure_analysis.summary import analyze_failures


def run_analyze_language_failures(
    *,
    evaluation: str | Path,
    output_dir: str | Path,
    overwrite: bool,
) -> dict[str, Any]:
    return analyze_failures(evaluation=evaluation, output_dir=output_dir, overwrite=overwrite)
