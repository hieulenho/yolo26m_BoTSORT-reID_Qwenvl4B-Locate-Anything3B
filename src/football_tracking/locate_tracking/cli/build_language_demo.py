"""CLI helper for language tracking demo manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.reporting.demo_manifest import build_demo_manifest


def run_build_language_demo(
    *,
    evaluation: str | Path,
    output_dir: str | Path,
    max_cases: int,
    overwrite: bool,
) -> dict[str, Any]:
    return build_demo_manifest(
        evaluation=evaluation,
        output_dir=output_dir,
        max_cases=max_cases,
        overwrite=overwrite,
    )
