"""CLI helper for validating language benchmark manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.service import run_validation


def run_validate_language_benchmark(
    *,
    manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return run_validation(manifest=manifest, output=output)
