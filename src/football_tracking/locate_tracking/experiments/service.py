"""CLI-facing service for language ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.experiments.runner import run_language_ablation


def run_ablation_service(
    *,
    config: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_language_ablation(
        config,
        output_dir=output_dir,
        overwrite=overwrite,
        dry_run=dry_run,
    )
