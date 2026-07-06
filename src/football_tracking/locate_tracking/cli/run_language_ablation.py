"""CLI helper for language benchmark ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.experiments.service import run_ablation_service


def run_language_ablation_cli(
    *,
    config: str | Path,
    output_dir: str | Path | None,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    return run_ablation_service(
        config=config,
        output_dir=output_dir,
        overwrite=overwrite,
        dry_run=dry_run,
    )
