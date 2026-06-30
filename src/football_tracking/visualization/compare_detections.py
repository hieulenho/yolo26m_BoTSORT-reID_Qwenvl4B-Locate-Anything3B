"""Visualization placeholders for comparing detector outputs."""

from __future__ import annotations

from pathlib import Path


def comparison_output_dir(figures_dir: Path) -> Path:
    path = figures_dir / "comparison"
    path.mkdir(parents=True, exist_ok=True)
    return path
