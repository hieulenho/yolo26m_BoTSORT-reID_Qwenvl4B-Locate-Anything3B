"""Speed/accuracy plotting helpers for tracker experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_speed_accuracy_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    return []
