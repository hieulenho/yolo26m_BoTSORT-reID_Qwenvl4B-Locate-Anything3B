"""Ablation plot placeholders.

Plots are created only when ablation results contain enough non-null official
metrics for a meaningful comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_ablation_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    return []
