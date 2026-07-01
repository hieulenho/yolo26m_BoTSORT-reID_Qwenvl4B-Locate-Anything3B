"""Markdown report for tracker ablation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_ablation_report(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tracker Ablation Report",
        "",
        f"Experiment count: {len(rows)}",
        "",
        "This report is generated from validation experiments only. It must not be used",
        "as evidence of test-set performance.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
