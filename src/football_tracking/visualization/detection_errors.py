"""Placeholder-safe rendering hooks for fine-tuned detector errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_error_directories(root: Path) -> dict[str, Path]:
    categories = [
        "false_positives",
        "false_negatives",
        "localization_errors",
        "crowded_frames",
        "small_players",
        "occlusions",
    ]
    paths = {category: root / category for category in categories}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def summarize_error_samples(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        category = str(sample.get("category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    return counts
