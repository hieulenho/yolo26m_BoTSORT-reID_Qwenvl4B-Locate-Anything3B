"""Create tracker adapters from experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.tracking.deepsort_adapter import (
    DeepSortTrackerAdapter,
    load_deepsort_config,
)
from football_tracking.tracking.sort_adapter import SortTrackerAdapter, load_sort_config


class TrackerFactoryError(RuntimeError):
    """Raised when a tracker cannot be created."""


def create_tracker(name: str, config: str | Path, device: str = "auto") -> Any:
    normalized = name.lower().strip()
    if normalized == "sort":
        return SortTrackerAdapter(load_sort_config(config))
    if normalized == "deepsort":
        return DeepSortTrackerAdapter(load_deepsort_config(config, device=device))
    raise TrackerFactoryError(f"Unsupported tracker: {name}")


def load_tracker_runtime_config(
    name: str,
    config: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    normalized = name.lower().strip()
    if normalized == "sort":
        return load_sort_config(config).to_dict()
    if normalized == "deepsort":
        return load_deepsort_config(config, device=device).to_dict()
    raise TrackerFactoryError(f"Unsupported tracker: {name}")
