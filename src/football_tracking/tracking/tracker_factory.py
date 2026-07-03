"""Create tracker adapters from experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.tracking.deepsort_adapter import (
    DeepSortTrackerAdapter,
    load_deepsort_config,
)
from football_tracking.tracking.sort_adapter import SortTrackerAdapter, load_sort_config
from football_tracking.tracking.ultralytics_adapter import (
    UltralyticsTrackerAdapter,
    load_ultralytics_tracker_config,
)


class TrackerFactoryError(RuntimeError):
    """Raised when a tracker cannot be created."""


def create_tracker(name: str, config: str | Path, device: str = "auto") -> Any:
    runtime_config = load_tracker_config_object(name, config, device=device)
    normalized = name.lower().strip()
    if normalized == "sort":
        return SortTrackerAdapter(runtime_config)
    if normalized == "deepsort":
        return DeepSortTrackerAdapter(runtime_config)
    if normalized in {"botsort", "bot-sort", "botsort_reid", "botsort-reid"}:
        return UltralyticsTrackerAdapter(runtime_config)
    if normalized in {"bytetrack", "byte-track"}:
        return UltralyticsTrackerAdapter(runtime_config)
    raise TrackerFactoryError(f"Unsupported tracker: {name}")


def load_tracker_config_object(
    name: str,
    config: str | Path,
    device: str = "auto",
) -> Any:
    normalized = name.lower().strip()
    if normalized == "sort":
        return load_sort_config(config)
    if normalized == "deepsort":
        return load_deepsort_config(config, device=device)
    if normalized in {"botsort", "bot-sort", "botsort_reid", "botsort-reid"}:
        return load_ultralytics_tracker_config(config, default_tracker_type="botsort")
    if normalized in {"bytetrack", "byte-track"}:
        return load_ultralytics_tracker_config(config, default_tracker_type="bytetrack")
    raise TrackerFactoryError(f"Unsupported tracker: {name}")


def load_tracker_runtime_config(
    name: str,
    config: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    runtime_config = load_tracker_config_object(name, config, device=device)
    if hasattr(runtime_config, "to_dict"):
        return runtime_config.to_dict()
    return dict(runtime_config)
