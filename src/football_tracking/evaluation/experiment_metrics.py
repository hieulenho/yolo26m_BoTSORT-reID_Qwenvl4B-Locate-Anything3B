"""Metric names and helpers for tracker experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from football_tracking.experiments.schemas import TRACKING_METRIC_NAMES


def empty_tracking_metrics() -> dict[str, float | int | None]:
    return {name: None for name in TRACKING_METRIC_NAMES}


def write_overall_metrics(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "confidence_threshold",
        "tracker_config_hash",
        "sequence_count",
        "frame_count",
        *TRACKING_METRIC_NAMES,
        "tracker_fps",
        "cached_pipeline_fps",
        "unique_predicted_ids",
        "smoke_only",
        "partial_sequences",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
