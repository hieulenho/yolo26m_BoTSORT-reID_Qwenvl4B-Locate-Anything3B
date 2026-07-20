"""Tracking run summary writers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def write_tracking_run_report(
    payload: dict[str, Any],
    metrics_dir: Path,
    tracker_name: str,
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tracker_slug = re.sub(r"[^a-z0-9]+", "_", tracker_name.lower()).strip("_")
    if not tracker_slug:
        raise ValueError("tracker_name must contain at least one letter or digit.")
    json_path = metrics_dir / f"{tracker_slug}_tracking_run.json"
    csv_path = metrics_dir / f"{tracker_slug}_tracking_per_sequence.csv"
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    rows = payload.get("sequences", [])
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sequence_name",
            "frame_count",
            "detection_count",
            "emitted_track_count",
            "unique_track_count",
            "detector_fps",
            "tracker_fps",
            "end_to_end_fps",
            "output_mot",
            "output_video",
            "smoke_only",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    temporary_csv.replace(csv_path)
    return {"json": json_path, "csv": csv_path}
