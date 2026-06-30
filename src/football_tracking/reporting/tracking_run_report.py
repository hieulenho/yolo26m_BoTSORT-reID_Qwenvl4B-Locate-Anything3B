"""Tracking run summary writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_tracking_run_report(
    payload: dict[str, Any],
    metrics_dir: Path,
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "deepsort_tracking_run.json"
    csv_path = metrics_dir / "deepsort_tracking_per_sequence.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    rows = payload.get("sequences", [])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
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
    return {"json": json_path, "csv": csv_path}
