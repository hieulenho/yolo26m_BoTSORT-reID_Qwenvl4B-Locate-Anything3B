"""Fine-tuned detector report writers."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.reporting.detector_tables import markdown_metric_table, metric_display


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items()}


def write_detector_metrics(
    payload: dict[str, Any],
    metrics_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = _safe_payload({**payload, "timestamp": datetime.now(UTC).isoformat()})
    json_path = metrics_dir / f"{prefix}.json"
    csv_path = metrics_dir / f"{prefix}.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow(payload)
    return {"json": json_path, "csv": csv_path}


def write_finetuned_report(
    metrics_dir: Path,
    payload: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    executed: bool = False,
) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = payload or {}
    path = metrics_dir / "yolov8m_finetuned_report.md"
    lines = [
        "# YOLOv8m Fine-Tuned Detector Report",
        "",
        "## Status",
        "",
        "executed" if executed else "not executed",
        "",
        "## Metrics",
        "",
        markdown_metric_table(payload),
        "",
        "## Protocol",
        "",
        "Training uses train split, validation uses val split, and final test evaluation "
        "uses test split once after configuration is fixed.",
        "",
        "mAP@50 is AP at IoU 0.50. mAP@50:95 is averaged from IoU 0.50 to 0.95 in steps of 0.05.",
        "",
        "## Baseline Comparison",
        "",
        metric_display(comparison) if comparison else "not available",
        "",
        "## Limitations",
        "",
        "DeepSORT is not implemented in Milestone 4. Tracking starts in a later milestone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
