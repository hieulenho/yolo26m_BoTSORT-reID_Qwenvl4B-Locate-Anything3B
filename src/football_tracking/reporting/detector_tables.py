"""Detector report table helpers."""

from __future__ import annotations

from typing import Any


def metric_display(value: Any) -> str:
    return "not available" if value is None else str(value)


def markdown_metric_table(payload: dict[str, Any]) -> str:
    rows = [
        ("Split", payload.get("split")),
        ("Precision", payload.get("precision")),
        ("Recall", payload.get("recall")),
        ("mAP@50", payload.get("map50")),
        ("mAP@50:95", payload.get("map50_95")),
        ("mAP@75", payload.get("map75")),
        ("Latency", payload.get("latency_per_image_seconds")),
        ("FPS", payload.get("detector_fps")),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {metric_display(value)} |" for name, value in rows)
    return "\n".join(lines)
