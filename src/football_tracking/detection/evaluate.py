"""Evaluate a YOLO baseline with the Ultralytics validator when available."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.detection.metrics import (
    BaselineMetrics,
    metrics_not_available,
    parse_ultralytics_metrics,
)


def evaluate_with_ultralytics(
    weights: str | Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    device: str,
    model: Any | None = None,
) -> BaselineMetrics:
    try:
        if model is None:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            model = YOLO(str(weights))
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch=batch,
            device=None if device == "auto" else device,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return metrics_not_available(f"Ultralytics evaluator failed: {exc}")
    return parse_ultralytics_metrics(result)
