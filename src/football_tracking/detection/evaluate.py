"""Evaluate YOLO detectors with the Ultralytics validator when available."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.detection.checkpoint import compute_file_hash
from football_tracking.detection.metrics import (
    BaselineMetrics,
    metrics_not_available,
    parse_ultralytics_metrics,
)
from football_tracking.detection.serialization import runtime_versions
from football_tracking.detection.training_config import (
    load_evaluation_config,
)
from football_tracking.reporting.detector_report import write_detector_metrics


def evaluate_with_ultralytics(
    weights: str | Path,
    val_args_or_data_yaml: dict[str, Any] | Path,
    split: str | None = None,
    imgsz: int | None = None,
    conf: float | None = None,
    iou: float | None = None,
    batch: int | None = None,
    device: str | None = None,
    model: Any | None = None,
) -> BaselineMetrics:
    if isinstance(val_args_or_data_yaml, dict):
        val_args = val_args_or_data_yaml
    else:
        val_args = {
            "data": str(val_args_or_data_yaml),
            "split": split,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "batch": batch,
            "device": None if device == "auto" else device,
            "verbose": False,
        }
        val_args = {key: value for key, value in val_args.items() if value is not None}
    try:
        if model is None:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            model = YOLO(str(weights))
        result = model.val(**val_args)
    except Exception as exc:  # noqa: BLE001
        return metrics_not_available(f"Ultralytics evaluator failed: {exc}")
    return parse_ultralytics_metrics(result)


def evaluate_detector(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    model: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_evaluation_config(config_path, overrides=overrides)
    if dry_run:
        return {
            "dry_run": True,
            "weights": str(config.weights),
            "data_yaml": str(config.data_yaml),
            "split": config.split,
            "val_args": config.sanitized_val_args(),
        }
    if not Path(str(config.weights)).is_file():
        metrics = metrics_not_available(f"Checkpoint does not exist: {config.weights}")
    else:
        metrics = evaluate_with_ultralytics(
            config.weights,
            config.sanitized_val_args(),
            model=model,
        )
    versions = runtime_versions()
    checkpoint_path = Path(str(config.weights))
    payload = {
        "model": "yolov8m_finetuned",
        "checkpoint": str(config.weights),
        "checkpoint_hash": compute_file_hash(checkpoint_path)
        if checkpoint_path.is_file()
        else None,
        "dataset": str(config.data_yaml),
        "split": config.split,
        "image_size": config.evaluation.get("imgsz"),
        "batch": config.evaluation.get("batch"),
        "confidence": config.evaluation.get("conf"),
        "iou": config.evaluation.get("iou"),
        **metrics.to_dict(),
        "device": config.evaluation.get("device"),
        "gpu": versions.get("gpu_name"),
        "cuda_available": versions.get("cuda_available"),
        "smoke_only": "smoke" in config.run_name.lower()
        or "smoke" in str(config.data_yaml).lower(),
    }
    paths = write_detector_metrics(payload, config.metrics_dir, config.output_prefix)
    return {"metrics": payload, "paths": {key: str(path) for key, path in paths.items()}}
