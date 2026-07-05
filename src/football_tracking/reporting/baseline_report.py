"""Write YOLOv8m pretrained baseline reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

COCO_PERSON_LIMITATIONS = (
    "YOLOv8m pretrained on COCO detects the broad person class. In football video, "
    "person can include players, goalkeepers, referees, staff, coaches, and people outside "
    "the field, while the MVP ground truth maps only player-like labels to player."
)


def _display(value: Any) -> Any:
    return "not available" if value is None else value


def write_baseline_report(
    payload: dict[str, Any],
    metrics_dir: Path,
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "yolov8m_pretrained_baseline.json"
    csv_path = metrics_dir / "yolov8m_pretrained_baseline.csv"
    markdown_path = metrics_dir / "yolov8m_pretrained_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    flat = {
        "split": payload.get("dataset", {}).get("split"),
        "model": payload.get("model", {}).get("weights"),
        "imgsz": payload.get("inference", {}).get("imgsz"),
        "conf": payload.get("inference", {}).get("conf"),
        "iou": payload.get("inference", {}).get("iou"),
        "precision": payload.get("metrics", {}).get("precision"),
        "recall": payload.get("metrics", {}).get("recall"),
        "map50": payload.get("metrics", {}).get("map50"),
        "map50_95": payload.get("metrics", {}).get("map50_95"),
        "map75": payload.get("metrics", {}).get("map75"),
        "image_count": payload.get("counts", {}).get("image_count"),
        "prediction_count": payload.get("counts", {}).get("prediction_count"),
        "device": payload.get("runtime", {}).get("device"),
        "gpu": payload.get("runtime", {}).get("gpu_name"),
        "detector_fps": payload.get("timing", {}).get("detector_fps"),
        "end_to_end_fps": payload.get("timing", {}).get("end_to_end_fps"),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    metrics = payload.get("metrics", {})
    timing = payload.get("timing", {})
    lines = [
        "# YOLOv8m Pretrained Baseline",
        "",
        f"- Dataset split: {_display(flat['split'])}",
        f"- Model: {_display(flat['model'])}",
        f"- Image size: {_display(flat['imgsz'])}",
        f"- Confidence threshold: {_display(flat['conf'])}",
        f"- IoU threshold: {_display(flat['iou'])}",
        f"- Precision: {_display(metrics.get('precision'))}",
        f"- Recall: {_display(metrics.get('recall'))}",
        f"- mAP@50: {_display(metrics.get('map50'))}",
        f"- mAP@50:95: {_display(metrics.get('map50_95'))}",
        f"- mAP@75: {_display(metrics.get('map75'))}",
        f"- Latency per image: {_display(timing.get('latency_per_image_seconds'))}",
        f"- Detector FPS: {_display(timing.get('detector_fps'))}",
        f"- End-to-end FPS: {_display(timing.get('end_to_end_fps'))}",
        f"- Image count: {_display(flat['image_count'])}",
        f"- Prediction count: {_display(flat['prediction_count'])}",
        f"- Device: {_display(flat['device'])}",
        f"- GPU: {_display(flat['gpu'])}",
        "",
        "## Metric Availability",
        "",
        metrics.get("reason") or "Metrics were parsed from the Ultralytics validator.",
        "",
        "## Limitations",
        "",
        COCO_PERSON_LIMITATIONS,
        "",
        "This baseline is kept for historical comparison. Current recommended runs use the "
        "fine-tuned YOLO26m football checkpoint and compare SORT, DeepSORT, and BoT-SORT ReID "
        "with TrackEval.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
