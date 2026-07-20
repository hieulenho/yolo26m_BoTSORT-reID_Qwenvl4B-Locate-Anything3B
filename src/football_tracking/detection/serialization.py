"""Write detection predictions and runtime metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from football_tracking.data.bbox import xyxy_to_yolo_normalized
from football_tracking.detection.schemas import Detection


def write_predictions_jsonl(detections: list[Detection], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for detection in detections:
            handle.write(json.dumps(detection.to_dict(), sort_keys=True) + "\n")
    return output_path


def write_yolo_prediction_labels(detections: list[Detection], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.sequence_name, detection.frame_index)].append(detection)
    paths: list[Path] = []
    for (sequence_name, frame_index), items in sorted(grouped.items()):
        path = output_dir / f"{sequence_name}_{frame_index:06d}.txt"
        lines = []
        for detection in items:
            x_center, y_center, width, height = xyxy_to_yolo_normalized(
                detection.bbox_xyxy,
                detection.image_width,
                detection.image_height,
            )
            lines.append(
                f"{detection.target_class_id} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                f"{detection.confidence:.6f}"
            )
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        paths.append(path)
    return paths


def write_predictions_summary_csv(detections: list[Detection], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(detection.sequence_name for detection in detections)
    confidences: dict[str, list[float]] = defaultdict(list)
    for detection in detections:
        confidences[detection.sequence_name].append(detection.confidence)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_name", "prediction_count", "mean_confidence"],
        )
        writer.writeheader()
        for sequence_name in sorted(counts):
            values = confidences[sequence_name]
            writer.writerow(
                {
                    "sequence_name": sequence_name,
                    "prediction_count": counts[sequence_name],
                    "mean_confidence": sum(values) / len(values) if values else None,
                }
            )
    return output_path


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        versions["physical_cpu_count"] = psutil.cpu_count(logical=False)
        versions["system_memory_total_bytes"] = int(memory.total)
    except Exception as exc:  # noqa: BLE001
        versions["physical_cpu_count"] = None
        versions["system_memory_total_bytes"] = None
        versions["system_info_error"] = str(exc)
    try:
        import torch  # type: ignore[import-not-found]

        versions["torch"] = torch.__version__
        versions["cuda_runtime"] = torch.version.cuda
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_device_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            versions["gpu_name"] = properties.name
            versions["gpu_memory_total_bytes"] = int(properties.total_memory)
        else:
            versions["gpu_name"] = None
            versions["gpu_memory_total_bytes"] = None
    except Exception as exc:  # noqa: BLE001
        versions["torch"] = None
        versions["cuda_runtime"] = None
        versions["cuda_available"] = False
        versions["cuda_device_count"] = 0
        versions["gpu_name"] = None
        versions["gpu_memory_total_bytes"] = None
        versions["torch_error"] = str(exc)
    try:
        import ultralytics  # type: ignore[import-not-found]

        versions["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        versions["ultralytics"] = None
        versions["ultralytics_error"] = str(exc)
    return versions


def write_run_metadata(metadata: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path
