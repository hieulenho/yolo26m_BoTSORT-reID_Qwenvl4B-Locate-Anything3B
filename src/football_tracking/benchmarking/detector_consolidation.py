"""Consolidate detector accuracy and timing under one compatibility contract."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import get_project_root, resolve_project_path


class DetectorConsolidationError(RuntimeError):
    """Raised when detector reports cannot be compared fairly."""


def consolidate_detector_benchmark(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    config = _mapping(yaml.safe_load(config_file.read_text(encoding="utf-8")), "config")
    expected = _mapping(config.get("expected"), "expected")
    sources = config.get("sources")
    output = _mapping(config.get("output"), "output")
    if not isinstance(sources, list) or not sources:
        raise DetectorConsolidationError("sources must be a non-empty list.")
    rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(sources):
        source = _mapping(value, f"sources[{index}]")
        name = str(source.get("name", "")).strip()
        if not name or name in seen:
            raise DetectorConsolidationError(
                f"sources[{index}].name is missing or duplicated: {name!r}"
            )
        seen.add(name)
        accuracy_path = _resolve(source.get("accuracy"))
        timing_path = _resolve(source.get("timing"))
        accuracy = _read_json(accuracy_path)
        timing = _read_json(timing_path)
        row = _detector_row(name, source, accuracy, timing)
        _validate_row(row, expected)
        rows.append(row)
        source_manifest.append(
            {
                "name": name,
                "accuracy": str(accuracy_path),
                "accuracy_sha256": file_sha256(accuracy_path),
                "timing": str(timing_path),
                "timing_sha256": file_sha256(timing_path),
            }
        )
    rows.sort(key=lambda item: float(item["map50_95"]), reverse=True)
    output_root = _resolve(output.get("root"), require_file=False)
    paths = {
        "summary_json": output_root / "detector_benchmark_summary.json",
        "summary_csv": output_root / "detector_benchmark_summary.csv",
        "report_md": output_root / "detector_benchmark_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise DetectorConsolidationError(
            "Detector benchmark output exists and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "config_sha256": file_sha256(config_file),
        "compatibility_contract": expected,
        "rows": rows,
        "sources": source_manifest,
        "timing_scope": {
            "detector_fps": "model inference only",
            "end_to_end_fps": "decode/preprocess/inference/postprocess/serialization",
        },
    }
    _write_text_atomic(
        paths["summary_json"], json.dumps(payload, indent=2, ensure_ascii=False)
    )
    _write_csv(paths["summary_csv"], rows)
    _write_text_atomic(paths["report_md"], _markdown(payload))
    figures = _figures(rows, output_root / "figures")
    return {
        "status": "ok",
        "detector_count": len(rows),
        "best_map50_95": rows[0]["name"],
        "paths": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
    }


def _detector_row(
    name: str,
    source: dict[str, Any],
    accuracy: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    accuracy_metrics = accuracy.get("metrics", accuracy)
    timing_metrics = timing.get("timing", {})
    accuracy_dataset = accuracy.get("dataset", {})
    timing_dataset = timing.get("dataset", {})
    if isinstance(accuracy_dataset, str):
        accuracy_dataset = {"data_yaml": accuracy_dataset, "split": accuracy.get("split")}
    accuracy_inference = accuracy.get("inference", {})
    return {
        "name": name,
        "display_name": str(source.get("display_name", name)),
        "training": str(source.get("training", "unknown")),
        "accuracy_weights": _weights(accuracy),
        "timing_weights": _weights(timing),
        "dataset": accuracy_dataset.get("data_yaml"),
        "split": accuracy.get("split") or accuracy_dataset.get("split"),
        "imgsz": accuracy.get("image_size") or accuracy_inference.get("imgsz"),
        "timing_dataset": timing_dataset.get("data_yaml"),
        "timing_split": timing_dataset.get("split"),
        "timing_imgsz": timing.get("inference", {}).get("imgsz"),
        "precision": accuracy_metrics.get("precision"),
        "recall": accuracy_metrics.get("recall"),
        "map50": accuracy_metrics.get("map50"),
        "map75": accuracy_metrics.get("map75"),
        "map50_95": accuracy_metrics.get("map50_95"),
        "detector_fps": timing_metrics.get("detector_fps"),
        "end_to_end_fps": timing_metrics.get("end_to_end_fps"),
        "latency_per_image_seconds": timing_metrics.get("latency_per_image_seconds"),
        "timed_image_count": timing.get("counts", {}).get("image_count"),
        "gpu_name": timing.get("runtime", {}).get("gpu_name"),
    }


def _validate_row(row: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = [
        key
        for key in (
            "precision",
            "recall",
            "map50",
            "map50_95",
            "detector_fps",
            "end_to_end_fps",
        )
        if row.get(key) is None
    ]
    errors = [f"missing metric(s): {', '.join(missing)}"] if missing else []
    expected_split = str(expected.get("split", "val"))
    expected_imgsz = int(expected.get("imgsz", 640))
    expected_images = int(expected.get("timed_image_count", 0))
    if str(row.get("split")) != expected_split or str(row.get("timing_split")) != expected_split:
        errors.append(
            f"split mismatch: accuracy={row.get('split')}, timing={row.get('timing_split')}"
        )
    if int(row.get("imgsz") or -1) != expected_imgsz:
        errors.append(f"accuracy imgsz={row.get('imgsz')} (expected {expected_imgsz})")
    if int(row.get("timing_imgsz") or -1) != expected_imgsz:
        errors.append(f"timing imgsz={row.get('timing_imgsz')} (expected {expected_imgsz})")
    if expected_images and int(row.get("timed_image_count") or -1) != expected_images:
        errors.append(
            f"timed_image_count={row.get('timed_image_count')} (expected {expected_images})"
        )
    expected_gpu = expected.get("gpu_name")
    if expected_gpu and row.get("gpu_name") != expected_gpu:
        errors.append(f"gpu_name={row.get('gpu_name')} (expected {expected_gpu})")
    if errors:
        raise DetectorConsolidationError(
            f"Incompatible detector source '{row['name']}': " + "; ".join(errors)
        )


def _weights(payload: dict[str, Any]) -> str | None:
    model = payload.get("model")
    if isinstance(model, dict):
        return model.get("weights")
    return payload.get("checkpoint")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DetectorConsolidationError(f"{name} must be a mapping.")
    return value


def _resolve(value: Any, *, require_file: bool = True) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise DetectorConsolidationError("Expected a non-empty path.")
    path = Path(value)
    resolved = (
        path.resolve()
        if path.is_absolute()
        else resolve_project_path(path, get_project_root())
    )
    if require_file and not resolved.is_file():
        raise DetectorConsolidationError(f"Required file does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DetectorConsolidationError(f"JSON root must be an object: {path}")
    return value


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Detector benchmark",
        "",
        "All rows use SportsMOT validation, 640 px input, 2,900 timed images, and the same GPU.",
        "",
        "| Detector | Training | P | R | mAP50 | mAP50-95 | Detector FPS | E2E FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['training']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['detector_fps']:.2f} | {row['end_to_end_fps']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _figures(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    names = [row["display_name"] for row in rows]
    paths: list[Path] = []
    for filename, title, metrics in (
        (
            "detector_accuracy.png",
            "Detector accuracy on SportsMOT val",
            ("precision", "recall", "map50", "map50_95"),
        ),
        (
            "detector_speed.png",
            "Detector speed on RTX 4060 Laptop 8GB",
            ("detector_fps", "end_to_end_fps"),
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 5.5))
        width = 0.8 / len(metrics)
        positions = list(range(len(rows)))
        for metric_index, metric in enumerate(metrics):
            offsets = [position - 0.4 + width / 2 + metric_index * width for position in positions]
            axis.bar(offsets, [row[metric] for row in rows], width=width, label=metric)
        axis.set_xticks(positions, names, rotation=20, ha="right")
        axis.set_title(title)
        axis.legend()
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths
