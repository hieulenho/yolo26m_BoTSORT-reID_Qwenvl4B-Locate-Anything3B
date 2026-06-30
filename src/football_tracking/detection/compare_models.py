"""Compare pretrained and fine-tuned detector reports without inventing metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_detector_reports(metrics_dir: Path, figures_dir: Path) -> dict[str, Any]:
    baseline = _load_optional_json(metrics_dir / "yolov8m_pretrained_baseline.json")
    val = _load_optional_json(metrics_dir / "yolov8m_finetuned_val.json")
    test = _load_optional_json(metrics_dir / "yolov8m_finetuned_test.json")
    rows = []
    warnings = []
    for name, payload in (
        ("pretrained", baseline),
        ("finetuned_val", val),
        ("finetuned_test", test),
    ):
        if payload is None:
            warnings.append(f"Missing metrics for {name}.")
            continue
        metrics = payload.get("metrics", payload)
        rows.append(
            {
                "model": name,
                "split": payload.get("split") or payload.get("dataset", {}).get("split"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "map50": metrics.get("map50"),
                "map50_95": metrics.get("map50_95"),
                "latency": metrics.get("latency_per_image_seconds"),
                "fps": metrics.get("detector_fps"),
            }
        )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "detector_comparison.json"
    csv_path = metrics_dir / "detector_comparison.csv"
    payload = {"rows": rows, "warnings": warnings}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "split",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "latency",
            "fps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_comparison_figures(rows, figures_dir)
    return {"json": str(json_path), "csv": str(csv_path), "warnings": warnings}


def _write_comparison_figures(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("precision", "precision_comparison.png"),
        ("recall", "recall_comparison.png"),
        ("map50", "map50_comparison.png"),
        ("map50_95", "map50_95_comparison.png"),
        ("latency", "latency_comparison.png"),
        ("fps", "fps_comparison.png"),
    ]
    for metric, filename in metrics:
        values = [(row["model"], row.get(metric)) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.bar([item[0] for item in values], [float(item[1]) for item in values])
        plt.title(metric)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(figures_dir / filename)
        plt.close()
