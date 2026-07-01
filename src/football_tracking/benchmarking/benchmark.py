"""Generate detector/tracker benchmark tables and figures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class BenchmarkError(RuntimeError):
    """Raised when benchmark generation fails."""


BENCHMARK_FIELDS = [
    "Detector",
    "Tracker",
    "mAP50",
    "mAP50-95",
    "Precision",
    "Recall",
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDSW",
    "FP",
    "FN",
    "FPS",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    project_root: Path
    config_path: Path
    detector_metrics: Path
    tracker_overall: Path
    tracker_per_sequence: Path
    output_root: Path
    figures_root: Path
    overwrite: bool
    log_level: str


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_benchmark_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> BenchmarkConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise BenchmarkError(f"Benchmark config does not exist: {resolved}")
    raw = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "benchmark config root")
    inputs = _mapping(raw.get("inputs"), "inputs")
    output = _mapping(raw.get("output"), "output")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    config = BenchmarkConfig(
        project_root=project_root,
        config_path=resolved,
        detector_metrics=_resolve_path(
            inputs.get("detector_metrics"),
            project_root,
            "inputs.detector_metrics",
        ),
        tracker_overall=_resolve_path(
            inputs.get("tracker_overall"),
            project_root,
            "inputs.tracker_overall",
        ),
        tracker_per_sequence=_resolve_path(
            inputs.get("tracker_per_sequence"),
            project_root,
            "inputs.tracker_per_sequence",
        ),
        output_root=_resolve_path(output.get("root"), project_root, "output.root"),
        figures_root=_resolve_path(output.get("figures_root"), project_root, "output.figures_root"),
        overwrite=bool(runtime.get("overwrite", True)),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if overrides and overrides.get("overwrite") is not None:
        config = replace(config, overwrite=bool(overrides["overwrite"]))
    return config


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar(payload: dict[str, Any] | None, key: str) -> Any:
    if not payload:
        return None
    metrics = payload.get("metrics", payload)
    return metrics.get(key)


def generate_benchmark(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_benchmark_config(config_path, overrides=overrides)
    detector_payload = _load_json(config.detector_metrics)
    tracker_rows = _load_json(config.tracker_overall) or []
    rows = [_benchmark_row(detector_payload, tracker) for tracker in tracker_rows]
    if dry_run:
        return {
            "dry_run": True,
            "row_count": len(rows),
            "detector_metrics": str(config.detector_metrics),
            "tracker_overall": str(config.tracker_overall),
            "output_root": str(config.output_root),
        }
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.figures_root.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_root / "benchmark.csv"
    json_path = config.output_root / "benchmark.json"
    md_path = config.output_root / "benchmark.md"
    _write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_benchmark_markdown(rows), encoding="utf-8")
    figures = _write_figures(rows, config.tracker_per_sequence, config.figures_root)
    return {
        "dry_run": False,
        "row_count": len(rows),
        "paths": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "figures": [str(path) for path in figures],
        },
        "warnings": _warnings(detector_payload, rows),
    }


def _benchmark_row(
    detector_payload: dict[str, Any] | None,
    tracker: dict[str, Any],
) -> dict[str, Any]:
    detector_metrics = (detector_payload or {}).get("metrics", detector_payload or {})
    return {
        "Detector": detector_metrics.get("model", "yolov8m_finetuned"),
        "Tracker": tracker.get("tracker"),
        "mAP50": _scalar(detector_payload, "map50"),
        "mAP50-95": _scalar(detector_payload, "map50_95"),
        "Precision": _scalar(detector_payload, "precision"),
        "Recall": _scalar(detector_payload, "recall"),
        "HOTA": tracker.get("HOTA"),
        "DetA": tracker.get("DetA"),
        "AssA": tracker.get("AssA"),
        "MOTA": tracker.get("MOTA"),
        "IDF1": tracker.get("IDF1"),
        "IDSW": tracker.get("IDSW"),
        "FP": tracker.get("FP"),
        "FN": tracker.get("FN"),
        "FPS": tracker.get("tracker_fps"),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Football Tracking Benchmark",
        "",
        "| " + " | ".join(BENCHMARK_FIELDS) + " |",
        "| " + " | ".join("---" for _field in BENCHMARK_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in BENCHMARK_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_figures(
    rows: list[dict[str, Any]],
    per_sequence_csv: Path,
    figures_root: Path,
) -> list[Path]:
    written: list[Path] = []
    for metric, filename in (
        ("Precision", "precision.png"),
        ("Recall", "recall.png"),
        ("mAP50", "map50.png"),
        ("mAP50-95", "map50_95.png"),
        ("HOTA", "hota.png"),
        ("IDF1", "idf1.png"),
        ("MOTA", "mota.png"),
        ("IDSW", "idsw.png"),
        ("FPS", "fps.png"),
    ):
        path = _write_bar(rows, metric, figures_root / filename)
        if path is not None:
            written.append(path)
    scatter = _write_scatter(rows, "FPS", "HOTA", figures_root / "speed_vs_hota.png")
    if scatter is not None:
        written.append(scatter)
    written.extend(_write_per_sequence_figures(per_sequence_csv, figures_root))
    return written


def _write_bar(rows: list[dict[str, Any]], metric: str, path: Path) -> Path | None:
    values = [(row.get("Tracker"), _numeric(row.get(metric))) for row in rows]
    values = [(name, value) for name, value in values if name and value is not None]
    if not values:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([str(name) for name, _value in values], [float(value) for _name, value in values])
    axis.set_title(metric)
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_scatter(
    rows: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    path: Path,
) -> Path | None:
    values = [
        (str(row.get("Tracker")), _numeric(row.get(x_metric)), _numeric(row.get(y_metric)))
        for row in rows
    ]
    values = [
        (name, x_value, y_value)
        for name, x_value, y_value in values
        if x_value is not None and y_value is not None
    ]
    if not values:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for name, x_value, y_value in values:
        axis.scatter([x_value], [y_value], s=60, label=name)
        axis.annotate(name, (x_value, y_value), textcoords="offset points", xytext=(5, 5))
    axis.set_xlabel(x_metric)
    axis.set_ylabel(y_metric)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_per_sequence_figures(per_sequence_csv: Path, figures_root: Path) -> list[Path]:
    if not per_sequence_csv.is_file():
        return []
    with per_sequence_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    written: list[Path] = []
    for metric, filename in (("HOTA", "per_sequence_hota.png"), ("IDF1", "per_sequence_idf1.png")):
        values = [
            (f"{row.get('tracker')}:{row.get('sequence')}", _numeric(row.get(metric)))
            for row in rows
        ]
        values = [(name, value) for name, value in values if value is not None]
        if not values:
            continue
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        path = figures_root / filename
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar([name for name, _value in values], [float(value) for _name, value in values])
        axis.set_title(metric)
        axis.tick_params(axis="x", labelrotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def _warnings(detector_payload: dict[str, Any] | None, rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if detector_payload is None:
        warnings.append("Detector metrics file is missing.")
    else:
        detector_metrics = detector_payload.get("metrics", detector_payload)
        if detector_metrics.get("reason"):
            warnings.append(str(detector_metrics["reason"]))
    if any(row.get("HOTA") is None for row in rows):
        warnings.append("At least one tracker metric is missing.")
    return warnings
