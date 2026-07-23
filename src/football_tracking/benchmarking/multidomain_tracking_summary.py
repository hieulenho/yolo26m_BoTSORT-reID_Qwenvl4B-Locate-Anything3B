"""Build a provenance-preserving tracking summary across heterogeneous domains."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256


class MultidomainTrackingSummaryError(RuntimeError):
    """Raised when benchmark evidence is missing or incompatible."""


_METRICS = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag")


def build_multidomain_tracking_summary(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    entries = config.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise MultidomainTrackingSummaryError("Config requires a non-empty benchmarks list.")
    rows = [_build_row(value, config_file.parent) for value in entries]
    output_dir = _resolve(config.get("output_dir"), config_file.parent)
    paths = {
        "json": output_dir / "multidomain_tracking_summary.json",
        "csv": output_dir / "multidomain_tracking_summary.csv",
        "markdown": output_dir / "multidomain_tracking_summary.md",
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise MultidomainTrackingSummaryError(f"Output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = _load_hardware(config.get("hardware_source"), config_file.parent)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "hardware": hardware,
        "comparison_scope": {
            "quality": "official or normalized GT evaluated with TrackEval",
            "fps": "measured end-to-end detector+tracker throughput on the recorded hardware",
            "cross_domain_warning": (
                "Dataset difficulty, class ontology, resolution, and frame rate differ; "
                "compare profiles within a dataset before comparing raw scores across domains."
            ),
            "ctc_warning": (
                "CTC rows use bounding boxes derived from TRA masks and TrackEval; they do "
                "not replace the official CTC DET/SEG/TRA evaluator."
            ),
        },
        "benchmarks": rows,
    }
    _write_json(paths["json"], payload)
    _write_csv(paths["csv"], rows)
    _write_text(paths["markdown"], _markdown(payload))
    figures = _write_figures(rows, output_dir / "figures")
    published_paths: dict[str, Path] = {}
    published_figures: list[Path] = []
    if config.get("publish_report_dir"):
        publish_report_dir = _resolve(config.get("publish_report_dir"), config_file.parent)
        publish_report_dir.mkdir(parents=True, exist_ok=True)
        for key in ("json", "csv"):
            destination = publish_report_dir / paths[key].name
            shutil.copy2(paths[key], destination)
            published_paths[key] = destination
        published_markdown = publish_report_dir / paths["markdown"].name
        _write_text(
            published_markdown,
            _markdown(payload).replace("(figures/", "(../assets/benchmarks/"),
        )
        published_paths["markdown"] = published_markdown
    if config.get("publish_figure_dir"):
        publish_figure_dir = _resolve(config.get("publish_figure_dir"), config_file.parent)
        publish_figure_dir.mkdir(parents=True, exist_ok=True)
        for figure in figures:
            destination = publish_figure_dir / figure.name
            shutil.copy2(figure, destination)
            published_figures.append(destination)
    return {
        "status": "ok",
        "benchmark_count": len(rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figures],
        "published_paths": {key: str(path) for key, path in published_paths.items()},
        "published_figures": [str(path) for path in published_figures],
        "benchmarks": rows,
    }


def _build_row(value: Any, base: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MultidomainTrackingSummaryError("Each benchmark entry must be a mapping.")
    benchmark_id = str(value.get("id", "")).strip()
    if not benchmark_id:
        raise MultidomainTrackingSummaryError("Each benchmark requires an id.")
    metrics_path = _resolve(value.get("metrics_source"), base)
    payload = _read_json(metrics_path)
    metric_row = _select_metrics(payload, value.get("selector"), benchmark_id)
    frame_count = int(value.get("frame_count") or metric_row.get("frame_count") or 0)
    sequence_count = int(value.get("sequence_count") or metric_row.get("sequence_count") or 0)
    fps, fps_sources = _read_fps(value.get("fps"), metric_row, base)
    row: dict[str, Any] = {
        "id": benchmark_id,
        "domain": str(value.get("domain", "unknown")),
        "dataset": str(value.get("dataset", "")),
        "profile": str(value.get("profile", "")),
        "detector": str(value.get("detector", "")),
        "tracker": str(
            value.get("tracker")
            or metric_row.get("tracker")
            or metric_row.get("tracker_name")
            or ""
        ),
        "evaluation_mode": str(value.get("evaluation_mode", "")),
        "sequence_count": sequence_count,
        "frame_count": frame_count,
    }
    for key in _METRICS:
        row[key] = _number(metric_row.get(key))
    row.update(
        {
            "end_to_end_fps": fps,
            "idsw_per_1000_frames": (
                round(1000.0 * float(row["IDSW"]) / frame_count, 6)
                if row["IDSW"] is not None and frame_count > 0
                else None
            ),
            "scope_note": str(value.get("scope_note", "")),
            "metrics_source": str(metrics_path),
            "metrics_sha256": file_sha256(metrics_path),
            "performance_sources": [str(path) for path in fps_sources],
            "performance_sha256": [file_sha256(path) for path in fps_sources],
        }
    )
    return row


def _select_metrics(
    payload: dict[str, Any], selector: Any, benchmark_id: str
) -> dict[str, Any]:
    if selector is None:
        metrics = payload.get("metrics", payload)
        if not isinstance(metrics, dict):
            raise MultidomainTrackingSummaryError(
                f"Metrics for '{benchmark_id}' must be an object."
            )
        return dict(metrics)
    if not isinstance(selector, dict):
        raise MultidomainTrackingSummaryError(f"selector for '{benchmark_id}' must be a mapping.")
    container = payload.get(str(selector.get("container", "")))
    if not isinstance(container, list):
        raise MultidomainTrackingSummaryError(
            f"Selector container for '{benchmark_id}' is not a list."
        )
    key = str(selector.get("key", ""))
    expected = str(selector.get("value", ""))
    matches = [row for row in container if isinstance(row, dict) and str(row.get(key)) == expected]
    if len(matches) != 1:
        raise MultidomainTrackingSummaryError(
            f"Selector for '{benchmark_id}' matched {len(matches)} rows; expected 1."
        )
    return dict(matches[0])


def _read_fps(
    value: Any, metric_row: dict[str, Any], base: Path
) -> tuple[float | None, list[Path]]:
    if value is None:
        return None, []
    if not isinstance(value, dict):
        raise MultidomainTrackingSummaryError("fps must be a mapping.")
    row_key = value.get("row_key")
    if row_key:
        return _number(metric_row.get(str(row_key))), []
    reports = value.get("reports")
    if not isinstance(reports, list) or not reports:
        raise MultidomainTrackingSummaryError("fps requires row_key or non-empty reports.")
    paths = [_resolve(path, base) for path in reports]
    frames_key = str(value.get("frames_key", "tracking.frame_count"))
    seconds_key = str(value.get("seconds_key", "tracking.timing.total_pipeline_seconds"))
    total_frames = 0.0
    total_seconds = 0.0
    for path in paths:
        report = _read_json(path)
        frames = _number(_nested(report, frames_key))
        seconds = _number(_nested(report, seconds_key))
        if frames is None or seconds is None or seconds <= 0:
            raise MultidomainTrackingSummaryError(f"Invalid FPS evidence in {path}")
        total_frames += frames
        total_seconds += seconds
    return round(total_frames / total_seconds, 6), paths


def _load_hardware(value: Any, base: Path) -> dict[str, Any]:
    if value is None:
        return {}
    path = _resolve(value, base)
    payload = _read_json(path)
    hardware = payload.get("hardware", {})
    if not isinstance(hardware, dict):
        raise MultidomainTrackingSummaryError(f"hardware must be an object in {path}")
    return {**hardware, "source": str(path), "source_sha256": file_sha256(path)}


def _nested(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for key in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MultidomainTrackingSummaryError(f"Evidence does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MultidomainTrackingSummaryError(f"Expected JSON object: {path}")
    return payload


def _resolve(value: Any, base: Path) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise MultidomainTrackingSummaryError("Expected a non-empty path.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _number(value: Any) -> float | None:
    return float(value) if value is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    hardware = payload["hardware"]
    lines = [
        "# Multi-domain tracking benchmark",
        "",
        f"Hardware: {hardware.get('gpu_name', 'unknown GPU')}, "
        f"{_gib(hardware.get('gpu_memory_total_bytes'))} GiB VRAM, "
        f"{hardware.get('physical_cpu_count', 'unknown')} physical CPU cores.",
        "",
        "| Domain / setting | Detector + tracker | HOTA | DetA | AssA | MOTA | "
        "IDF1 | IDSW | IDSW/1k frames | E2E FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["benchmarks"]:
        lines.append(
            f"| {row['domain']} / {row['profile']} | {row['detector']} + {row['tracker']} | "
            f"{_fmt(row['HOTA'])} | {_fmt(row['DetA'])} | {_fmt(row['AssA'])} | "
            f"{_fmt(row['MOTA'])} | {_fmt(row['IDF1'])} | {_integer(row['IDSW'])} | "
            f"{_fmt(row['idsw_per_1000_frames'])} | {_fmt(row['end_to_end_fps'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading the table",
            "",
            f"- {payload['comparison_scope']['cross_domain_warning']}",
            f"- {payload['comparison_scope']['ctc_warning']}",
            "- Zero-shot failures are retained as results, not removed from the comparison.",
            "- FPS is tied to the hardware above and is not portable to another GPU/CPU.",
            "",
        ]
    )
    for row in payload["benchmarks"]:
        lines.append(f"- **{row['id']}**: {row['scope_note']}")
    lines.extend(
        [
            "",
            "![Tracking quality](figures/multidomain_tracking_quality.png)",
            "",
            "![Throughput](figures/multidomain_tracking_fps.png)",
            "",
            "![Identity errors](figures/multidomain_tracking_identity.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _write_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["id"]).replace("_", "\n") for row in rows]
    positions = np.arange(len(rows))
    figures: list[Path] = []

    figure, axis = plt.subplots(figsize=(12, 5.5))
    width = 0.25
    for offset, key, label, color in (
        (-width, "HOTA", "HOTA", "#2463A8"),
        (0.0, "MOTA", "MOTA", "#2E8B57"),
        (width, "IDF1", "IDF1", "#F28E2B"),
    ):
        axis.bar(
            positions + offset,
            [float(row[key] or 0) for row in rows],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "multidomain_tracking_quality.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    figure, axis = plt.subplots(figsize=(11, 5))
    bars = axis.bar(labels, [float(row["end_to_end_fps"] or 0) for row in rows], color="#2E8B57")
    axis.set_ylabel("End-to-end FPS")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    figure.tight_layout()
    path = output_dir / "multidomain_tracking_fps.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, [float(row["IDSW"] or 0) for row in rows], color="#B83A3A")
    axes[0].set_ylabel("Official TrackEval IDSW")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [float(row["idsw_per_1000_frames"] or 0) for row in rows], color="#8B5CF6")
    axes[1].set_ylabel("IDSW per 1,000 evaluated frames")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "multidomain_tracking_identity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)
    return figures


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _integer(value: Any) -> str:
    return "n/a" if value is None else str(int(float(value)))


def _gib(value: Any) -> str:
    return "unknown" if value is None else f"{float(value) / (1024**3):.1f}"


__all__ = ["MultidomainTrackingSummaryError", "build_multidomain_tracking_summary"]
