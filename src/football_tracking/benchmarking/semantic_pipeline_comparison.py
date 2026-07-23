"""Compare semantic Pipelines A/B/C using the same reference ground truth."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256, runtime_versions


class SemanticPipelineComparisonError(RuntimeError):
    """Raised when semantic pipeline comparison inputs are invalid."""


def build_semantic_pipeline_comparison(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    pipelines = config.get("pipelines")
    if not isinstance(pipelines, list) or not pipelines:
        raise SemanticPipelineComparisonError(
            "Comparison config requires a non-empty pipelines list."
        )
    frame_count = int(config.get("frame_count") or 0)
    rows = [_read_pipeline(row, config_file.parent, frame_count) for row in pipelines]
    output_dir = _resolve(config.get("output_dir"), config_file.parent)
    paths = {
        "json": output_dir / "semantic_pipeline_comparison.json",
        "csv": output_dir / "semantic_pipeline_comparison.csv",
        "markdown": output_dir / "semantic_pipeline_comparison.md",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise SemanticPipelineComparisonError(
                "Comparison output exists and overwrite=false: "
                + ", ".join(str(path) for path in existing)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "frame_count": frame_count,
        "hardware": runtime_versions(),
        "measurement_scope": {
            "quality": str(
                config.get("quality_scope", "same human-reviewed GT manifest")
            ),
            "tracking_fps": "video tracking stage only",
            "effective_cold_fps": (
                "frame_count / (tracking wall time + cold semantic wall time)"
            ),
            "semantic_execution": "selected-track batch analysis, not per-frame VLM",
            "combined_peak_vram": (
                "maximum component peak because Qwen and Locate run sequentially"
            ),
        },
        "pipelines": rows,
    }
    _write_json(paths["json"], payload)
    _write_csv(paths["csv"], rows)
    _write_text(paths["markdown"], _markdown(payload))
    figures = _write_figures(rows, output_dir / "figures")
    return {
        "status": "ok",
        "pipeline_count": len(rows),
        "paths": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
        "pipelines": rows,
    }


def _read_pipeline(value: Any, base: Path, frame_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SemanticPipelineComparisonError("Each pipeline entry must be a mapping.")
    pipeline_id = str(value.get("id", "")).strip()
    if not pipeline_id:
        raise SemanticPipelineComparisonError("Each pipeline requires an id.")
    evaluation_path = _resolve(value.get("evaluation"), base)
    if not evaluation_path.is_file():
        raise SemanticPipelineComparisonError(
            f"Evaluation does not exist: {evaluation_path}"
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = evaluation.get("summary", {})
    performance = summary.get("performance_means", {})
    qwen_load = _number(performance.get("qwen_model_load_seconds"))
    qwen_inference = _number(performance.get("qwen_inference_seconds"))
    locate_load = _number(performance.get("locate_model_load_seconds"))
    locate_execution = _number(performance.get("locate_inference_seconds"))
    semantic_cold_seconds = sum(
        value for value in (qwen_load, qwen_inference, locate_load, locate_execution)
        if value is not None
    )
    tracking_fps = _number(performance.get("tracking_end_to_end_fps"))
    tracking_seconds = (
        frame_count / tracking_fps if frame_count > 0 and tracking_fps else None
    )
    effective_cold_fps = (
        frame_count / (tracking_seconds + semantic_cold_seconds)
        if tracking_seconds is not None and semantic_cold_seconds > 0
        else tracking_fps
    )
    qwen_peak = _number(performance.get("qwen_peak_allocated_bytes"))
    locate_peak = _number(performance.get("locate_peak_allocated_bytes"))
    component_peaks = [value for value in (qwen_peak, locate_peak) if value is not None]
    return {
        "pipeline": pipeline_id,
        "name": str(value.get("name", pipeline_id)),
        "components": str(value.get("components", "")),
        "semantic_accuracy": summary.get("semantic_track_accuracy"),
        "semantic_macro_f1": summary.get("semantic_macro_f1"),
        "semantic_coverage": summary.get("semantic_coverage"),
        "selective_accuracy": summary.get("semantic_selective_accuracy"),
        "fine_label_accuracy": summary.get("fine_semantic_track_accuracy"),
        "fine_candidate_accuracy": summary.get("fine_candidate_accuracy"),
        "fine_candidate_coverage": summary.get("fine_candidate_coverage"),
        "unknown_rejection_f1": summary.get("unknown_rejection_f1"),
        "hallucination_rate": summary.get("semantic_hallucination_rate"),
        "gt_tracks": summary.get("semantic_gt_track_count"),
        "accepted_tracks": summary.get("semantic_accepted_track_count"),
        "tracking_fps": tracking_fps,
        "qwen_load_seconds": qwen_load,
        "qwen_inference_seconds": qwen_inference,
        "locate_load_seconds": locate_load,
        "locate_execution_seconds": locate_execution,
        "semantic_cold_seconds": round(semantic_cold_seconds, 6),
        "effective_cold_fps": (
            round(effective_cold_fps, 6) if effective_cold_fps is not None else None
        ),
        "qwen_peak_gib": _gib(qwen_peak),
        "locate_peak_gib": _gib(locate_peak),
        "sequential_peak_gib": _gib(max(component_peaks) if component_peaks else None),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": file_sha256(evaluation_path),
    }


def _resolve(value: Any, base: Path) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise SemanticPipelineComparisonError("Expected a non-empty path.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _number(value: Any) -> float | None:
    return float(value) if value is not None else None


def _gib(value: float | None) -> float | None:
    return round(value / (1024**3), 4) if value is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Semantic pipeline comparison",
        "",
        f"Quality scope: {payload['measurement_scope']['quality']}.",
        "Unknown predictions count as errors in accuracy and are excluded from selective accuracy.",
        "",
        "| Pipeline | Accuracy | Macro F1 | Fine accepted | Fine candidate | "
        "Unknown F1 | Hallucination | Coverage | Semantic cold (s) | "
        "Effective cold FPS | Peak VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["pipelines"]:
        lines.append(
            "| {pipeline} | {accuracy} | {f1} | {fine} | {fine_candidate} | {unknown} | "
            "{hallucination} | {coverage} | {seconds} | {fps} | {vram} |".format(
                pipeline=row["pipeline"],
                accuracy=_percent(row["semantic_accuracy"]),
                f1=_percent(row["semantic_macro_f1"]),
                fine=_percent(row["fine_label_accuracy"]),
                fine_candidate=_percent(row["fine_candidate_accuracy"]),
                unknown=_percent(row["unknown_rejection_f1"]),
                hallucination=_percent(row["hallucination_rate"]),
                coverage=_percent(row["semantic_coverage"]),
                seconds=_format(row["semantic_cold_seconds"]),
                fps=_format(row["effective_cold_fps"]),
                vram=_format(row["sequential_peak_gib"]),
            )
        )
    lines.extend(
        [
            "",
            "Effective cold FPS includes one cold semantic pass over the selected tracks. ",
            "For a persistent realtime service, semantic discovery is cached and "
            "tracking FPS is the relevant continuous rate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _write_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["pipeline"] for row in rows]
    positions = np.arange(len(rows))

    quality, axis = plt.subplots(figsize=(10, 5.2))
    width = 0.22
    for offset, key, label, color in (
        (-width, "semantic_accuracy", "Accuracy", "#2463A8"),
        (0.0, "fine_label_accuracy", "Fine accuracy", "#8B5CF6"),
        (width, "unknown_rejection_f1", "Unknown rejection F1", "#2E8B57"),
    ):
        values = [100.0 * float(row.get(key) or 0.0) for row in rows]
        axis.bar(positions + offset, values, width, label=label, color=color)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Percent (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    quality.tight_layout()
    quality_path = output_dir / "semantic_pipeline_quality.png"
    quality.savefig(quality_path, dpi=180, bbox_inches="tight")
    plt.close(quality)

    risk, risk_axis = plt.subplots(figsize=(9, 4.8))
    risk_bars = risk_axis.bar(
        labels,
        [100.0 * float(row.get("hallucination_rate") or 0.0) for row in rows],
        color="#B83A3A",
    )
    risk_axis.set_ylabel("Accepted hallucination (%)")
    risk_axis.set_ylim(0, 105)
    risk_axis.grid(axis="y", alpha=0.25)
    risk_axis.bar_label(risk_bars, fmt="%.2f", padding=3)
    risk.tight_layout()
    risk_path = output_dir / "semantic_pipeline_hallucination.png"
    risk.savefig(risk_path, dpi=180, bbox_inches="tight")
    plt.close(risk)

    performance, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    qwen = [
        float(row.get("qwen_load_seconds") or 0.0)
        + float(row.get("qwen_inference_seconds") or 0.0)
        for row in rows
    ]
    locate = [
        float(row.get("locate_load_seconds") or 0.0)
        + float(row.get("locate_execution_seconds") or 0.0)
        for row in rows
    ]
    axes[0].bar(labels, qwen, label="Qwen", color="#2463A8")
    axes[0].bar(labels, locate, bottom=qwen, label="LocateAnything", color="#F28E2B")
    axes[0].set_ylabel("Cold semantic wall time (s)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(
        labels,
        [float(row.get("effective_cold_fps") or 0.0) for row in rows],
        color="#2E8B57",
    )
    axes[1].set_ylabel("Effective cold pipeline FPS")
    axes[1].grid(axis="y", alpha=0.25)
    performance.tight_layout()
    performance_path = output_dir / "semantic_pipeline_performance.png"
    performance.savefig(performance_path, dpi=180, bbox_inches="tight")
    plt.close(performance)
    return [quality_path, risk_path, performance_path]


__all__ = [
    "SemanticPipelineComparisonError",
    "build_semantic_pipeline_comparison",
]
