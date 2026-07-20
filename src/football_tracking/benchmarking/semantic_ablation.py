"""Consolidate Qwen semantic ablations with timing and ground-truth metrics."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256, runtime_versions


class SemanticAblationError(RuntimeError):
    """Raised when a semantic ablation manifest is invalid."""


def build_semantic_ablation_report(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SemanticAblationError("Semantic ablation config requires non-empty runs.")
    output_dir = _resolve(config.get("output_dir"), config_file.parent)
    rows = [_read_run(row, config_file.parent) for row in runs]
    paths = {
        "json": output_dir / "semantic_ablation_summary.json",
        "csv": output_dir / "semantic_ablation_summary.csv",
        "markdown": output_dir / "semantic_ablation_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SemanticAblationError(
            "Semantic ablation output exists and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "hardware": runtime_versions(),
        "runs": rows,
    }
    _write_json(paths["json"], payload)
    _write_csv(paths["csv"], rows)
    _write_text(paths["markdown"], _markdown(payload))
    figures = _write_figures(rows, output_dir / "figures")
    return {
        "status": "ok",
        "run_count": len(rows),
        "paths": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
        "runs": rows,
    }


def _read_run(value: Any, base: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or not str(value.get("name", "")).strip():
        raise SemanticAblationError("Each semantic run requires a name.")
    answer_path = _resolve(value.get("answer"), base)
    if not answer_path.is_file():
        raise SemanticAblationError(f"Qwen answer does not exist: {answer_path}")
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    evaluation_path = (
        _resolve(value["evaluation"], base) if value.get("evaluation") else None
    )
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path is not None and evaluation_path.is_file()
        else {}
    )
    coverage = answer.get("coverage", {})
    timing = answer.get("timing", {})
    memory = answer.get("cuda_memory", {})
    metrics = evaluation.get("summary", {})
    expected = int(coverage.get("expected_track_count") or 0)
    predicted = int(coverage.get("predicted_track_count") or 0)
    return {
        "name": str(value["name"]),
        "description": str(value.get("description", "")),
        "quantization": str(answer.get("quantization", "unknown")),
        "batch_count": int(answer.get("batch_count") or 0),
        "image_count": int(answer.get("image_count") or 0),
        "expected_tracks": expected,
        "predicted_tracks": predicted,
        "model_coverage": round(predicted / expected, 6) if expected else 0.0,
        "semantic_accuracy_gt": metrics.get("semantic_track_accuracy"),
        "semantic_macro_f1_gt": metrics.get("semantic_macro_f1"),
        "selective_accuracy_gt": metrics.get("semantic_selective_accuracy"),
        "model_load_seconds": timing.get("model_load_seconds"),
        "inference_seconds": timing.get("inference_seconds"),
        "peak_allocated_gib": _gib(memory.get("peak_allocated_bytes")),
        "peak_reserved_gib": _gib(memory.get("peak_reserved_bytes")),
        "answer_path": str(answer_path),
        "answer_sha256": file_sha256(answer_path),
        "evaluation_path": str(evaluation_path) if evaluation_path else None,
    }


def _resolve(value: Any, base: Path) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise SemanticAblationError("Expected a non-empty path.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _gib(value: Any) -> float | None:
    return round(float(value) / (1024**3), 4) if value is not None else None


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


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Semantic ablation",
        "",
        "Accuracy fields are populated only when a ground-truth evaluation is supplied.",
        "",
        "| Run | Coverage | GT accuracy | Selective accuracy | Inference (s) | VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            "| {name} | {coverage:.2%} | {accuracy} | {selective} | {seconds} | {vram} |".format(
                name=row["name"],
                coverage=row["model_coverage"],
                accuracy=_format_metric(row["semantic_accuracy_gt"]),
                selective=_format_metric(row["selective_accuracy_gt"]),
                seconds=_format_number(row["inference_seconds"]),
                vram=_format_number(row["peak_allocated_gib"]),
            )
        )
    return "\n".join(lines) + "\n"


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _write_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in rows]
    coverage = [100.0 * float(row["model_coverage"]) for row in rows]
    seconds = [float(row["inference_seconds"] or 0.0) for row in rows]
    vram = [float(row["peak_allocated_gib"] or 0.0) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(names, coverage, color="#2463A8")
    axes[0].set_ylabel("Track coverage (%)")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    scatter = axes[1].scatter(seconds, coverage, s=100, c=vram, cmap="viridis")
    for name, x, y in zip(names, seconds, coverage, strict=True):
        axes[1].annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Inference time (s)")
    axes[1].set_ylabel("Track coverage (%)")
    axes[1].grid(alpha=0.25)
    figure.colorbar(scatter, ax=axes[1], label="Peak allocated VRAM (GiB)")
    figure.tight_layout()
    path = output_dir / "semantic_coverage_efficiency.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [path]


__all__ = ["SemanticAblationError", "build_semantic_ablation_report"]
