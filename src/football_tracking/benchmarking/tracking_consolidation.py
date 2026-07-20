"""Consolidate compatible MOT benchmark artifacts into one audited report."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import runtime_versions
from football_tracking.paths import get_project_root, resolve_project_path


class TrackingConsolidationError(RuntimeError):
    """Raised when benchmark artifacts cannot be compared safely."""


SUMMARY_FIELDS = (
    "tracker",
    "display_name",
    "confidence_threshold",
    "sequence_count",
    "frame_count",
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "IDSW",
    "FP",
    "FN",
    "Frag",
    "tracker_fps",
    "cached_pipeline_fps",
    "unique_predicted_ids",
    "tracker_config_hash",
    "source_summary",
    "source_per_sequence",
)


@dataclass(frozen=True)
class TrackingSource:
    tracker: str
    display_name: str
    summary: Path
    per_sequence: Path


@dataclass(frozen=True)
class ConsolidationConfig:
    config_path: Path
    output_root: Path
    sources: tuple[TrackingSource, ...]
    expected_sequence_count: int
    expected_frame_count: int
    expected_confidence: float
    detector_cache_root: Path
    dataset: Path
    split: str
    overwrite: bool
    write_figures: bool


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrackingConsolidationError(f"{name} must be a mapping.")
    return value


def _resolve(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrackingConsolidationError(f"{name} must be a non-empty path.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, root)


def load_consolidation_config(path: str | Path) -> ConsolidationConfig:
    root = get_project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = resolve_project_path(config_path, root)
    if not config_path.is_file():
        raise TrackingConsolidationError(f"Config does not exist: {config_path}")
    payload = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    benchmark = _mapping(payload.get("benchmark"), "benchmark")
    expected = _mapping(payload.get("expected"), "expected")
    output = _mapping(payload.get("output"), "output")
    source_values = payload.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise TrackingConsolidationError("sources must be a non-empty list.")

    sources: list[TrackingSource] = []
    seen: set[str] = set()
    for index, value in enumerate(source_values):
        item = _mapping(value, f"sources[{index}]")
        tracker = str(item.get("tracker", "")).strip()
        if not tracker:
            raise TrackingConsolidationError(f"sources[{index}].tracker is required.")
        if tracker in seen:
            raise TrackingConsolidationError(f"Duplicate tracker source: {tracker}")
        seen.add(tracker)
        sources.append(
            TrackingSource(
                tracker=tracker,
                display_name=str(item.get("display_name") or tracker),
                summary=_resolve(item.get("summary"), root, f"sources[{index}].summary"),
                per_sequence=_resolve(
                    item.get("per_sequence"), root, f"sources[{index}].per_sequence"
                ),
            )
        )

    return ConsolidationConfig(
        config_path=config_path,
        output_root=_resolve(output.get("root"), root, "output.root"),
        sources=tuple(sources),
        expected_sequence_count=int(expected.get("sequence_count")),
        expected_frame_count=int(expected.get("frame_count")),
        expected_confidence=float(expected.get("confidence_threshold")),
        detector_cache_root=_resolve(
            benchmark.get("detector_cache_root"), root, "benchmark.detector_cache_root"
        ),
        dataset=_resolve(benchmark.get("dataset"), root, "benchmark.dataset"),
        split=str(benchmark.get("split", "all")),
        overwrite=bool(output.get("overwrite", False)),
        write_figures=bool(output.get("write_figures", True)),
    )


def consolidate_tracking_benchmark(
    config_path: str | Path,
    *,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Validate and consolidate tracker summaries and per-sequence metrics."""

    config = load_consolidation_config(config_path)
    allow_overwrite = config.overwrite if overwrite is None else overwrite
    rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for source in config.sources:
        row = _load_summary_row(source)
        filtered_sequence_rows = _load_per_sequence_rows(source)
        _validate_source(config, source, row, filtered_sequence_rows)
        row = {
            **row,
            "tracker": source.tracker,
            "display_name": source.display_name,
            "source_summary": str(source.summary),
            "source_per_sequence": str(source.per_sequence),
        }
        rows.append(row)
        per_sequence_rows.extend(filtered_sequence_rows)
        source_manifest.append(
            {
                "tracker": source.tracker,
                "display_name": source.display_name,
                "summary": str(source.summary),
                "summary_sha256": _sha256(source.summary),
                "per_sequence": str(source.per_sequence),
                "per_sequence_sha256": _sha256(source.per_sequence),
                "tracker_config_hash": row.get("tracker_config_hash"),
            }
        )

    _validate_cross_source_sequences(config, per_sequence_rows)
    rows.sort(key=lambda item: float(item.get("HOTA") or 0), reverse=True)
    output_paths = _output_paths(config.output_root)
    _prepare_outputs(output_paths.values(), allow_overwrite)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths["summary_csv"], rows, SUMMARY_FIELDS)
    output_paths["summary_json"].write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sequence_fields = _ordered_fields(per_sequence_rows)
    _write_csv(output_paths["per_sequence_csv"], per_sequence_rows, sequence_fields)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config.config_path),
        "config_sha256": _sha256(config.config_path),
        "dataset": str(config.dataset),
        "split": config.split,
        "detector_cache_root": str(config.detector_cache_root),
        "compatibility_contract": {
            "sequence_count": config.expected_sequence_count,
            "frame_count": config.expected_frame_count,
            "confidence_threshold": config.expected_confidence,
            "smoke_only": False,
            "partial_sequences": False,
        },
        "timing_scope": {
            "tracker_fps": "tracker update only using cached detections",
            "cached_pipeline_fps": "cache read, filtering, tracker update, and MOT writing",
            "excludes": ["detector inference", "video decoding", "rendering", "VLM inference"],
        },
        "hardware_reference": runtime_versions(),
        "sources": source_manifest,
    }
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_paths["report_md"].write_text(
        _markdown_report(rows, config, output_paths), encoding="utf-8"
    )

    figures: list[Path] = []
    if config.write_figures:
        figures = _write_figures(rows, config.output_root / "figures")
    return {
        "status": "ok",
        "tracker_count": len(rows),
        "best_hota_tracker": rows[0]["tracker"] if rows else None,
        "paths": {name: str(path) for name, path in output_paths.items()},
        "figures": [str(path) for path in figures],
    }


def _load_summary_row(source: TrackingSource) -> dict[str, Any]:
    if not source.summary.is_file():
        raise TrackingConsolidationError(f"Summary does not exist: {source.summary}")
    payload = json.loads(source.summary.read_text(encoding="utf-8"))
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("rows", payload.get("trackers", []))
    )
    if not isinstance(rows, list):
        raise TrackingConsolidationError(f"Summary must contain a list: {source.summary}")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("tracker") == source.tracker
    ]
    if len(matches) != 1:
        raise TrackingConsolidationError(
            f"Expected one '{source.tracker}' row in {source.summary}, found {len(matches)}."
        )
    return dict(matches[0])


def _load_per_sequence_rows(source: TrackingSource) -> list[dict[str, Any]]:
    if not source.per_sequence.is_file():
        raise TrackingConsolidationError(
            f"Per-sequence metrics do not exist: {source.per_sequence}"
        )
    with source.per_sequence.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    matches = [row for row in rows if row.get("tracker") == source.tracker]
    for row in matches:
        row["tracker"] = source.tracker
        row["display_name"] = source.display_name
    return matches


def _validate_source(
    config: ConsolidationConfig,
    source: TrackingSource,
    row: dict[str, Any],
    sequence_rows: list[dict[str, Any]],
) -> None:
    errors: list[str] = []
    if int(row.get("sequence_count", -1)) != config.expected_sequence_count:
        errors.append(
            f"sequence_count={row.get('sequence_count')} "
            f"(expected {config.expected_sequence_count})"
        )
    if int(row.get("frame_count", -1)) != config.expected_frame_count:
        errors.append(
            f"frame_count={row.get('frame_count')} "
            f"(expected {config.expected_frame_count})"
        )
    if abs(float(row.get("confidence_threshold", -1)) - config.expected_confidence) > 1e-9:
        errors.append(
            f"confidence_threshold={row.get('confidence_threshold')} "
            f"(expected {config.expected_confidence})"
        )
    if bool(row.get("smoke_only", False)):
        errors.append("smoke_only=true")
    if bool(row.get("partial_sequences", False)):
        errors.append("partial_sequences=true")
    sequence_names = {item.get("sequence") for item in sequence_rows if item.get("sequence")}
    if len(sequence_names) != config.expected_sequence_count:
        errors.append(
            f"per-sequence rows cover {len(sequence_names)} sequence(s) "
            f"(expected {config.expected_sequence_count})"
        )
    frame_total = sum(int(float(item.get("frame_count") or 0)) for item in sequence_rows)
    if frame_total != config.expected_frame_count:
        errors.append(
            f"per-sequence frame total={frame_total} (expected {config.expected_frame_count})"
        )
    if errors:
        raise TrackingConsolidationError(
            f"Incompatible benchmark source '{source.tracker}': " + "; ".join(errors)
        )


def _validate_cross_source_sequences(
    config: ConsolidationConfig, rows: list[dict[str, Any]]
) -> None:
    by_tracker: dict[str, dict[str, int]] = {}
    for row in rows:
        tracker = str(row.get("tracker"))
        sequence = str(row.get("sequence"))
        by_tracker.setdefault(tracker, {})[sequence] = int(float(row.get("frame_count") or 0))
    reference_tracker = next(iter(by_tracker), None)
    if reference_tracker is None:
        raise TrackingConsolidationError("No per-sequence rows were loaded.")
    reference = by_tracker[reference_tracker]
    for tracker, sequence_frames in by_tracker.items():
        if sequence_frames != reference:
            raise TrackingConsolidationError(
                f"Sequence/frame set for '{tracker}' differs from '{reference_tracker}'."
            )
    if sum(reference.values()) != config.expected_frame_count:
        raise TrackingConsolidationError("Cross-source frame total does not match expected value.")


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "summary_json": root / "tracker_benchmark_summary.json",
        "summary_csv": root / "tracker_benchmark_summary.csv",
        "per_sequence_csv": root / "tracker_benchmark_per_sequence.csv",
        "manifest_json": root / "benchmark_manifest.json",
        "report_md": root / "tracker_benchmark_report.md",
    }


def _prepare_outputs(paths: Any, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise TrackingConsolidationError(
            "Output exists and overwrite=false: " + ", ".join(str(path) for path in existing)
        )


def _ordered_fields(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    preferred = (
        "tracker",
        "display_name",
        "sequence",
        "frame_count",
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "IDF1",
        "IDSW",
        "FP",
        "FN",
        "Frag",
        "tracker_fps",
    )
    available = {key for row in rows for key in row}
    return tuple(key for key in preferred if key in available) + tuple(
        sorted(available - set(preferred))
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_report(
    rows: list[dict[str, Any]], config: ConsolidationConfig, paths: dict[str, Path]
) -> str:
    columns = ("display_name", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "tracker_fps")
    lines = [
        "# Tracking benchmark",
        "",
        f"- Dataset: `{config.dataset}` (`{config.split}`, "
        f"{config.expected_sequence_count} sequences, "
        f"{config.expected_frame_count:,} frames)",
        f"- Shared detection cache: `{config.detector_cache_root}`",
        f"- Detection confidence threshold: `{config.expected_confidence}`",
        "- FPS scope: tracker-only on cached detections; detector, decode, render, "
        "and VLM are excluded.",
        "",
        "| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Tracker FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "HOTA is the primary balance between detection and association. AssA and IDF1 "
            "measure identity continuity; lower IDSW is better. FPS must be read together "
            "with the hardware and timing scope "
            "stored in the benchmark manifest.",
            "",
            f"Machine-readable manifest: `{paths['manifest_json']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_figures(rows: list[dict[str, Any]], root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    names = [str(row["display_name"]) for row in rows]
    written: list[Path] = []

    quality_metrics = ("HOTA", "DetA", "AssA", "MOTA", "IDF1")
    figure, axis = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(names))
    width = 0.15
    for index, metric in enumerate(quality_metrics):
        values = [float(row[metric]) for row in rows]
        axis.bar(x + (index - 2) * width, values, width, label=metric)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=18, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title("SportsMOT tracking quality on shared YOLO26m detections")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=5)
    written.append(_save_figure(figure, root / "tracking_quality.png", plt))

    error_metrics = ("IDSW", "Frag", "FP", "FN")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for axis, metric in zip(axes.flat, error_metrics, strict=True):
        values = [float(row[metric]) for row in rows]
        axis.bar(names, values, color="#b33a3a")
        axis.set_title(metric)
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Tracking error counts (lower is better)")
    figure.tight_layout()
    written.append(_save_figure(figure, root / "tracking_errors.png", plt))

    figure, axis = plt.subplots(figsize=(11, 5.8))
    tracker_values = [float(row["tracker_fps"]) for row in rows]
    pipeline_values = [float(row["cached_pipeline_fps"]) for row in rows]
    axis.bar(x - 0.18, tracker_values, 0.36, label="Tracker only")
    axis.bar(x + 0.18, pipeline_values, 0.36, label="Cached pipeline")
    axis.set_yscale("log")
    axis.set_ylabel("FPS (log scale)")
    axis.set_title("Runtime on cached detections")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    written.append(_save_figure(figure, root / "tracking_fps.png", plt))

    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    for row in rows:
        fps = float(row["tracker_fps"])
        hota = float(row["HOTA"])
        axis.scatter(fps, hota, s=65)
        axis.annotate(
            str(row["display_name"]),
            (fps, hota),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Tracker FPS (log scale)")
    axis.set_ylabel("HOTA")
    axis.set_title("Accuracy-speed trade-off")
    axis.grid(alpha=0.25)
    written.append(_save_figure(figure, root / "speed_vs_hota.png", plt))
    return written


def _save_figure(figure: Any, path: Path, pyplot: Any) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)
    return path
