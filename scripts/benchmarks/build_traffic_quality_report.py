"""Build an auditable traffic tracking and semantic quality report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class TrafficQualityReportError(RuntimeError):
    """Raised when a source artifact violates the report contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrafficQualityReportError(f"{name} must be a mapping.")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrafficQualityReportError(f"Required artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _resolve(value: Any, root: Path) -> Path:
    if not str(value or "").strip():
        raise TrafficQualityReportError("Artifact path must not be empty.")
    return resolve_project_path(Path(str(value)), root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detector_checkpoint(runtime: dict[str, Any]) -> str:
    detector = _mapping(runtime.get("detector", {}), "runtime.detector")
    checkpoint = detector.get("checkpoint")
    if checkpoint:
        return str(checkpoint)
    primary = detector.get("primary")
    if isinstance(primary, dict) and primary.get("checkpoint"):
        return str(primary["checkpoint"])
    return str(_mapping(runtime.get("route", {}), "runtime.route").get("checkpoint", ""))


def _tracker_name(runtime: dict[str, Any]) -> tuple[str, bool]:
    config = _mapping(
        runtime.get("tracker_runtime_config", {}),
        "runtime.tracker_runtime_config",
    )
    tracker_type = str(config.get("tracker_type", "")).strip().casefold()
    if not tracker_type:
        raise TrafficQualityReportError("tracker_runtime_config.tracker_type is missing.")
    with_reid = bool(config.get("with_reid", False))
    return tracker_type, with_reid


def _tracking_row(
    entry: dict[str, Any],
    *,
    root: Path,
    expected_frames: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    evaluation_path = _resolve(entry.get("evaluation"), root)
    runtime_path = _resolve(entry.get("runtime"), root)
    evaluation = _read_json(evaluation_path)
    runtime = _read_json(runtime_path)
    issues: list[dict[str, str]] = []
    if not bool(evaluation.get("available")):
        raise TrafficQualityReportError(f"TrackEval result is unavailable: {evaluation_path}")
    ignore = _mapping(evaluation.get("ignore_region_filter", {}), "ignore filter")
    if not bool(ignore.get("enabled")):
        raise TrafficQualityReportError(
            f"Ignore-region filtering is required for UA-DETRAC: {evaluation_path}"
        )
    if int(runtime.get("frames", 0)) != expected_frames:
        raise TrafficQualityReportError(
            f"Runtime frame count mismatch in {runtime_path}: "
            f"{runtime.get('frames')} != {expected_frames}"
        )
    metrics = _mapping(evaluation.get("metrics", {}), "TrackEval metrics")
    timing = _mapping(runtime.get("timing", {}), "runtime.timing")
    tracker_type, with_reid = _tracker_name(runtime)
    declared_id = str(entry.get("id", "")).strip()
    if tracker_type not in declared_id:
        issues.append(
            {
                "severity": "WARNING",
                "code": "tracker_label_audit",
                "message": (
                    f"Configured report id '{declared_id}' was verified from runtime as "
                    f"'{tracker_type}' (ReID={with_reid})."
                ),
            }
        )
    detector = _detector_checkpoint(runtime)
    hardware = _mapping(runtime.get("hardware", {}), "runtime.hardware")
    resources = _mapping(runtime.get("resources", {}), "runtime.resources")
    cuda_memory = _mapping(runtime.get("cuda_memory", {}), "runtime.cuda_memory")
    row = {
        "id": declared_id,
        "label": str(entry.get("label", declared_id)),
        "short_label": str(entry.get("short_label", declared_id)),
        "detector": detector,
        "tracker_type": tracker_type,
        "with_reid": with_reid,
        **{key: metrics.get(key) for key in (
            "HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag"
        )},
        "processing_fps": timing.get("processing_fps"),
        "end_to_end_fps": timing.get("end_to_end_fps"),
        "detector_fps": timing.get("detector_fps"),
        "tracker_fps": timing.get("tracker_fps"),
        "peak_process_rss_gib": round(
            float(resources.get("peak_process_rss_bytes", 0)) / 1024**3,
            4,
        ),
        "peak_cuda_allocated_gib": round(
            float(cuda_memory.get("peak_allocated_bytes", 0)) / 1024**3,
            4,
        ),
        "gpu": hardware.get("gpu_name"),
        "evaluation_path": str(evaluation_path),
        "runtime_path": str(runtime_path),
    }
    for metric in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        value = float(row[metric])
        if not 0.0 <= value <= 100.0:
            raise TrafficQualityReportError(f"{declared_id}.{metric} is invalid: {value}")
    for metric in ("processing_fps", "end_to_end_fps"):
        if float(row[metric]) <= 0:
            raise TrafficQualityReportError(f"{declared_id}.{metric} must be positive.")
    return row, issues


def _semantic_summary(config: dict[str, Any], root: Path) -> dict[str, Any]:
    metrics_path = _resolve(config.get("metrics"), root)
    mapping_path = _resolve(config.get("mapping"), root)
    cache_path = _resolve(config.get("cache"), root)
    render_path = _resolve(config.get("render_metadata"), root)
    worker_paths = [_resolve(path, root) for path in config.get("worker_reports", [])]
    metrics = _mapping(_read_json(metrics_path).get("summary", {}), "semantic summary")
    mapping = _read_json(mapping_path)
    cache = _read_json(cache_path)
    render = _read_json(render_path)
    workers = [_read_json(path) for path in worker_paths]
    processed = sum(int(row.get("processed_event_count", 0)) for row in workers)
    elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in workers)
    locate_elapsed = sum(
        float(_mapping(row.get("locateanything", {}), "locate report").get("elapsed_seconds", 0.0))
        for row in workers
    )
    cache_summary = _mapping(cache.get("summary", {}), "semantic cache summary")
    render_summary = _mapping(render.get("semantics_summary", {}), "semantic render summary")
    return {
        "name": str(config.get("name", "semantic pipeline")),
        "processed_track_count": processed,
        "scored_track_count": int(mapping.get("scored_track_count", 0)),
        "unscored_track_count": int(mapping.get("unscored_track_count", 0)),
        "semantic_accuracy": metrics.get("semantic_track_accuracy"),
        "semantic_macro_f1": metrics.get("semantic_macro_f1"),
        "semantic_coverage": metrics.get("semantic_coverage"),
        "semantic_hallucination_rate": metrics.get("semantic_hallucination_rate"),
        "unknown_rejection_f1": metrics.get("unknown_rejection_f1"),
        "fine_label_accuracy": metrics.get("fine_semantic_track_accuracy"),
        "fine_label_gt_track_count": metrics.get("fine_semantic_gt_track_count"),
        "fine_label_acceptance_coverage": cache_summary.get("fine_coverage"),
        "semantic_elapsed_seconds": round(elapsed, 6),
        "locate_elapsed_seconds": round(locate_elapsed, 6),
        "qwen_and_overhead_seconds": round(max(0.0, elapsed - locate_elapsed), 6),
        "seconds_per_processed_track": round(elapsed / max(processed, 1), 6),
        "processed_tracks_per_minute": round(60.0 * processed / max(elapsed, 1e-9), 6),
        "render_track_coverage": render_summary.get("track_coverage"),
        "render_box_coverage": render_summary.get("box_coverage"),
        "labeled_box_coverage": render_summary.get("label_box_coverage"),
        "sources": {
            "metrics": str(metrics_path),
            "mapping": str(mapping_path),
            "cache": str(cache_path),
            "render_metadata": str(render_path),
            "worker_reports": [str(path) for path in worker_paths],
        },
    }


def _write_tracking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [key for key in rows[0] if not key.endswith("_path")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    output_dir: Path,
    tracking: list[dict[str, Any]],
    semantic: dict[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = figure_dir / "tracking_accuracy_comparison.png"
    ordered = sorted(tracking, key=lambda row: float(row["HOTA"]), reverse=True)
    labels = [str(row["short_label"]) for row in ordered]
    x = list(range(len(ordered)))
    width = 0.2
    fig, axis = plt.subplots(figsize=(14, 6.2))
    for offset, metric, color in (
        (-1.5, "HOTA", "#1f77b4"),
        (-0.5, "DetA", "#2ca02c"),
        (0.5, "AssA", "#ff7f0e"),
        (1.5, "IDF1", "#9467bd"),
    ):
        axis.bar(
            [value + offset * width for value in x],
            [float(row[metric]) for row in ordered],
            width=width,
            label=metric,
            color=color,
        )
    axis.set_ylabel("TrackEval score (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_title("Official UA-DETRAC tracking accuracy (ignore regions applied)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=190)
    plt.close(fig)

    tradeoff_path = figure_dir / "tracking_identity_speed_tradeoff.png"
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    annotation_offsets = {
        "n + BoT-R": (-8, -34),
        "s + BoT-R": (8, 12),
        "n + DeepOC-R": (14, 10),
        "m + BoT-R": (14, 42),
        "n + TT-R": (8, 6),
        "s + TT-R": (6, 6),
        "n + Byte": (8, 5),
        "s + Byte": (8, 5),
        "n + OC": (8, 5),
        "n + TT": (8, 5),
    }
    for row in tracking:
        idsw = int(row["IDSW"])
        axis.scatter(
            float(row["processing_fps"]),
            float(row["IDF1"]),
            s=65 + 18 * idsw,
            c="#d62728" if idsw else "#2ca02c",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.7,
        )
        axis.annotate(
            f"{row['short_label']}\nIDSW={idsw}",
            (float(row["processing_fps"]), float(row["IDF1"])),
            xytext=annotation_offsets.get(str(row["short_label"]), (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Processing FPS")
    axis.set_ylabel("IDF1 (%)")
    axis.set_title("Identity accuracy versus throughput")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(tradeoff_path, dpi=190)
    plt.close(fig)

    detector_path = figure_dir / "detector_size_ablation.png"
    detector_rows = [
        next(row for row in tracking if row["id"] == run_id)
        for run_id in (
            "yolo26n_botsort_reid",
            "yolo26s_botsort_reid",
            "yolo26m_botsort_reid",
        )
    ]
    labels = [Path(str(row["detector"])).stem for row in detector_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(labels, [float(row["DetA"]) for row in detector_rows], color="#1f77b4")
    axes[0].set_title("Detection accuracy inside MOT")
    axes[0].set_ylabel("DetA (%)")
    axes[1].bar(labels, [float(row["processing_fps"]) for row in detector_rows], color="#ff7f0e")
    axes[1].set_title("End-to-end processing speed")
    axes[1].set_ylabel("FPS")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("YOLO26 detector-size ablation with the same BoT-SORT ReID tracker")
    fig.tight_layout()
    fig.savefig(detector_path, dpi=190)
    plt.close(fig)

    semantic_path = figure_dir / "semantic_quality_and_cost.png"
    quality = [
        100.0 * float(semantic["semantic_accuracy"]),
        100.0 * float(semantic["semantic_macro_f1"]),
        100.0 * float(semantic["semantic_coverage"]),
        100.0 * (1.0 - float(semantic["semantic_hallucination_rate"])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(
        ["Accuracy", "Macro-F1", "Coverage", "Non-hallucination"],
        quality,
        color=["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e"],
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Official parent-class score (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        ["Locate", "Qwen + overhead"],
        [semantic["locate_elapsed_seconds"], semantic["qwen_and_overhead_seconds"]],
        color=["#17becf", "#8c564b"],
    )
    axes[1].set_ylabel("Measured cumulative seconds")
    axes[1].set_title(f"24 tracks: {semantic['seconds_per_processed_track']:.1f} s/track")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("LocateAnything -> Qwen semantic quality and measured cost")
    fig.tight_layout()
    fig.savefig(semantic_path, dpi=190)
    plt.close(fig)
    return [accuracy_path, tradeoff_path, detector_path, semantic_path]


def _markdown(payload: dict[str, Any]) -> str:
    best = payload["selection"]["best_quality"]
    fastest = payload["selection"]["best_speed"]
    semantic = payload["semantic"]
    rows = payload["tracking"]
    gpu_memory_gib = (
        float(payload["hardware"].get("gpu_memory_total_bytes", 0)) / 1024**3
    )
    system_memory_gib = (
        float(payload["hardware"].get("system_memory_total_bytes", 0)) / 1024**3
    )
    lines = [
        "# Traffic quality benchmark",
        "",
        "## Result summary",
        "",
        f"- Best quality: **{best['label']}**, HOTA {best['HOTA']:.3f}, "
        f"IDF1 {best['IDF1']:.3f}, IDSW {best['IDSW']}, {best['processing_fps']:.2f} FPS.",
        f"- Best speed: **{fastest['label']}**, {fastest['processing_fps']:.2f} FPS, "
        f"HOTA {fastest['HOTA']:.3f}, IDSW {fastest['IDSW']}.",
        f"- Semantic parent class: {100 * semantic['semantic_accuracy']:.1f}% accuracy, "
        f"{100 * semantic['semantic_macro_f1']:.1f}% Macro-F1 on "
        f"{semantic['scored_track_count']} official GT tracks.",
        "",
        "## Tracking table",
        "",
        "| Detector + tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['label']} | {row['HOTA']:.3f} | {row['DetA']:.3f} | "
        f"{row['AssA']:.3f} | {row['MOTA']:.3f} | {row['IDF1']:.3f} | "
        f"{row['IDSW']} | {row['FP']} | {row['FN']} | {row['processing_fps']:.2f} |"
        for row in sorted(rows, key=lambda item: float(item["HOTA"]), reverse=True)
    )
    lines.extend(
        [
            "",
            "## Semantic table",
            "",
            "| Pipeline | Processed | Scored GT | Unscored | Accuracy | Macro-F1 | "
            "Coverage | Hallucination | Fine-label GT | Time/track |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {semantic['name']} | {semantic['processed_track_count']} | "
            f"{semantic['scored_track_count']} | {semantic['unscored_track_count']} | "
            f"{100 * semantic['semantic_accuracy']:.1f}% | "
            f"{100 * semantic['semantic_macro_f1']:.1f}% | "
            f"{100 * semantic['semantic_coverage']:.1f}% | "
            f"{100 * semantic['semantic_hallucination_rate']:.1f}% | "
            f"{semantic['fine_label_gt_track_count']} | "
            f"{semantic['seconds_per_processed_track']:.2f} s |",
            "",
            "`Unscored` means the class-incomplete UA-DETRAC GT does not annotate "
            "that predicted object. It is not treated as semantic unknown or as a "
            "semantic model error.",
            "",
            "## Metric scope",
            "",
            "- HOTA balances detection and identity association; IDF1 measures ID consistency.",
            "- IDSW, FP, FN and Frag come from TrackEval against official GT.",
            "- Ignore regions from the UA-DETRAC XML are applied to GT and predictions.",
            "- Fine-label accuracy and unknown-rejection F1 are N/A because this GT has "
            "neither vehicle subtype/color labels nor a reviewed unknown class.",
            "- FPS is the measured processing loop with rendering and MP4 output on the "
            "hardware recorded below; semantic VLM inference runs after tracking.",
            "",
            "## Hardware",
            "",
            f"- GPU: {payload['hardware'].get('gpu_name')}",
            f"- GPU memory: {gpu_memory_gib:.1f} GiB",
            f"- CPU: {payload['hardware'].get('processor')}",
            f"- RAM: {system_memory_gib:.1f} GiB",
            (
                f"- PyTorch/CUDA: {payload['hardware'].get('torch')} / "
                f"{payload['hardware'].get('cuda_runtime')}"
            ),
            "",
            "## Figures",
            "",
            "![Tracking accuracy](figures/tracking_accuracy_comparison.png)",
            "",
            "![Identity-speed tradeoff](figures/tracking_identity_speed_tradeoff.png)",
            "",
            "![Detector ablation](figures/detector_size_ablation.png)",
            "",
            "![Semantic quality and cost](figures/semantic_quality_and_cost.png)",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(config_path: Path, *, overwrite: bool) -> dict[str, Any]:
    root = get_project_root()
    resolved_config = _resolve(config_path, root)
    config = _mapping(yaml.safe_load(resolved_config.read_text(encoding="utf-8")), "config")
    if int(config.get("schema_version", 0)) != 1:
        raise TrafficQualityReportError("Unsupported report schema_version.")
    dataset = _mapping(config.get("dataset"), "dataset")
    expected_frames = int(dataset.get("frame_count", 0))
    manifest_path = _resolve(dataset.get("manifest"), root)
    manifest = _read_json(manifest_path)
    if int(manifest.get("sequence_count", 0)) != 1:
        raise TrafficQualityReportError("This focused report expects one GT sequence.")

    entries = config.get("tracking_runs")
    if not isinstance(entries, list) or len(entries) < 3:
        raise TrafficQualityReportError("At least three tracking runs are required.")
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for raw in entries:
        row, row_issues = _tracking_row(
            _mapping(raw, "tracking run"),
            root=root,
            expected_frames=expected_frames,
        )
        rows.append(row)
        issues.extend(row_issues)
    semantic = _semantic_summary(_mapping(config.get("semantic"), "semantic"), root)
    hardware = _read_json(Path(rows[0]["runtime_path"])).get("hardware", {})
    if semantic["fine_label_gt_track_count"] == 0:
        issues.append(
            {
                "severity": "WARNING",
                "code": "fine_label_gt_missing",
                "message": "Fine-label accuracy is not measurable from UA-DETRAC GT.",
            }
        )
    if semantic["unknown_rejection_f1"] is None:
        issues.append(
            {
                "severity": "WARNING",
                "code": "unknown_gt_missing",
                "message": "Unknown-rejection F1 is not measurable from this GT.",
            }
        )
    issues.append(
        {
            "severity": "WARNING",
            "code": "single_sequence_scope",
            "message": "Traffic quality conclusions currently cover one 750-frame sequence.",
        }
    )
    best_quality = max(rows, key=lambda row: (float(row["HOTA"]), float(row["IDF1"])))
    best_speed = max(rows, key=lambda row: float(row["processing_fps"]))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(resolved_config),
        "dataset": dataset,
        "hardware": hardware,
        "tracking": rows,
        "semantic": semantic,
        "selection": {
            "best_quality": best_quality,
            "best_speed": best_speed,
        },
        "measurement_contract": {
            "tracking": "Official UA-DETRAC MOT GT; four ignore regions applied symmetrically.",
            "semantic": (
                "Parent vehicle class only; unmatched predictions ignored because "
                "GT is class-incomplete."
            ),
            "fps": "Measured processing loop with detector, tracker, render, and MP4 write.",
            "vlm": "LocateAnything-3B 8-bit then Qwen3-VL-4B 8-bit, sequential VRAM use.",
        },
        "issues": issues,
    }
    output = _resolve(_mapping(config.get("output"), "output").get("root"), root)
    publish_value = _mapping(config.get("output"), "output").get("publish_root")
    publish_root = _resolve(publish_value, root) if publish_value else None
    output_files = {
        "json": output / "traffic_quality_summary.json",
        "csv": output / "tracking_comparison.csv",
        "markdown": output / "traffic_quality_report.md",
        "audit": output / "artifact_audit.json",
    }
    if not overwrite:
        existing = [path for path in output_files.values() if path.exists()]
        if existing:
            raise TrafficQualityReportError(f"Report output exists: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    figures = _write_figures(output, rows, semantic)
    output_files["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_tracking_csv(output_files["csv"], rows)
    output_files["markdown"].write_text(_markdown(payload), encoding="utf-8")
    source_paths = [manifest_path, resolved_config]
    source_paths.extend(Path(row["evaluation_path"]) for row in rows)
    source_paths.extend(Path(row["runtime_path"]) for row in rows)
    for value in semantic["sources"].values():
        if isinstance(value, list):
            source_paths.extend(Path(item) for item in value)
        else:
            source_paths.append(Path(value))
    audit = {
        "status": "ok",
        "error_count": 0,
        "warning_count": len(issues),
        "issues": issues,
        "sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys(source_paths)
        ],
    }
    output_files["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    published: dict[str, str] = {}
    if publish_root is not None:
        publish_root.mkdir(parents=True, exist_ok=True)
        published_figures = publish_root / "figures"
        published_figures.mkdir(parents=True, exist_ok=True)
        for key, source in output_files.items():
            destination = publish_root / source.name
            shutil.copy2(source, destination)
            published[key] = str(destination)
        for source in figures:
            shutil.copy2(source, published_figures / source.name)
    return {
        "status": "ok",
        "paths": {key: str(path) for key, path in output_files.items()},
        "figures": [str(path) for path in figures],
        "published": published,
        "best_quality": best_quality["id"],
        "best_speed": best_speed["id"],
        "warning_count": len(issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/traffic_quality_report.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_report(args.config, overwrite=args.overwrite)
    except (OSError, ValueError, TrafficQualityReportError) as exc:
        print(f"Error: {exc}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
