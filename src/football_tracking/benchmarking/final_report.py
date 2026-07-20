"""Validate canonical benchmark artifacts and build the final project report."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import get_project_root, resolve_project_path


class FinalReportError(RuntimeError):
    """Raised when canonical benchmark artifacts fail their integrity contract."""


def build_final_benchmark_report(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = get_project_root()
    config_file = _resolve(config_path, root)
    config = _mapping(yaml.safe_load(config_file.read_text(encoding="utf-8")), "config")
    inputs = _mapping(config.get("inputs"), "inputs")
    expected = _mapping(config.get("expected"), "expected")
    output = _mapping(config.get("output"), "output")

    paths = {
        "detector": _resolve(inputs.get("detector_summary"), root),
        "tracking": _resolve(inputs.get("tracking_summary"), root),
        "tracking_per_sequence": _resolve(inputs.get("tracking_per_sequence"), root),
        "tracking_manifest": _resolve(inputs.get("tracking_manifest"), root),
        "idsw": _resolve(inputs.get("idsw_summary"), root),
        "semantic": _resolve(inputs.get("semantic_summary"), root),
    }
    runtime_entries = inputs.get("runtime_routes")
    if not isinstance(runtime_entries, list) or not runtime_entries:
        raise FinalReportError("inputs.runtime_routes must be a non-empty list.")
    runtime_sources = [
        {
            "id": str(_mapping(item, "runtime route").get("id", "")).strip(),
            "name": str(_mapping(item, "runtime route").get("name", "")).strip(),
            "metrics": _resolve(_mapping(item, "runtime route").get("metrics"), root),
        }
        for item in runtime_entries
    ]

    detector = _read_json(paths["detector"])
    tracking = _read_json(paths["tracking"])
    tracking_manifest = _read_json(paths["tracking_manifest"])
    idsw = _read_json(paths["idsw"])
    semantic = _read_json(paths["semantic"])
    runtime = [_runtime_row(item, expected) for item in runtime_sources]

    issues: list[dict[str, Any]] = []
    _validate_detector(detector, issues)
    _validate_tracking(
        tracking,
        paths["tracking_per_sequence"],
        tracking_manifest,
        expected,
        issues,
    )
    _validate_idsw(idsw, tracking, issues)
    _validate_semantic(semantic, expected, issues)
    _validate_runtime(runtime, expected, issues)
    issues.extend(_known_limitations())
    errors = [item for item in issues if item["severity"] == "ERROR"]
    if errors:
        details = "; ".join(f"{item['code']}: {item['message']}" for item in errors)
        raise FinalReportError(f"Final benchmark audit failed: {details}")

    output_root = _resolve(output.get("root"), root, require_file=False)
    publish_root = _resolve(output.get("publish_figures"), root, require_file=False)
    publish_report_root = _resolve(
        output.get("publish_report_root"),
        root,
        require_file=False,
    )
    output_paths = {
        "json": output_root / "final_experiment_summary.json",
        "markdown": output_root / "final_experiment_report.md",
        "runtime_csv": output_root / "realtime_route_summary.csv",
        "audit_json": output_root / "artifact_audit.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FinalReportError(
            "Final report output exists and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    publish_root.mkdir(parents=True, exist_ok=True)
    publish_report_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "config_sha256": file_sha256(config_file),
        "hardware": runtime[0]["hardware"],
        "detectors": detector.get("rows", []),
        "trackers": tracking,
        "semantic_pipelines": semantic.get("pipelines", []),
        "realtime_routes": runtime,
        "idsw_taxonomy": idsw.get("summaries", []),
        "measurement_contract": {
            "detector": "SportsMOT val, 2,900 images, 640 px, same GPU",
            "tracking": "SportsMOT 30 sequences, 20,171 frames, shared detections",
            "semantic": "31 human-reviewed tracks from one football video",
            "realtime": "120-frame file-source smoke, rendering and MP4 writing enabled",
            "idsw_taxonomy": "heuristic diagnostic categories; official IDSW is TrackEval",
        },
        "sources": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in paths.items()
        },
    }
    audit = {
        "status": "ok",
        "error_count": 0,
        "warning_count": len([item for item in issues if item["severity"] == "WARNING"]),
        "issues": issues,
        "source_count": len(paths) + len(runtime_sources),
    }
    _write_json(output_paths["json"], payload)
    _write_json(output_paths["audit_json"], audit)
    _write_csv(output_paths["runtime_csv"], runtime)
    figures = _write_figures(payload, output_root / "figures")
    published = _publish_figures(figures, publish_root)
    canonical_figure_links = [Path("figures") / path.name for path in figures]
    _write_text(
        output_paths["markdown"],
        _markdown(payload, audit, canonical_figure_links),
    )
    published_paths = {
        "markdown": publish_report_root / output_paths["markdown"].name,
        "json": publish_report_root / output_paths["json"].name,
        "runtime_csv": publish_report_root / output_paths["runtime_csv"].name,
        "audit_json": publish_report_root / output_paths["audit_json"].name,
    }
    published_figure_links = [
        Path("..") / "assets" / "benchmarks" / path.name for path in published
    ]
    _write_text(
        published_paths["markdown"],
        _markdown(payload, audit, published_figure_links),
    )
    for key in ("json", "runtime_csv", "audit_json"):
        shutil.copy2(output_paths[key], published_paths[key])
    return {
        "status": "ok",
        "paths": {key: str(value) for key, value in output_paths.items()},
        "figures": [str(path) for path in figures],
        "published_figures": [str(path) for path in published],
        "published_reports": {key: str(path) for key, path in published_paths.items()},
        "warnings": audit["warning_count"],
    }


def _validate_detector(payload: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        _issue(issues, "ERROR", "detector_rows", "At least two detector rows are required.")
        return
    names = [str(row.get("name")) for row in rows if isinstance(row, dict)]
    if len(names) != len(set(names)):
        _issue(issues, "ERROR", "detector_duplicate", "Detector names are duplicated.")
    for row in rows:
        if not isinstance(row, dict):
            _issue(issues, "ERROR", "detector_schema", "Detector row is not an object.")
            continue
        for metric in ("precision", "recall", "map50", "map50_95"):
            _range_metric(issues, row.get(metric), f"detector.{row.get('name')}.{metric}")
        for metric in ("detector_fps", "end_to_end_fps"):
            _positive_metric(issues, row.get(metric), f"detector.{row.get('name')}.{metric}")
    _validate_source_hashes(payload.get("sources"), issues, "detector")


def _validate_tracking(
    rows: Any,
    per_sequence_path: Path,
    manifest: dict[str, Any],
    expected: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(rows, list):
        _issue(issues, "ERROR", "tracking_schema", "Tracking summary must be a list.")
        return
    expected_trackers = int(expected.get("tracker_count", 0))
    expected_sequences = int(expected.get("sequence_count", 0))
    expected_frames = int(expected.get("frame_count", 0))
    if expected_trackers and len(rows) != expected_trackers:
        _issue(
            issues,
            "ERROR",
            "tracking_tracker_count",
            f"Found {len(rows)} trackers, expected {expected_trackers}.",
        )
    names: set[str] = set()
    for row in rows:
        tracker = str(row.get("tracker", ""))
        if not tracker or tracker in names:
            _issue(issues, "ERROR", "tracking_duplicate", f"Invalid tracker id: {tracker!r}.")
        names.add(tracker)
        if int(row.get("sequence_count", -1)) != expected_sequences:
            _issue(issues, "ERROR", "tracking_sequences", f"{tracker} sequence count mismatch.")
        if int(row.get("frame_count", -1)) != expected_frames:
            _issue(issues, "ERROR", "tracking_frames", f"{tracker} frame count mismatch.")
        for metric in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
            _range_metric(issues, row.get(metric), f"tracking.{tracker}.{metric}", scale=100.0)
        _positive_metric(issues, row.get("tracker_fps"), f"tracking.{tracker}.tracker_fps")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with per_sequence_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[str(row.get("tracker", ""))].append(row)
    for tracker in names:
        sequence_rows = grouped.get(tracker, [])
        frame_total = sum(int(float(row.get("frame_count") or 0)) for row in sequence_rows)
        sequence_names = {str(row.get("sequence")) for row in sequence_rows}
        if len(sequence_names) != expected_sequences or frame_total != expected_frames:
            _issue(
                issues,
                "ERROR",
                "tracking_per_sequence",
                f"{tracker} has {len(sequence_names)} sequences and {frame_total} frames.",
            )
    _validate_source_hashes(manifest.get("sources"), issues, "tracking")


def _validate_idsw(
    payload: dict[str, Any],
    tracking_rows: Any,
    issues: list[dict[str, Any]],
) -> None:
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        _issue(issues, "ERROR", "idsw_schema", "IDSW summaries must be a list.")
        return
    official_trackers = {
        str(row.get("tracker")) for row in tracking_rows if isinstance(row, dict)
    }
    overall = [row for row in summaries if row.get("sequence") == "__overall__"]
    taxonomy_trackers = {str(row.get("tracker")) for row in overall}
    if taxonomy_trackers != official_trackers:
        _issue(
            issues,
            "ERROR",
            "idsw_tracker_set",
            "IDSW taxonomy tracker set does not match the official tracking table.",
        )
    categories = (
        "fragmentation",
        "identity_swap",
        "re_identification_failure",
        "association_error",
        "appearance_confusion",
    )
    for row in overall:
        total = int(row.get("total_id_switches_recomputed", -1))
        count_sum = sum(int(row.get(f"{name}_count", 0)) for name in categories)
        percent_sum = sum(float(row.get(f"{name}_percent", 0.0)) for name in categories)
        tracker = str(row.get("tracker"))
        if total != count_sum:
            _issue(issues, "ERROR", "idsw_count_sum", f"{tracker} category counts do not sum.")
        if abs(percent_sum - 100.0) > 0.02:
            _issue(
                issues,
                "ERROR",
                "idsw_percent_sum",
                f"{tracker} category percentages sum to {percent_sum:.3f}%.",
            )


def _validate_semantic(
    payload: dict[str, Any],
    expected: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    rows = payload.get("pipelines")
    if not isinstance(rows, list) or not rows:
        _issue(issues, "ERROR", "semantic_schema", "Semantic pipeline rows are missing.")
        return
    expected_ids = {"A", "B", "C"}
    actual_ids = {str(row.get("pipeline")) for row in rows}
    if actual_ids != expected_ids:
        _issue(issues, "ERROR", "semantic_pipelines", f"Expected A/B/C, found {actual_ids}.")
    expected_gt = int(expected.get("semantic_gt_tracks", 0))
    for row in rows:
        pipeline = str(row.get("pipeline"))
        if expected_gt and int(row.get("gt_tracks", -1)) != expected_gt:
            _issue(issues, "ERROR", "semantic_gt", f"Pipeline {pipeline} GT count mismatch.")
        for metric in (
            "semantic_accuracy",
            "semantic_macro_f1",
            "semantic_coverage",
            "selective_accuracy",
        ):
            _range_metric(issues, row.get(metric), f"semantic.{pipeline}.{metric}")
        _positive_metric(
            issues,
            row.get("semantic_cold_seconds"),
            f"semantic.{pipeline}.cold_seconds",
        )
        evaluation = Path(str(row.get("evaluation", "")))
        expected_hash = str(row.get("evaluation_sha256", ""))
        if not evaluation.is_file() or file_sha256(evaluation) != expected_hash:
            _issue(
                issues,
                "ERROR",
                "semantic_provenance",
                f"Pipeline {pipeline} evaluation hash does not match.",
            )


def _runtime_row(source: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    if not source["id"] or not source["name"]:
        raise FinalReportError("Each runtime route requires id and name.")
    payload = _read_json(source["metrics"])
    timing = _mapping(payload.get("timing"), f"runtime {source['id']}.timing")
    hardware = _mapping(payload.get("hardware"), f"runtime {source['id']}.hardware")
    detector_summary = _runtime_detector_summary(payload.get("detector"))
    return {
        "route": source["id"],
        "name": source["name"],
        "checkpoint": payload.get("route", {}).get("checkpoint"),
        "checkpoint_type": payload.get("route", {}).get("checkpoint_type"),
        "tracker": payload.get("tracker"),
        "frames": payload.get("frames"),
        "detections": payload.get("detections"),
        "tracker_detections": payload.get("tracker_detections"),
        "detection_only_boxes": payload.get("detection_only_boxes"),
        "track_boxes": payload.get("track_boxes"),
        "unique_tracks": payload.get("unique_tracks"),
        "end_to_end_fps": timing.get("end_to_end_fps"),
        "processing_fps": timing.get("processing_fps"),
        "steady_state_processing_fps": timing.get("steady_state_processing_fps"),
        "latency_ms_p95": timing.get("frame_latency_ms_p95"),
        "startup_seconds": timing.get("startup_seconds"),
        **detector_summary,
        "gpu_name": hardware.get("gpu_name"),
        "gpu_memory_total_bytes": hardware.get("gpu_memory_total_bytes"),
        "system_memory_total_bytes": hardware.get("system_memory_total_bytes"),
        "source": str(source["metrics"]),
        "source_sha256": file_sha256(source["metrics"]),
        "hardware": hardware,
        "expected_frames": int(expected.get("runtime_frames", 0)),
    }


def _runtime_detector_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "detector_backend": "not_recorded",
            "primary_detector": "not_recorded",
            "supplemental_schedule": "none",
            "supplemental_inference_calls": 0,
        }
    backend = str(value.get("backend", "unknown"))
    primary = value.get("primary") if backend == "routed_composite" else value
    if not isinstance(primary, dict):
        primary = {}
    primary_name = str(
        primary.get("checkpoint_name")
        or primary.get("detector_name")
        or "unknown"
    )
    supplemental = value.get("supplemental", [])
    if not isinstance(supplemental, list):
        supplemental = []
    schedules: list[str] = []
    inference_calls = 0
    for item in supplemental:
        if not isinstance(item, dict):
            continue
        interval = max(1, int(item.get("every_n_frames", 1)))
        calls = max(0, int(item.get("inference_calls", 0)))
        name = str(item.get("checkpoint_name") or item.get("detector_name") or "supplemental")
        schedules.append(f"{name}: every {interval} frame(s), {calls} call(s)")
        inference_calls += calls
    return {
        "detector_backend": backend,
        "primary_detector": primary_name,
        "supplemental_schedule": "; ".join(schedules) if schedules else "none",
        "supplemental_inference_calls": inference_calls,
    }


def _validate_runtime(
    rows: list[dict[str, Any]],
    expected: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    expected_routes = int(expected.get("runtime_route_count", 0))
    expected_frames = int(expected.get("runtime_frames", 0))
    if expected_routes and len(rows) != expected_routes:
        _issue(issues, "ERROR", "runtime_route_count", "Runtime route count mismatch.")
    route_ids = [row["route"] for row in rows]
    if len(route_ids) != len(set(route_ids)):
        _issue(issues, "ERROR", "runtime_duplicate", "Runtime route ids are duplicated.")
    hardware_signatures = {
        (row.get("gpu_name"), row.get("gpu_memory_total_bytes")) for row in rows
    }
    if len(hardware_signatures) != 1:
        _issue(issues, "ERROR", "runtime_hardware", "Runtime routes used different GPUs.")
    for row in rows:
        if expected_frames and int(row.get("frames") or -1) != expected_frames:
            _issue(
                issues,
                "ERROR",
                "runtime_frames",
                f"Route {row['route']} did not process {expected_frames} frames.",
            )
        for metric in (
            "end_to_end_fps",
            "processing_fps",
            "steady_state_processing_fps",
            "startup_seconds",
        ):
            _positive_metric(issues, row.get(metric), f"runtime.{row['route']}.{metric}")


def _validate_source_hashes(
    sources: Any,
    issues: list[dict[str, Any]],
    prefix: str,
) -> None:
    if not isinstance(sources, list) or not sources:
        _issue(issues, "ERROR", f"{prefix}_sources", f"{prefix} source manifest is missing.")
        return
    for source in sources:
        if not isinstance(source, dict):
            _issue(issues, "ERROR", f"{prefix}_source_schema", "Source is not an object.")
            continue
        hash_fields = [key for key in source if key.endswith("_sha256")]
        for hash_field in hash_fields:
            path_field = hash_field.removesuffix("_sha256")
            path = Path(str(source.get(path_field, "")))
            expected_hash = str(source.get(hash_field, ""))
            if not path.is_file():
                _issue(issues, "ERROR", f"{prefix}_source_missing", f"Missing source: {path}")
            elif file_sha256(path) != expected_hash:
                _issue(
                    issues,
                    "ERROR",
                    f"{prefix}_source_hash",
                    f"Source hash changed: {path}",
                )


def _known_limitations() -> list[dict[str, Any]]:
    return [
        {
            "severity": "WARNING",
            "code": "semantic_gt_scope",
            "message": "Semantic A/B/C ground truth covers 31 tracks from one football video.",
        },
        {
            "severity": "WARNING",
            "code": "runtime_scope",
            "message": "Realtime route timings use a 120-frame local file, not a live camera.",
        },
        {
            "severity": "WARNING",
            "code": "cross_domain_gt_pending",
            "message": (
                "Traffic, medical, and education routes need human GT before accuracy claims."
            ),
        },
        {
            "severity": "WARNING",
            "code": "idsw_taxonomy_heuristic",
            "message": "IDSW categories are diagnostic heuristics, not official TrackEval metrics.",
        },
    ]


def _markdown(payload: dict[str, Any], audit: dict[str, Any], figures: list[Path]) -> str:
    detectors = payload["detectors"]
    trackers = payload["trackers"]
    semantics = payload["semantic_pipelines"]
    runtime = payload["realtime_routes"]
    lines = [
        "# Final experiment report",
        "",
        f"Artifact audit: **PASS** with {audit['warning_count']} scoped limitation(s).",
        "",
        "## Hardware",
        "",
        f"- GPU: {payload['hardware'].get('gpu_name')}",
        f"- VRAM: {_gib(payload['hardware'].get('gpu_memory_total_bytes')):.2f} GiB",
        f"- System RAM: {_gib(payload['hardware'].get('system_memory_total_bytes')):.2f} GiB",
        f"- PyTorch: {payload['hardware'].get('torch')}",
        "",
        "## Detector",
        "",
        "| Detector | Training | Precision | Recall | mAP50 | mAP50-95 | Detector FPS | E2E FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detectors:
        lines.append(
            f"| {row['display_name']} | {row['training']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['detector_fps']:.2f} | {row['end_to_end_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Tracking",
            "",
            "| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Tracker FPS | Cached FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trackers:
        lines.append(
            f"| {row['display_name']} | {row['HOTA']:.3f} | {row['DetA']:.3f} | "
            f"{row['AssA']:.3f} | {row['MOTA']:.3f} | {row['IDF1']:.3f} | "
            f"{int(row['IDSW'])} | {row['tracker_fps']:.2f} | "
            f"{row['cached_pipeline_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Recommended profiles: OC-SORT for realtime, TrackTrack for balanced quality, "
            "and BoT-SORT ReID stable when minimizing official ID switches is the priority.",
            "",
            "## Semantic A/B/C",
            "",
            "| Pipeline | E2E accuracy | Macro F1 | Coverage | Selective accuracy | "
            "Accepted / GT | Cold semantic (s) | Peak VRAM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in semantics:
        lines.append(
            f"| {row['pipeline']} | {row['semantic_accuracy']:.2%} | "
            f"{row['semantic_macro_f1']:.2%} | {row['semantic_coverage']:.2%} | "
            f"{row['selective_accuracy']:.2%} | "
            f"{row['accepted_tracks']} / {row['gt_tracks']} | "
            f"{row['semantic_cold_seconds']:.2f} | {row['sequential_peak_gib']:.2f} GiB |"
        )
    lines.extend(
        [
            "",
            "## Realtime routes",
            "",
            "| Route | Detector | Detections | Detect-only | E2E FPS | Steady FPS | "
            "P95 latency | Startup |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runtime:
        lines.append(
            f"| {row['name']} | {Path(str(row['checkpoint'])).name} | "
            f"{row['detections']} | {row['detection_only_boxes']} | "
            f"{row['end_to_end_fps']:.2f} | {row['steady_state_processing_fps']:.2f} | "
            f"{row['latency_ms_p95']:.2f} ms | {row['startup_seconds']:.2f} s |"
        )
    lines.extend(["", "Detector scheduling:"])
    for row in runtime:
        lines.append(
            f"- {row['name']}: `{row['primary_detector']}` every frame; "
            f"supplemental = {row['supplemental_schedule']}."
        )
    lines.extend(
        [
            "",
            "## IDSW diagnostic taxonomy",
            "",
            "Counts below are recomputed diagnostic events. Percentages partition each "
            "tracker's recomputed total; they do not replace official TrackEval IDSW.",
            "",
            "| Tracker | Recomputed | Fragmentation | Identity swap | ReID failure | "
            "Association | Appearance |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    tracker_names = {str(row["tracker"]): str(row["display_name"]) for row in trackers}
    for row in payload["idsw_taxonomy"]:
        tracker_name = tracker_names.get(str(row["tracker"]), str(row["tracker"]))
        lines.append(
            f"| {tracker_name} | {row['total_id_switches_recomputed']} | "
            f"{_count_percent(row, 'fragmentation')} | "
            f"{_count_percent(row, 'identity_swap')} | "
            f"{_count_percent(row, 're_identification_failure')} | "
            f"{_count_percent(row, 'association_error')} | "
            f"{_count_percent(row, 'appearance_confusion')} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Detector and tracking scores are compared against SportsMOT ground truth.",
            "- Semantic scores use 31 manually reviewed track labels from video 1.",
            "- Detection-only classes are rendered without a track ID; only `track` classes "
            "enter MOT.",
            "- Cross-domain routing is integration-tested, but cross-domain accuracy still "
            "needs GT.",
            "- IDSW taxonomy is heuristic; use the official TrackEval IDSW column for ranking.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    return "\n".join(lines) + "\n"


def _count_percent(row: dict[str, Any], prefix: str) -> str:
    return f"{int(row[f'{prefix}_count'])} ({float(row[f'{prefix}_percent']):.1f}%)"


def _write_figures(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    architecture, axis = plt.subplots(figsize=(15, 5.4))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 5.4)
    axis.axis("off")
    boxes = (
        (0.3, 3.2, 1.7, "Video / stream"),
        (2.4, 3.2, 1.8, "Shots + keyframes"),
        (4.6, 3.2, 1.8, "Qwen discovery"),
        (6.8, 3.2, 1.8, "Vocabulary + router"),
        (9.0, 3.2, 1.8, "YOLO / YOLOE"),
        (11.2, 3.2, 1.6, "Tracker"),
        (13.2, 3.2, 1.5, "Render + JSON"),
        (4.6, 0.8, 2.0, "Track crops + Qwen"),
        (7.2, 0.8, 2.0, "Locate on events"),
        (9.8, 0.8, 2.0, "Fusion + unknown"),
    )
    for x, y, width, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            0.9,
            boxstyle="round,pad=0.04",
            facecolor="#EAF1E5" if y > 2 else "#E7EFF7",
            edgecolor="#315B2B" if y > 2 else "#245B85",
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 0.45, label, ha="center", va="center", fontsize=10)
    for start, end in (
        ((2.0, 3.65), (2.4, 3.65)),
        ((4.2, 3.65), (4.6, 3.65)),
        ((6.4, 3.65), (6.8, 3.65)),
        ((8.6, 3.65), (9.0, 3.65)),
        ((10.8, 3.65), (11.2, 3.65)),
        ((12.8, 3.65), (13.2, 3.65)),
        ((12.0, 3.2), (10.8, 1.7)),
        ((6.6, 1.25), (7.2, 1.25)),
        ((9.2, 1.25), (9.8, 1.25)),
        ((11.8, 1.25), (13.8, 3.2)),
    ):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.text(7.5, 5.0, "Adaptive multi-domain tracking architecture", ha="center", fontsize=16)
    path = output_dir / "adaptive_architecture.png"
    architecture.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(architecture)
    figures.append(path)

    runtime = payload["realtime_routes"]
    labels = [row["route"] for row in runtime]
    positions = np.arange(len(runtime))
    width = 0.24
    performance, axis = plt.subplots(figsize=(9.5, 5.2))
    for offset, key, label, color in (
        (-width, "end_to_end_fps", "E2E FPS", "#2F5597"),
        (0.0, "processing_fps", "Processing FPS", "#70AD47"),
        (width, "steady_state_processing_fps", "Steady FPS", "#ED7D31"),
    ):
        values = [float(row[key]) for row in runtime]
        bars = axis.bar(positions + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Frames per second")
    axis.set_title("Realtime detector routes on RTX 4060 Laptop GPU")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    performance.tight_layout()
    path = output_dir / "realtime_route_fps.png"
    performance.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(performance)
    figures.append(path)

    trackers = payload["trackers"]
    tradeoff, axis = plt.subplots(figsize=(10, 6))
    for row in trackers:
        axis.scatter(
            float(row["cached_pipeline_fps"]),
            float(row["HOTA"]),
            s=70,
            alpha=0.85,
        )
        axis.annotate(
            str(row["display_name"]),
            (float(row["cached_pipeline_fps"]), float(row["HOTA"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Cached pipeline FPS")
    axis.set_ylabel("HOTA")
    axis.set_title("Tracker quality-speed trade-off on 30 SportsMOT sequences")
    axis.grid(alpha=0.25)
    tradeoff.tight_layout()
    path = output_dir / "tracker_quality_speed.png"
    tradeoff.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(tradeoff)
    figures.append(path)

    detectors = payload["detectors"]
    detector_labels = [str(row.get("display_name") or row["name"]) for row in detectors]
    detector_positions = np.arange(len(detectors))
    detector_figure, detector_axes = plt.subplots(1, 2, figsize=(14, 5.4))
    quality_bars = detector_axes[0].bar(
        detector_positions,
        [float(row["map50_95"]) for row in detectors],
        color="#2F5597",
    )
    detector_axes[0].bar_label(quality_bars, fmt="%.3f", padding=2, fontsize=8)
    detector_axes[0].set_xticks(detector_positions, detector_labels, rotation=18, ha="right")
    detector_axes[0].set_ylim(0, 1.0)
    detector_axes[0].set_ylabel("mAP50-95")
    detector_axes[0].set_title("Detector accuracy on SportsMOT validation")
    detector_axes[0].grid(axis="y", alpha=0.25)
    speed_bars = detector_axes[1].bar(
        detector_positions,
        [float(row["detector_fps"]) for row in detectors],
        color="#70AD47",
    )
    detector_axes[1].bar_label(speed_bars, fmt="%.1f", padding=2, fontsize=8)
    detector_axes[1].set_xticks(detector_positions, detector_labels, rotation=18, ha="right")
    detector_axes[1].set_ylabel("Detector FPS")
    detector_axes[1].set_title("Detector throughput on the same GPU")
    detector_axes[1].grid(axis="y", alpha=0.25)
    detector_figure.tight_layout()
    path = output_dir / "detector_quality_speed.png"
    detector_figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(detector_figure)
    figures.append(path)

    semantics = payload["semantic_pipelines"]
    semantic_labels = [str(row["pipeline"]) for row in semantics]
    semantic_positions = np.arange(len(semantics))
    semantic_figure, semantic_axes = plt.subplots(1, 2, figsize=(13, 5.2))
    semantic_width = 0.25
    for offset, key, label, color in (
        (-semantic_width, "semantic_accuracy", "E2E accuracy", "#2F5597"),
        (0.0, "semantic_macro_f1", "Macro F1", "#70AD47"),
        (semantic_width, "semantic_coverage", "Coverage", "#ED7D31"),
    ):
        values = [100.0 * float(row[key]) for row in semantics]
        bars = semantic_axes[0].bar(
            semantic_positions + offset,
            values,
            semantic_width,
            label=label,
            color=color,
        )
        semantic_axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    semantic_axes[0].set_xticks(semantic_positions, semantic_labels)
    semantic_axes[0].set_ylim(0, 110)
    semantic_axes[0].set_ylabel("Percent")
    semantic_axes[0].set_title("Semantic quality on 31 reviewed tracks")
    semantic_axes[0].grid(axis="y", alpha=0.25)
    semantic_axes[0].legend(fontsize=8)
    cold_bars = semantic_axes[1].bar(
        semantic_positions,
        [float(row["semantic_cold_seconds"]) for row in semantics],
        color="#A64D79",
    )
    semantic_axes[1].bar_label(cold_bars, fmt="%.1f s", padding=2, fontsize=8)
    semantic_axes[1].set_xticks(semantic_positions, semantic_labels)
    semantic_axes[1].set_ylabel("Cold semantic wall time (seconds)")
    semantic_axes[1].set_title("Sequential semantic cost")
    semantic_axes[1].grid(axis="y", alpha=0.25)
    for position, row in zip(semantic_positions, semantics, strict=True):
        semantic_axes[1].text(
            position,
            float(row["semantic_cold_seconds"]) * 0.52,
            f"{float(row['sequential_peak_gib']):.2f} GiB",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    semantic_figure.tight_layout()
    path = output_dir / "semantic_quality_cost.png"
    semantic_figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(semantic_figure)
    figures.append(path)

    taxonomy = payload["idsw_taxonomy"]
    taxonomy_names = {str(row["tracker"]): str(row["display_name"]) for row in trackers}
    taxonomy_labels = [
        taxonomy_names.get(str(row["tracker"]), str(row["tracker"]))
        for row in taxonomy
    ]
    taxonomy_positions = np.arange(len(taxonomy))
    taxonomy_figure, taxonomy_axis = plt.subplots(figsize=(12, 6.3))
    left = np.zeros(len(taxonomy))
    for prefix, label, color in (
        ("fragmentation", "Fragmentation", "#4472C4"),
        ("identity_swap", "Identity swap", "#ED7D31"),
        ("re_identification_failure", "ReID failure", "#A5A5A5"),
        ("association_error", "Association error", "#FFC000"),
        ("appearance_confusion", "Appearance confusion", "#70AD47"),
    ):
        values = np.array([float(row[f"{prefix}_percent"]) for row in taxonomy])
        taxonomy_axis.barh(
            taxonomy_positions,
            values,
            left=left,
            label=label,
            color=color,
        )
        left += values
    taxonomy_axis.set_yticks(taxonomy_positions, taxonomy_labels)
    taxonomy_axis.invert_yaxis()
    taxonomy_axis.set_xlim(0, 100)
    taxonomy_axis.set_xlabel("Share of recomputed diagnostic ID switches (%)")
    taxonomy_axis.set_title("IDSW failure taxonomy by tracker")
    taxonomy_axis.grid(axis="x", alpha=0.2)
    taxonomy_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    taxonomy_figure.tight_layout()
    path = output_dir / "idsw_taxonomy.png"
    taxonomy_figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(taxonomy_figure)
    figures.append(path)
    return figures


def _publish_figures(figures: list[Path], output_dir: Path) -> list[Path]:
    published: list[Path] = []
    for source in figures:
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        published.append(destination)
    return published


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict | list):
        raise FinalReportError(f"JSON root must be an object or list: {path}")
    return value


def _resolve(value: Any, root: Path, *, require_file: bool = True) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise FinalReportError("Expected a non-empty path.")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, root)
    if require_file and not resolved.is_file():
        raise FinalReportError(f"Required file does not exist: {resolved}")
    return resolved


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalReportError(f"{name} must be a mapping.")
    return value


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def _range_metric(
    issues: list[dict[str, Any]],
    value: Any,
    name: str,
    *,
    scale: float = 1.0,
) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _issue(issues, "ERROR", "metric_missing", f"{name} is missing or non-numeric.")
        return
    if not 0.0 <= numeric <= scale:
        _issue(issues, "ERROR", "metric_range", f"{name}={numeric} is outside [0,{scale}].")


def _positive_metric(issues: list[dict[str, Any]], value: Any, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _issue(issues, "ERROR", "metric_missing", f"{name} is missing or non-numeric.")
        return
    if numeric <= 0:
        _issue(issues, "ERROR", "metric_nonpositive", f"{name}={numeric} must be positive.")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [key for key in rows[0] if key not in {"hardware", "expected_frames"}]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _gib(value: Any) -> float:
    return float(value or 0) / (1024**3)


__all__ = ["FinalReportError", "build_final_benchmark_report"]
