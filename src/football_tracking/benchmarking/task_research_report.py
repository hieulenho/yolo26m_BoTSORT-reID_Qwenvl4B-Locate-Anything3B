"""Build the audited research report for the explicit single-Qwen task pipeline."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import get_project_root, resolve_project_path


class TaskResearchReportError(RuntimeError):
    """Raised when a canonical input is missing or violates the report contract."""


def build_task_research_report(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = get_project_root()
    config_file = _resolve(config_path, root)
    config = _mapping(yaml.safe_load(config_file.read_text(encoding="utf-8")), "config")
    inputs = _mapping(config.get("inputs"), "inputs")
    output = _mapping(config.get("output"), "output")
    validation = _mapping(config.get("validation", {}), "validation")

    input_paths = {name: _resolve(value, root) for name, value in inputs.items()}
    detector_payload = _read_json(input_paths["detector_summary"])
    tracker_rows = _object_list(
        _read_json_value(input_paths["tracker_summary"]),
        "tracker summary",
    )
    idsw_payload = _read_json(input_paths["idsw_summary"])
    runtime_payload = _read_json(input_paths["task_runtime"])
    webcam_payload = _read_json(input_paths["physical_webcam"])
    task_webcam_smoke = _read_json(input_paths["task_physical_webcam_smoke"])
    threshold_payload = _read_json(input_paths["open_threshold"])
    semantic_payload = _read_json(input_paths["semantic_qwen"])
    panel_cache = _read_json(input_paths["semantic_panel_cache"])
    panel_worker = _read_json(input_paths["semantic_panel_worker"])
    review_status = _read_json(input_paths["semantic_review_status"])
    legacy_semantic = _read_json(input_paths["legacy_semantic_ablation"])

    detector_rows = _object_list(detector_payload.get("rows"), "detector rows")
    task_gt_rows, task_sources = _task_gt_rows(config.get("task_gt"), root)
    idsw_rows = _idsw_overall(idsw_payload, tracker_rows)
    semantic_row = _semantic_row(semantic_payload, panel_cache, panel_worker)
    _validate_runtime(
        runtime_payload,
        minimum_repeats=int(validation.get("minimum_runtime_repeats", 1)),
    )
    _validate_review_status(review_status)

    output_root = _resolve(output.get("root"), root, require_file=False)
    publish_root = _resolve(output.get("publish_root"), root, require_file=False)
    paths = {
        "json": output_root / "research_summary.json",
        "markdown": output_root / "research_report.md",
        "detectors_csv": output_root / "detector_metrics.csv",
        "trackers_csv": output_root / "sportsmot_tracker_metrics.csv",
        "task_gt_csv": output_root / "multidomain_tracking_metrics.csv",
        "runtime_csv": output_root / "runtime_metrics.csv",
        "runtime_profiles_csv": output_root / "runtime_profile_metrics.csv",
        "webcam_profiles_csv": output_root / "physical_webcam_metrics.csv",
        "task_webcam_smoke_csv": output_root / "task_webcam_smoke_metrics.csv",
        "open_threshold_csv": output_root / "open_threshold_metrics.csv",
        "semantic_csv": output_root / "semantic_metrics.csv",
        "idsw_csv": output_root / "idsw_taxonomy.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise TaskResearchReportError(f"Research report exists: {existing[0]}")
    output_root.mkdir(parents=True, exist_ok=True)
    publish_root.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "report_config": config_file,
        **input_paths,
        **task_sources,
    }
    sources = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in source_paths.items()
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "hardware": runtime_payload.get("hardware", {}),
        "measurement_contract": {
            "detector": "SportsMOT val, 2,900 images, same 640 px protocol",
            "sportsmot_tracking": "30 sequences, 20,171 frames, shared detections, TrackEval",
            "multidomain_tracking": "official or normalized GT evaluated by TrackEval",
            "runtime": (
                "three repeated foreground runs per profile: detector + tracker + "
                "queue bookkeeping + MP4 render; Qwen measured separately"
            ),
            "semantic": (
                "20 predicted tracks matched to official UA-DETRAC GT; "
                "selected-track batch inference"
            ),
            "idsw_taxonomy": (
                "heuristic diagnostic partition; official IDSW remains the "
                "TrackEval value"
            ),
        },
        "detectors": detector_rows,
        "sportsmot_trackers": tracker_rows,
        "multidomain_tracking": task_gt_rows,
        "runtime": runtime_payload,
        "physical_webcam": webcam_payload,
        "task_physical_webcam_smoke": task_webcam_smoke,
        "open_threshold_ablation": threshold_payload,
        "semantic_qwen": semantic_row,
        "legacy_semantic_ablation": legacy_semantic.get("pipelines", []),
        "idsw_taxonomy": idsw_rows,
        "semantic_gt_review": review_status,
        "scientific_status": {
            "tracking_gt_complete": True,
            "detector_gt_complete": True,
            "physical_webcam_measured": True,
            "task_physical_webcam_smoke_complete": (
                int(task_webcam_smoke.get("run_count", 0)) == 3
            ),
            "task_runtime_repeated": all(
                int(row.get("repeat_count", 0)) >= 3
                for row in runtime_payload.get("profiles", [])
            ),
            "single_qwen_smoke_complete": panel_worker.get("status") == "ok",
            "multidomain_semantic_gt_complete": (
                review_status.get("remaining_track_count", 1) == 0
            ),
        },
        "limitations": [
            (
                "Three runtime repeats estimate run-to-run spread, but they are "
                "not confidence intervals for every possible hardware state."
            ),
            (
                "AnimalTrack open-vocabulary quality is detector-limited and "
                "should not be generalized to all wildlife."
            ),
            (
                "The current single-Qwen panel smoke verifies execution, not "
                "classroom semantic accuracy."
            ),
            (
                "The current TaskConfig camera smoke was capture-limited to about "
                "1 source FPS (camera reported 0 FPS), so it validates integration "
                "but is not a throughput comparison."
            ),
            "Multidomain semantic labels remain draft until human review is complete.",
            "IDSW subtypes are heuristic; only total TrackEval IDSW is an official metric.",
        ],
        "sources": sources,
    }

    _write_json(paths["json"], payload)
    _write_csv(paths["detectors_csv"], detector_rows)
    _write_csv(paths["trackers_csv"], tracker_rows)
    _write_csv(paths["task_gt_csv"], task_gt_rows)
    _write_csv(paths["runtime_csv"], runtime_payload.get("runs", []))
    _write_csv(paths["runtime_profiles_csv"], runtime_payload.get("profiles", []))
    _write_csv(paths["webcam_profiles_csv"], webcam_payload.get("profiles", []))
    _write_csv(
        paths["task_webcam_smoke_csv"],
        task_webcam_smoke.get("profiles", []),
    )
    _write_csv(paths["open_threshold_csv"], threshold_payload.get("runs", []))
    _write_csv(paths["semantic_csv"], [semantic_row])
    _write_csv(paths["idsw_csv"], idsw_rows)
    figures = _write_figures(payload, output_root / "figures")
    paths["markdown"].write_text(_markdown(payload, figures), encoding="utf-8")

    if publish_root.exists() and overwrite:
        for candidate in publish_root.glob("research_*"):
            if candidate.is_file():
                candidate.unlink()
    publish_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["markdown"], publish_root / "research_report.md")
    shutil.copy2(paths["json"], publish_root / "research_summary.json")
    for name, path in paths.items():
        if name.endswith("_csv"):
            shutil.copy2(path, publish_root / path.name)
    publish_figures = publish_root / "figures"
    publish_figures.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        shutil.copy2(figure, publish_figures / figure.name)
    return {
        "status": "ok",
        "paths": {name: str(path) for name, path in paths.items()},
        "figures": [str(path) for path in figures],
        "published": str(publish_root),
    }


def _task_gt_rows(value: Any, root: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    entries = _object_list(value, "task_gt")
    rows: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    for item_value in entries:
        item = item_value
        benchmark_id = str(item.get("id", "")).strip()
        if not benchmark_id:
            raise TaskResearchReportError("Each task_gt item requires an id.")
        evaluation_path = _resolve(item.get("evaluation"), root)
        runtime_root = _resolve(item.get("runtime_root"), root, require_file=False)
        evaluation = _read_json(evaluation_path)
        if not evaluation.get("available"):
            raise TaskResearchReportError(f"TrackEval is unavailable for {benchmark_id}.")
        metrics = _mapping(evaluation.get("metrics"), f"{benchmark_id}.metrics")
        runtime_paths = sorted(runtime_root.glob("*/realtime_metrics.json"))
        if not runtime_paths:
            raise TaskResearchReportError(f"No runtime metrics found for {benchmark_id}.")
        runtime = _aggregate_runtime(runtime_paths)
        rows.append(
            {
                "id": benchmark_id,
                "domain": item.get("domain"),
                "dataset": item.get("dataset"),
                "detector": item.get("detector"),
                "tracker": item.get("tracker"),
                "profile": item.get("profile"),
                "sequence_count": len(evaluation.get("per_sequence", {})),
                "frame_count": runtime["frame_count"],
                **{
                    name: metrics.get(name)
                    for name in (
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
                    )
                },
                "processing_fps": runtime["processing_fps"],
                "p95_latency_ms": runtime["p95_latency_ms"],
                "peak_vram_gb": runtime["peak_vram_gb"],
            }
        )
        sources[f"task_gt_{benchmark_id}"] = evaluation_path
        for index, path in enumerate(runtime_paths, start=1):
            sources[f"task_runtime_{benchmark_id}_{index}"] = path
    return rows, sources


def _aggregate_runtime(paths: list[Path]) -> dict[str, float | int | None]:
    frames = 0
    processing_seconds = 0.0
    p95_values: list[float] = []
    peak_vram: list[float] = []
    for path in paths:
        payload = _read_json(path)
        frame_count = int(payload.get("frames", 0))
        fps = float(_mapping(payload.get("timing"), "timing").get("processing_fps", 0.0))
        if frame_count <= 0 or fps <= 0:
            raise TaskResearchReportError(f"Invalid runtime frame/FPS values: {path}")
        frames += frame_count
        processing_seconds += frame_count / fps
        latency = payload.get("timing", {}).get("frame_latency_ms_p95")
        if latency is not None:
            p95_values.append(float(latency))
        peak = payload.get("cuda_memory", {}).get("peak_allocated_bytes")
        if peak is not None:
            peak_vram.append(float(peak) / (1024**3))
    return {
        "frame_count": frames,
        "processing_fps": frames / processing_seconds,
        "p95_latency_ms": max(p95_values) if p95_values else None,
        "peak_vram_gb": max(peak_vram) if peak_vram else None,
    }


def _idsw_overall(payload: dict[str, Any], tracker_rows: list[Any]) -> list[dict[str, Any]]:
    summaries = _object_list(payload.get("summaries"), "IDSW summaries")
    rows = [dict(row) for row in summaries if row.get("sequence") == "__overall__"]
    tracker_lookup = {
        str(row.get("tracker")): row
        for row in tracker_rows
        if isinstance(row, dict)
    }
    tracker_names = set(tracker_lookup)
    if {str(row.get("tracker")) for row in rows} != tracker_names:
        raise TaskResearchReportError("IDSW taxonomy does not cover the tracker table.")
    categories = (
        "fragmentation", "identity_swap", "re_identification_failure",
        "association_error", "appearance_confusion",
    )
    for row in rows:
        row["official_idsw"] = tracker_lookup[str(row.get("tracker"))].get("IDSW")
        total = int(row.get("total_id_switches_recomputed", -1))
        count_sum = sum(int(row.get(f"{category}_count", 0)) for category in categories)
        percent_sum = sum(float(row.get(f"{category}_percent", 0.0)) for category in categories)
        if total != count_sum or abs(percent_sum - 100.0) > 0.02:
            raise TaskResearchReportError(f"Invalid IDSW partition for {row.get('tracker')}.")
    return rows


def _semantic_row(
    payload: dict[str, Any],
    panel_cache: dict[str, Any],
    panel_worker: dict[str, Any],
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"), "semantic summary")
    performance = _mapping(summary.get("performance_means"), "semantic performance")
    cache_runtime = _mapping(panel_cache.get("runtime"), "panel cache runtime")
    panel_timing = _mapping(cache_runtime.get("timing"), "panel timing")
    panel_cuda = _mapping(cache_runtime.get("cuda_memory"), "panel cuda")
    return {
        "model": cache_runtime.get("model", "Qwen/Qwen3-VL-4B-Instruct"),
        "quantization": cache_runtime.get("quantization", "8bit"),
        "gt_tracks": summary.get("semantic_gt_track_count"),
        "accuracy": summary.get("semantic_track_accuracy"),
        "macro_f1": summary.get("semantic_macro_f1"),
        "coverage": summary.get("semantic_coverage"),
        "selective_accuracy": summary.get("semantic_selective_accuracy"),
        "unknown_rejection_f1": summary.get("unknown_rejection_f1"),
        "hallucination_rate": summary.get("semantic_hallucination_rate"),
        "official_qwen_load_seconds": performance.get("qwen_model_load_seconds"),
        "official_qwen_inference_seconds": performance.get("qwen_inference_seconds"),
        "official_peak_vram_gb": (
            float(performance.get("qwen_peak_allocated_bytes")) / (1024**3)
            if performance.get("qwen_peak_allocated_bytes") is not None else None
        ),
        "panel_smoke_status": panel_worker.get("status"),
        "panel_model_load_seconds": panel_timing.get("model_load_seconds"),
        "panel_inference_seconds": panel_timing.get("inference_seconds"),
        "panel_peak_vram_gb": (
            float(panel_cuda.get("peak_allocated_bytes")) / (1024**3)
            if panel_cuda.get("peak_allocated_bytes") is not None else None
        ),
        "panel_processed_events": panel_worker.get("processed_event_count"),
    }


def _validate_runtime(payload: dict[str, Any], *, minimum_repeats: int = 1) -> None:
    runs = _list(payload.get("runs"), "runtime runs")
    if len(runs) < 4:
        raise TaskResearchReportError("Runtime matrix must contain at least four runs.")
    if any(float(row.get("processing_fps") or 0.0) <= 0 for row in runs):
        raise TaskResearchReportError("Runtime matrix contains non-positive FPS.")
    profiles = _object_list(payload.get("profiles"), "runtime profiles")
    incomplete = [
        str(row.get("profile"))
        for row in profiles
        if int(row.get("repeat_count", 0)) < minimum_repeats
    ]
    if incomplete:
        raise TaskResearchReportError(
            "Runtime profiles do not satisfy the minimum repeat count "
            f"({minimum_repeats}): {', '.join(incomplete)}"
        )


def _validate_review_status(payload: dict[str, Any]) -> None:
    total = int(payload.get("track_count", 0))
    reviewed = int(payload.get("reviewed_track_count", 0))
    remaining = int(payload.get("remaining_track_count", 0))
    if total <= 0 or reviewed + remaining != total:
        raise TaskResearchReportError("Semantic review-status counts are inconsistent.")


def _write_figures(payload: dict[str, Any], root: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError:
        return []
    root.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    def save(figure: Any, name: str) -> None:
        path = root / name
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figures.append(path)

    detectors = payload["detectors"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [row.get("display_name", row.get("name")) for row in detectors]
    x = list(range(len(names)))
    ax.bar(
        [v - 0.2 for v in x],
        [100 * float(r["recall"]) for r in detectors],
        0.4,
        label="Recall",
    )
    ax.bar(
        [v + 0.2 for v in x],
        [100 * float(r["map50_95"]) for r in detectors],
        0.4,
        label="mAP50-95",
    )
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "research_detector_quality.png")

    trackers = payload["sportsmot_trackers"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = [row.get("display_name", row.get("tracker")) for row in trackers]
    x = list(range(len(names)))
    ax.bar([v - 0.2 for v in x], [float(r["HOTA"]) for r in trackers], 0.4, label="HOTA")
    ax.bar([v + 0.2 for v in x], [float(r["IDF1"]) for r in trackers], 0.4, label="IDF1")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "research_tracker_quality.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for row in trackers:
        ax.scatter(float(row["cached_pipeline_fps"]), float(row["HOTA"]), s=55)
        display_name = str(row.get("display_name", row["tracker"]))
        annotation_offsets = {
            "FastTracker": (6, 8),
            "ByteTrack": (6, -12),
        }
        ax.annotate(
            display_name,
            (float(row["cached_pipeline_fps"]), float(row["HOTA"])),
            xytext=annotation_offsets.get(display_name, (4, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cached detector + tracker FPS (log scale)")
    ax.set_ylabel("HOTA")
    ax.grid(alpha=0.25)
    save(fig, "research_tracker_tradeoff.png")

    task_rows = payload["multidomain_tracking"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [row["id"] for row in task_rows]
    x = list(range(len(labels)))
    ax.bar([v - 0.2 for v in x], [float(r["HOTA"]) for r in task_rows], 0.4, label="HOTA")
    ax.bar([v + 0.2 for v in x], [float(r["IDF1"]) for r in task_rows], 0.4, label="IDF1")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "research_multidomain_tracking.png")

    ctc = {
        row["id"]: row
        for row in task_rows
        if row["id"] in {"ctc_none", "ctc_clahe"}
    }
    if len(ctc) == 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        labels = ["No preprocessing", "CLAHE"]
        rows = [ctc["ctc_none"], ctc["ctc_clahe"]]
        x = [0, 1]
        ax.bar(
            [v - 0.22 for v in x],
            [float(r["HOTA"]) for r in rows],
            0.22,
            label="HOTA",
        )
        ax.bar(x, [float(r["MOTA"]) for r in rows], 0.22, label="MOTA")
        ax.bar(
            [v + 0.22 for v in x],
            [float(r["processing_fps"]) for r in rows],
            0.22,
            label="FPS",
        )
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save(fig, "research_preprocessing_ablation.png")

    traffic_gate = {
        row["id"]: row
        for row in task_rows
        if row["id"] in {"ua_detrac_tracktrack", "ua_detrac_class_gate"}
    }
    if len(traffic_gate) == 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        labels = ["Class-agnostic", "Hard class gate"]
        rows = [
            traffic_gate["ua_detrac_tracktrack"],
            traffic_gate["ua_detrac_class_gate"],
        ]
        x = [0, 1]
        ax.bar(
            [value - 0.18 for value in x],
            [float(row["HOTA"]) for row in rows],
            0.36,
            label="HOTA",
        )
        ax.bar(
            [value + 0.18 for value in x],
            [float(row["IDF1"]) for row in rows],
            0.36,
            label="IDF1",
        )
        for index, row in enumerate(rows):
            ax.text(
                index,
                max(float(row["HOTA"]), float(row["IDF1"])) + 1.0,
                f"IDSW={int(row['IDSW'])}",
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 85)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save(fig, "research_class_gate_ablation.png")

    runtime_rows = payload["runtime"]["profiles"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["profile"] for row in runtime_rows]
    means = [float(row.get("processing_fps_mean") or 0) for row in runtime_rows]
    deviations = [
        float(row.get("processing_fps_std") or 0) for row in runtime_rows
    ]
    ax.bar(
        range(len(labels)),
        means,
        yerr=deviations,
        capsize=3,
    )
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Processing FPS")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "research_runtime_fps.png")

    threshold_rows = payload["open_threshold_ablation"]["runs"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [row["name"] for row in threshold_rows]
    x = list(range(len(labels)))
    ax.bar(
        x,
        [
            100 * float(row.get("track_frame_coverage", 0))
            for row in threshold_rows
        ],
        label="Track frame coverage",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "research_open_threshold_ablation.png")

    semantic = payload["semantic_qwen"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metric_names = ["Accuracy", "Macro-F1", "Coverage", "Unknown F1", "Hallucination"]
    values = [
        semantic["accuracy"],
        semantic["macro_f1"],
        semantic["coverage"],
        semantic["unknown_rejection_f1"],
        semantic["hallucination_rate"],
    ]
    colors = ["#2878B5", "#2878B5", "#2878B5", "#2878B5", "#D95319"]
    ax.bar(metric_names, [100 * float(value or 0) for value in values], color=colors)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "research_semantic_quality.png")

    idsw_rows = sorted(payload["idsw_taxonomy"], key=lambda row: str(row.get("tracker")))
    categories = [
        "fragmentation",
        "identity_swap",
        "re_identification_failure",
        "association_error",
        "appearance_confusion",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = [0.0] * len(idsw_rows)
    for category in categories:
        values = [float(row.get(f"{category}_percent", 0.0)) for row in idsw_rows]
        ax.bar(
            range(len(idsw_rows)),
            values,
            bottom=bottom,
            label=category.replace("_", " "),
        )
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    ax.set_xticks(
        range(len(idsw_rows)),
        [row["tracker"] for row in idsw_rows],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Heuristic event share (%)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "research_idsw_taxonomy.png")

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_axis_off()
    boxes = [
        (0.02, "TaskConfig\nintent + ontology"),
        (0.22, "Detector\nYOLO / YOLOE"),
        (0.42, "TrackTrack\nID + motion"),
        (0.62, "Evidence queue\ncontext + crops"),
        (0.82, "Qwen3-VL-4B 8-bit\nsemantic cache"),
    ]
    for x_value, label in boxes:
        patch = FancyBboxPatch(
            (x_value, 0.35),
            0.16,
            0.3,
            boxstyle="round,pad=0.02",
            facecolor="#E8F0E3",
            edgecolor="#315A24",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x_value + 0.08, 0.5, label, ha="center", va="center", fontsize=9)
    for index in range(len(boxes) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (boxes[index][0] + 0.16, 0.5),
                (boxes[index + 1][0], 0.5),
                arrowstyle="->",
                mutation_scale=12,
                color="#333333",
            )
        )
    ax.text(
        0.5,
        0.12,
        "Foreground CV renders immediately; semantic labels arrive "
        "asynchronously and are fused over time.",
        ha="center",
        fontsize=9,
    )
    save(fig, "research_pipeline_architecture.png")
    return figures


def _markdown(payload: dict[str, Any], figures: list[Path]) -> str:
    hardware = payload["hardware"]
    lines = [
        "# Single-Qwen Realtime Tracking Research Report",
        "",
        (
            f"Hardware: {hardware.get('gpu')} "
            f"({float(hardware.get('gpu_memory_gb') or 0):.1f} GB VRAM), "
            f"{hardware.get('logical_cpu_count')} logical CPU threads."
        ),
        "",
        "![Pipeline](figures/research_pipeline_architecture.png)",
        "",
        "## Measurement contract",
        "",
    ]
    lines.extend(
        f"- **{name}:** {description}"
        for name, description in payload["measurement_contract"].items()
    )
    lines.extend(
        [
            "",
            "## SportsMOT detector evaluation",
            "",
            (
                "| Detector | Training | Precision | Recall | mAP50 | mAP50-95 | "
                "Detector FPS | E2E FPS |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["detectors"]:
        lines.append(
            f"| {row.get('display_name', row['name'])} | {row['training']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['detector_fps']:.2f} | {row['end_to_end_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## SportsMOT tracking",
            "",
            "| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Cached FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["sportsmot_trackers"]:
        lines.append(
            f"| {row.get('display_name', row['tracker'])} | "
            f"{row['HOTA']:.3f} | {row['DetA']:.3f} | "
            f"{row['AssA']:.3f} | {row['MOTA']:.3f} | "
            f"{row['IDF1']:.3f} | {row['IDSW']} | "
            f"{row['cached_pipeline_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Multidomain tracking with GT",
            "",
            "| Task | Domain | Frames | HOTA | MOTA | IDF1 | IDSW | FPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["multidomain_tracking"]:
        lines.append(
            f"| {row['id']} | {row['domain']} | {row['frame_count']} | "
            f"{row['HOTA']:.3f} | {row['MOTA']:.3f} | "
            f"{row['IDF1']:.3f} | {row['IDSW']} | "
            f"{row['processing_fps']:.2f} |"
        )
    traffic_gate = {
        row["id"]: row
        for row in payload["multidomain_tracking"]
        if row["id"] in {"ua_detrac_tracktrack", "ua_detrac_class_gate"}
    }
    if len(traffic_gate) == 2:
        no_gate = traffic_gate["ua_detrac_tracktrack"]
        hard_gate = traffic_gate["ua_detrac_class_gate"]
        lines.extend(
            [
                "",
                "## Traffic class-association ablation",
                "",
                (
                    "Both rows use the same video, detector, TrackTrack thresholds, "
                    "preprocessing setting, and official GT. Only the hard detector-class "
                    "gate changes."
                ),
                "",
                "| Association | HOTA | MOTA | IDF1 | IDSW | Frag | FPS |",
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    f"| Class-agnostic + temporal class stabilization | "
                    f"{no_gate['HOTA']:.3f} | {no_gate['MOTA']:.3f} | "
                    f"{no_gate['IDF1']:.3f} | {no_gate['IDSW']} | "
                    f"{no_gate['Frag']} | {no_gate['processing_fps']:.2f} |"
                ),
                (
                    f"| Hard detector-class gate | {hard_gate['HOTA']:.3f} | "
                    f"{hard_gate['MOTA']:.3f} | {hard_gate['IDF1']:.3f} | "
                    f"{hard_gate['IDSW']} | {hard_gate['Frag']} | "
                    f"{hard_gate['processing_fps']:.2f} |"
                ),
                "",
                (
                    "The class-agnostic profile is selected for production because it "
                    f"raises HOTA by {no_gate['HOTA'] - hard_gate['HOTA']:.3f}, raises "
                    f"IDF1 by {no_gate['IDF1'] - hard_gate['IDF1']:.3f}, and reduces "
                    f"IDSW by {int(hard_gate['IDSW']) - int(no_gate['IDSW'])} on this "
                    "controlled sequence."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Repeated foreground runtime",
            "",
            (
                "| Profile | Runs | Processing FPS | Source progress FPS | "
                "P95 latency | Drop rate | Peak VRAM |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["runtime"]["profiles"]:
        lines.append(
            f"| {row['profile']} | {row['repeat_count']} | "
            f"{row['processing_fps_mean']:.2f} +/- "
            f"{row['processing_fps_std']:.2f} | "
            f"{row['source_progress_fps_mean']:.2f} +/- "
            f"{row['source_progress_fps_std']:.2f} | "
            f"{row['p95_latency_ms_mean']:.2f} ms | "
            f"{100 * row['drop_rate_mean']:.2f}% | "
            f"{row['peak_vram_gb_mean']:.3f} GiB |"
        )
    lines.extend(
        [
            "",
            "## Physical webcam foreground benchmark",
            "",
            (
                "| Profile | Runs | Processing FPS | Source progress FPS | "
                "P95 latency | Drop rate |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["physical_webcam"]["profiles"]:
        lines.append(
            f"| {row['profile']} | {row['repeat_count']} | "
            f"{row['processing_fps_mean']:.2f} | "
            f"{row['source_progress_fps_mean']:.2f} | "
            f"{row['p95_latency_ms_mean']:.2f} ms | "
            f"{100 * row['drop_rate_mean']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Current TaskConfig webcam integration smoke",
            "",
            (
                "This 150-frame integration run used the current TaskConfig path. "
                "The camera driver exposed no nominal FPS and delivered about one "
                "frame per second, so these values must not be read as model capacity."
            ),
            "",
            "| Profile | Frames/s from camera | Processing FPS | P95 latency |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["task_physical_webcam_smoke"]["profiles"]:
        lines.append(
            f"| {row['profile']} | {row['source_progress_fps_mean']:.2f} | "
            f"{row['processing_fps_mean']:.2f} | "
            f"{row['p95_latency_ms_mean']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "## Open-vocabulary tracking-threshold ablation",
            "",
            (
                "| Profile | Detections | Detection coverage | Track boxes | "
                "Track coverage | IDs | FPS |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["open_threshold_ablation"]["runs"]:
        lines.append(
            f"| {row['name']} | {row['detection_count']} | "
            f"{100 * row['detection_frame_coverage']:.1f}% | "
            f"{row['track_box_count']} | "
            f"{100 * row['track_frame_coverage']:.1f}% | "
            f"{row['unique_track_count']} | {row['processing_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## ID-switch diagnostic taxonomy",
            "",
            (
                "Counts and percentages below are heuristic diagnostics. Official "
                "TrackEval IDSW remains the ranking metric."
            ),
            "",
            (
                "| Tracker | Official IDSW | Diagnostic events | Fragmentation | "
                "Identity swap | ReID failure | Association | Appearance |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["idsw_taxonomy"]:
        lines.append(
            f"| {row['tracker']} | {row['official_idsw']} | "
            f"{row['total_id_switches_recomputed']} | "
            f"{row['fragmentation_count']} ({row['fragmentation_percent']:.1f}%) | "
            f"{row['identity_swap_count']} ({row['identity_swap_percent']:.1f}%) | "
            f"{row['re_identification_failure_count']} "
            f"({row['re_identification_failure_percent']:.1f}%) | "
            f"{row['association_error_count']} "
            f"({row['association_error_percent']:.1f}%) | "
            f"{row['appearance_confusion_count']} "
            f"({row['appearance_confusion_percent']:.1f}%) |"
        )
    semantic = payload["semantic_qwen"]
    lines.extend([
        "",
        "## Qwen semantic evaluation",
        "",
        (
            "| GT tracks | Accuracy | Known-class Macro-F1 | Coverage | "
            "Selective accuracy | "
            "Unknown F1 | Hallucination |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {semantic['gt_tracks']} | {100 * semantic['accuracy']:.1f}% | "
            f"{100 * semantic['macro_f1']:.1f}% | "
            f"{100 * semantic['coverage']:.1f}% | "
            f"{100 * semantic['selective_accuracy']:.1f}% | "
            f"{100 * semantic['unknown_rejection_f1']:.1f}% | "
            f"{100 * semantic['hallucination_rate']:.1f}% |"
        ),
        "",
        (
            "The current evidence-panel smoke processed "
            f"{semantic['panel_processed_events']} event with status "
            f"`{semantic['panel_smoke_status']}`; it is an execution test, "
            "not an accuracy estimate."
        ),
        "",
        "## Main conclusions",
        "",
        (
            "- TrackTrack gives the strongest SportsMOT HOTA in the saved "
            "shared-detection benchmark; BoT-SORT ReID has fewer official ID "
            "switches but much lower throughput."
        ),
        (
            "- On the controlled UA-DETRAC ablation, class-agnostic association "
            "followed by temporal class stabilization outperforms a hard class gate "
            "and is therefore the production multiclass policy."
        ),
        (
            "- Open-vocabulary detection is not automatically more accurate: "
            "YOLOE zebra tracking is detector-limited in the current zero-shot "
            "profile."
        ),
        (
            "- CLAHE improves the held-out microscopy tracking score but costs "
            "foreground FPS, so preprocessing remains task-specific."
        ),
        (
            "- The physical webcam foreground path sustains the 30 FPS source "
            "rate; Qwen labels are asynchronous and have a separate semantic "
            "latency."
        ),
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.extend(
            [
                f"### {figure.stem.replace('_', ' ').title()}",
                "",
                f"![{figure.stem}](figures/{figure.name})",
                "",
            ]
        )
    return "\n".join(lines)


def _resolve(value: Any, root: Path, *, require_file: bool = True) -> Path:
    if value is None:
        raise TaskResearchReportError("A required path is missing from the report config.")
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, root)
    if require_file and not resolved.is_file():
        raise TaskResearchReportError(f"Required report input does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(_read_json_value(path), str(path))


def _read_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskResearchReportError(f"{name} must be an object.")
    return dict(value)


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise TaskResearchReportError(f"{name} must be a non-empty list.")
    return list(value)


def _object_list(value: Any, name: str) -> list[dict[str, Any]]:
    rows = _list(value, name)
    if any(not isinstance(row, dict) for row in rows):
        raise TaskResearchReportError(f"{name} must contain only objects.")
    return [dict(row) for row in rows]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TaskResearchReportError(f"Cannot write empty CSV: {path}")
    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
            if not isinstance(row.get(key), (dict, list))
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "TaskResearchReportError",
    "build_task_research_report",
]
