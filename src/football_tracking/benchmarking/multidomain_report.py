"""Build an auditable report for licensed multi-domain video trials."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


class MultidomainReportError(RuntimeError):
    """Raised when a trial artifact is missing or malformed."""


_DOMAIN_FAMILIES = {
    "traffic": {"traffic", "urban transport", "street traffic", "transport"},
    "education": {"education", "classroom", "lecture hall", "school"},
    "wildlife": {"wildlife", "nature", "animal", "bird"},
    "sports": {"sports", "football", "soccer"},
}
_CLASS_ALIASES = {
    "automobile": "car",
    "lorry": "truck",
    "motorbike": "motorcycle",
    "pedestrian": "person",
    "lecturer": "teacher",
    "student": "person",
    "teacher": "person",
    "seabird": "bird",
}


def _name(value: Any) -> str:
    return re.sub(r"[_-]+", " ", str(value or "").strip().lower())


def _domain_family(value: Any) -> str:
    normalized = _name(value)
    for family, aliases in _DOMAIN_FAMILIES.items():
        if normalized in aliases or any(alias in normalized for alias in aliases):
            return family
    return normalized


def _class_name(value: Any) -> str:
    normalized = _name(value)
    return _CLASS_ALIASES.get(normalized, normalized)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MultidomainReportError(f"Required artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MultidomainReportError(f"Expected a JSON object: {path}")
    return payload


def _gib(value: Any) -> float | None:
    try:
        return round(float(value) / (1024**3), 3)
    except (TypeError, ValueError):
        return None


def compute_mot_stability_proxy(path: Path | None, fps: float) -> dict[str, Any]:
    """Return GT-free continuity proxies; these are not IDSW measurements."""
    if path is None or not path.is_file():
        return {
            "status": "unavailable",
            "reason": "MOT prediction file is missing",
        }
    frames_by_track: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            columns = line.strip().split(",")
            if len(columns) < 2:
                continue
            try:
                frame_id = int(float(columns[0]))
                track_id = int(float(columns[1]))
            except ValueError:
                continue
            frames_by_track.setdefault(track_id, []).append(frame_id)
    if not frames_by_track:
        return {"status": "empty", "track_count": 0}

    lengths = [len(set(frames)) for frames in frames_by_track.values()]
    spans: list[int] = []
    gap_events = 0
    for frames in frames_by_track.values():
        ordered = sorted(set(frames))
        spans.append(ordered[-1] - ordered[0] + 1)
        gap_events += sum(
            right - left > 1 for left, right in zip(ordered, ordered[1:], strict=False)
        )
    short_threshold = max(1, int(round(fps)))
    short_count = sum(length < short_threshold for length in lengths)
    continuity = [
        length / span for length, span in zip(lengths, spans, strict=True) if span > 0
    ]
    return {
        "status": "ok",
        "scope": "prediction_only_proxy_not_ground_truth_idsw",
        "track_count": len(lengths),
        "observation_count": sum(lengths),
        "short_track_threshold_frames": short_threshold,
        "short_track_count": short_count,
        "short_track_ratio": round(short_count / len(lengths), 6),
        "median_track_length_frames": round(float(statistics.median(lengths)), 3),
        "mean_track_length_frames": round(float(statistics.mean(lengths)), 3),
        "mean_track_continuity": round(float(statistics.mean(continuity)), 6),
        "within_id_gap_events": gap_events,
    }


def _record(sample: dict[str, Any], run_root: Path) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    report = _read_json(run_root / sample_id / "adaptive_run_report.json")
    ground_truth = dict(sample.get("ground_truth", {}))
    video = dict(sample.get("video", {}))
    scene = dict(report.get("scene", {}))
    tracking = dict(report.get("tracking", {}))
    qwen = dict(report.get("qwen_track_semantics", {}))
    locate = dict(report.get("locateanything_verification", {}))
    fusion = dict(report.get("semantic_fusion", {}))
    render = dict(report.get("render", {}))

    fused_path = run_root / sample_id / "fused_track_semantics.json"
    fused = _read_json(fused_path)
    fused_tracks = [row for row in fused.get("tracks", []) if isinstance(row, dict)]
    observed_classes = {
        _class_name(row.get("canonical_name"))
        for row in scene.get("objects", [])
        if isinstance(row, dict)
    }
    observed_classes.update(
        _class_name(row.get("class_label")) for row in fused_tracks if row.get("accepted")
    )
    expected_classes = {_class_name(value) for value in ground_truth.get("base_classes", [])}
    matched_classes = expected_classes & observed_classes
    fine_outputs = sorted(
        {
            _name(row.get("fine_label"))
            for row in fused_tracks
            if row.get("fine_accepted") and _name(row.get("fine_label")) != "unknown"
        }
    )
    timing = dict(tracking.get("timing", {}))
    tracker_diagnostics = dict(tracking.get("tracker_diagnostics") or {})
    fps = float(video.get("fps") or 0.0)
    mot_value = tracking.get("output_mot")
    mot_stability = compute_mot_stability_proxy(
        Path(mot_value) if mot_value else None, fps
    )
    qwen_timing = dict(qwen.get("timing", {}))
    qwen_cuda = dict(qwen.get("cuda_memory", {}))
    locate_cuda = dict(locate.get("cuda_memory", {}))
    render_semantics = dict(render.get("semantics", {}))
    expected_domain = _domain_family(ground_truth.get("domain"))
    predicted_domain = _domain_family(scene.get("domain"))
    semantic_metadata = next(
        (
            str(item.get("path"))
            for item in report.get("artifacts", [])
            if str(item.get("path", "")).endswith("_semantic.semantic.metadata.json")
        ),
        None,
    )
    semantic_video = (
        semantic_metadata.replace(".semantic.metadata.json", ".mp4") if semantic_metadata else None
    )
    return {
        "sample_id": sample_id,
        "source_page": sample.get("source_page"),
        "license": sample.get("license"),
        "source_duration_seconds": video.get("duration_seconds"),
        "source_frame_count": video.get("frame_count"),
        "source_resolution": (
            f"{video.get('width')}x{video.get('height')}"
            if video.get("width") and video.get("height")
            else None
        ),
        "expected_domain": expected_domain,
        "predicted_domain_raw": scene.get("domain"),
        "predicted_domain_family": predicted_domain,
        "domain_match": expected_domain == predicted_domain,
        "expected_base_classes": sorted(expected_classes),
        "observed_base_classes": sorted(value for value in observed_classes if value),
        "matched_base_classes": sorted(matched_classes),
        "video_level_class_recall": (
            round(len(matched_classes) / len(expected_classes), 6) if expected_classes else None
        ),
        "processed_frames": tracking.get("frame_count"),
        "detection_count": tracking.get("detection_count"),
        "unique_track_count": tracking.get("unique_track_count"),
        "raw_class_switches": tracker_diagnostics.get("raw_class_switches"),
        "stable_class_switches": tracker_diagnostics.get("stable_class_switches"),
        "suppressed_class_switches": tracker_diagnostics.get(
            "suppressed_class_switches"
        ),
        "tracking_stability_proxy": mot_stability,
        "short_track_ratio": mot_stability.get("short_track_ratio"),
        "median_track_length_frames": mot_stability.get("median_track_length_frames"),
        "within_id_gap_events": mot_stability.get("within_id_gap_events"),
        "steady_state_fps": timing.get("steady_state_fps"),
        "cold_start_fps": timing.get("cold_start_fps"),
        "qwen_discovery_seconds": scene.get("inference_seconds"),
        "qwen_semantic_seconds": qwen_timing.get("inference_seconds"),
        "qwen_peak_vram_gib": _gib(qwen_cuda.get("peak_allocated_bytes")),
        "locate_peak_vram_gib": _gib(locate_cuda.get("peak_allocated_bytes")),
        "modeled_track_count": fusion.get("track_count"),
        "base_semantic_coverage_modeled": fusion.get("coverage"),
        "fine_semantic_coverage_modeled": fusion.get("fine_coverage"),
        "render_track_coverage_all": render_semantics.get("track_coverage"),
        "accepted_fine_labels": fine_outputs,
        "semantic_track_accuracy": None,
        "accuracy_status": "requires_human_per_track_ground_truth",
        "semantic_video": semantic_video,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _write_charts(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [row["sample_id"].replace("_", "\n") for row in rows]
    chart_paths: list[str] = []

    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = range(len(rows))
    axis.bar(
        [index - 0.18 for index in x],
        [row["steady_state_fps"] or 0 for row in rows],
        0.36,
        label="Steady-state",
    )
    axis.bar(
        [index + 0.18 for index in x],
        [row["cold_start_fps"] or 0 for row in rows],
        0.36,
        label="Cold-start",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Frames per second")
    axis.set_title("Adaptive detector + tracker throughput")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "multidomain_fps.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    chart_paths.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        x,
        [100 * float(row["short_track_ratio"] or 0) for row in rows],
        0.55,
        color="#b45309",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Tracks shorter than one second (%)")
    axis.set_title("Prediction-only fragmentation proxy")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "multidomain_track_fragmentation_proxy.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    chart_paths.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        [index - 0.2 for index in x],
        [row["raw_class_switches"] or 0 for row in rows],
        0.4,
        label="Raw class changes",
    )
    axis.bar(
        [index + 0.2 for index in x],
        [row["stable_class_switches"] or 0 for row in rows],
        0.4,
        label="Final emitted class changes",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Class transitions")
    axis.set_title("Temporal class stabilization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "multidomain_class_stability.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    chart_paths.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        [index - 0.24 for index in x],
        [row["video_level_class_recall"] or 0 for row in rows],
        0.24,
        label="Video-level class recall",
    )
    axis.bar(
        x,
        [row["base_semantic_coverage_modeled"] or 0 for row in rows],
        0.24,
        label="Base coverage (modeled)",
    )
    axis.bar(
        [index + 0.24 for index in x],
        [row["fine_semantic_coverage_modeled"] or 0 for row in rows],
        0.24,
        label="Fine coverage (modeled)",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Ratio")
    axis.set_title("Vocabulary and semantic coverage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "multidomain_semantic_coverage.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    chart_paths.append(str(path))
    return chart_paths


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Public Multi-Domain Trial",
        "",
        "These runs use licensed public videos and the same RTX 4060 Laptop GPU (8 GB).",
        "Video-level labels test domain/vocabulary discovery only. They are not per-track GT,",
        "so semantic track accuracy remains unreported until manual track annotation exists.",
        "",
        "| Sample | Length | Domain | Class recall | Tracks | Steady FPS | Short tracks | "
        "Base coverage | Fine coverage | Raw/stable class changes | Accepted fine labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fine = ", ".join(row["accepted_fine_labels"]) or "none"
        lines.append(
            f"| {row['sample_id']} | {float(row['source_duration_seconds'] or 0):.1f}s | "
            f"{'yes' if row['domain_match'] else 'no'} | "
            f"{100 * (row['video_level_class_recall'] or 0):.1f}% | "
            f"{row['unique_track_count']} | {float(row['steady_state_fps'] or 0):.2f} | "
            f"{100 * float(row['short_track_ratio'] or 0):.1f}% | "
            f"{100 * float(row['base_semantic_coverage_modeled'] or 0):.1f}% | "
            f"{100 * float(row['fine_semantic_coverage_modeled'] or 0):.1f}% | "
            f"{row['raw_class_switches'] or 0}/{row['stable_class_switches'] or 0} | "
            f"{fine} |"
        )
    lines.extend(
        [
            "",
            f"- Domain-family accuracy: "
            f"**{100 * summary['domain_family_accuracy']:.1f}%** "
            f"({summary['domain_match_count']}/{summary['sample_count']}).",
            "- Mean steady detector+tracker throughput: "
            f"**{summary['mean_steady_state_fps']:.2f} FPS**.",
            "- Fine labels are emitted selectively: unsupported vehicle/role "
            "subtypes stay unknown.",
            "- Short-track ratio and within-ID gaps are prediction-only continuity proxies; "
            "they are not GT-based IDSW.",
            "- Next accuracy step: annotate class_label and fine_label for every evaluated track.",
            "",
            "![Throughput](multidomain_fps.png)",
            "",
            "![Semantic coverage](multidomain_semantic_coverage.png)",
            "",
            "![Class stability](multidomain_class_stability.png)",
            "",
            "![Fragmentation proxy](multidomain_track_fragmentation_proxy.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_multidomain_trial_report(
    manifest_path: Path,
    run_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "multidomain_trial_summary.json"
    if output_json.exists() and not overwrite:
        raise MultidomainReportError(f"Output already exists: {output_json}")
    samples = [row for row in manifest.get("samples", []) if isinstance(row, dict)]
    if not samples:
        raise MultidomainReportError("Sample manifest contains no samples.")
    rows = [_record(sample, run_root) for sample in samples]
    domain_matches = sum(bool(row["domain_match"]) for row in rows)
    fps_values = [
        float(row["steady_state_fps"]) for row in rows if row["steady_state_fps"] is not None
    ]
    summary = {
        "sample_count": len(rows),
        "domain_match_count": domain_matches,
        "domain_family_accuracy": domain_matches / len(rows),
        "mean_steady_state_fps": sum(fps_values) / len(fps_values) if fps_values else 0.0,
        "semantic_track_accuracy": None,
        "accuracy_status": "requires_human_per_track_ground_truth",
    }
    payload = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "run_root": str(run_root.resolve()),
        "hardware": _read_json(run_root / samples[0]["sample_id"] / "adaptive_run_report.json").get(
            "hardware", {}
        ),
        "summary": summary,
        "samples": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "multidomain_trial_summary.csv"
    _write_csv(csv_path, rows)
    chart_paths = _write_charts(output_dir, rows)
    markdown_path = output_dir / "multidomain_trial_report.md"
    _write_markdown(markdown_path, rows, summary)
    return {
        "status": "ok",
        "summary": summary,
        "paths": {
            "json": str(output_json),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "charts": chart_paths,
        },
    }


__all__ = [
    "MultidomainReportError",
    "build_multidomain_trial_report",
    "compute_mot_stability_proxy",
]
