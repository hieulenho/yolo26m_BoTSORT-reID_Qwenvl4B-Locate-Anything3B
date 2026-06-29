"""Dataset audit statistics and Milestone 3 quality assurance outputs."""

from __future__ import annotations

import csv
import json
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox, xyxy_to_xywh
from football_tracking.data.schemas import SequenceInfo
from football_tracking.data.statistics import compute_dataset_statistics, json_safe
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.visualization.draw_annotations import draw_annotation_samples


class AuditConfigError(RuntimeError):
    """Raised when the dataset audit config is invalid."""


@dataclass(frozen=True)
class AuditConfig:
    project_root: Path
    config_path: Path
    data_config_path: Path
    manifest_path: Path
    yolo_dataset_path: Path
    mot_root: Path
    splits: list[str]
    include_empty_frames: bool
    check_images: bool
    check_labels: bool
    check_track_ids: bool
    check_split_leakage: bool
    sample_sequences: int
    sample_frames_per_sequence: int
    random_seed: int
    small_max_ratio: float
    medium_max_ratio: float
    metrics_dir: Path
    figures_dir: Path
    samples_dir: Path
    log_level: str
    fail_on_error: bool


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditConfigError(f"{section} must be a mapping.")
    return value


def _resolve(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuditConfigError(f"{section} must be a non-empty path string.")
    return resolve_project_path(value, project_root=project_root)


def load_audit_config(config_path: str | Path) -> AuditConfig:
    project_root = get_project_root()
    resolved_config_path = resolve_project_path(config_path, project_root=project_root)
    if not resolved_config_path.is_file():
        raise AuditConfigError(f"Audit config does not exist: {resolved_config_path}")
    raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config root")
    dataset = _mapping(root.get("dataset"), "dataset")
    audit = _mapping(root.get("audit"), "audit")
    output = _mapping(root.get("output"), "output")
    runtime = _mapping(root.get("runtime", {}), "runtime")
    box_bins = _mapping(audit.get("box_size_bins", {}), "audit.box_size_bins")

    splits = audit.get("splits", ["train", "val", "test"])
    if not isinstance(splits, list) or not all(
        split in {"train", "val", "test"} for split in splits
    ):
        raise AuditConfigError("audit.splits must contain only train, val, and test.")
    sample_sequences = int(audit.get("sample_sequences", 5))
    sample_frames = int(audit.get("sample_frames_per_sequence", 5))
    if sample_sequences <= 0 or sample_frames <= 0:
        raise AuditConfigError("audit sample counts must be positive.")
    small_max_ratio = float(box_bins.get("small_max_ratio", 0.01))
    medium_max_ratio = float(box_bins.get("medium_max_ratio", 0.05))
    if not 0.0 <= small_max_ratio <= medium_max_ratio:
        raise AuditConfigError("box size ratios must satisfy 0 <= small <= medium.")

    return AuditConfig(
        project_root=project_root,
        config_path=resolved_config_path,
        data_config_path=_resolve(dataset.get("config"), project_root, "dataset.config"),
        manifest_path=_resolve(dataset.get("manifest"), project_root, "dataset.manifest"),
        yolo_dataset_path=_resolve(
            dataset.get("yolo_dataset"),
            project_root,
            "dataset.yolo_dataset",
        ),
        mot_root=_resolve(dataset.get("mot_root"), project_root, "dataset.mot_root"),
        splits=[str(split) for split in splits],
        include_empty_frames=bool(audit.get("include_empty_frames", True)),
        check_images=bool(audit.get("check_images", True)),
        check_labels=bool(audit.get("check_labels", True)),
        check_track_ids=bool(audit.get("check_track_ids", True)),
        check_split_leakage=bool(audit.get("check_split_leakage", True)),
        sample_sequences=sample_sequences,
        sample_frames_per_sequence=sample_frames,
        random_seed=int(audit.get("random_seed", 42)),
        small_max_ratio=small_max_ratio,
        medium_max_ratio=medium_max_ratio,
        metrics_dir=_resolve(output.get("metrics_dir"), project_root, "output.metrics_dir"),
        figures_dir=_resolve(output.get("figures_dir"), project_root, "output.figures_dir"),
        samples_dir=_resolve(output.get("samples_dir"), project_root, "output.samples_dir"),
        log_level=str(runtime.get("log_level", "INFO")),
        fail_on_error=bool(runtime.get("fail_on_error", True)),
    )


def create_dataset_audit(sequences: list[SequenceInfo]) -> dict[str, Any]:
    frame_count = sum(sequence.frame_count for sequence in sequences)
    target_boxes = []
    track_frames: dict[tuple[str, int | str], set[int]] = defaultdict(set)
    frame_with_object: set[tuple[str, int]] = set()
    ignored_count = 0
    invalid_count = 0
    clipped_count = 0
    for sequence in sequences:
        for frame in sequence.annotations:
            for annotation in frame.objects:
                if annotation.is_ignored or annotation.target_class_id is None:
                    ignored_count += 1
                    continue
                if clip_xyxy_to_image(
                    annotation.bbox_xyxy,
                    frame.width,
                    frame.height,
                ) != annotation.bbox_xyxy:
                    clipped_count += 1
                if not is_valid_bbox(annotation.bbox_xyxy):
                    invalid_count += 1
                    continue
                xywh = xyxy_to_xywh(annotation.bbox_xyxy)
                target_boxes.append(xywh)
                frame_with_object.add((sequence.name, frame.frame_index))
                track_frames[(sequence.name, annotation.track_id)].add(frame.frame_index)
    track_lengths = [len(frames) for frames in track_frames.values()]
    widths = [box.width for box in target_boxes]
    heights = [box.height for box in target_boxes]
    areas = [bbox_area_xywh.width * bbox_area_xywh.height for bbox_area_xywh in target_boxes]
    return {
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "frames_with_object": len(frame_with_object),
        "empty_frames": frame_count - len(frame_with_object),
        "player_box_count": len(target_boxes),
        "unique_tracks_by_sequence": len(track_lengths),
        "objects_per_frame_mean": len(target_boxes) / frame_count if frame_count else 0.0,
        "box_width_mean": mean(widths) if widths else 0.0,
        "box_height_mean": mean(heights) if heights else 0.0,
        "box_area_mean": mean(areas) if areas else 0.0,
        "track_length_min": min(track_lengths) if track_lengths else 0,
        "track_length_mean": mean(track_lengths) if track_lengths else 0.0,
        "track_length_median": median(track_lengths) if track_lengths else 0.0,
        "track_length_max": max(track_lengths) if track_lengths else 0,
        "single_frame_track_count": sum(length == 1 for length in track_lengths),
        "clipped_box_count": clipped_count,
        "invalid_box_count": invalid_count,
        "ignored_class_count": ignored_count,
    }


def write_dataset_audit(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (output_dir / "dataset_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in audit.items():
            writer.writerow({"metric": key, "value": value})


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    if not fieldnames:
        fieldnames = ["status"]
        records = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_errors(path: Path, errors: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(errors), indent=2), encoding="utf-8")


def _pyplot_for_output(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config = path.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_histogram(values: list[float | int], title: str, xlabel: str, path: Path) -> None:
    if not values:
        return
    plt = _pyplot_for_output(path)
    plt.figure()
    plt.hist(values, bins="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_bar(values: dict[str, int | float], title: str, ylabel: str, path: Path) -> None:
    filtered = {key: value for key, value in values.items() if value}
    if not filtered:
        return
    plt = _pyplot_for_output(path)
    plt.figure()
    plt.bar(list(filtered), list(filtered.values()))
    plt.title(title)
    plt.xlabel("category")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_dataset_audit_figures(samples: dict[str, Any], figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    figure_specs = [
        (
            "bbox_widths",
            "Bounding Box Width Distribution",
            "width (px)",
            "bbox_width_distribution.png",
        ),
        (
            "bbox_heights",
            "Bounding Box Height Distribution",
            "height (px)",
            "bbox_height_distribution.png",
        ),
        (
            "bbox_area_ratios",
            "Bounding Box Area Ratio Distribution",
            "box area / image area",
            "bbox_area_ratio_distribution.png",
        ),
        (
            "bbox_aspect_ratios",
            "Bounding Box Aspect Ratio Distribution",
            "width / height",
            "bbox_aspect_ratio_distribution.png",
        ),
        ("track_lengths", "Track Length Distribution", "frames", "track_length_distribution.png"),
        (
            "objects_per_frame",
            "Objects Per Frame Distribution",
            "objects per frame",
            "objects_per_frame_distribution.png",
        ),
    ]
    for key, title, xlabel, filename in figure_specs:
        path = figures_dir / filename
        _plot_histogram(samples.get(key, []), title, xlabel, path)
        if path.is_file():
            written.append(path)
    bar_specs = [
        ("objects_per_split", "Objects Per Split", "objects", "objects_per_split.png"),
        ("tracks_per_split", "Tracks Per Split", "tracks", "tracks_per_split.png"),
        (
            "box_size_counts",
            "Small Medium Large Boxes",
            "boxes",
            "small_medium_large_boxes.png",
        ),
        (
            "ignored_classes",
            "Ignored Classes Distribution",
            "objects",
            "ignored_classes_distribution.png",
        ),
    ]
    for key, title, ylabel, filename in bar_specs:
        path = figures_dir / filename
        _plot_bar(samples.get(key, {}), title, ylabel, path)
        if path.is_file():
            written.append(path)
    return written


def write_extended_dataset_audit(
    summary: dict[str, Any],
    per_sequence: list[dict[str, Any]],
    per_split: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    errors: dict[str, Any],
    metrics_dir: Path,
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": metrics_dir / "dataset_audit_summary.json",
        "per_sequence": metrics_dir / "dataset_audit_per_sequence.csv",
        "per_split": metrics_dir / "dataset_audit_per_split.csv",
        "tracks": metrics_dir / "dataset_audit_tracks.csv",
        "errors": metrics_dir / "dataset_audit_errors.json",
    }
    paths["summary"].write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    _write_records_csv(paths["per_sequence"], per_sequence)
    _write_records_csv(paths["per_split"], per_split)
    _write_records_csv(paths["tracks"], tracks)
    _write_errors(paths["errors"], errors)
    return paths


def run_dataset_audit(config_path: str | Path, max_sequences: int | None = None) -> dict[str, Any]:
    """Run the Milestone 3 dataset audit from an audit config."""

    from football_tracking.data.class_mapping import load_class_mapping
    from football_tracking.data.prepare import _load_sequences, load_data_config
    from football_tracking.data.split_sequences import split_sequences

    audit_config = load_audit_config(config_path)
    data_config = load_data_config(audit_config.data_config_path)
    class_mapping = load_class_mapping(data_config.class_mapping_path)
    sequences = _load_sequences(data_config, class_mapping, max_sequences=max_sequences)
    if not sequences:
        raise AuditConfigError(f"No sequences were loaded from {data_config.raw_dir}")

    split_manifest = split_sequences(
        sequences,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        seed=data_config.seed,
        strategy=data_config.split_strategy,
        predefined_split_file=data_config.predefined_split_file,
    )
    stats = compute_dataset_statistics(
        sequences,
        split_manifest=split_manifest,
        small_max_ratio=audit_config.small_max_ratio,
        medium_max_ratio=audit_config.medium_max_ratio,
        split_names=audit_config.splits,
    )
    summary = {
        "dataset_name": data_config.dataset_name,
        "dataset_path": str(data_config.raw_dir),
        "config_path": str(data_config.config_path),
        "audit_config_path": str(audit_config.config_path),
        "python_version": platform.python_version(),
        "created_at": datetime.now(UTC).isoformat(),
        "totals": stats["totals"],
        "per_split": stats["per_split"],
        "bbox_statistics": stats["bbox_statistics"],
        "track_statistics": stats["track_statistics"],
        "class_statistics": stats["class_statistics"],
        "warnings": stats["warnings"],
        "errors": stats["errors"],
        "inputs": {
            "manifest": str(audit_config.manifest_path),
            "yolo_dataset": str(audit_config.yolo_dataset_path),
            "mot_root": str(audit_config.mot_root),
            "check_images": audit_config.check_images,
            "check_labels": audit_config.check_labels,
            "check_track_ids": audit_config.check_track_ids,
            "check_split_leakage": audit_config.check_split_leakage,
        },
    }
    output_paths = write_extended_dataset_audit(
        summary=summary,
        per_sequence=stats["per_sequence"],
        per_split=stats["per_split"],
        tracks=stats["tracks"],
        errors=stats["errors"],
        metrics_dir=audit_config.metrics_dir,
    )
    figure_paths = write_dataset_audit_figures(stats["samples"], audit_config.figures_dir)
    sample_paths = draw_annotation_samples(
        sequences,
        output_dir=audit_config.samples_dir,
        split_manifest=split_manifest,
        num_sequences=audit_config.sample_sequences,
        frames_per_sequence=audit_config.sample_frames_per_sequence,
        seed=audit_config.random_seed,
        draw_ignored=True,
    )
    result = dict(summary)
    result["output_paths"] = {key: str(path) for key, path in output_paths.items()}
    result["figure_paths"] = [str(path) for path in figure_paths]
    result["sample_paths"] = [str(path) for path in sample_paths]
    return result
