"""Dataset manifest generation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import DatasetManifestEntry, SequenceInfo, SplitManifest


def _target_objects(sequence: SequenceInfo) -> list[tuple[int, int | str]]:
    objects: list[tuple[int, int | str]] = []
    for frame in sequence.annotations:
        for annotation in frame.objects:
            if annotation.is_ignored or annotation.target_class_id is None:
                continue
            if not is_valid_bbox(annotation.bbox_xyxy):
                continue
            objects.append((frame.frame_index, annotation.track_id))
    return objects


def _track_lengths(sequence: SequenceInfo) -> list[int]:
    frames_by_track: dict[int | str, set[int]] = defaultdict(set)
    for frame_index, track_id in _target_objects(sequence):
        frames_by_track[track_id].add(frame_index)
    return [len(frames) for frames in frames_by_track.values()]


def _sequence_record(
    sequence: SequenceInfo,
    split: str,
    yolo_output_dir: Path,
    mot_output_dir: Path,
) -> dict[str, Any]:
    target_objects = _target_objects(sequence)
    track_lengths = _track_lengths(sequence)
    annotated_frames = {frame_index for frame_index, _track_id in target_objects}
    ignored_count = sum(
        annotation.is_ignored
        for frame in sequence.annotations
        for annotation in frame.objects
    )
    unknown_count = sum(
        bool(annotation.metadata.get("unknown_class"))
        for frame in sequence.annotations
        for annotation in frame.objects
    )
    invalid_count = sum(
        not is_valid_bbox(annotation.bbox_xyxy)
        for frame in sequence.annotations
        for annotation in frame.objects
    )
    return {
        "sequence_name": sequence.name,
        "split": split,
        "source_path": str(sequence.source_path),
        "annotation_path": str(sequence.annotations_path),
        "video_path": sequence.metadata.get("video_path"),
        "frames_path": str(sequence.frames_dir),
        "frame_count": sequence.frame_count,
        "annotated_frame_count": len(annotated_frames),
        "empty_frame_count": sequence.frame_count - len(annotated_frames),
        "width": sequence.width,
        "height": sequence.height,
        "fps": sequence.fps,
        "object_count": len(target_objects),
        "unique_track_count": len({track_id for _frame, track_id in target_objects}),
        "min_track_length": min(track_lengths) if track_lengths else 0,
        "max_track_length": max(track_lengths) if track_lengths else 0,
        "mean_track_length": mean(track_lengths) if track_lengths else 0.0,
        "ignored_object_count": ignored_count,
        "clipped_box_count": sum(sequence.metadata.get("clipped_box_count", 0) for _ in [0]),
        "invalid_box_count": invalid_count,
        "unknown_class_count": unknown_count,
        "output_yolo_path": str(yolo_output_dir),
        "output_mot_path": str(mot_output_dir / split / sequence.name),
    }


def build_manifest_entries(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    yolo_output_dir: Path,
    mot_output_dir: Path,
) -> list[DatasetManifestEntry]:
    entries: list[DatasetManifestEntry] = []
    for sequence in sequences:
        split = split_manifest.split_for_sequence(sequence.name) or "unknown"
        record = _sequence_record(sequence, split, yolo_output_dir, mot_output_dir)
        entries.append(
            DatasetManifestEntry(
                sequence_name=sequence.name,
                split=split,
                frame_count=sequence.frame_count,
                annotated_frame_count=record["annotated_frame_count"],
                width=sequence.width,
                height=sequence.height,
                fps=sequence.fps,
                object_count=record["object_count"],
                unique_track_count=record["unique_track_count"],
                source_path=sequence.source_path,
                output_yolo_path=yolo_output_dir,
                output_mot_path=mot_output_dir / split / sequence.name,
            )
        )
    return entries


def write_dataset_manifest(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    output_dir: Path,
    dataset_name: str,
    adapter: str,
    seed: int,
    config_path: Path,
    class_mapping_path: Path,
    yolo_output_dir: Path,
    mot_output_dir: Path,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _sequence_record(
            sequence,
            split_manifest.split_for_sequence(sequence.name) or "unknown",
            yolo_output_dir,
            mot_output_dir,
        )
        for sequence in sequences
    ]
    split_counts = Counter(record["split"] for record in records)
    payload = {
        "dataset_name": dataset_name,
        "adapter": adapter,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "config_path": str(config_path),
        "class_mapping": str(class_mapping_path),
        "split_counts": dict(split_counts),
        "total_frames": sum(record["frame_count"] for record in records),
        "total_objects": sum(record["object_count"] for record in records),
        "total_unique_tracks_by_sequence": sum(record["unique_track_count"] for record in records),
        "warnings": warnings or [],
        "errors": errors or [],
        "sequences": records,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if records:
        csv_path = output_dir / "dataset_manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    return payload
