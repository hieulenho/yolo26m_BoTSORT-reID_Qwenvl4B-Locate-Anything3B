"""Basic dataset audit statistics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from football_tracking.data.bbox import is_valid_bbox, xyxy_to_xywh
from football_tracking.data.schemas import SequenceInfo


def create_dataset_audit(sequences: list[SequenceInfo]) -> dict[str, Any]:
    frame_count = sum(sequence.frame_count for sequence in sequences)
    target_boxes = []
    track_frames: dict[tuple[str, int | str], set[int]] = defaultdict(set)
    frame_with_object: set[tuple[str, int]] = set()
    ignored_count = 0
    invalid_count = 0
    for sequence in sequences:
        for frame in sequence.annotations:
            for annotation in frame.objects:
                if annotation.is_ignored or annotation.target_class_id is None:
                    ignored_count += 1
                    continue
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
        "clipped_box_count": sum(
            sequence.metadata.get("clipped_box_count", 0) for sequence in sequences
        ),
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
