"""Dataset statistics for audit reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from football_tracking.data.bbox import bbox_area, clip_xyxy_to_image, is_valid_bbox
from football_tracking.data.class_mapping import normalize_class_name
from football_tracking.data.schemas import BoundingBoxXYXY, SequenceInfo, SplitManifest

DEFAULT_SMALL_MAX_RATIO = 0.01
DEFAULT_MEDIUM_MAX_RATIO = 0.05


def json_safe(value: Any) -> Any:
    """Return a JSON-compatible value without NaN or infinity."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def numeric_stats(values: list[float], reason: str) -> dict[str, float | str | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None, "reason": reason}
    return {
        "min": min(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
        "reason": None,
    }


def _touches_image_boundary(box: BoundingBoxXYXY, width: int, height: int) -> bool:
    return box.x1 <= 0 or box.y1 <= 0 or box.x2 >= width or box.y2 >= height


def _split_name(split_manifest: SplitManifest | None, sequence_name: str) -> str:
    if split_manifest is None:
        return "unknown"
    return split_manifest.split_for_sequence(sequence_name) or "unknown"


def _empty_split_records(splits: list[str]) -> dict[str, dict[str, Any]]:
    return {
        split: {
            "split": split,
            "sequence_count": 0,
            "frame_count": 0,
            "annotated_frame_count": 0,
            "empty_frame_count": 0,
            "object_count": 0,
            "track_count": 0,
        }
        for split in splits
    }


def _split_leakage(split_manifest: SplitManifest | None) -> list[dict[str, Any]]:
    if split_manifest is None:
        return []
    leaks: list[dict[str, Any]] = []
    splits = split_manifest.as_mapping()
    split_names = list(splits)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = sorted(set(splits[left_name]) & set(splits[right_name]))
            if overlap:
                leaks.append({"left": left_name, "right": right_name, "sequences": overlap})
    return leaks


def compute_dataset_statistics(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest | None = None,
    small_max_ratio: float = DEFAULT_SMALL_MAX_RATIO,
    medium_max_ratio: float = DEFAULT_MEDIUM_MAX_RATIO,
    split_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute audit statistics without writing files."""

    splits = split_names or ["train", "val", "test"]
    per_split = _empty_split_records([*splits, "unknown"] if "unknown" not in splits else splits)
    per_sequence: list[dict[str, Any]] = []
    track_records: list[dict[str, Any]] = []
    errors: dict[str, Any] = {
        "invalid_boxes": [],
        "duplicate_frame_tracks": [],
        "split_leakage": _split_leakage(split_manifest),
        "missing_images": [],
    }
    warnings: list[str] = []

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    area_ratios: list[float] = []
    aspect_ratios: list[float] = []
    track_lengths: list[int] = []
    objects_per_frame: list[int] = []
    target_boxes = 0
    invalid_box_count = 0
    clipped_box_count = 0
    boundary_touch_count = 0
    small_count = 0
    medium_count = 0
    large_count = 0
    frames_with_objects: set[tuple[str, int]] = set()
    track_frames: dict[tuple[str, int | str], set[int]] = defaultdict(set)
    source_classes: Counter[str] = Counter()
    target_classes: Counter[str] = Counter()
    ignored_classes: Counter[str] = Counter()
    unknown_class_count = 0
    goalkeeper_mapped_count = 0
    referee_ignored_count = 0

    for sequence in sequences:
        split = _split_name(split_manifest, sequence.name)
        if split not in per_split:
            per_split[split] = {
                "split": split,
                "sequence_count": 0,
                "frame_count": 0,
                "annotated_frame_count": 0,
                "empty_frame_count": 0,
                "object_count": 0,
                "track_count": 0,
            }
        per_split[split]["sequence_count"] += 1
        per_split[split]["frame_count"] += sequence.frame_count

        sequence_frame_object_counts: dict[int, int] = {
            frame_index: 0 for frame_index in range(1, sequence.frame_count + 1)
        }
        sequence_tracks: dict[int | str, set[int]] = defaultdict(set)
        sequence_invalid = 0
        sequence_clipped = 0
        sequence_ignored = 0
        sequence_unknown = 0
        sequence_boundary_touch = 0
        seen_frame_track: set[tuple[int, int | str]] = set()
        duplicate_count = 0

        for frame in sequence.annotations:
            if not frame.image_path.is_file():
                errors["missing_images"].append(
                    {
                        "sequence_name": sequence.name,
                        "frame_index": frame.frame_index,
                        "path": str(frame.image_path),
                    }
                )
            sequence_frame_object_counts.setdefault(frame.frame_index, 0)
            for annotation in frame.objects:
                normalized_source = normalize_class_name(annotation.source_class)
                source_classes[normalized_source] += 1
                if annotation.target_class:
                    target_classes[annotation.target_class] += 1
                if annotation.is_ignored or annotation.target_class_id is None:
                    ignored_classes[normalized_source] += 1
                    sequence_ignored += 1
                    if "referee" in normalized_source:
                        referee_ignored_count += 1
                    if annotation.metadata.get("unknown_class"):
                        unknown_class_count += 1
                        sequence_unknown += 1
                    continue
                if "goalkeeper" in normalized_source and annotation.target_class == "player":
                    goalkeeper_mapped_count += 1

                key = (frame.frame_index, annotation.track_id)
                if key in seen_frame_track:
                    duplicate_count += 1
                    errors["duplicate_frame_tracks"].append(
                        {
                            "sequence_name": sequence.name,
                            "frame_index": frame.frame_index,
                            "track_id": annotation.track_id,
                        }
                    )
                    continue
                seen_frame_track.add(key)

                box = annotation.bbox_xyxy
                clipped = clip_xyxy_to_image(box, frame.width, frame.height)
                if clipped != box:
                    clipped_box_count += 1
                    sequence_clipped += 1
                if not is_valid_bbox(clipped):
                    invalid_box_count += 1
                    sequence_invalid += 1
                    errors["invalid_boxes"].append(
                        {
                            "sequence_name": sequence.name,
                            "frame_index": frame.frame_index,
                            "track_id": annotation.track_id,
                            "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                        }
                    )
                    continue
                if _touches_image_boundary(clipped, frame.width, frame.height):
                    boundary_touch_count += 1
                    sequence_boundary_touch += 1

                width = clipped.x2 - clipped.x1
                height = clipped.y2 - clipped.y1
                area = bbox_area(clipped)
                image_area = frame.width * frame.height
                area_ratio = area / image_area if image_area > 0 else 0.0
                widths.append(width)
                heights.append(height)
                areas.append(area)
                area_ratios.append(area_ratio)
                aspect_ratios.append(width / height)
                if area_ratio <= small_max_ratio:
                    small_count += 1
                elif area_ratio <= medium_max_ratio:
                    medium_count += 1
                else:
                    large_count += 1

                target_boxes += 1
                frames_with_objects.add((sequence.name, frame.frame_index))
                sequence_frame_object_counts[frame.frame_index] += 1
                track_frames[(sequence.name, annotation.track_id)].add(frame.frame_index)
                sequence_tracks[annotation.track_id].add(frame.frame_index)

        annotated_frame_count = sum(count > 0 for count in sequence_frame_object_counts.values())
        sequence_object_count = sum(sequence_frame_object_counts.values())
        sequence_track_lengths = [len(frames) for frames in sequence_tracks.values()]
        per_split[split]["annotated_frame_count"] += annotated_frame_count
        per_split[split]["empty_frame_count"] += sequence.frame_count - annotated_frame_count
        per_split[split]["object_count"] += sequence_object_count
        per_split[split]["track_count"] += len(sequence_tracks)
        objects_per_frame.extend(sequence_frame_object_counts.values())
        per_sequence.append(
            {
                "sequence_name": sequence.name,
                "split": split,
                "frame_count": sequence.frame_count,
                "annotated_frame_count": annotated_frame_count,
                "empty_frame_count": sequence.frame_count - annotated_frame_count,
                "object_count": sequence_object_count,
                "unique_track_count": len(sequence_tracks),
                "min_track_length": min(sequence_track_lengths) if sequence_track_lengths else None,
                "max_track_length": max(sequence_track_lengths) if sequence_track_lengths else None,
                "ignored_object_count": sequence_ignored,
                "unknown_class_count": sequence_unknown,
                "clipped_box_count": sequence_clipped,
                "invalid_box_count": sequence_invalid,
                "boundary_touch_box_count": sequence_boundary_touch,
                "duplicate_frame_track_count": duplicate_count,
            }
        )

    single_frame_tracks = 0
    tracks_with_gap = 0
    continuous_tracks = 0
    total_frame_gaps = 0
    for (sequence_name, track_id), frames in sorted(track_frames.items()):
        sorted_frames = sorted(frames)
        gaps = [
            right - left - 1
            for left, right in zip(sorted_frames, sorted_frames[1:], strict=False)
            if right - left > 1
        ]
        length = len(sorted_frames)
        track_lengths.append(length)
        single_frame_tracks += length == 1
        tracks_with_gap += bool(gaps)
        continuous_tracks += not gaps
        total_frame_gaps += sum(gaps)
        track_records.append(
            {
                "sequence_name": sequence_name,
                "track_id": track_id,
                "length": length,
                "first_frame": sorted_frames[0],
                "last_frame": sorted_frames[-1],
                "frame_gap_count": sum(gaps),
                "has_frame_gap": bool(gaps),
                "is_continuous": not gaps,
            }
        )

    frame_count = sum(sequence.frame_count for sequence in sequences)
    max_objects_per_frame = max(objects_per_frame) if objects_per_frame else None
    objects_per_frame_mean = mean(objects_per_frame) if objects_per_frame else None
    if frame_count == 0:
        warnings.append("No frames were available for object-per-frame statistics.")
    if not target_boxes:
        warnings.append("No valid target player boxes were available for bbox statistics.")
    if not track_lengths:
        warnings.append("No valid target tracks were available for track statistics.")

    totals = {
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "frames_with_annotation": len(frames_with_objects),
        "empty_frames": frame_count - len(frames_with_objects),
        "player_box_count": target_boxes,
        "unique_tracks_by_sequence": len(track_frames),
        "unique_tracks_global_sequence_track_id": len(track_frames),
        "objects_per_frame_mean": objects_per_frame_mean,
        "max_objects_per_frame": max_objects_per_frame,
    }
    bbox_statistics = {
        "width": numeric_stats(widths, "no valid target boxes"),
        "height": numeric_stats(heights, "no valid target boxes"),
        "area": numeric_stats(areas, "no valid target boxes"),
        "area_ratio": numeric_stats(area_ratios, "no valid target boxes"),
        "aspect_ratio": numeric_stats(aspect_ratios, "no valid target boxes"),
        "small_object_count": small_count,
        "medium_object_count": medium_count,
        "large_object_count": large_count,
        "clipped_box_count": clipped_box_count,
        "invalid_box_count": invalid_box_count,
        "boundary_touch_box_count": boundary_touch_count,
        "box_size_bins": {
            "small_max_ratio": small_max_ratio,
            "medium_max_ratio": medium_max_ratio,
        },
    }
    track_statistics = {
        "track_length": numeric_stats(
            [float(length) for length in track_lengths],
            "no valid target tracks",
        ),
        "single_frame_track_count": single_frame_tracks,
        "total_frame_gap_count": total_frame_gaps,
        "tracks_with_frame_gap": tracks_with_gap,
        "continuous_track_count": continuous_tracks,
        "duplicate_frame_track_count": len(errors["duplicate_frame_tracks"]),
    }
    class_statistics = {
        "source_class_distribution": dict(sorted(source_classes.items())),
        "target_class_distribution": dict(sorted(target_classes.items())),
        "ignored_class_distribution": dict(sorted(ignored_classes.items())),
        "ignored_class_count": sum(ignored_classes.values()),
        "unknown_class_count": unknown_class_count,
        "goalkeeper_mapped_to_player_count": goalkeeper_mapped_count,
        "referee_ignored_count": referee_ignored_count,
    }
    samples = {
        "bbox_widths": widths,
        "bbox_heights": heights,
        "bbox_areas": areas,
        "bbox_area_ratios": area_ratios,
        "bbox_aspect_ratios": aspect_ratios,
        "track_lengths": track_lengths,
        "objects_per_frame": objects_per_frame,
        "objects_per_split": {
            split_name: record["object_count"] for split_name, record in per_split.items()
        },
        "tracks_per_split": {
            split_name: record["track_count"] for split_name, record in per_split.items()
        },
        "box_size_counts": {"small": small_count, "medium": medium_count, "large": large_count},
        "ignored_classes": dict(sorted(ignored_classes.items())),
    }

    return json_safe(
        {
            "totals": totals,
            "per_sequence": per_sequence,
            "per_split": list(per_split.values()),
            "tracks": track_records,
            "bbox_statistics": bbox_statistics,
            "track_statistics": track_statistics,
            "class_statistics": class_statistics,
            "warnings": warnings,
            "errors": errors,
            "samples": samples,
        }
    )
