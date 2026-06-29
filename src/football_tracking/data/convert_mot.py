"""Convert internal annotations to MOTChallenge format."""

from __future__ import annotations

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox, xyxy_to_xywh
from football_tracking.data.convert_yolo import (
    ConversionError,
    _ensure_output_root,
    _link_or_copy_image,
)
from football_tracking.data.schemas import ObjectAnnotation, SequenceInfo, SplitManifest

LOGGER = logging.getLogger(__name__)


def _mot_track_id(track_id: int | str, mapping: dict[int | str, int]) -> int:
    if isinstance(track_id, int) and track_id > 0:
        return track_id
    try:
        value = int(str(track_id))
    except ValueError:
        if track_id not in mapping:
            mapping[track_id] = len(mapping) + 1
        return mapping[track_id]
    if value <= 0:
        if track_id not in mapping:
            mapping[track_id] = len(mapping) + 1
        return mapping[track_id]
    return value


def _eligible_mot_objects(
    sequence: SequenceInfo,
    clip_boxes: bool,
    confidence_default: float,
    visibility_default: float,
) -> list[tuple[int, int, ObjectAnnotation]]:
    rows: list[tuple[int, int, ObjectAnnotation]] = []
    track_mapping: dict[int | str, int] = {}
    seen: set[tuple[int, int]] = set()
    for frame in sequence.annotations:
        mot_frame_index = frame.frame_index if frame.frame_index >= 1 else frame.frame_index + 1
        for annotation in frame.objects:
            if annotation.is_ignored or annotation.target_class_id is None:
                continue
            box = (
                clip_xyxy_to_image(annotation.bbox_xyxy, frame.width, frame.height)
                if clip_boxes
                else annotation.bbox_xyxy
            )
            if not is_valid_bbox(box):
                continue
            mot_track_id = _mot_track_id(annotation.track_id, track_mapping)
            key = (mot_frame_index, mot_track_id)
            if key in seen:
                LOGGER.warning(
                    "Skipping duplicate MOT annotation in %s frame=%s track=%s",
                    sequence.name,
                    mot_frame_index,
                    mot_track_id,
                )
                continue
            seen.add(key)
            rows.append(
                (
                    mot_frame_index,
                    mot_track_id,
                    ObjectAnnotation(
                        frame_index=annotation.frame_index,
                        track_id=mot_track_id,
                        source_class=annotation.source_class,
                        target_class=annotation.target_class,
                        target_class_id=annotation.target_class_id,
                        bbox_xyxy=box,
                        confidence=annotation.confidence or confidence_default,
                        visibility=max(0.0, min(1.0, annotation.visibility or visibility_default)),
                        is_ignored=annotation.is_ignored,
                        metadata=annotation.metadata,
                    ),
                )
            )
    return sorted(rows, key=lambda row: (row[0], row[1]))


def _write_seqinfo(sequence: SequenceInfo, sequence_dir: Path, image_extension: str) -> None:
    parser = ConfigParser()
    parser.optionxform = str
    parser["Sequence"] = {
        "name": sequence.name,
        "imDir": "img1",
        "frameRate": str(sequence.fps),
        "seqLength": str(sequence.frame_count),
        "imWidth": str(sequence.width),
        "imHeight": str(sequence.height),
        "imExt": image_extension,
    }
    with (sequence_dir / "seqinfo.ini").open("w", encoding="utf-8") as handle:
        parser.write(handle, space_around_delimiters=False)


def convert_to_mot(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    output_dir: Path,
    image_extension: str,
    frame_index_base: int = 1,
    confidence_default: float = 1.0,
    visibility_default: float = 1.0,
    clip_boxes: bool = True,
    prefer_symlink: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if frame_index_base != 1:
        raise ConversionError("MOT frame_index_base must be 1.")
    _ensure_output_root(output_dir, overwrite=overwrite, dry_run=dry_run)
    stats: dict[str, Any] = {"gt_rows": 0, "sequences": 0, "dry_run": dry_run}

    seqmaps: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sequence in sequences:
        split_name = split_manifest.split_for_sequence(sequence.name)
        if split_name is None:
            continue
        seqmaps[split_name].append(sequence.name)
        sequence_dir = output_dir / split_name / sequence.name
        if not dry_run:
            (sequence_dir / "img1").mkdir(parents=True, exist_ok=True)
            (sequence_dir / "gt").mkdir(parents=True, exist_ok=True)
        rows = _eligible_mot_objects(
            sequence,
            clip_boxes=clip_boxes,
            confidence_default=confidence_default,
            visibility_default=visibility_default,
        )
        stats["gt_rows"] += len(rows)
        stats["sequences"] += 1
        if dry_run:
            continue

        for frame in sequence.annotations:
            if frame.image_path.is_file():
                destination = (
                    sequence_dir / "img1" / f"{frame.frame_index:06d}{frame.image_path.suffix}"
                )
                _link_or_copy_image(
                    frame.image_path,
                    destination,
                    prefer_symlink=prefer_symlink,
                    copy_images=False,
                )
        lines: list[str] = []
        for frame_index, track_id, annotation in rows:
            xywh = xyxy_to_xywh(annotation.bbox_xyxy)
            lines.append(
                f"{frame_index},{track_id},{xywh.x:.2f},{xywh.y:.2f},"
                f"{xywh.width:.2f},{xywh.height:.2f},{confidence_default:g},1,"
                f"{annotation.visibility:.2f}"
            )
        (sequence_dir / "gt" / "gt.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        _write_seqinfo(sequence, sequence_dir, image_extension)

    if not dry_run:
        seqmap_dir = output_dir / "seqmaps"
        seqmap_dir.mkdir(parents=True, exist_ok=True)
        for split_name, names in seqmaps.items():
            content = "name\n" + "\n".join(sorted(names)) + ("\n" if names else "")
            (seqmap_dir / f"{split_name}.txt").write_text(content, encoding="utf-8")
    stats["seqmaps"] = {split: sorted(names) for split, names in seqmaps.items()}
    return stats
