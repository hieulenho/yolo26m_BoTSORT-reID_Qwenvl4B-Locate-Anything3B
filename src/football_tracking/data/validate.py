"""Validation checks for raw annotations and converted outputs."""

from __future__ import annotations

import json
import math
from configparser import ConfigParser
from pathlib import Path

from football_tracking.data.bbox import is_valid_bbox
from football_tracking.data.schemas import (
    SequenceInfo,
    SplitManifest,
    ValidationIssue,
    ValidationReport,
)


def _issue(
    severity: str,
    code: str,
    message: str,
    sequence_name: str | None = None,
    frame_index: int | None = None,
    track_id: int | str | None = None,
    path: Path | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        sequence_name=sequence_name,
        frame_index=frame_index,
        track_id=track_id,
        path=path,
    )


def validate_sequences(
    sequences: list[SequenceInfo],
    require_images: bool = True,
    invalid_box_policy: str = "warn_and_skip",
    fail_on_duplicate_track_in_frame: bool = True,
) -> ValidationReport:
    report = ValidationReport([])
    seen_sequences: set[str] = set()
    invalid_severity = "ERROR" if invalid_box_policy == "fail" else "WARNING"
    duplicate_severity = "ERROR" if fail_on_duplicate_track_in_frame else "WARNING"

    for sequence in sequences:
        if not sequence.name:
            report.issues.append(_issue("ERROR", "sequence_name_empty", "Sequence name is empty."))
        if sequence.name in seen_sequences:
            report.issues.append(
                _issue(
                    "ERROR",
                    "duplicate_sequence_name",
                    "Duplicate sequence name.",
                    sequence.name,
                )
            )
        seen_sequences.add(sequence.name)
        if sequence.frame_count <= 0:
            report.issues.append(
                _issue("ERROR", "frame_count", "frame_count must be > 0.", sequence.name)
            )
        if sequence.width <= 0 or sequence.height <= 0:
            report.issues.append(
                _issue(
                    "ERROR",
                    "resolution",
                    "width and height must be > 0.",
                    sequence.name,
                )
            )
        if sequence.fps <= 0:
            report.issues.append(_issue("ERROR", "fps", "fps must be > 0.", sequence.name))

        seen_frame_track: set[tuple[int, int | str]] = set()
        for frame in sequence.annotations:
            if frame.frame_index < 1 or frame.frame_index > sequence.frame_count:
                report.issues.append(
                    _issue(
                        "ERROR",
                        "frame_index",
                        "Frame index is outside sequence length.",
                        sequence.name,
                        frame.frame_index,
                    )
                )
            if require_images and not frame.image_path.is_file():
                report.issues.append(
                    _issue(
                        "ERROR",
                        "image_missing",
                        "Frame image does not exist.",
                        sequence.name,
                        frame.frame_index,
                        path=frame.image_path,
                    )
                )
            if frame.width != sequence.width or frame.height != sequence.height:
                report.issues.append(
                    _issue(
                        "WARNING",
                        "frame_resolution",
                        "Frame resolution differs from sequence resolution.",
                        sequence.name,
                        frame.frame_index,
                    )
                )
            for annotation in frame.objects:
                key = (frame.frame_index, annotation.track_id)
                if key in seen_frame_track:
                    report.issues.append(
                        _issue(
                            duplicate_severity,
                            "duplicate_frame_track",
                            "Duplicate sequence/frame/track annotation.",
                            sequence.name,
                            frame.frame_index,
                            annotation.track_id,
                        )
                    )
                seen_frame_track.add(key)
                if not is_valid_bbox(annotation.bbox_xyxy):
                    report.issues.append(
                        _issue(
                            invalid_severity,
                            "invalid_bbox",
                            "Bounding box is invalid.",
                            sequence.name,
                            frame.frame_index,
                            annotation.track_id,
                        )
                    )
                if annotation.target_class_id is None and not annotation.is_ignored:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "class_unmapped",
                            "Object is neither mapped nor ignored.",
                            sequence.name,
                            frame.frame_index,
                            annotation.track_id,
                        )
                    )
                if isinstance(annotation.track_id, int) and annotation.track_id <= 0:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "track_id",
                            "Track ID must be positive.",
                            sequence.name,
                            frame.frame_index,
                            annotation.track_id,
                        )
                    )
                if isinstance(annotation.track_id, str) and not annotation.track_id.strip():
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "track_id",
                            "Track ID string must not be empty.",
                            sequence.name,
                            frame.frame_index,
                            annotation.track_id,
                        )
                    )
                if not 0.0 <= annotation.visibility <= 1.0:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "visibility",
                            "Visibility must be in [0, 1].",
                            sequence.name,
                        )
                    )
                if not 0.0 <= annotation.confidence <= 1.0:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "confidence",
                            "Confidence must be in [0, 1].",
                            sequence.name,
                        )
                    )
    return report


def validate_split_leakage(split_manifest: SplitManifest) -> ValidationReport:
    report = ValidationReport([])
    splits = split_manifest.as_mapping()
    for left_name, left_values in splits.items():
        for right_name, right_values in splits.items():
            if left_name >= right_name:
                continue
            overlap = set(left_values) & set(right_values)
            if overlap:
                report.issues.append(
                    _issue(
                        "ERROR",
                        "split_leakage",
                        f"Sequences appear in both {left_name} and {right_name}: {sorted(overlap)}",
                    )
                )
    return report


def validate_yolo_dataset(path: Path) -> ValidationReport:
    report = ValidationReport([])
    label_root = path / "labels"
    image_root = path / "images"
    for split_name in ("train", "val", "test"):
        for image_path in (image_root / split_name).glob("*"):
            label_path = label_root / split_name / f"{image_path.stem}.txt"
            if not label_path.is_file():
                report.issues.append(
                    _issue("ERROR", "yolo_label_missing", "Missing YOLO label.", path=label_path)
                )
        for label_path in (label_root / split_name).glob("*.txt"):
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "yolo_fields",
                            f"YOLO line {line_number} must have 5 fields.",
                            path=label_path,
                        )
                    )
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "yolo_numeric",
                            "YOLO fields must be numeric.",
                            path=label_path,
                        )
                    )
                    continue
                if class_id != 0 or any(value < 0.0 or value > 1.0 for value in values):
                    report.issues.append(
                        _issue(
                            "ERROR",
                            "yolo_range",
                            "YOLO values are outside expected range.",
                            path=label_path,
                        )
                    )
    return report


def validate_mot_dataset(path: Path) -> ValidationReport:
    report = ValidationReport([])
    for gt_path in path.glob("*/*/gt/gt.txt"):
        seqinfo_path = gt_path.parent.parent / "seqinfo.ini"
        seq_length = None
        if seqinfo_path.is_file():
            parser = ConfigParser()
            parser.read(seqinfo_path, encoding="utf-8")
            seq_length = parser.getint("Sequence", "seqLength", fallback=None)
        lines = gt_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            parts = line.split(",")
            if len(parts) != 9:
                report.issues.append(
                    _issue(
                        "ERROR",
                        "mot_fields",
                        f"MOT line {line_number} must have 9 fields.",
                        path=gt_path,
                    )
                )
                continue
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                left, top, width, height, confidence, _klass, visibility = [
                    float(value) for value in parts[2:]
                ]
            except ValueError:
                report.issues.append(
                    _issue("ERROR", "mot_numeric", "MOT fields must be numeric.", path=gt_path)
                )
                continue
            values = (left, top, width, height, confidence, visibility)
            if any(not math.isfinite(value) for value in values):
                report.issues.append(
                    _issue("ERROR", "mot_nan", "MOT values must be finite.", path=gt_path)
                )
            if frame < 1 or track_id <= 0 or width <= 0 or height <= 0:
                report.issues.append(
                    _issue(
                        "ERROR",
                        "mot_range",
                        "MOT frame, track, width, or height is invalid.",
                        path=gt_path,
                    )
                )
            if seq_length is not None and frame > seq_length:
                report.issues.append(
                    _issue(
                        "ERROR",
                        "mot_frame_length",
                        "MOT frame exceeds seqLength.",
                        path=gt_path,
                    )
                )
    return report


def write_validation_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
