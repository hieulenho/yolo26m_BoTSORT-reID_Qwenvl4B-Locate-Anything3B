"""Validation for MOT tracking prediction outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.tracking.sequence_runner import read_seqinfo


@dataclass(frozen=True)
class TrackValidationIssue:
    severity: str
    code: str
    message: str
    path: Path | None = None
    line_number: int | None = None


@dataclass
class TrackValidationReport:
    issues: list[TrackValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "WARNING" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def extend(self, other: TrackValidationReport) -> None:
        self.issues.extend(other.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {"errors": self.error_count, "warnings": self.warning_count},
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "path": str(issue.path) if issue.path else None,
                    "line_number": issue.line_number,
                }
                for issue in self.issues
            ],
        }


def validate_mot_prediction_file(
    mot_path: Path,
    seqinfo_path: Path | None = None,
    metadata_path: Path | None = None,
) -> TrackValidationReport:
    report = TrackValidationReport()
    if not mot_path.is_file():
        report.issues.append(
            TrackValidationIssue(
                "ERROR",
                "missing_mot",
                f"Missing MOT output: {mot_path}",
                mot_path,
            )
        )
        return report
    if metadata_path is not None and not metadata_path.is_file():
        report.issues.append(
            TrackValidationIssue(
                "ERROR",
                "missing_metadata",
                f"Missing metadata output: {metadata_path}",
                metadata_path,
            )
        )
    seq_length = None
    if seqinfo_path is not None and seqinfo_path.is_file():
        seq_length = int(read_seqinfo(seqinfo_path)["seqLength"])
    seen: set[tuple[int, int]] = set()
    previous_key: tuple[int, int] | None = None
    for line_number, line in enumerate(mot_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 9:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "field_count",
                    "MOT prediction row must contain 9 fields.",
                    mot_path,
                    line_number,
                )
            )
            continue
        try:
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            left, top, width, height, confidence, mot_class, visibility = [
                float(value) for value in fields[2:]
            ]
        except ValueError:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "parse_error",
                    "MOT prediction row contains non-numeric fields.",
                    mot_path,
                    line_number,
                )
            )
            continue
        values = (left, top, width, height, confidence, mot_class, visibility)
        if any(not math.isfinite(value) for value in values):
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "non_finite",
                    "MOT prediction row contains NaN or infinity.",
                    mot_path,
                    line_number,
                )
            )
        if frame < 1 or track_id <= 0 or width <= 0 or height <= 0:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "invalid_geometry",
                    "Frame, track id, width, and height must be positive.",
                    mot_path,
                    line_number,
                )
            )
        if seq_length is not None and frame > seq_length:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "frame_exceeds_sequence",
                    f"Frame {frame} exceeds seqLength {seq_length}.",
                    mot_path,
                    line_number,
                )
            )
        key = (frame, track_id)
        if key in seen:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "duplicate_frame_track",
                    f"Duplicate frame-track pair: {key}",
                    mot_path,
                    line_number,
                )
            )
        seen.add(key)
        if previous_key is not None and key < previous_key:
            report.issues.append(
                TrackValidationIssue(
                    "ERROR",
                    "not_sorted",
                    "MOT rows must be sorted by frame then track id.",
                    mot_path,
                    line_number,
                )
            )
        previous_key = key
    return report


def write_track_validation_report(report: TrackValidationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
