"""Validation and compatibility checks for detection caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.detection.cache_reader import DetectionCacheReader, DetectionCacheReaderError
from football_tracking.detection.cache_schema import SCHEMA_VERSION, CachedFrameDetections
from football_tracking.detection.cache_writer import sequence_cache_dir
from football_tracking.detection.serialization import file_sha256
from football_tracking.tracking.sequence_runner import SequenceSource


@dataclass(frozen=True)
class DetectionCacheValidationIssue:
    severity: str
    code: str
    message: str
    path: Path | None = None
    frame_index: int | None = None


@dataclass
class DetectionCacheValidationReport:
    issues: list[DetectionCacheValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "WARNING" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        path: Path | None = None,
        frame_index: int | None = None,
    ) -> None:
        self.issues.append(
            DetectionCacheValidationIssue(severity, code, message, path, frame_index)
        )

    def extend(self, other: DetectionCacheValidationReport) -> None:
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
                    "frame_index": issue.frame_index,
                }
                for issue in self.issues
            ],
        }


def _check_frame_geometry(
    frame: CachedFrameDetections,
    report: DetectionCacheValidationReport,
    path: Path,
) -> None:
    seen_boxes: set[tuple[float, float, float, float, float]] = set()
    for detection in frame.detections:
        box = detection.bbox_xyxy
        if (
            box.x1 < 0.0
            or box.y1 < 0.0
            or box.x2 > frame.image_width
            or box.y2 > frame.image_height
        ):
            report.add(
                "ERROR",
                "bbox_outside_image",
                "Detection bbox must be inside image bounds.",
                path,
                frame.frame_index,
            )
        key = (box.x1, box.y1, box.x2, box.y2, detection.confidence)
        if key in seen_boxes:
            report.add(
                "WARNING",
                "duplicate_detection",
                "Duplicate detection row in the same frame.",
                path,
                frame.frame_index,
            )
        seen_boxes.add(key)


def validate_detection_cache_sequence(
    cache_dir: Path,
    expected_frame_count: int | None = None,
    expected_max_frame: int | None = None,
    require_complete: bool = True,
    allow_partial: bool = False,
    confidence_threshold: float | None = None,
    expected_checkpoint_hash: str | None = None,
) -> DetectionCacheValidationReport:
    report = DetectionCacheValidationReport()
    reader = DetectionCacheReader(cache_dir)
    try:
        metadata = reader.read_metadata()
        frames = reader.read_frames()
    except DetectionCacheReaderError as exc:
        report.add("ERROR", "cache_read_error", str(exc), cache_dir)
        return report
    except Exception as exc:  # noqa: BLE001
        report.add("ERROR", "cache_schema_error", str(exc), cache_dir)
        return report

    if metadata.schema_version != SCHEMA_VERSION:
        report.add("ERROR", "schema_version", "Unsupported cache schema version.", cache_dir)
    if metadata.file_hash != file_sha256(reader.detections_path):
        report.add(
            "ERROR",
            "file_hash_mismatch",
            "Cache file hash does not match metadata.",
            cache_dir,
        )
    if expected_checkpoint_hash and metadata.checkpoint_hash != expected_checkpoint_hash:
        report.add(
            "ERROR",
            "checkpoint_hash_mismatch",
            "Cache checkpoint hash mismatch.",
            cache_dir,
        )
    if confidence_threshold is not None and metadata.confidence_floor > confidence_threshold:
        report.add(
            "ERROR",
            "confidence_floor_too_high",
            "Cache confidence_floor is higher than experiment confidence threshold.",
            cache_dir,
        )
    if require_complete and not metadata.complete and not allow_partial:
        report.add(
            "ERROR",
            "partial_cache",
            "Cache is partial but full cache is required.",
            cache_dir,
        )

    frame_indices = [frame.frame_index for frame in frames]
    if len(frame_indices) != len(set(frame_indices)):
        report.add("ERROR", "duplicate_frame", "Cache contains duplicate frame indices.", cache_dir)
    if frame_indices != sorted(frame_indices):
        report.add(
            "ERROR",
            "unsorted_frames",
            "Cache frames must be sorted by frame_index.",
            cache_dir,
        )

    required_count = expected_max_frame or expected_frame_count or metadata.expected_frame_count
    if required_count is not None:
        required = set(range(1, int(required_count) + 1))
        missing = sorted(required.difference(frame_indices))
        if missing:
            report.add(
                "ERROR",
                "missing_frame",
                f"Cache is missing {len(missing)} frame(s); first missing frame is {missing[0]}.",
                cache_dir,
                missing[0],
            )
    if metadata.cached_frame_count != len(frames):
        report.add(
            "ERROR",
            "cached_frame_count_mismatch",
            "metadata.cached_frame_count does not match JSONL rows.",
            cache_dir,
        )
    if (
        metadata.max_frame is not None
        and frame_indices
        and metadata.max_frame != max(frame_indices)
    ):
        report.add(
            "ERROR",
            "max_frame_mismatch",
            "metadata.max_frame does not match JSONL.",
            cache_dir,
        )

    for frame in frames:
        if frame.sequence_name != metadata.sequence_name:
            report.add(
                "ERROR",
                "sequence_mismatch",
                "Frame sequence_name does not match metadata.",
                cache_dir,
                frame.frame_index,
            )
        _check_frame_geometry(frame, report, reader.detections_path)
    return report


def validate_cache_for_sources(
    cache_root: Path,
    split: str,
    sources: list[SequenceSource],
    confidence_threshold: float,
    max_frames_per_sequence: int | None = None,
    allow_partial_sequences: bool = False,
    expected_checkpoint_hash: str | None = None,
) -> DetectionCacheValidationReport:
    report = DetectionCacheValidationReport()
    for source in sources:
        expected_max_frame = None
        if max_frames_per_sequence is not None:
            expected_max_frame = min(
                max_frames_per_sequence,
                source.frame_count or max_frames_per_sequence,
            )
        sequence_report = validate_detection_cache_sequence(
            sequence_cache_dir(cache_root, split, source.name),
            expected_frame_count=source.frame_count,
            expected_max_frame=expected_max_frame,
            require_complete=max_frames_per_sequence is None,
            allow_partial=allow_partial_sequences or max_frames_per_sequence is not None,
            confidence_threshold=confidence_threshold,
            expected_checkpoint_hash=expected_checkpoint_hash,
        )
        report.extend(sequence_report)
    return report
