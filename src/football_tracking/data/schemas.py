"""Core data models for dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["train", "val", "test"]
Severity = Literal["INFO", "WARNING", "ERROR"]


@dataclass(frozen=True)
class BoundingBoxXYXY:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class BoundingBoxXYWH:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ObjectAnnotation:
    frame_index: int
    track_id: int | str
    source_class: str
    target_class: str | None
    target_class_id: int | None
    bbox_xyxy: BoundingBoxXYXY
    confidence: float = 1.0
    visibility: float = 1.0
    is_ignored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameAnnotation:
    sequence_name: str
    frame_index: int
    image_path: Path
    width: int
    height: int
    objects: list[ObjectAnnotation] = field(default_factory=list)


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    source_path: Path
    frames_dir: Path | None
    video_path: Path | None
    annotations_path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    annotations: list[FrameAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetManifestEntry:
    sequence_name: str
    split: str
    frame_count: int
    annotated_frame_count: int
    empty_frame_count: int
    width: int
    height: int
    fps: float
    object_count: int
    unique_track_count: int
    ignored_object_count: int
    clipped_box_count: int
    invalid_box_count: int
    source_path: Path
    output_yolo_path: Path
    output_mot_path: Path


@dataclass(frozen=True)
class SequenceCandidate:
    name: str
    source_path: Path
    frames_dir: Path
    annotations_path: Path
    video_path: Path | None = None
    metadata_path: Path | None = None
    seqinfo_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    sequence_name: str | None = None
    frame_index: int | None = None
    track_id: int | str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "WARNING" for issue in self.issues)

    @property
    def info_count(self) -> int:
        return sum(issue.severity == "INFO" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "info": self.info_count,
                "warnings": self.warning_count,
                "errors": self.error_count,
            },
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "sequence_name": issue.sequence_name,
                    "frame_index": issue.frame_index,
                    "track_id": issue.track_id,
                    "path": str(issue.path) if issue.path is not None else None,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class DataPipelineConfig:
    project_root: Path
    config_path: Path
    dataset_name: str
    adapter: str
    raw_dir: Path
    interim_dir: Path
    class_mapping_path: Path
    target_class: str
    extract_frames: bool
    image_extension: str
    jpeg_quality: int
    split_strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    predefined_split_file: Path | None
    yolo_output_dir: Path
    yolo_decimal_places: int
    yolo_copy_images: bool
    yolo_prefer_symlink: bool
    mot_output_dir: Path
    mot_frame_index_base: int
    mot_confidence_default: float
    mot_visibility_default: float
    clip_boxes: bool
    invalid_box_policy: str
    unknown_class_policy: str
    fail_on_duplicate_track_in_frame: bool
    visualization_output_dir: Path
    visualization_num_sequences: int
    visualization_frames_per_sequence: int
    visualization_seed: int
    visualization_draw_ignored: bool
    visualization_line_thickness: int
    visualization_font_scale: float
    dry_run: bool
    overwrite: bool
    fail_fast: bool
    log_level: str


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    strategy: str
    train: list[str]
    val: list[str]
    test: list[str]

    def as_mapping(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def split_for_sequence(self, sequence_name: str) -> str | None:
        for split_name, sequence_names in self.as_mapping().items():
            if sequence_name in sequence_names:
                return split_name
        return None
