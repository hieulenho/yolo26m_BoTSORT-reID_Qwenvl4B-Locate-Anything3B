"""Orchestrate Milestone 2 data preparation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.audit import create_dataset_audit, write_dataset_audit
from football_tracking.data.class_mapping import (
    ClassMapping,
    apply_mapping_to_object,
    load_class_mapping,
)
from football_tracking.data.convert_mot import convert_to_mot
from football_tracking.data.convert_yolo import convert_to_yolo
from football_tracking.data.manifest import write_dataset_manifest
from football_tracking.data.schemas import (
    DataPipelineConfig,
    FrameAnnotation,
    SequenceInfo,
    SplitManifest,
    ValidationReport,
)
from football_tracking.data.soccernet_adapter import SoccerNetAdapter
from football_tracking.data.split_sequences import split_sequences, write_split_manifest
from football_tracking.data.validate import (
    validate_mot_dataset,
    validate_sequences,
    validate_split_leakage,
    validate_yolo_dataset,
    write_validation_report,
)
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.visualization.draw_annotations import draw_annotation_samples

LOGGER = logging.getLogger(__name__)


class DataConfigError(RuntimeError):
    """Raised when data pipeline configuration is invalid."""


@dataclass(frozen=True)
class PreparationResult:
    sequences: list[SequenceInfo]
    split_manifest: SplitManifest
    validation_report: ValidationReport
    yolo_stats: dict[str, Any]
    mot_stats: dict[str, Any]
    manifest: dict[str, Any] | None
    audit: dict[str, Any] | None
    visualization_paths: list[Path]
    dry_run: bool

    def summary(self) -> dict[str, Any]:
        return {
            "sequence_count": len(self.sequences),
            "frame_count": sum(sequence.frame_count for sequence in self.sequences),
            "validation_errors": self.validation_report.error_count,
            "validation_warnings": self.validation_report.warning_count,
            "yolo": self.yolo_stats,
            "mot": self.mot_stats,
            "dry_run": self.dry_run,
        }


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataConfigError(f"{section} must be a mapping.")
    return value


def _resolve_optional_path(value: Any, project_root: Path) -> Path | None:
    if value in (None, "null", ""):
        return None
    return resolve_project_path(str(value), project_root=project_root)


def load_data_config(config_path: str | Path) -> DataPipelineConfig:
    project_root = get_project_root()
    resolved_config_path = resolve_project_path(config_path, project_root=project_root)
    if not resolved_config_path.is_file():
        raise DataConfigError(f"Data config does not exist: {resolved_config_path}")
    raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config root")

    dataset = _mapping(root.get("dataset"), "dataset")
    classes = _mapping(root.get("classes"), "classes")
    frames = _mapping(root.get("frames"), "frames")
    split = _mapping(root.get("split"), "split")
    yolo_config = _mapping(root.get("yolo"), "yolo")
    mot_config = _mapping(root.get("mot"), "mot")
    validation = _mapping(root.get("validation"), "validation")
    visualization = _mapping(root.get("visualization", {}), "visualization")
    runtime = _mapping(root.get("runtime", {}), "runtime")

    train_ratio = float(split.get("train_ratio", 0.7))
    val_ratio = float(split.get("val_ratio", 0.15))
    test_ratio = float(split.get("test_ratio", 0.15))
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise DataConfigError("train_ratio + val_ratio + test_ratio must equal 1.0.")
    mot_frame_index_base = int(mot_config.get("frame_index_base", 1))
    if mot_frame_index_base != 1:
        raise DataConfigError("mot.frame_index_base must be 1 for MOTChallenge output.")

    return DataPipelineConfig(
        project_root=project_root,
        config_path=resolved_config_path,
        dataset_name=str(dataset.get("name", "dataset")),
        adapter=str(dataset.get("adapter", "soccernet")),
        raw_dir=resolve_project_path(str(dataset["raw_dir"]), project_root=project_root),
        interim_dir=resolve_project_path(str(dataset["interim_dir"]), project_root=project_root),
        class_mapping_path=resolve_project_path(
            str(classes["mapping_config"]),
            project_root=project_root,
        ),
        target_class=str(classes.get("target_class", "player")),
        extract_frames=bool(frames.get("extract_from_video", False)),
        image_extension=str(frames.get("image_extension", ".jpg")),
        jpeg_quality=int(frames.get("jpeg_quality", 95)),
        split_strategy=str(split.get("strategy", "sequence")),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=int(split.get("seed", 42)),
        predefined_split_file=_resolve_optional_path(
            split.get("predefined_split_file"),
            project_root,
        ),
        yolo_output_dir=resolve_project_path(
            str(yolo_config["output_dir"]),
            project_root=project_root,
        ),
        yolo_decimal_places=int(yolo_config.get("decimal_places", 6)),
        yolo_copy_images=bool(yolo_config.get("copy_images", False)),
        yolo_prefer_symlink=bool(yolo_config.get("prefer_symlink", True)),
        mot_output_dir=resolve_project_path(
            str(mot_config["output_dir"]),
            project_root=project_root,
        ),
        mot_frame_index_base=mot_frame_index_base,
        mot_confidence_default=float(mot_config.get("confidence_default", 1.0)),
        mot_visibility_default=float(mot_config.get("visibility_default", 1.0)),
        clip_boxes=bool(validation.get("clip_boxes", True)),
        invalid_box_policy=str(validation.get("invalid_box_policy", "warn_and_skip")),
        unknown_class_policy=str(validation.get("unknown_class_policy", "warn_and_skip")),
        fail_on_duplicate_track_in_frame=bool(
            validation.get("fail_on_duplicate_track_in_frame", True)
        ),
        visualization_output_dir=resolve_project_path(
            str(visualization.get("output_dir", "outputs/figures/annotation_samples")),
            project_root=project_root,
        ),
        visualization_num_sequences=int(visualization.get("num_sequences", 2)),
        visualization_frames_per_sequence=int(visualization.get("frames_per_sequence", 2)),
        visualization_seed=int(visualization.get("seed", split.get("seed", 42))),
        visualization_draw_ignored=bool(visualization.get("draw_ignored", False)),
        visualization_line_thickness=int(visualization.get("line_thickness", 2)),
        visualization_font_scale=float(visualization.get("font_scale", 0.5)),
        dry_run=bool(runtime.get("dry_run", False)),
        overwrite=bool(runtime.get("overwrite", False)),
        fail_fast=bool(runtime.get("fail_fast", True)),
        log_level=str(runtime.get("log_level", "INFO")),
    )


def _adapter_for_config(config: DataPipelineConfig) -> SoccerNetAdapter:
    if config.adapter != "soccernet":
        raise DataConfigError(f"Unsupported dataset adapter: {config.adapter}")
    return SoccerNetAdapter()


def _map_sequence_classes(sequence: SequenceInfo, class_mapping: ClassMapping) -> SequenceInfo:
    mapped_frames: list[FrameAnnotation] = []
    unknown_count = 0
    ignored_count = 0
    for frame in sequence.annotations:
        mapped_objects = []
        for annotation in frame.objects:
            mapped = apply_mapping_to_object(annotation, class_mapping)
            unknown_count += bool(mapped.metadata.get("unknown_class"))
            ignored_count += mapped.is_ignored
            mapped_objects.append(mapped)
        mapped_frames.append(
            FrameAnnotation(
                sequence_name=frame.sequence_name,
                frame_index=frame.frame_index,
                image_path=frame.image_path,
                width=frame.width,
                height=frame.height,
                objects=mapped_objects,
            )
        )
    metadata = dict(sequence.metadata)
    metadata["unknown_class_count"] = unknown_count
    metadata["ignored_object_count"] = ignored_count
    return SequenceInfo(
        name=sequence.name,
        source_path=sequence.source_path,
        frames_dir=sequence.frames_dir,
        annotations_path=sequence.annotations_path,
        fps=sequence.fps,
        width=sequence.width,
        height=sequence.height,
        frame_count=sequence.frame_count,
        annotations=mapped_frames,
        metadata=metadata,
    )


def _load_sequences(
    config: DataPipelineConfig,
    class_mapping: ClassMapping,
    max_sequences: int | None,
) -> list[SequenceInfo]:
    adapter = _adapter_for_config(config)
    candidates = adapter.discover_sequences(config.raw_dir)
    if max_sequences is not None:
        candidates = candidates[:max_sequences]
    sequences: list[SequenceInfo] = []
    for candidate in candidates:
        started = time.perf_counter()
        try:
            sequence = adapter.load_sequence(candidate.source_path)
            sequences.append(_map_sequence_classes(sequence, class_mapping))
            LOGGER.info(
                "Loaded sequence %s with %s frames in %.2fs",
                candidate.name,
                len(sequence.annotations),
                time.perf_counter() - started,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to load sequence %s", candidate.source_path)
            if config.fail_fast:
                raise
    return sequences


def prepare_data(
    config_path: str | Path,
    dry_run: bool | None = None,
    overwrite: bool | None = None,
    fail_fast: bool | None = None,
    max_sequences: int | None = None,
) -> PreparationResult:
    config = load_data_config(config_path)
    if dry_run is not None:
        config = _replace_config(config, dry_run=dry_run)
    if overwrite is not None:
        config = _replace_config(config, overwrite=overwrite)
    if fail_fast is not None:
        config = _replace_config(config, fail_fast=fail_fast)

    started = time.perf_counter()
    class_mapping = load_class_mapping(config.class_mapping_path)
    sequences = _load_sequences(config, class_mapping, max_sequences=max_sequences)
    if not sequences:
        raise DataConfigError(f"No sequences were loaded from {config.raw_dir}")

    validation_report = validate_sequences(
        sequences,
        require_images=True,
        invalid_box_policy=config.invalid_box_policy,
        fail_on_duplicate_track_in_frame=config.fail_on_duplicate_track_in_frame,
    )
    if validation_report.has_errors and config.fail_fast:
        report_path = config.project_root / "outputs/metrics/data_validation.json"
        write_validation_report(validation_report, report_path)
        raise DataConfigError(
            "Raw data validation failed. See outputs/metrics/data_validation.json."
        )

    split_manifest = split_sequences(
        sequences,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        strategy=config.split_strategy,
        predefined_split_file=config.predefined_split_file,
    )
    validation_report.extend(validate_split_leakage(split_manifest))
    if not config.dry_run:
        write_split_manifest(split_manifest, config.interim_dir / "splits.json")

    yolo_stats = convert_to_yolo(
        sequences,
        split_manifest,
        output_dir=config.yolo_output_dir,
        class_names={0: "player"},
        decimal_places=config.yolo_decimal_places,
        copy_images=config.yolo_copy_images,
        prefer_symlink=config.yolo_prefer_symlink,
        clip_boxes=config.clip_boxes,
        overwrite=config.overwrite,
        dry_run=config.dry_run,
    )
    mot_stats = convert_to_mot(
        sequences,
        split_manifest,
        output_dir=config.mot_output_dir,
        image_extension=config.image_extension,
        frame_index_base=config.mot_frame_index_base,
        confidence_default=config.mot_confidence_default,
        visibility_default=config.mot_visibility_default,
        clip_boxes=config.clip_boxes,
        prefer_symlink=config.yolo_prefer_symlink,
        overwrite=config.overwrite,
        dry_run=config.dry_run,
    )

    manifest = None
    audit = None
    visualization_paths: list[Path] = []
    if not config.dry_run:
        validation_report.extend(validate_yolo_dataset(config.yolo_output_dir))
        validation_report.extend(validate_mot_dataset(config.mot_output_dir))
        write_validation_report(
            validation_report,
            config.project_root / "outputs/metrics/data_validation.json",
        )
        manifest = write_dataset_manifest(
            sequences,
            split_manifest,
            output_dir=config.interim_dir,
            dataset_name=config.dataset_name,
            adapter=config.adapter,
            seed=config.seed,
            config_path=config.config_path,
            class_mapping_path=config.class_mapping_path,
            yolo_output_dir=config.yolo_output_dir,
            mot_output_dir=config.mot_output_dir,
            warnings=[
                issue.message for issue in validation_report.issues if issue.severity == "WARNING"
            ],
            errors=[
                issue.message for issue in validation_report.issues if issue.severity == "ERROR"
            ],
        )
        audit = create_dataset_audit(sequences)
        write_dataset_audit(audit, config.project_root / "outputs/metrics")
        visualization_paths = draw_annotation_samples(
            sequences,
            output_dir=config.visualization_output_dir,
            num_sequences=config.visualization_num_sequences,
            frames_per_sequence=config.visualization_frames_per_sequence,
            seed=config.visualization_seed,
            draw_ignored=config.visualization_draw_ignored,
            line_thickness=config.visualization_line_thickness,
            font_scale=config.visualization_font_scale,
        )

    LOGGER.info("Data preparation completed in %.2fs", time.perf_counter() - started)
    return PreparationResult(
        sequences=sequences,
        split_manifest=split_manifest,
        validation_report=validation_report,
        yolo_stats=yolo_stats,
        mot_stats=mot_stats,
        manifest=manifest,
        audit=audit,
        visualization_paths=visualization_paths,
        dry_run=config.dry_run,
    )


def validate_data(config_path: str | Path, max_sequences: int | None = None) -> ValidationReport:
    config = load_data_config(config_path)
    class_mapping = load_class_mapping(config.class_mapping_path)
    sequences = _load_sequences(config, class_mapping, max_sequences=max_sequences)
    report = validate_sequences(
        sequences,
        require_images=True,
        invalid_box_policy=config.invalid_box_policy,
        fail_on_duplicate_track_in_frame=config.fail_on_duplicate_track_in_frame,
    )
    write_validation_report(report, config.project_root / "outputs/metrics/data_validation.json")
    return report


def audit_data(config_path: str | Path, max_sequences: int | None = None) -> dict[str, Any]:
    config = load_data_config(config_path)
    class_mapping = load_class_mapping(config.class_mapping_path)
    sequences = _load_sequences(config, class_mapping, max_sequences=max_sequences)
    audit = create_dataset_audit(sequences)
    write_dataset_audit(audit, config.project_root / "outputs/metrics")
    return audit


def visualize_annotations(
    config_path: str | Path,
    num_samples: int | None = None,
    max_sequences: int | None = None,
) -> list[Path]:
    config = load_data_config(config_path)
    class_mapping = load_class_mapping(config.class_mapping_path)
    sequences = _load_sequences(config, class_mapping, max_sequences=max_sequences)
    frames_per_sequence = max(1, num_samples or config.visualization_frames_per_sequence)
    return draw_annotation_samples(
        sequences,
        output_dir=config.visualization_output_dir,
        num_sequences=config.visualization_num_sequences,
        frames_per_sequence=frames_per_sequence,
        seed=config.visualization_seed,
        draw_ignored=config.visualization_draw_ignored,
        line_thickness=config.visualization_line_thickness,
        font_scale=config.visualization_font_scale,
    )


def _replace_config(config: DataPipelineConfig, **changes: Any) -> DataPipelineConfig:
    values = config.__dict__.copy()
    values.update(changes)
    return DataPipelineConfig(**values)
