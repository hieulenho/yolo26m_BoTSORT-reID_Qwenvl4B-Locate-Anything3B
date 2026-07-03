"""Create and validate reusable detector caches for tracker experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.cache_schema import (
    SCHEMA_VERSION,
    CachedDetection,
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.cache_validation import (
    validate_cache_for_sources,
    validate_detection_cache_sequence,
)
from football_tracking.detection.cache_writer import DetectionCacheWriter, sequence_cache_dir
from football_tracking.detection.detector import resolve_device
from football_tracking.detection.detector_factory import create_detector, detector_name_from_config
from football_tracking.detection.postprocessing import postprocess_detections
from football_tracking.detection.serialization import runtime_versions
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.checkpoint_resolver import (
    ResolvedCheckpoint,
    resolve_detector_checkpoint,
)
from football_tracking.tracking.sequence_runner import (
    SequenceSource,
    discover_mot_sequences,
    iter_source_frames,
)
from football_tracking.utils.progress import progress_iter


class DetectionCacheError(RuntimeError):
    """Raised when detection cache creation or validation fails."""


@dataclass(frozen=True)
class DetectionCacheConfig:
    project_root: Path
    config_path: Path
    model: dict[str, Any]
    dataset_name: str
    mot_root: Path
    split: str
    seqmap: Path | None
    imgsz: int
    conf_floor: float
    iou: float
    max_det: int
    device: str
    half: bool
    batch: int
    class_ids: tuple[int, ...] | None
    target_class_id: int
    target_class_name: str
    preserve_source_classes: bool
    source_class_names: dict[int, str]
    cache_root: Path
    cache_format: str
    save_npz: bool
    include_empty_frames: bool
    overwrite: bool
    validate_after_write: bool
    max_sequences: int | None
    max_frames_per_sequence: int | None
    warmup_iterations: int
    log_level: str
    smoke_only: bool


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DetectionCacheError(f"{section} must be a mapping.")
    return value


def _resolve_path(
    value: Any,
    project_root: Path,
    section: str,
    required: bool = True,
) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DetectionCacheError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_detection_cache_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> DetectionCacheConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise DetectionCacheError(f"Detection cache config does not exist: {resolved}")
    root = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "cache config root")
    model = dict(_mapping(root.get("model"), "model"))
    dataset = _mapping(root.get("dataset"), "dataset")
    inference = _mapping(root.get("inference"), "inference")
    cache = _mapping(root.get("cache"), "cache")
    runtime = _mapping(root.get("runtime", {}), "runtime")
    class_values = inference.get("class_ids", [0])
    class_ids = None if class_values is None else tuple(int(value) for value in class_values)
    config = DetectionCacheConfig(
        project_root=project_root,
        config_path=resolved,
        model=model,
        dataset_name=str(dataset.get("name", "SportsMOT football")),
        mot_root=_resolve_path(dataset.get("mot_root"), project_root, "dataset.mot_root"),
        split=str(dataset.get("split", "val")),
        seqmap=_resolve_path(dataset.get("seqmap"), project_root, "dataset.seqmap", required=False),
        imgsz=int(inference.get("imgsz", 960)),
        conf_floor=float(inference.get("conf_floor", 0.001)),
        iou=float(inference.get("iou", 0.70)),
        max_det=int(inference.get("max_det", 300)),
        device=str(inference.get("device", "auto")),
        half=bool(inference.get("half", False)),
        batch=int(inference.get("batch", 1)),
        class_ids=class_ids,
        target_class_id=int(inference.get("target_class_id", 0)),
        target_class_name=str(inference.get("target_class_name", "player")),
        preserve_source_classes=bool(inference.get("preserve_source_classes", False)),
        source_class_names={
            int(key): str(value)
            for key, value in (inference.get("source_class_names", {}) or {}).items()
        },
        cache_root=_resolve_path(cache.get("root"), project_root, "cache.root"),
        cache_format=str(cache.get("format", "jsonl")),
        save_npz=bool(cache.get("save_npz", False)),
        include_empty_frames=bool(cache.get("include_empty_frames", True)),
        overwrite=bool(cache.get("overwrite", False)),
        validate_after_write=bool(cache.get("validate_after_write", True)),
        max_sequences=runtime.get("max_sequences"),
        max_frames_per_sequence=runtime.get(
            "max_frames_per_sequence",
            runtime.get("max_frames"),
        ),
        warmup_iterations=int(runtime.get("warmup_iterations", 0)),
        log_level=str(runtime.get("log_level", "INFO")),
        smoke_only=bool(runtime.get("smoke_only", False)),
    )
    if config.max_sequences is not None:
        config = replace(config, max_sequences=int(config.max_sequences))
    if config.max_frames_per_sequence is not None:
        config = replace(config, max_frames_per_sequence=int(config.max_frames_per_sequence))
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_config(config)
    return config


def _apply_overrides(
    config: DetectionCacheConfig,
    overrides: dict[str, Any],
) -> DetectionCacheConfig:
    changes: dict[str, Any] = {}
    if overrides.get("device") is not None:
        changes["device"] = str(overrides["device"])
    if overrides.get("max_sequences") is not None:
        changes["max_sequences"] = int(overrides["max_sequences"])
    if overrides.get("max_frames") is not None:
        changes["max_frames_per_sequence"] = int(overrides["max_frames"])
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config


def _validate_config(config: DetectionCacheConfig) -> None:
    if config.cache_format != "jsonl":
        raise DetectionCacheError("Only cache.format=jsonl is supported.")
    if config.save_npz:
        raise DetectionCacheError("cache.save_npz is not implemented for this milestone.")
    if not 0.0 <= config.conf_floor <= 1.0:
        raise DetectionCacheError("inference.conf_floor must be in [0, 1].")
    if not 0.0 <= config.iou <= 1.0:
        raise DetectionCacheError("inference.iou must be in [0, 1].")
    if config.imgsz <= 0 or config.max_det <= 0 or config.batch <= 0:
        raise DetectionCacheError("imgsz, max_det, and batch must be positive.")
    if config.max_sequences is not None and config.max_sequences <= 0:
        raise DetectionCacheError("runtime.max_sequences must be positive when set.")
    if config.max_frames_per_sequence is not None and config.max_frames_per_sequence <= 0:
        raise DetectionCacheError("runtime.max_frames_per_sequence must be positive when set.")
    if not config.mot_root.is_dir():
        raise DetectionCacheError(f"dataset.mot_root does not exist: {config.mot_root}")
    if config.seqmap is not None and not config.seqmap.is_file():
        raise DetectionCacheError(f"dataset.seqmap does not exist: {config.seqmap}")


def _discover_sources(config: DetectionCacheConfig) -> list[SequenceSource]:
    return discover_mot_sequences(
        config.mot_root,
        config.split,
        config.seqmap,
        max_sequences=config.max_sequences,
    )


def _dry_run_plan(
    config: DetectionCacheConfig,
    checkpoint: ResolvedCheckpoint,
    sources: list[SequenceSource],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint.to_dict(),
        "cache_root": str(config.cache_root),
        "split": config.split,
        "confidence_floor": config.conf_floor,
        "nms_iou": config.iou,
        "sequence_count": len(sources),
        "sequences": [
            {
                "name": source.name,
                "frame_count": source.frame_count,
                "cache_dir": str(sequence_cache_dir(config.cache_root, config.split, source.name)),
            }
            for source in sources
        ],
        "action": "validated config, checkpoint, and sequence list; detector was not run",
    }


def _predict_frame(detector: Any, frame: Any, config: DetectionCacheConfig) -> Any:
    if hasattr(detector, "predict_frame"):
        return detector.predict_frame(
            frame,
            imgsz=config.imgsz,
            conf=config.conf_floor,
            iou=config.iou,
            max_det=config.max_det,
            device=config.device,
        )
    model = detector.load_model()
    kwargs = {
        "imgsz": config.imgsz,
        "conf": config.conf_floor,
        "iou": config.iou,
        "max_det": config.max_det,
        "device": getattr(detector, "device", config.device),
        "verbose": False,
    }
    if getattr(detector, "half", config.half):
        kwargs["half"] = True
    raw = model(frame, **kwargs)
    if isinstance(raw, list | tuple):
        return raw[0] if raw else None
    return raw


def _to_cached_frame(
    raw_prediction: Any,
    frame_index: int,
    sequence_name: str,
    image_path: Path | None,
    image_width: int,
    image_height: int,
    config: DetectionCacheConfig,
    checkpoint_type: str,
    detector_seconds: float,
    postprocess_seconds: float,
) -> CachedFrameDetections:
    keep_only_person = checkpoint_type == "pretrained_coco"
    detections = postprocess_detections(
        raw_prediction,
        frame_index=frame_index,
        sequence_name=sequence_name,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=config.conf_floor,
        coco_person_class_id=0,
        target_class_id=config.target_class_id,
        target_class_name=config.target_class_name,
        keep_only_person=keep_only_person,
        allowed_class_ids=config.class_ids,
        source_class_names=config.source_class_names,
        preserve_source_class=config.preserve_source_classes,
        image_path=image_path,
    )
    cached = [
        CachedDetection(
            bbox_xyxy=detection.bbox_xyxy,
            confidence=detection.confidence,
            class_id=detection.target_class_id,
            class_name=detection.target_class_name,
            source_class_id=detection.source_class_id,
            source_class_name=detection.source_class_name,
            metadata=detection.metadata,
        )
        for detection in detections
    ]
    return CachedFrameDetections(
        sequence_name=sequence_name,
        frame_index=frame_index,
        image_path=str(image_path) if image_path is not None else "",
        image_width=image_width,
        image_height=image_height,
        detections=cached,
        detector_seconds=detector_seconds,
        preprocessing_seconds=None,
        postprocessing_seconds=postprocess_seconds,
    )


def _load_detector_once(
    config: DetectionCacheConfig,
    checkpoint: ResolvedCheckpoint,
    detector: Any | None,
) -> Any:
    if detector is not None:
        return detector
    return create_detector(
        config.model,
        checkpoint.checkpoint,
        device=resolve_device(config.device),
        half=config.half,
    )


def create_detection_cache(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    detector: Any | None = None,
) -> dict[str, Any]:
    config = load_detection_cache_config(config_path, overrides=overrides)
    config = replace(config, device=resolve_device(config.device))
    checkpoint = resolve_detector_checkpoint(config.model, config.project_root)
    sources = _discover_sources(config)
    if dry_run:
        return _dry_run_plan(config, checkpoint, sources)
    if not sources:
        raise DetectionCacheError("No sequences were found for detection caching.")

    detector = _load_detector_once(config, checkpoint, detector)
    if hasattr(detector, "load_model"):
        detector.load_model()

    runtime = runtime_versions()
    sequence_results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in sources:
        try:
            sequence_results.append(
                _create_sequence_cache(config, checkpoint, source, detector, runtime)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source.name}: {exc}")
            raise
    return {
        "dry_run": False,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint.to_dict(),
        "cache_root": str(config.cache_root),
        "split": config.split,
        "sequence_count": len(sequence_results),
        "cached_frame_count": sum(row["cached_frame_count"] for row in sequence_results),
        "detection_count": sum(row["detection_count"] for row in sequence_results),
        "smoke_only": config.smoke_only or checkpoint.smoke_only,
        "sequences": sequence_results,
        "errors": errors,
    }


def _create_sequence_cache(
    config: DetectionCacheConfig,
    checkpoint: ResolvedCheckpoint,
    source: SequenceSource,
    detector: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    frames: list[CachedFrameDetections] = []
    detection_count = 0
    started = time.perf_counter()
    expected_frames = (
        min(source.frame_count, config.max_frames_per_sequence)
        if source.frame_count is not None and config.max_frames_per_sequence is not None
        else source.frame_count
    )
    for frame_item in progress_iter(
        iter_source_frames(source, max_frames=config.max_frames_per_sequence),
        total=expected_frames,
        desc=f"cache {source.name}",
        unit="frame",
    ):
        frame = frame_item.image
        image_height, image_width = frame.shape[:2]
        detector_started = time.perf_counter()
        raw_prediction = _predict_frame(detector, frame, config)
        detector_seconds = time.perf_counter() - detector_started
        postprocess_started = time.perf_counter()
        cached_frame = _to_cached_frame(
            raw_prediction,
            frame_item.frame_index,
            source.name,
            frame_item.image_path,
            image_width,
            image_height,
            config,
            checkpoint.checkpoint_type,
            detector_seconds,
            0.0,
        )
        cached_frame = replace(
            cached_frame,
            postprocessing_seconds=time.perf_counter() - postprocess_started,
        )
        if cached_frame.detections or config.include_empty_frames:
            frames.append(cached_frame)
        detection_count += len(cached_frame.detections)
    max_frame = max((frame.frame_index for frame in frames), default=None)
    complete = source.frame_count is None or len(frames) >= source.frame_count
    partial = not complete
    writer = DetectionCacheWriter(
        sequence_cache_dir(config.cache_root, config.split, source.name),
        overwrite=config.overwrite,
    )
    file_hash = writer.write_frames(frames)
    metadata = DetectionCacheMetadata(
        schema_version=SCHEMA_VERSION,
        dataset=config.dataset_name,
        split=config.split,
        sequence_name=source.name,
        expected_frame_count=source.frame_count,
        cached_frame_count=len(frames),
        checkpoint=str(checkpoint.checkpoint),
        checkpoint_type=checkpoint.checkpoint_type,
        checkpoint_hash=checkpoint.checkpoint_hash,
        detector_name=detector_name_from_config(config.model, checkpoint.checkpoint),
        image_size=config.imgsz,
        confidence_floor=config.conf_floor,
        nms_iou=config.iou,
        max_det=config.max_det,
        class_filter=list(config.class_ids) if config.class_ids is not None else [],
        device=config.device,
        python_version=runtime.get("python"),
        torch_version=runtime.get("torch"),
        ultralytics_version=runtime.get("ultralytics"),
        created_at=datetime.now(UTC).isoformat(),
        source_sequence_path=str(source.source_path),
        complete=complete,
        partial=partial,
        max_frame=max_frame,
        file_hash=file_hash,
        warnings=list(source.warnings) + list(checkpoint.warnings),
        errors=[],
    )
    writer.write_metadata(metadata)
    validation = validate_detection_cache_sequence(
        writer.cache_dir,
        expected_frame_count=source.frame_count,
        expected_max_frame=max_frame,
        require_complete=config.max_frames_per_sequence is None,
        allow_partial=config.max_frames_per_sequence is not None,
    )
    if config.validate_after_write:
        writer.write_validation(validation.to_dict())
        if validation.has_errors:
            raise DetectionCacheError(f"Cache validation failed for {source.name}.")
    return {
        "sequence_name": source.name,
        "cache_dir": str(writer.cache_dir),
        "detections_jsonl": str(writer.detections_path),
        "metadata_json": str(writer.metadata_path),
        "validation_json": str(writer.validation_path),
        "expected_frame_count": source.frame_count,
        "cached_frame_count": len(frames),
        "detection_count": detection_count,
        "file_hash": file_hash,
        "complete": complete,
        "partial": partial,
        "seconds": time.perf_counter() - started,
        "validation": validation.to_dict(),
    }


def validate_detection_cache(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_detection_cache_config(config_path, overrides=overrides)
    checkpoint = resolve_detector_checkpoint(config.model, config.project_root)
    sources = _discover_sources(config)
    report = validate_cache_for_sources(
        config.cache_root,
        config.split,
        sources,
        confidence_threshold=config.conf_floor,
        max_frames_per_sequence=config.max_frames_per_sequence,
        allow_partial_sequences=config.max_frames_per_sequence is not None,
        expected_checkpoint_hash=checkpoint.checkpoint_hash,
    )
    validation_paths: list[str] = []
    for source in sources:
        cache_dir = sequence_cache_dir(config.cache_root, config.split, source.name)
        validation_path = cache_dir / "validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_report = validate_detection_cache_sequence(
            cache_dir,
            expected_frame_count=source.frame_count,
            expected_max_frame=(
                min(config.max_frames_per_sequence, source.frame_count)
                if config.max_frames_per_sequence is not None and source.frame_count is not None
                else config.max_frames_per_sequence
            ),
            require_complete=config.max_frames_per_sequence is None,
            allow_partial=config.max_frames_per_sequence is not None,
            confidence_threshold=config.conf_floor,
            expected_checkpoint_hash=checkpoint.checkpoint_hash,
        )
        validation_path.write_text(
            json.dumps(sequence_report.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        validation_paths.append(str(validation_path))
    return {
        "report": report.to_dict(),
        "checkpoint": checkpoint.to_dict(),
        "cache_root": str(config.cache_root),
        "split": config.split,
        "sequence_count": len(sources),
        "validation_paths": validation_paths,
    }
