"""YOLOv8m pretrained baseline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.detector import (
    KNOWN_ULTRALYTICS_CHECKPOINTS,
    YOLOv8Detector,
    validate_checkpoint,
)
from football_tracking.detection.evaluate import evaluate_with_ultralytics
from football_tracking.detection.metrics import BaselineMetrics, metrics_not_available
from football_tracking.detection.postprocessing import postprocess_detections
from football_tracking.detection.preprocessing import (
    ImageMetadata,
    inspect_image,
    parse_sequence_frame,
)
from football_tracking.detection.schemas import Detection
from football_tracking.detection.serialization import (
    file_sha256,
    runtime_versions,
    write_predictions_jsonl,
    write_predictions_summary_csv,
    write_run_metadata,
    write_yolo_prediction_labels,
)
from football_tracking.detection.timing import TimingStats, maybe_synchronize_cuda
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.reporting.baseline_report import (
    COCO_PERSON_LIMITATIONS,
    write_baseline_report,
)
from football_tracking.visualization.draw_detections import draw_detection_samples


class BaselineConfigError(RuntimeError):
    """Raised when the baseline config is invalid."""


@dataclass(frozen=True)
class BaselineConfig:
    project_root: Path
    config_path: Path
    weights: str | Path
    data_yaml: Path
    split: str
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str
    half: bool
    batch: int
    workers: int
    verbose: bool
    coco_person_class_id: int
    target_class_id: int
    target_class_name: str
    keep_only_person: bool
    use_ultralytics_validator: bool
    save_predictions: bool
    save_visualizations: bool
    save_error_samples: bool
    max_visualization_samples: int
    max_images: int | None
    max_sequences: int | None
    warmup_iterations: int
    log_level: str
    overwrite: bool
    output_root: Path
    metrics_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    sequence_name: str
    frame_index: int


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BaselineConfigError(f"{section} must be a non-empty path string.")
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return resolve_project_path(value, project_root)


def _resolve_weights(value: Any, project_root: Path) -> str | Path:
    if not isinstance(value, str) or not value.strip():
        raise BaselineConfigError("model.weights must be a non-empty string.")
    if value in KNOWN_ULTRALYTICS_CHECKPOINTS:
        return value
    raw_path = Path(value)
    path = raw_path if raw_path.is_absolute() else resolve_project_path(raw_path, project_root)
    if not path.is_file():
        raise BaselineConfigError(f"model.weights does not exist: {path}")
    return path


def load_baseline_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> BaselineConfig:
    project_root = get_project_root()
    raw_config_path = Path(config_path)
    resolved_config_path = (
        raw_config_path.resolve()
        if raw_config_path.is_absolute()
        else resolve_project_path(config_path, project_root=project_root)
    )
    if not resolved_config_path.is_file():
        raise BaselineConfigError(f"Baseline config does not exist: {resolved_config_path}")
    raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config root")
    model = _mapping(root.get("model"), "model")
    dataset = _mapping(root.get("dataset"), "dataset")
    inference = _mapping(root.get("inference"), "inference")
    classes = _mapping(root.get("classes"), "classes")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    runtime = _mapping(root.get("runtime", {}), "runtime")
    output = _mapping(root.get("output"), "output")

    config = BaselineConfig(
        project_root=project_root,
        config_path=resolved_config_path,
        weights=_resolve_weights(model.get("weights", "yolov8m.pt"), project_root),
        data_yaml=_resolve_path(dataset.get("data_yaml"), project_root, "dataset.data_yaml"),
        split=str(dataset.get("split", "val")),
        imgsz=int(inference.get("imgsz", 960)),
        conf=float(inference.get("conf", 0.25)),
        iou=float(inference.get("iou", 0.7)),
        max_det=int(inference.get("max_det", 300)),
        device=str(inference.get("device", "auto")),
        half=bool(inference.get("half", False)),
        batch=int(inference.get("batch", 1)),
        workers=int(inference.get("workers", 4)),
        verbose=bool(inference.get("verbose", False)),
        coco_person_class_id=int(classes.get("coco_person_class_id", 0)),
        target_class_id=int(classes.get("target_class_id", 0)),
        target_class_name=str(classes.get("target_class_name", "player")),
        keep_only_person=bool(classes.get("keep_only_person", True)),
        use_ultralytics_validator=bool(evaluation.get("use_ultralytics_validator", True)),
        save_predictions=bool(evaluation.get("save_predictions", True)),
        save_visualizations=bool(evaluation.get("save_visualizations", True)),
        save_error_samples=bool(evaluation.get("save_error_samples", True)),
        max_visualization_samples=int(evaluation.get("max_visualization_samples", 30)),
        max_images=runtime.get("max_images"),
        max_sequences=runtime.get("max_sequences"),
        warmup_iterations=int(runtime.get("warmup_iterations", 3)),
        log_level=str(runtime.get("log_level", "INFO")),
        overwrite=bool(runtime.get("overwrite", False)),
        output_root=_resolve_path(output.get("root"), project_root, "output.root"),
        metrics_dir=_resolve_path(output.get("metrics_dir"), project_root, "output.metrics_dir"),
        figures_dir=_resolve_path(output.get("figures_dir"), project_root, "output.figures_dir"),
    )
    if config.max_images is not None:
        config = replace(config, max_images=int(config.max_images))
    if config.max_sequences is not None:
        config = replace(config, max_sequences=int(config.max_sequences))
    if overrides:
        config = _apply_overrides(config, overrides)
    validate_baseline_config(config)
    return config


def _apply_overrides(config: BaselineConfig, overrides: dict[str, Any]) -> BaselineConfig:
    changes: dict[str, Any] = {}
    allowed = {
        "split",
        "device",
        "imgsz",
        "conf",
        "iou",
        "batch",
        "max_images",
        "max_sequences",
        "overwrite",
    }
    for key, value in overrides.items():
        if key in allowed and value is not None:
            changes[key] = value
    if overrides.get("no_visualization"):
        changes["save_visualizations"] = False
        changes["save_error_samples"] = False
    if not changes:
        return config
    return replace(config, **changes)


def validate_baseline_config(config: BaselineConfig) -> None:
    if config.split not in {"train", "val", "test"}:
        raise BaselineConfigError("dataset.split must be one of train, val, or test.")
    if not config.data_yaml.is_file():
        raise BaselineConfigError(f"dataset.data_yaml does not exist: {config.data_yaml}")
    if not 0.0 <= config.conf <= 1.0:
        raise BaselineConfigError("inference.conf must be in [0, 1].")
    if not 0.0 <= config.iou <= 1.0:
        raise BaselineConfigError("inference.iou must be in [0, 1].")
    if config.imgsz <= 0:
        raise BaselineConfigError("inference.imgsz must be positive.")
    if config.max_det <= 0:
        raise BaselineConfigError("inference.max_det must be positive.")
    if config.batch <= 0:
        raise BaselineConfigError("inference.batch must be positive.")
    if config.max_images is not None and config.max_images <= 0:
        raise BaselineConfigError("runtime.max_images must be positive when set.")
    if config.max_sequences is not None and config.max_sequences <= 0:
        raise BaselineConfigError("runtime.max_sequences must be positive when set.")
    try:
        validate_checkpoint(config.weights)
    except Exception as exc:  # noqa: BLE001
        raise BaselineConfigError(str(exc)) from exc


def _dataset_root(data_yaml: Path, raw: dict[str, Any]) -> Path:
    root = Path(str(raw.get("path", data_yaml.parent)))
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def collect_dataset_images(config: BaselineConfig) -> list[DatasetImage]:
    raw = yaml.safe_load(config.data_yaml.read_text(encoding="utf-8"))
    dataset = _mapping(raw, "YOLO dataset YAML")
    if config.split not in dataset:
        raise BaselineConfigError(f"YOLO dataset YAML has no {config.split!r} split.")
    root = _dataset_root(config.data_yaml, dataset)
    split_value = dataset[config.split]
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
    images: list[DatasetImage] = []
    for split_path_value in split_paths:
        split_path = Path(str(split_path_value))
        image_root = split_path if split_path.is_absolute() else root / split_path
        if not image_root.exists():
            continue
        image_paths = sorted(
            item for item in image_root.rglob("*") if item.suffix.lower() in suffixes
        )
        for path in image_paths:
            sequence_name, frame_index = parse_sequence_frame(path)
            images.append(
                DatasetImage(
                    path=path,
                    sequence_name=sequence_name,
                    frame_index=frame_index,
                )
            )
    if config.max_sequences is not None:
        selected: set[str] = set()
        filtered: list[DatasetImage] = []
        for image in images:
            if image.sequence_name not in selected:
                if len(selected) >= config.max_sequences:
                    continue
                selected.add(image.sequence_name)
            filtered.append(image)
        images = filtered
    if config.max_images is not None:
        images = images[: config.max_images]
    return images


def _dry_run_plan(config: BaselineConfig) -> dict[str, Any]:
    images = collect_dataset_images(config)
    return {
        "dry_run": True,
        "model": {"weights": str(config.weights), "checkpoint_valid": True},
        "dataset": {
            "data_yaml": str(config.data_yaml),
            "split": config.split,
            "image_count": len(images),
        },
        "inference": {
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "batch": config.batch,
            "max_det": config.max_det,
            "device": config.device,
        },
        "outputs": {
            "root": str(config.output_root),
            "metrics_dir": str(config.metrics_dir),
            "figures_dir": str(config.figures_dir),
        },
        "action": "validated config/dataset/checkpoint; no inference was run",
    }


def _metadata_for_images(images: list[DatasetImage]) -> tuple[list[ImageMetadata], float]:
    started = time.perf_counter()
    metadata = [
        inspect_image(image.path, image.sequence_name, image.frame_index)
        for image in images
    ]
    return metadata, time.perf_counter() - started


def _run_warmup(detector: Any, images: list[DatasetImage], config: BaselineConfig) -> float:
    if config.warmup_iterations <= 0 or not images or not hasattr(detector, "warmup"):
        return 0.0
    started = time.perf_counter()
    detector.warmup(images[0].path, iterations=config.warmup_iterations)
    return time.perf_counter() - started


def _run_inference(
    detector: Any,
    images: list[DatasetImage],
    image_metadata: list[ImageMetadata],
    config: BaselineConfig,
    timings: TimingStats,
) -> list[Detection]:
    detections: list[Detection] = []
    for start_index in range(0, len(images), config.batch):
        image_batch = images[start_index : start_index + config.batch]
        metadata_batch = image_metadata[start_index : start_index + config.batch]
        maybe_synchronize_cuda(config.device)
        started = time.perf_counter()
        raw_results = detector.predict_batch(
            [image.path for image in image_batch],
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            max_det=config.max_det,
            batch=config.batch,
            verbose=config.verbose,
        )
        maybe_synchronize_cuda(config.device)
        timings.inference_seconds += time.perf_counter() - started
        started = time.perf_counter()
        for raw_result, metadata in zip(raw_results, metadata_batch, strict=False):
            detections.extend(
                postprocess_detections(
                    raw_result,
                    frame_index=metadata.frame_index,
                    sequence_name=metadata.sequence_name,
                    image_width=metadata.width,
                    image_height=metadata.height,
                    confidence_threshold=config.conf,
                    coco_person_class_id=config.coco_person_class_id,
                    target_class_id=config.target_class_id,
                    target_class_name=config.target_class_name,
                    keep_only_person=config.keep_only_person,
                    image_path=metadata.image_path,
                )
            )
        timings.postprocessing_seconds += time.perf_counter() - started
    return detections


def _write_outputs(
    detections: list[Detection],
    config: BaselineConfig,
    timings: TimingStats,
    metrics: BaselineMetrics,
    started_at: str,
    ended_at: str,
    image_count: int,
    visualization_paths: list[Path],
) -> dict[str, Any]:
    serialization_started = time.perf_counter()
    prediction_path = write_predictions_jsonl(
        detections,
        config.output_root / "predictions.jsonl",
    )
    label_paths = write_yolo_prediction_labels(detections, config.output_root / "yolo_labels")
    summary_csv = write_predictions_summary_csv(
        detections,
        config.output_root / "predictions_summary.csv",
    )
    versions = runtime_versions()
    weights_path = Path(str(config.weights))
    metadata = {
        "model_checkpoint": str(config.weights),
        "model_file_hash": file_sha256(weights_path),
        "config": str(config.config_path),
        "versions": versions,
        "cuda_available": versions.get("cuda_available"),
        "gpu_name": versions.get("gpu_name"),
        "dataset_split": config.split,
        "start_time": started_at,
        "end_time": ended_at,
        "image_count": image_count,
        "prediction_count": len(detections),
        "runtime": timings.to_dict(image_count),
    }
    metadata_path = write_run_metadata(metadata, config.output_root / "run_metadata.json")
    timings.serialization_seconds += time.perf_counter() - serialization_started
    report_payload = {
        "dataset": {"data_yaml": str(config.data_yaml), "split": config.split},
        "model": {"weights": str(config.weights), "task": "detect"},
        "inference": {
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "batch": config.batch,
            "max_det": config.max_det,
        },
        "metrics": metrics.to_dict(),
        "timing": timings.to_dict(image_count),
        "counts": {"image_count": image_count, "prediction_count": len(detections)},
        "runtime": {
            "device": config.device,
            "cuda_available": versions.get("cuda_available"),
            "gpu_name": versions.get("gpu_name"),
        },
        "limitations": COCO_PERSON_LIMITATIONS,
        "outputs": {
            "predictions_jsonl": str(prediction_path),
            "yolo_labels_dir": str(config.output_root / "yolo_labels"),
            "predictions_summary_csv": str(summary_csv),
            "run_metadata": str(metadata_path),
            "visualizations": [str(path) for path in visualization_paths],
        },
    }
    report_paths = write_baseline_report(report_payload, config.metrics_dir)
    return {
        "predictions_jsonl": str(prediction_path),
        "yolo_label_files": [str(path) for path in label_paths],
        "predictions_summary_csv": str(summary_csv),
        "run_metadata": str(metadata_path),
        "report_paths": {key: str(path) for key, path in report_paths.items()},
        "report": report_payload,
    }


def run_baseline(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    detector: Any | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    config = load_baseline_config(config_path, overrides=overrides)
    if dry_run:
        return _dry_run_plan(config)

    images = collect_dataset_images(config)
    if not images:
        raise BaselineConfigError(f"No images found for split {config.split} in {config.data_yaml}")
    if config.output_root.exists() and not config.overwrite:
        existing = [path for path in config.output_root.rglob("*") if path.is_file()]
        if existing:
            raise BaselineConfigError(
                f"Output already exists and overwrite=false: {config.output_root}"
            )

    started_at = datetime.now(UTC).isoformat()
    pipeline_started = time.perf_counter()
    timings = TimingStats()
    image_metadata, timings.preprocessing_seconds = _metadata_for_images(images)
    detector = detector or YOLOv8Detector(config.weights, device=config.device, half=config.half)
    load_started = time.perf_counter()
    if hasattr(detector, "load_model"):
        detector.load_model()
    timings.model_load_seconds = time.perf_counter() - load_started
    timings.warmup_seconds = _run_warmup(detector, images, config)
    detections = _run_inference(detector, images, image_metadata, config, timings)

    visualization_paths: list[Path] = []
    if config.save_visualizations:
        visualization_paths = draw_detection_samples(
            detections,
            output_dir=config.figures_dir / "predictions",
            max_samples=config.max_visualization_samples,
        )

    if evaluate and config.use_ultralytics_validator:
        metrics = evaluate_with_ultralytics(
            config.weights,
            config.data_yaml,
            config.split,
            config.imgsz,
            config.conf,
            config.iou,
            config.batch,
            config.device,
            model=getattr(detector, "model", None),
        )
    else:
        metrics = metrics_not_available("Evaluation was not requested.")
    timings.total_pipeline_seconds = time.perf_counter() - pipeline_started
    ended_at = datetime.now(UTC).isoformat()
    outputs = _write_outputs(
        detections,
        config,
        timings,
        metrics,
        started_at,
        ended_at,
        len(images),
        visualization_paths,
    )
    return {
        "dry_run": False,
        "image_count": len(images),
        "prediction_count": len(detections),
        "metrics": metrics.to_dict(),
        "timing": timings.to_dict(len(images)),
        "outputs": outputs,
    }


def evaluate_baseline(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_baseline_config(config_path, overrides=overrides)
    if dry_run:
        return _dry_run_plan(config)
    metrics = evaluate_with_ultralytics(
        config.weights,
        config.data_yaml,
        config.split,
        config.imgsz,
        config.conf,
        config.iou,
        config.batch,
        config.device,
    )
    versions = runtime_versions()
    payload = {
        "dataset": {"data_yaml": str(config.data_yaml), "split": config.split},
        "model": {"weights": str(config.weights), "task": "detect"},
        "inference": {"imgsz": config.imgsz, "conf": config.conf, "iou": config.iou},
        "metrics": metrics.to_dict(),
        "timing": {},
        "counts": {"image_count": None, "prediction_count": None},
        "runtime": {
            "device": config.device,
            "cuda_available": versions.get("cuda_available"),
            "gpu_name": versions.get("gpu_name"),
        },
        "limitations": COCO_PERSON_LIMITATIONS,
    }
    report_paths = write_baseline_report(payload, config.metrics_dir)
    return {
        "metrics": metrics.to_dict(),
        "report_paths": {key: str(path) for key, path in report_paths.items()},
    }
