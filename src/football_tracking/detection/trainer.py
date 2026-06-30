"""YOLOv8m training wrapper and preflight checks."""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.detection.checkpoint import (
    CheckpointError,
    copy_final_checkpoints,
    validate_checkpoint,
)
from football_tracking.detection.experiment import ExperimentManifest
from football_tracking.detection.training_config import (
    TrainingConfig,
    TrainingConfigError,
    load_training_config,
    load_yolo_dataset_yaml,
    resolve_yolo_split_path,
)


class TrainingError(RuntimeError):
    """Raised when training fails."""


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass
class PreflightReport:
    issues: list[PreflightIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "WARNING" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def add(self, severity: str, code: str, message: str, path: Path | None = None) -> None:
        self.issues.append(
            PreflightIssue(
                severity=severity,
                code=code,
                message=message,
                path=str(path) if path else None,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {"errors": self.error_count, "warnings": self.warning_count},
            "metadata": self.metadata,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def _split_image_and_label_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    dataset_yaml = load_yolo_dataset_yaml(data_yaml)
    image_dir = resolve_yolo_split_path(data_yaml, dataset_yaml, split)
    label_dir = Path(str(image_dir).replace(f"{Path('images')}", f"{Path('labels')}"))
    if "images" in image_dir.parts:
        parts = list(image_dir.parts)
        parts[parts.index("images")] = "labels"
        label_dir = Path(*parts)
    return image_dir, label_dir


def _image_paths(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in suffixes)


def _label_paths(path: Path) -> list[Path]:
    return sorted(path.rglob("*.txt")) if path.is_dir() else []


def _sequence_name_from_image(path: Path) -> str:
    stem = path.stem
    sequence_name, separator, frame_token = stem.rpartition("_")
    if separator and frame_token.isdigit():
        return sequence_name
    return stem


def _is_sportsmot_config(config: TrainingConfig) -> bool:
    return "sportsmot" in str(config.data_yaml).lower() or "sportsmot" in config.run_name.lower()


def _validate_label_file(path: Path, class_count: int, report: PreflightReport) -> int:
    objects = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            report.add("ERROR", "label_malformed", f"Line {line_number} must have 5 fields.", path)
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            report.add(
                "ERROR",
                "label_numeric",
                f"Line {line_number} has non-numeric fields.",
                path,
            )
            continue
        if class_id < 0 or class_id >= class_count:
            report.add(
                "ERROR",
                "label_class_id",
                f"Line {line_number} class id is out of range.",
                path,
            )
        if any(value < 0.0 or value > 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
            report.add(
                "ERROR",
                "label_normalized_range",
                f"Line {line_number} has invalid normalized values.",
                path,
            )
        objects += 1
    return objects


def _runtime_metadata(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_version": platform.python_version(),
        "device": device,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_total_bytes": None,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
    }
    try:
        import torch  # type: ignore[import-not-found]

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["gpu_memory_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:  # noqa: BLE001
        payload["torch_error"] = str(exc)
    try:
        import ultralytics  # type: ignore[import-not-found]

        payload["ultralytics_version"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        payload["ultralytics_error"] = str(exc)
    return payload


def run_training_preflight(config: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    report.metadata.update(_runtime_metadata(str(config.training.get("device", "auto"))))
    if not platform.python_version().startswith("3.12."):
        report.add("ERROR", "python_version", "Python 3.12.x is required.")
    if (
        config.training.get("device") not in {"auto", "cpu"}
        and not report.metadata.get("cuda_available")
    ):
        report.add(
            "ERROR",
            "cuda_unavailable",
            "CUDA device requested but torch CUDA is unavailable.",
        )
    if config.training.get("device") == "auto" and not report.metadata.get("cuda_available"):
        report.add(
            "WARNING",
            "cuda_auto_unavailable",
            "device=auto resolved on a machine where torch CUDA is unavailable.",
        )
    if not config.data_yaml.is_file():
        report.add("ERROR", "dataset_yaml_missing", "Dataset YAML is missing.", config.data_yaml)
        return write_preflight_report(config, report)
    try:
        dataset_yaml = load_yolo_dataset_yaml(config.data_yaml)
    except TrainingConfigError as exc:
        report.add("ERROR", "dataset_yaml_invalid", str(exc), config.data_yaml)
        return write_preflight_report(config, report)

    names = dataset_yaml.get("names", {})
    class_count = (
        len(names)
        if isinstance(names, dict | list)
        else int(dataset_yaml.get("nc", 0) or 0)
    )
    if class_count != 1:
        report.add(
            "ERROR",
            "class_count",
            "Dataset YAML must contain exactly one class: player.",
            config.data_yaml,
        )
    if isinstance(names, dict) and names.get(0) != "player" and names.get("0") != "player":
        report.add("ERROR", "class_name", "Dataset class 0 must be player.", config.data_yaml)

    split_records: dict[str, dict[str, Any]] = {}
    total_objects = 0
    for split in (config.train_split, config.validation_split):
        try:
            image_dir, label_dir = _split_image_and_label_dirs(config.data_yaml, split)
        except TrainingConfigError as exc:
            report.add("ERROR", "split_missing", str(exc), config.data_yaml)
            continue
        images = _image_paths(image_dir)
        labels = _label_paths(label_dir)
        split_records[split] = {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "image_count": len(images),
            "label_count": len(labels),
        }
        if not images:
            report.add(
                "ERROR",
                f"{split}_images_missing",
                f"{split} images are missing.",
                image_dir,
            )
        if not label_dir.is_dir():
            report.add(
                "ERROR",
                f"{split}_labels_missing",
                f"{split} labels are missing.",
                label_dir,
            )
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for missing in sorted(image_stems - label_stems):
            report.add(
                "ERROR",
                "label_missing",
                f"Missing label for image stem {missing}.",
                label_dir,
            )
        for extra in sorted(label_stems - image_stems):
            report.add(
                "ERROR",
                "image_missing_for_label",
                f"Missing image for label stem {extra}.",
                label_dir,
            )
        for label_path in labels:
            total_objects += _validate_label_file(label_path, max(1, class_count), report)

    if total_objects <= 0:
        report.add("ERROR", "no_player_annotations", "No player annotations were found.")
    train_stems = set()
    val_stems = set()
    if config.train_split in split_records and config.validation_split in split_records:
        train_images = _image_paths(Path(split_records[config.train_split]["image_dir"]))
        val_images = _image_paths(Path(split_records[config.validation_split]["image_dir"]))
        train_stems = {path.name for path in train_images}
        val_stems = {path.name for path in val_images}
        leakage = sorted(train_stems & val_stems)
        if leakage:
            report.add("ERROR", "split_leakage", f"Train/val leakage detected: {leakage[:10]}")
        train_sequences = {_sequence_name_from_image(path) for path in train_images}
        val_sequences = {_sequence_name_from_image(path) for path in val_images}
        sequence_leakage = sorted(train_sequences & val_sequences)
        if sequence_leakage:
            report.add(
                "ERROR",
                "sequence_split_leakage",
                f"Train/val sequence leakage detected: {sequence_leakage[:10]}",
            )
    if config.run_dir.exists() and not config.overwrite and not config.training.get("resume"):
        report.add(
            "ERROR",
            "output_exists",
            "Output run directory exists and overwrite=false.",
            config.run_dir,
        )
    config.output_project.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    if not config.output_project.exists():
        report.add(
            "ERROR",
            "output_not_writable",
            "Output project directory is not writable.",
            config.output_project,
        )
    known_weights = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
    if str(config.weights) not in known_weights:
        checkpoint = Path(str(config.weights))
        if not checkpoint.is_file():
            report.add("ERROR", "checkpoint_missing", "Checkpoint does not exist.", checkpoint)
    report.metadata.update(
        {
            "dataset_yaml": str(config.data_yaml),
            "class_count": class_count,
            "split_records": split_records,
            "ground_truth_count": total_objects,
            "train_val_overlap_count": len(train_stems & val_stems),
            "train_val_sequence_overlap_count": len(
                {
                    _sequence_name_from_image(path)
                    for path in _image_paths(Path(split_records[config.train_split]["image_dir"]))
                }
                & {
                    _sequence_name_from_image(path)
                    for path in _image_paths(
                        Path(split_records[config.validation_split]["image_dir"])
                    )
                }
            )
            if config.train_split in split_records and config.validation_split in split_records
            else 0,
        }
    )
    return write_preflight_report(config, report)


def write_preflight_report(config: TrainingConfig, report: PreflightReport) -> PreflightReport:
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    path = config.metrics_dir / "training_preflight.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    if _is_sportsmot_config(config):
        sportsmot_path = config.metrics_dir / "training_preflight_sportsmot.json"
        sportsmot_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    return report


def _summarize_training_result(result: Any) -> Any:
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        return {
            key: float(value) if isinstance(value, int | float) else value
            for key, value in results_dict.items()
        }
    if isinstance(result, str):
        return result
    text = str(result)
    return text if len(text) <= 1000 else f"{text[:1000]}..."


class YOLOv8Trainer:
    def __init__(self, config: TrainingConfig, model_factory: Any | None = None) -> None:
        self.config = config
        self.model_factory = model_factory
        self.model: Any | None = None

    def validate_environment(self) -> PreflightReport:
        return run_training_preflight(self.config)

    @classmethod
    def load_config(
        cls,
        config_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> YOLOv8Trainer:
        return cls(load_training_config(config_path, overrides=overrides))

    def resolve_device(self) -> str:
        return str(self.config.training.get("device", "auto"))

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model
        try:
            if self.model_factory is not None:
                self.model = self.model_factory(str(self.config.weights))
            else:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                self.model = YOLO(str(self.config.weights))
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"Failed to load YOLO model: {exc}") from exc
        return self.model

    def run_preflight(self) -> PreflightReport:
        return run_training_preflight(self.config)

    def train(self, dry_run: bool = False) -> dict[str, Any]:
        preflight = self.run_preflight()
        args = self.config.sanitized_train_args()
        if dry_run or self.config.dry_run:
            return {"dry_run": True, "preflight": preflight.to_dict(), "train_args": args}
        if preflight.has_errors:
            raise TrainingError(
                "Training preflight failed. See outputs/metrics/training_preflight.json."
            )
        manifest = ExperimentManifest(
            experiment_name=self.config.experiment_name,
            run_dir=self.config.run_dir,
            project_root=self.config.project_root,
            payload={
                "seed": self.config.seed,
                "deterministic": self.config.deterministic,
                "dataset_yaml": str(self.config.data_yaml),
                "training_config": str(self.config.config_path),
                "resolved_training_args": args,
                "initial_checkpoint": str(self.config.weights),
            },
        )
        manifest.write()
        started = time.perf_counter()
        try:
            result = self.load_model().train(**args)
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message += (
                    " Try reducing batch, using batch=-1, lowering imgsz from 960 to 640, "
                    "setting cache=false, or closing other GPU applications."
                )
            manifest.finish("failed", [message])
            raise TrainingError(message) from exc
        except Exception as exc:  # noqa: BLE001
            manifest.finish("failed", [str(exc)])
            raise TrainingError(f"Training failed: {exc}") from exc
        duration = time.perf_counter() - started
        artifacts = self.collect_artifacts()
        manifest.payload.update(
            {
                "training_duration_seconds": duration,
                "best_checkpoint": str(artifacts.get("best_checkpoint"))
                if artifacts.get("best_checkpoint")
                else None,
                "last_checkpoint": str(artifacts.get("last_checkpoint"))
                if artifacts.get("last_checkpoint")
                else None,
            }
        )
        manifest.finish("completed")
        try:
            copied = copy_final_checkpoints(
                self.config.run_dir,
                self.config.models_dir,
                self.config.run_name,
            )
        except CheckpointError:
            copied = {}
        return {
            "dry_run": False,
            "result": _summarize_training_result(result),
            "artifacts": artifacts,
            "copied": copied,
        }

    def resume(self, checkpoint: Path) -> dict[str, Any]:
        validate_checkpoint(checkpoint)
        config = self.config
        args = config.sanitized_train_args()
        args["resume"] = str(checkpoint)
        return {"checkpoint": str(checkpoint), "train_args": args}

    def collect_artifacts(self) -> dict[str, Path | None]:
        weights_dir = self.config.run_dir / "weights"
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"
        return {
            "run_dir": self.config.run_dir,
            "results_csv": self.config.run_dir / "results.csv",
            "best_checkpoint": best if best.is_file() else None,
            "last_checkpoint": last if last.is_file() else None,
            "config_snapshot": self.config.run_dir / "args.yaml",
        }

    def export_metadata(self) -> Path:
        manifest = ExperimentManifest(
            experiment_name=self.config.experiment_name,
            run_dir=self.config.run_dir,
            project_root=self.config.project_root,
            payload={"training_config": str(self.config.config_path)},
        )
        return manifest.write()
