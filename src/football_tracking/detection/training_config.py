"""Training and evaluation configuration for YOLOv8m fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.detection.detector import KNOWN_ULTRALYTICS_CHECKPOINTS
from football_tracking.paths import get_project_root, resolve_project_path

VALID_SPLITS = {"train", "val", "test"}


class TrainingConfigError(RuntimeError):
    """Raised when training or evaluation config is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    project_root: Path
    config_path: Path
    experiment_name: str
    seed: int
    deterministic: bool
    weights: str | Path
    task: str
    data_yaml: Path
    train_split: str
    validation_split: str
    test_split: str
    training: dict[str, Any]
    augmentation: dict[str, Any]
    output_project: Path
    run_name: str
    models_dir: Path
    metrics_dir: Path
    figures_dir: Path
    overwrite: bool
    dry_run: bool
    smoke_test: bool
    max_train_images: int | None
    max_val_images: int | None
    log_level: str

    @property
    def run_dir(self) -> Path:
        return self.output_project / self.run_name

    def sanitized_train_args(self) -> dict[str, Any]:
        args = {
            "data": str(self.data_yaml),
            "epochs": self.training.get("epochs"),
            "imgsz": self.training.get("imgsz"),
            "batch": self.training.get("batch"),
            "patience": self.training.get("patience"),
            "device": None
            if self.training.get("device") == "auto"
            else self.training.get("device"),
            "workers": self.training.get("workers"),
            "fraction": self.training.get("fraction"),
            "val": self.training.get("val"),
            "cache": self.training.get("cache"),
            "pretrained": self.training.get("pretrained"),
            "optimizer": self.training.get("optimizer"),
            "lr0": self.training.get("lr0"),
            "lrf": self.training.get("lrf"),
            "weight_decay": self.training.get("weight_decay"),
            "warmup_epochs": self.training.get("warmup_epochs"),
            "close_mosaic": self.training.get("close_mosaic"),
            "amp": self.training.get("amp"),
            "resume": self.training.get("resume"),
            "save": self.training.get("save"),
            "save_period": self.training.get("save_period"),
            "plots": self.training.get("plots"),
            "verbose": self.training.get("verbose"),
            "project": str(self.output_project),
            "name": self.run_name,
            "exist_ok": self.overwrite or bool(self.training.get("resume")),
        }
        args.update(self.augmentation)
        return {key: value for key, value in args.items() if value is not None}


@dataclass(frozen=True)
class EvaluationConfig:
    project_root: Path
    config_path: Path
    weights: str | Path
    data_yaml: Path
    split: str
    evaluation: dict[str, Any]
    output_project: Path
    run_name: str
    metrics_dir: Path
    figures_dir: Path
    overwrite: bool
    max_images: int | None
    warmup_iterations: int
    log_level: str

    @property
    def output_prefix(self) -> str:
        return self.run_name

    def sanitized_val_args(self) -> dict[str, Any]:
        args = {
            "data": str(self.data_yaml),
            "split": self.split,
            "imgsz": self.evaluation.get("imgsz"),
            "batch": self.evaluation.get("batch"),
            "conf": self.evaluation.get("conf"),
            "iou": self.evaluation.get("iou"),
            "max_det": self.evaluation.get("max_det"),
            "device": None
            if self.evaluation.get("device") == "auto"
            else self.evaluation.get("device"),
            "half": self.evaluation.get("half"),
            "plots": self.evaluation.get("plots"),
            "save_json": self.evaluation.get("save_json"),
            "save_txt": self.evaluation.get("save_txt"),
            "save_conf": self.evaluation.get("save_conf"),
            "verbose": self.evaluation.get("verbose"),
            "project": str(self.output_project),
            "name": self.run_name,
            "exist_ok": self.overwrite,
        }
        return {key: value for key, value in args.items() if value is not None}


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{section} must be a non-empty path string.")
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else resolve_project_path(raw, project_root)


def _resolve_weights(value: Any, project_root: Path, require_exists: bool = False) -> str | Path:
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str) and value.strip():
        if value in KNOWN_ULTRALYTICS_CHECKPOINTS:
            return value
        raw = Path(value)
    else:
        raise TrainingConfigError("model.weights must be a non-empty string.")
    path = raw.resolve() if raw.is_absolute() else resolve_project_path(raw, project_root)
    if require_exists and not path.is_file():
        raise TrainingConfigError(f"Checkpoint does not exist: {path}")
    return path


def _optional_int(value: Any, name: str) -> int | None:
    if value in (None, "null", ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise TrainingConfigError(f"{name} must be positive when set.")
    return parsed


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingConfigError(f"Config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(loaded, "config root")


def _resolve_config_path(config_path: str | Path, project_root: Path) -> Path:
    raw = Path(config_path)
    return raw.resolve() if raw.is_absolute() else resolve_project_path(raw, project_root)


def _validate_batch(value: Any) -> int | float:
    if isinstance(value, bool):
        raise TrainingConfigError("training.batch must be numeric.")
    if isinstance(value, int):
        if value == -1 or value > 0:
            return value
        raise TrainingConfigError("training.batch must be positive, -1, or a float in (0, 1].")
    if isinstance(value, float):
        if 0.0 < value <= 1.0:
            return value
        raise TrainingConfigError("training.batch float must be in (0, 1].")
    raise TrainingConfigError("training.batch must be an int or float.")


def _validate_device(device: Any) -> str:
    value = str(device)
    lowered = value.lower()
    if lowered == "auto" or lowered == "cpu" or lowered == "cuda":
        return value
    if lowered.startswith("cuda:") and lowered.removeprefix("cuda:").isdigit():
        return value
    if value.isdigit():
        return value
    raise TrainingConfigError(f"Unsupported device value: {device}")


def _validate_fraction(value: Any) -> float:
    if isinstance(value, bool):
        raise TrainingConfigError("training.fraction must be numeric.")
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise TrainingConfigError("training.fraction must be in (0, 1].")
    return fraction


def _validate_training_values(training: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(training)
    epochs = int(cleaned.get("epochs", 100))
    if epochs <= 0:
        raise TrainingConfigError("training.epochs must be > 0.")
    imgsz = int(cleaned.get("imgsz", 960))
    if imgsz <= 0:
        raise TrainingConfigError("training.imgsz must be > 0.")
    patience = int(cleaned.get("patience", 20))
    if patience < 0:
        raise TrainingConfigError("training.patience must be >= 0.")
    workers = int(cleaned.get("workers", 4))
    if workers < 0:
        raise TrainingConfigError("training.workers must be >= 0.")
    cleaned["epochs"] = epochs
    cleaned["imgsz"] = imgsz
    cleaned["batch"] = _validate_batch(cleaned.get("batch", -1))
    cleaned["patience"] = patience
    cleaned["workers"] = workers
    cleaned["device"] = _validate_device(cleaned.get("device", "auto"))
    if cleaned.get("fraction") is not None:
        cleaned["fraction"] = _validate_fraction(cleaned["fraction"])
    if cleaned.get("val") is not None:
        cleaned["val"] = bool(cleaned["val"])
    return cleaned


def _validate_eval_values(evaluation: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(evaluation)
    imgsz = int(cleaned.get("imgsz", 960))
    batch = int(cleaned.get("batch", 4))
    max_det = int(cleaned.get("max_det", 300))
    if imgsz <= 0:
        raise TrainingConfigError("evaluation.imgsz must be > 0.")
    if batch <= 0:
        raise TrainingConfigError("evaluation.batch must be > 0.")
    if max_det <= 0:
        raise TrainingConfigError("evaluation.max_det must be > 0.")
    conf = float(cleaned.get("conf", 0.001))
    iou = float(cleaned.get("iou", 0.70))
    if not 0.0 <= conf <= 1.0:
        raise TrainingConfigError("evaluation.conf must be in [0, 1].")
    if not 0.0 <= iou <= 1.0:
        raise TrainingConfigError("evaluation.iou must be in [0, 1].")
    cleaned["imgsz"] = imgsz
    cleaned["batch"] = batch
    cleaned["max_det"] = max_det
    cleaned["conf"] = conf
    cleaned["iou"] = iou
    cleaned["device"] = _validate_device(cleaned.get("device", "auto"))
    return cleaned


def load_training_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    project_root = get_project_root()
    resolved_config = _resolve_config_path(config_path, project_root)
    root = _load_yaml(resolved_config)
    experiment = _mapping(root.get("experiment"), "experiment")
    model = _mapping(root.get("model"), "model")
    dataset = _mapping(root.get("dataset"), "dataset")
    training = _validate_training_values(_mapping(root.get("training"), "training"))
    augmentation = dict(_mapping(root.get("augmentation", {}), "augmentation"))
    output = _mapping(root.get("output"), "output")
    runtime = _mapping(root.get("runtime", {}), "runtime")

    config = TrainingConfig(
        project_root=project_root,
        config_path=resolved_config,
        experiment_name=str(experiment.get("name", "yolov8m_players_finetuned")),
        seed=int(experiment.get("seed", 42)),
        deterministic=bool(experiment.get("deterministic", True)),
        weights=_resolve_weights(model.get("weights", "yolov8m.pt"), project_root),
        task=str(model.get("task", "detect")),
        data_yaml=_resolve_path(dataset.get("data_yaml"), project_root, "dataset.data_yaml"),
        train_split=str(dataset.get("train_split", "train")),
        validation_split=str(dataset.get("validation_split", "val")),
        test_split=str(dataset.get("test_split", "test")),
        training=training,
        augmentation=augmentation,
        output_project=_resolve_path(output.get("project"), project_root, "output.project"),
        run_name=str(output.get("run_name", "yolov8m_players")),
        models_dir=_resolve_path(
            output.get("models_dir", "models/detector"),
            project_root,
            "output.models_dir",
        ),
        metrics_dir=_resolve_path(
            output.get("metrics_dir", "outputs/metrics"),
            project_root,
            "output.metrics_dir",
        ),
        figures_dir=_resolve_path(
            output.get("figures_dir", "outputs/figures/yolov8m_finetuned"),
            project_root,
            "output.figures_dir",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        dry_run=bool(runtime.get("dry_run", False)),
        smoke_test=bool(runtime.get("smoke_test", False)),
        max_train_images=_optional_int(runtime.get("max_train_images"), "runtime.max_train_images"),
        max_val_images=_optional_int(runtime.get("max_val_images"), "runtime.max_val_images"),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if config.train_split not in VALID_SPLITS or config.validation_split not in VALID_SPLITS:
        raise TrainingConfigError("dataset train/validation split must be train, val, or test.")
    if overrides:
        config = apply_training_overrides(config, overrides)
    return config


def apply_training_overrides(config: TrainingConfig, overrides: dict[str, Any]) -> TrainingConfig:
    training = dict(config.training)
    if overrides.get("device") is not None:
        training["device"] = _validate_device(overrides["device"])
    if overrides.get("epochs") is not None:
        training["epochs"] = int(overrides["epochs"])
    if overrides.get("batch") is not None:
        training["batch"] = _validate_batch(overrides["batch"])
    if overrides.get("imgsz") is not None:
        training["imgsz"] = int(overrides["imgsz"])
    if overrides.get("workers") is not None:
        training["workers"] = int(overrides["workers"])
    if overrides.get("fraction") is not None:
        training["fraction"] = _validate_fraction(overrides["fraction"])
    if overrides.get("val") is not None:
        training["val"] = bool(overrides["val"])
    config = replace(config, training=_validate_training_values(training))
    if overrides.get("overwrite"):
        config = replace(config, overwrite=True)
    if overrides.get("dry_run"):
        config = replace(config, dry_run=True)
    if overrides.get("resume") is not None:
        training = dict(config.training)
        training["resume"] = bool(overrides["resume"])
        config = replace(config, training=training)
    return config


def load_evaluation_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> EvaluationConfig:
    project_root = get_project_root()
    resolved_config = _resolve_config_path(config_path, project_root)
    root = _load_yaml(resolved_config)
    model = _mapping(root.get("model"), "model")
    dataset = _mapping(root.get("dataset"), "dataset")
    evaluation = _validate_eval_values(_mapping(root.get("evaluation"), "evaluation"))
    output = _mapping(root.get("output"), "output")
    runtime = _mapping(root.get("runtime", {}), "runtime")

    config = EvaluationConfig(
        project_root=project_root,
        config_path=resolved_config,
        weights=_resolve_weights(model.get("weights"), project_root, require_exists=False),
        data_yaml=_resolve_path(dataset.get("data_yaml"), project_root, "dataset.data_yaml"),
        split=str(dataset.get("split", "val")),
        evaluation=evaluation,
        output_project=_resolve_path(output.get("project"), project_root, "output.project"),
        run_name=str(output.get("run_name", "yolov8m_finetuned_val")),
        metrics_dir=_resolve_path(
            output.get("metrics_dir", "outputs/metrics"),
            project_root,
            "output.metrics_dir",
        ),
        figures_dir=_resolve_path(
            output.get("figures_dir", "outputs/figures/yolov8m_finetuned"),
            project_root,
            "output.figures_dir",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        max_images=_optional_int(runtime.get("max_images"), "runtime.max_images"),
        warmup_iterations=int(runtime.get("warmup_iterations", 3)),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if overrides:
        config = apply_evaluation_overrides(config, overrides)
    if config.split not in VALID_SPLITS:
        raise TrainingConfigError("dataset.split must be train, val, or test.")
    return config


def apply_evaluation_overrides(
    config: EvaluationConfig,
    overrides: dict[str, Any],
) -> EvaluationConfig:
    evaluation = dict(config.evaluation)
    if overrides.get("checkpoint") is not None:
        config = replace(
            config,
            weights=_resolve_weights(overrides["checkpoint"], config.project_root),
        )
    if overrides.get("device") is not None:
        evaluation["device"] = _validate_device(overrides["device"])
    if overrides.get("batch") is not None:
        evaluation["batch"] = int(overrides["batch"])
    if overrides.get("imgsz") is not None:
        evaluation["imgsz"] = int(overrides["imgsz"])
    if overrides.get("split") is not None:
        config = replace(config, split=str(overrides["split"]))
    if overrides.get("max_images") is not None:
        config = replace(config, max_images=int(overrides["max_images"]))
    if overrides.get("overwrite"):
        config = replace(config, overwrite=True)
    return replace(config, evaluation=_validate_eval_values(evaluation))


def load_yolo_dataset_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingConfigError(f"Dataset YAML does not exist: {path}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "YOLO dataset YAML")


def resolve_yolo_split_path(data_yaml: Path, dataset_yaml: dict[str, Any], split: str) -> Path:
    if split not in dataset_yaml:
        raise TrainingConfigError(f"Dataset YAML does not define split: {split}")
    root = Path(str(dataset_yaml.get("path", data_yaml.parent)))
    root = root if root.is_absolute() else (data_yaml.parent / root).resolve()
    value = dataset_yaml[split]
    split_path = Path(str(value[0] if isinstance(value, list) else value))
    return split_path if split_path.is_absolute() else root / split_path
