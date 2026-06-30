from pathlib import Path

import pytest
import yaml

from football_tracking.detection.training_config import (
    TrainingConfigError,
    load_training_config,
)


def _dataset_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "player"},
                "nc": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, **updates: object) -> Path:
    payload = {
        "experiment": {"name": "test", "seed": 42, "deterministic": True},
        "model": {"weights": "yolov8m.pt", "task": "detect"},
        "dataset": {
            "data_yaml": str(_dataset_yaml(tmp_path)),
            "train_split": "train",
            "validation_split": "val",
            "test_split": "test",
        },
        "training": {
            "epochs": 1,
            "imgsz": 640,
            "batch": -1,
            "patience": 1,
            "device": "auto",
            "workers": 0,
            "cache": False,
            "pretrained": True,
            "optimizer": "auto",
            "lr0": None,
            "amp": True,
            "plots": False,
        },
        "augmentation": {"degrees": 0.0, "mixup": 0.0, "mosaic": None},
        "output": {
            "project": str(tmp_path / "training"),
            "run_name": "run",
            "models_dir": str(tmp_path / "models"),
            "metrics_dir": str(tmp_path / "metrics"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "runtime": {"overwrite": True, "dry_run": False, "smoke_test": True},
    }
    for key, value in updates.items():
        section, field = key.split("__", 1)
        payload[section][field] = value
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_training_config_sanitizes_null_fields_and_accepts_auto_batch(tmp_path: Path) -> None:
    config = load_training_config(_config(tmp_path))
    args = config.sanitized_train_args()

    assert args["batch"] == -1
    assert "lr0" not in args
    assert "mosaic" not in args


def test_training_config_accepts_fractional_batch(tmp_path: Path) -> None:
    config = load_training_config(_config(tmp_path, training__batch=0.7))

    assert config.training["batch"] == 0.7


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"training__epochs": 0}, "epochs"),
        ({"training__imgsz": 0}, "imgsz"),
        ({"training__batch": 0}, "batch"),
        ({"dataset__validation_split": "dev"}, "split"),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TrainingConfigError, match=message):
        load_training_config(_config(tmp_path, **updates))
