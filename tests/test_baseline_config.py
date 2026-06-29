from pathlib import Path

import pytest
import yaml

from football_tracking.detection.baseline import BaselineConfigError, load_baseline_config


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    dataset = tmp_path / "dataset.yaml"
    images = tmp_path / "images" / "val"
    images.mkdir(parents=True)
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "train": "images/val",
                "test": "images/val",
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "model": {"weights": "yolov8m.pt", "task": "detect"},
        "dataset": {"data_yaml": str(dataset), "split": "val"},
        "inference": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 100,
            "device": "auto",
            "half": False,
            "batch": 1,
            "workers": 0,
            "verbose": False,
        },
        "classes": {
            "coco_person_class_id": 0,
            "target_class_id": 0,
            "target_class_name": "player",
            "keep_only_person": True,
        },
        "evaluation": {
            "use_ultralytics_validator": True,
            "save_predictions": True,
            "save_visualizations": False,
            "save_error_samples": False,
            "max_visualization_samples": 2,
        },
        "runtime": {
            "max_images": None,
            "max_sequences": None,
            "warmup_iterations": 0,
            "overwrite": True,
        },
        "output": {
            "root": str(tmp_path / "out"),
            "metrics_dir": str(tmp_path / "metrics"),
            "figures_dir": str(tmp_path / "figures"),
        },
    }
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split("__", 1)
        payload[section][key] = value
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_baseline_config_validates_values_and_overrides(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    config = load_baseline_config(path, overrides={"split": "test", "conf": 0.5})

    assert config.split == "test"
    assert config.conf == 0.5
    assert config.weights == "yolov8m.pt"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"inference__conf": 2.0}, "conf"),
        ({"inference__imgsz": 0}, "imgsz"),
        ({"dataset__split": "dev"}, "split"),
        ({"model__weights": "missing.pt"}, "weights"),
    ],
)
def test_baseline_config_rejects_invalid_values(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    path = _write_config(tmp_path, **override)

    with pytest.raises(BaselineConfigError, match=message):
        load_baseline_config(path)
