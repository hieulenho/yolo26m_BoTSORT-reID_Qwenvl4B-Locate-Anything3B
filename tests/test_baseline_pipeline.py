import json
import shutil
from pathlib import Path

import yaml

from football_tracking.detection.baseline import run_baseline


class _FakeDetector:
    def __init__(self) -> None:
        self.load_count = 0

    def load_model(self) -> None:
        self.load_count += 1

    def predict_batch(
        self,
        image_paths: list[Path],
        **_kwargs: object,
    ) -> list[list[dict[str, object]]]:
        return [
            [
                {"xyxy": [0, 0, 1, 1], "conf": 0.9, "cls": 0},
                {"xyxy": [1, 1, 5, 5], "conf": 0.7, "cls": 32},
            ]
            for _path in image_paths
        ]


def _baseline_config(tmp_path: Path) -> Path:
    image_root = tmp_path / "images" / "val"
    image_root.mkdir(parents=True)
    shutil.copy2(
        Path("tests/fixtures/mini_tracking_dataset/sequence_001/frames/000001.ppm"),
        image_root / "sequence_001_000001.ppm",
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/val",
                "val": "images/val",
                "test": "images/val",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "model": {"weights": "yolov8m.pt", "task": "detect"},
        "dataset": {"data_yaml": str(dataset_yaml), "split": "val"},
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
            "save_visualizations": True,
            "save_error_samples": True,
            "max_visualization_samples": 3,
        },
        "runtime": {
            "max_images": None,
            "max_sequences": None,
            "warmup_iterations": 0,
            "overwrite": True,
        },
        "output": {
            "root": str(tmp_path / "detections"),
            "metrics_dir": str(tmp_path / "metrics"),
            "figures_dir": str(tmp_path / "figures"),
        },
    }
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_baseline_pipeline_with_fake_detector_writes_predictions_and_report(tmp_path: Path) -> None:
    detector = _FakeDetector()
    result = run_baseline(_baseline_config(tmp_path), detector=detector, evaluate=False)

    predictions = tmp_path / "detections" / "predictions.jsonl"
    report = tmp_path / "metrics" / "yolov8m_pretrained_report.md"
    payload = json.loads(predictions.read_text(encoding="utf-8").splitlines()[0])

    assert detector.load_count == 1
    assert result["prediction_count"] == 1
    assert payload["target_class_name"] == "player"
    assert report.is_file()
    assert (tmp_path / "figures" / "predictions" / "sequence_001_000001.ppm").is_file()
