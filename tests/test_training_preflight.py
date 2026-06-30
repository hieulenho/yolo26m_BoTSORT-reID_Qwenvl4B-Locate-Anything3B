from pathlib import Path

import yaml

from football_tracking.detection.trainer import run_training_preflight
from football_tracking.detection.training_config import load_training_config
from tests.test_training_config import _config


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="ascii")


def _valid_dataset(tmp_path: Path) -> None:
    for split in ("train", "val"):
        _write_ppm(tmp_path / "images" / split / f"{split}_001.ppm")
        label = tmp_path / "labels" / split / f"{split}_001.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def test_preflight_passes_cpu_when_dataset_is_valid(tmp_path: Path) -> None:
    _valid_dataset(tmp_path)
    config = load_training_config(_config(tmp_path))

    report = run_training_preflight(config)

    assert report.error_count == 0
    assert report.metadata["ground_truth_count"] == 2


def test_preflight_reports_missing_dataset_yaml(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["dataset"]["data_yaml"] = str(tmp_path / "missing.yaml")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_training_config(config_path)

    report = run_training_preflight(config)

    assert report.has_errors
    assert any(issue.code == "dataset_yaml_missing" for issue in report.issues)


def test_preflight_reports_malformed_label_and_leakage(tmp_path: Path) -> None:
    _write_ppm(tmp_path / "images" / "train" / "same.ppm")
    _write_ppm(tmp_path / "images" / "val" / "same.ppm")
    for split in ("train", "val"):
        label = tmp_path / "labels" / split / "same.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("5 2.0 0.5 1.0\n", encoding="utf-8")
    config = load_training_config(_config(tmp_path))

    report = run_training_preflight(config)

    codes = {issue.code for issue in report.issues}
    assert "label_malformed" in codes
    assert "split_leakage" in codes
