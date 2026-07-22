from pathlib import Path

from football_tracking.data.prepare import prepare_data


def test_prepare_data_fixture_creates_yolo_and_mot_outputs() -> None:
    result = prepare_data("configs/legacy/football/data_test.yaml", overwrite=True, fail_fast=False)

    assert result.validation_report.error_count == 0
    assert result.yolo_stats["images"] == 9
    assert result.mot_stats["sequences"] == 3
    assert Path("data/yolo/mini_tracking_fixture/dataset.yaml").is_file()
    assert Path("data/mot/mini_tracking_fixture/seqmaps/train.txt").is_file()
    assert Path("data/interim/mini_tracking_fixture/dataset_manifest.json").is_file()
    assert result.visualization_paths


def test_prepare_data_dry_run_does_not_write_outputs() -> None:
    result = prepare_data(
        "configs/legacy/football/data_test.yaml", dry_run=True, overwrite=True, fail_fast=False
    )

    assert result.dry_run is True
    assert result.yolo_stats["dry_run"] is True
