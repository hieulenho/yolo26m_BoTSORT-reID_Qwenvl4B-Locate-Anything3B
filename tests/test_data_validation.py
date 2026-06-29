from pathlib import Path

from football_tracking.data.schemas import SplitManifest
from football_tracking.data.validate import (
    validate_mot_dataset,
    validate_sequences,
    validate_split_leakage,
)
from tests.test_yolo_conversion import _load_sequences


def test_validation_detects_duplicate_frame_track() -> None:
    report = validate_sequences(
        _load_sequences(),
        invalid_box_policy="warn_and_skip",
        fail_on_duplicate_track_in_frame=True,
    )

    assert any(issue.code == "duplicate_frame_track" for issue in report.issues)


def test_validation_detects_invalid_bbox() -> None:
    report = validate_sequences(
        _load_sequences(),
        invalid_box_policy="fail",
        fail_on_duplicate_track_in_frame=False,
    )

    assert any(
        issue.code == "invalid_bbox" and issue.severity == "ERROR" for issue in report.issues
    )


def test_validation_detects_split_leakage() -> None:
    split = SplitManifest(1, "sequence", ["sequence_001"], ["sequence_001"], [])

    report = validate_split_leakage(split)

    assert report.has_errors


def test_validation_detects_bad_mot_frame_index(tmp_path: Path) -> None:
    gt_dir = tmp_path / "train" / "sequence_001" / "gt"
    gt_dir.mkdir(parents=True)
    (gt_dir / "gt.txt").write_text("0,1,1,1,10,10,1,1,1\n", encoding="utf-8")

    report = validate_mot_dataset(tmp_path)

    assert any(issue.code == "mot_range" for issue in report.issues)
