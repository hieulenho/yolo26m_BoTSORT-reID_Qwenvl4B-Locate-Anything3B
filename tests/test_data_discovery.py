from pathlib import Path

import pytest

from football_tracking.data.discover import DatasetDiscoveryError, discover_sequence_candidates

FIXTURE_ROOT = Path("tests/fixtures/mini_tracking_dataset")


def test_discovers_sequences() -> None:
    candidates = discover_sequence_candidates(FIXTURE_ROOT)

    assert [candidate.name for candidate in candidates] == [
        "sequence_001",
        "sequence_002",
        "sequence_003",
    ]


def test_missing_annotation_reports_error(tmp_path: Path) -> None:
    frames_dir = tmp_path / "sequence_missing" / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "000001.ppm").write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    with pytest.raises(DatasetDiscoveryError, match="missing annotation"):
        discover_sequence_candidates(tmp_path)


def test_empty_raw_directory_reports_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetDiscoveryError, match="No sequence directories"):
        discover_sequence_candidates(tmp_path)
