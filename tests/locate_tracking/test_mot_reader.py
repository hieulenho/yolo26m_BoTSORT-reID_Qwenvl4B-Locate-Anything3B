from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.locate_tracking.artifacts.mot_reader import (
    MotReaderError,
    read_mot_track_file,
)


def _mot(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_mot_reader_valid_row_and_ltwh_to_xyxy(tmp_path: Path) -> None:
    mot = read_mot_track_file(_mot(tmp_path / "tracks.txt", "1,7,10,20,30,40,0.9,1,1\n"))

    row = mot.observations[0]
    assert row.frame_index == 1
    assert row.track_id == 7
    assert row.bbox_ltwh == (10.0, 20.0, 30.0, 40.0)
    assert row.bbox_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert row.confidence == 0.9


def test_mot_reader_multiple_rows_blank_lines_and_sorting(tmp_path: Path) -> None:
    mot = read_mot_track_file(
        _mot(
            tmp_path / "tracks.txt",
            "\n2,3,0,0,10,10,-1,1,1\n1,2,0,0,10,10,0.5,1,1\n",
        )
    )

    assert [(row.frame_index, row.track_id) for row in mot.observations] == [(1, 2), (2, 3)]
    assert mot.observations[1].confidence is None


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("1,1,0,0,10", "at least 6"),
        ("1,1,abc,0,10,10", "left must be numeric"),
        ("0,1,0,0,10,10", "frame_index"),
        ("1,-1,0,0,10,10", "track_id"),
        ("1,1,0,0,0,10", "width"),
        ("1,1,0,0,10,-1", "height"),
        ("1,1,nan,0,10,10", "finite"),
        ("1,1,inf,0,10,10", "finite"),
    ],
)
def test_mot_reader_rejects_malformed_rows(tmp_path: Path, line: str, reason: str) -> None:
    with pytest.raises(MotReaderError, match=reason):
        read_mot_track_file(_mot(tmp_path / "tracks.txt", f"{line}\n"))


def test_mot_reader_rejects_duplicate_frame_track_pair(tmp_path: Path) -> None:
    with pytest.raises(MotReaderError, match="duplicate"):
        read_mot_track_file(_mot(tmp_path / "tracks.txt", "1,1,0,0,10,10,-1\n1,1,2,2,10,10,-1\n"))
