from __future__ import annotations

import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.mot_writer import (
    MotPredictionWriter,
    MotWriterError,
    format_mot_row,
)
from football_tracking.tracking.schemas import TrackOutput


def _track(frame: int, track_id: int, confidence: float | None = 0.9) -> TrackOutput:
    return TrackOutput.from_xyxy(
        frame_index=frame,
        sequence_name="seq",
        track_id=track_id,
        bbox_xyxy=BoundingBoxXYXY(10, 20, 40, 70),
        confidence=confidence,
    )


def test_format_mot_row_has_nine_fields_and_null_confidence_is_minus_one() -> None:
    row = format_mot_row(_track(1, 7, None))
    fields = row.split(",")

    assert len(fields) == 9
    assert fields[:2] == ["1", "7"]
    assert fields[6] == "-1.000000"
    assert fields[7:] == ["1", "1.000000"]


def test_mot_writer_sorts_rows_and_rejects_duplicates(tmp_path) -> None:
    writer = MotPredictionWriter(tmp_path / "seq.txt")
    writer.add_tracks([_track(2, 1), _track(1, 1)])
    writer.write()

    assert writer.path.read_text(encoding="utf-8").splitlines()[0].startswith("1,1,")

    duplicate = MotPredictionWriter(tmp_path / "dup.txt")
    duplicate.add_tracks([_track(1, 1), _track(1, 1)])
    with pytest.raises(MotWriterError, match="Duplicate"):
        duplicate.write()
