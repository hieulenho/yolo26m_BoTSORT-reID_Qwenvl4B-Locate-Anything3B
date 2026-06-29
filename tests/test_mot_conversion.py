from pathlib import Path

from football_tracking.data.convert_mot import convert_to_mot
from football_tracking.data.schemas import SplitManifest
from tests.test_yolo_conversion import _load_sequences


def test_mot_conversion_outputs_gt_with_frame_base_one(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], [], [])

    convert_to_mot(sequences[:1], split, tmp_path, image_extension=".ppm", overwrite=True)

    gt_path = tmp_path / "train" / "sequence_001" / "gt" / "gt.txt"
    first_line = gt_path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "1,1,10.00,8.00,12.00,26.00,1,1,0.95"
    assert len(first_line.split(",")) == 9


def test_mot_conversion_writes_seqinfo_and_seqmap(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], [], [])

    convert_to_mot(sequences[:1], split, tmp_path, image_extension=".ppm", overwrite=True)

    seqinfo = (tmp_path / "train" / "sequence_001" / "seqinfo.ini").read_text(encoding="utf-8")
    seqmap = (tmp_path / "seqmaps" / "train.txt").read_text(encoding="utf-8")
    assert "name=sequence_001" in seqinfo
    assert "seqLength=3" in seqinfo
    assert seqmap.splitlines() == ["name", "sequence_001"]


def test_mot_conversion_sorts_by_frame_and_track_id(tmp_path: Path) -> None:
    sequences = _load_sequences()
    split = SplitManifest(1, "sequence", ["sequence_001"], [], [])

    convert_to_mot(sequences[:1], split, tmp_path, image_extension=".ppm", overwrite=True)

    rows = [
        tuple(int(value) for value in line.split(",")[:2])
        for line in (tmp_path / "train" / "sequence_001" / "gt" / "gt.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows == sorted(rows)
