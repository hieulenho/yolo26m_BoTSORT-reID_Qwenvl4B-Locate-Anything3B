from __future__ import annotations

from pathlib import Path

import numpy as np

from football_tracking.tracking.sequence_runner import (
    discover_mot_sequences,
    iter_mot_frames,
    mot_sequence_source,
    read_seqmap,
)


def _write_sequence(root: Path, name: str = "seq") -> Path:
    import cv2  # type: ignore[import-not-found]

    sequence_dir = root / name
    img_dir = sequence_dir / "img1"
    img_dir.mkdir(parents=True)
    for frame in (1, 2):
        cv2.imwrite(str(img_dir / f"{frame:06d}.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={name}",
                "imDir=img1",
                "frameRate=25",
                "seqLength=2",
                "imWidth=30",
                "imHeight=20",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    return sequence_dir


def test_mot_sequence_source_reads_seqinfo_and_frames_in_order(tmp_path) -> None:
    sequence_dir = _write_sequence(tmp_path)

    source = mot_sequence_source(sequence_dir)
    frames = list(iter_mot_frames(source))

    assert source.name == "seq"
    assert source.fps == 25
    assert [frame.frame_index for frame in frames] == [1, 2]


def test_discover_mot_sequences_uses_seqmap(tmp_path) -> None:
    split_dir = tmp_path / "val"
    _write_sequence(split_dir, "seq_b")
    _write_sequence(split_dir, "seq_a")
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text("name\nseq_a\n", encoding="utf-8")

    assert read_seqmap(seqmap) == ["seq_a"]
    sources = discover_mot_sequences(tmp_path, "val", seqmap)

    assert [source.name for source in sources] == ["seq_a"]
