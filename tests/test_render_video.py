from __future__ import annotations

from pathlib import Path

import numpy as np

from football_tracking.rendering.video_renderer import render_videos


def _write_mot_sequence(root: Path, name: str = "seq") -> None:
    import cv2  # type: ignore[import-not-found]

    sequence_dir = root / "val" / name
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)
    for frame in (1, 2):
        image = np.full((32, 48, 3), frame * 30, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"{frame:06d}.jpg"), image)
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={name}",
                "imDir=img1",
                "frameRate=25",
                "seqLength=2",
                "imWidth=48",
                "imHeight=32",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )


def test_render_videos_writes_annotated_mp4(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_mot_sequence(tmp_path / "data" / "mot" / "sportsmot_football")
    seqmap = tmp_path / "data" / "mot" / "sportsmot_football" / "seqmaps" / "val.txt"
    seqmap.parent.mkdir(parents=True)
    seqmap.write_text("name\nseq\n", encoding="utf-8")
    track_dir = tmp_path / "outputs" / "tracks" / "comparison" / "deepsort" / "val"
    track_dir.mkdir(parents=True)
    (track_dir / "seq.txt").write_text(
        "1,1,5,6,12,10,0.9,1,1\n2,1,7,6,12,10,0.8,1,1\n",
        encoding="utf-8",
    )
    config = tmp_path / "render.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset:",
                "  mot_root: data/mot/sportsmot_football",
                "  split: val",
                "  seqmap: data/mot/sportsmot_football/seqmaps/val.txt",
                "tracking:",
                "  tracker_name: deepsort",
                "  tracks_root: outputs/tracks/comparison",
                "output:",
                "  videos_root: outputs/videos/rendered",
                "runtime:",
                "  max_sequences: 1",
                "  max_frames_per_sequence: 2",
                "  overwrite: true",
            ]
        ),
        encoding="utf-8",
    )

    result = render_videos(config)

    video_path = tmp_path / "outputs" / "videos" / "rendered" / "deepsort" / "val" / "seq.mp4"
    assert result["sequence_count"] == 1
    assert video_path.is_file()
    assert video_path.with_suffix(".metadata.json").is_file()
