from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (60, 100):
        frame = np.full((48, 64, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_render_team_position_video_can_hide_unlabeled_tracks(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    tracks = tmp_path / "tracks.txt"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "rendered.mp4"

    _write_video(source)
    tracks.write_text(
        "\n".join(
            [
                "1,1,5,5,12,16,0.9,1,1",
                "1,2,30,6,12,16,0.9,1,1",
                "2,1,7,5,12,16,0.9,1,1",
                "2,2,32,6,12,16,0.9,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "variant_id": "test",
                "variant_name": "test",
                "benchmark_name": "test",
                "pipeline_type": "yolo_botsort_qwen",
                "track_predictions": [
                    {
                        "sequence_name": "seq",
                        "track_id": 1,
                        "status": "resolved",
                        "team_label": "light_blue",
                        "role_label": "player",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/legacy/render_team_position_video.py",
            "--source-video",
            str(source),
            "--tracks",
            str(tracks),
            "--predictions",
            str(predictions),
            "--sequence-name",
            "seq",
            "--output-video",
            str(output),
            "--hide-unlabeled",
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert output.is_file()
    assert "drawn_boxes" in result.stdout
    assert metadata["track_count_with_predictions"] == 1
    assert metadata["total_track_boxes"] == 4
    assert metadata["drawn_boxes"] == 2
    assert metadata["skipped_unlabeled_boxes"] == 2
    assert metadata["unlabeled_track_ids"] == [2]
