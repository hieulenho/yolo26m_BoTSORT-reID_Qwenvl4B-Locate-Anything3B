from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from football_tracking.team_benchmark.label_completion import (
    build_track_label_completion,
)


def _write_color_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (96, 64),
    )
    assert writer.isOpened()
    for _ in range(20):
        frame = np.full((64, 96, 3), (40, 120, 40), dtype=np.uint8)
        cv2.rectangle(frame, (5, 10), (20, 50), (220, 200, 120), -1)
        cv2.rectangle(frame, (27, 10), (42, 50), (150, 55, 30), -1)
        cv2.rectangle(frame, (49, 10), (64, 50), (35, 35, 35), -1)
        cv2.rectangle(frame, (71, 10), (86, 50), (35, 35, 35), -1)
        writer.write(frame)
    writer.release()


def test_label_completion_covers_every_track_and_propagates_referee(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    tracks = tmp_path / "tracks.txt"
    annotations = tmp_path / "annotations.csv"
    _write_color_video(video)

    rows: list[str] = []
    for frame in range(1, 21):
        for track_id, x in ((1, 5), (2, 27), (3, 49), (4, 71)):
            rows.append(f"{frame},{track_id},{x},10,16,40,0.9,1,1")
    tracks.write_text("\n".join(rows) + "\n", encoding="utf-8")
    annotations.write_text(
        "\n".join(
            [
                "sequence_name,track_id,start_frame,end_frame,team_label,role_label",
                "seq,1,1,20,light_blue,player",
                "seq,2,1,20,dark_blue,player",
                "seq,3,1,20,referee_black,referee",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_track_label_completion(
        sequence_name="seq",
        source_video=video,
        tracks_path=tracks,
        annotation_csv=annotations,
        samples_per_track=3,
    )
    predictions = {row["track_id"]: row for row in result["track_predictions"]}

    assert set(predictions) == {1, 2, 3, 4}
    assert all(row["status"] == "resolved" for row in predictions.values())
    assert predictions[4]["team_label"] == "referee_black"
    assert predictions[4]["role_label"] == "referee"
    assert predictions[4]["metadata"]["not_model_claim"] is True
    assert result["metadata"]["prediction_count"] == 4
    json.dumps(result)
