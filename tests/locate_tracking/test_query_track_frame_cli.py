from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from football_tracking.locate_tracking.cli.__main__ import main


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 32),
    )
    assert writer.isOpened()
    writer.write(np.full((32, 32, 3), 120, dtype=np.uint8))
    writer.release()
    return path


def _tracks(path: Path) -> Path:
    path.write_text("1,7,3,3,10,10,-1,1,1\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grounding_config(path: Path, cache_dir: Path) -> Path:
    path.write_text(
        f"""
backend:
  name: mock
  model_id: mock-grounding
  mock_response: "<ref>player</ref><box><90><90><410><410></box>"
cache:
  enabled: true
  directory: "{cache_dir.as_posix()}"
output:
  directory: "{(path.parent / "grounding").as_posix()}"
runtime:
  overwrite: true
""",
        encoding="utf-8",
    )
    return path


def _association_config(path: Path, output_dir: Path) -> Path:
    path.write_text(
        f"""
association:
  min_iou: 0.10
  min_track_coverage: 0.50
  score:
    iou_weight: 0.70
    track_coverage_weight: 0.20
    center_similarity_weight: 0.10
  decision:
    min_score: 0.20
    ambiguity_margin: 0.05
    top_k: 5
geometry:
  clip_tracks_to_frame: true
output:
  directory: "{output_dir.as_posix()}"
  save_candidates: true
  save_overlay: false
runtime:
  overwrite: true
""",
        encoding="utf-8",
    )
    return path


def test_match_grounding_frame_cli_smoke_and_keeps_mot_unchanged(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
    grounding_config = _grounding_config(tmp_path / "grounding.yaml", tmp_path / "cache")
    grounding_output = tmp_path / "grounding.json"
    assert (
        main(
            [
                "locate-image",
                "--config",
                str(grounding_config),
                "--image",
                str(image),
                "--query",
                "player",
                "--output",
                str(grounding_output),
                "--overwrite",
            ]
        )
        == 0
    )
    tracks = _tracks(tmp_path / "tracks.txt")
    before = _sha(tracks)
    association_config = _association_config(tmp_path / "assoc.yaml", tmp_path / "queries")
    output = tmp_path / "association.json"

    assert (
        main(
            [
                "match-grounding-frame",
                "--association-config",
                str(association_config),
                "--grounding-result",
                str(grounding_output),
                "--tracks",
                str(tracks),
                "--frame-index",
                "1",
                "--frame-width",
                "32",
                "--frame-height",
                "32",
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        == 0
    )

    assert output.is_file()
    assert _sha(tracks) == before


def test_query_track_frame_cli_smoke(tmp_path: Path) -> None:
    grounding_config = _grounding_config(tmp_path / "grounding.yaml", tmp_path / "cache")
    association_config = _association_config(tmp_path / "assoc.yaml", tmp_path / "queries")
    output_dir = tmp_path / "query"

    assert (
        main(
            [
                "query-track-frame",
                "--grounding-config",
                str(grounding_config),
                "--association-config",
                str(association_config),
                "--source-video",
                str(_video(tmp_path / "video.avi")),
                "--tracks",
                str(_tracks(tmp_path / "tracks.txt")),
                "--frame-index",
                "1",
                "--query",
                "player",
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]
        )
        == 0
    )

    assert (output_dir / "association.json").is_file()
    assert (output_dir / "grounding.json").is_file()
    assert (output_dir / "frame_000001.jpg").is_file()
