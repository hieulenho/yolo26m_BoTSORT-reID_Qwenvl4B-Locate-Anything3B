from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.cli.__main__ import main
from tests.locate_tracking.appearance_test_utils import (
    one_track_semantic_memory_fixture,
    tiny_tracks,
    tiny_video,
)


def test_verify_language_track_cli_smoke_with_mock_backend(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "source.avi")
    tracks = tiny_tracks(tmp_path / "tracks.txt")
    semantic = one_track_semantic_memory_fixture(tmp_path / "semantic_memory.json")
    output = tmp_path / "appearance"

    code = main(
        [
            "verify-language-track",
            "--source-video",
            str(video),
            "--tracks",
            str(tracks),
            "--semantic-memory",
            str(semantic),
            "--output-dir",
            str(output),
            "--backend",
            "mock",
            "--overwrite",
        ]
    )

    assert code == 0
    assert (output / "appearance_manifest.json").is_file()
    assert (output / "appearance_scores.json").is_file()
    assert (output / "fusion_result.json").is_file()
    assert (output / "appearance_summary.md").is_file()
    assert json.loads((output / "fusion_result.json").read_text())["status"] == "resolved"


def test_verify_language_track_cli_missing_semantic_memory_returns_error(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "source.avi")
    tracks = tiny_tracks(tmp_path / "tracks.txt")

    code = main(
        [
            "verify-language-track",
            "--source-video",
            str(video),
            "--tracks",
            str(tracks),
            "--semantic-memory",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "appearance"),
            "--backend",
            "mock",
        ]
    )

    assert code == 2


def test_verify_language_track_cli_missing_source_video_returns_error(tmp_path: Path) -> None:
    tracks = tiny_tracks(tmp_path / "tracks.txt")
    semantic = one_track_semantic_memory_fixture(tmp_path / "semantic_memory.json")

    code = main(
        [
            "verify-language-track",
            "--source-video",
            str(tmp_path / "missing.avi"),
            "--tracks",
            str(tracks),
            "--semantic-memory",
            str(semantic),
            "--output-dir",
            str(tmp_path / "appearance"),
            "--backend",
            "mock",
        ]
    )

    assert code == 2
