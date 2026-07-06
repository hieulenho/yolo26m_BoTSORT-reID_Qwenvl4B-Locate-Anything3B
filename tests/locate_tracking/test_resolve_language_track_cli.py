from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.cli.__main__ import main
from tests.locate_tracking.semantic_test_utils import resolved_frame


def _write_resolution(path: Path, frame_index: int, track_id: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(resolved_frame(frame_index, track_id).to_dict()),
        encoding="utf-8",
    )
    return path


def test_aggregate_language_track_cli_smoke(tmp_path: Path) -> None:
    first = _write_resolution(tmp_path / "f1" / "association.json", 1, 7)
    second = _write_resolution(tmp_path / "f2" / "association.json", 2, 7)
    output = tmp_path / "semantic"

    code = main(
        [
            "aggregate-language-track",
            "--query",
            "player",
            "--frame-resolution",
            str(first),
            "--frame-resolution",
            str(second),
            "--sampled-frames",
            "1,2",
            "--output-dir",
            str(output),
            "--min-usable-frames",
            "1",
            "--min-support-frames",
            "1",
            "--overwrite",
        ]
    )

    assert code == 0
    assert (output / "semantic_memory.json").is_file()
    assert (output / "final_resolution.json").is_file()
    assert (output / "session.json").is_file()
    assert (output / "semantic_summary.md").is_file()
    final = json.loads((output / "final_resolution.json").read_text(encoding="utf-8"))
    assert final["status"] == "resolved"
    assert final["selected_track_id"] == 7


def test_aggregate_language_track_cli_missing_artifact_returns_error(tmp_path: Path) -> None:
    code = main(
        [
            "aggregate-language-track",
            "--query",
            "player",
            "--frame-resolution",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "semantic"),
        ]
    )

    assert code == 2


def test_aggregate_language_track_cli_invalid_frame_list_returns_error(tmp_path: Path) -> None:
    first = _write_resolution(tmp_path / "f1" / "association.json", 1, 7)
    code = main(
        [
            "aggregate-language-track",
            "--query",
            "player",
            "--frame-resolution",
            str(first),
            "--sampled-frames",
            "1,bad",
            "--output-dir",
            str(tmp_path / "semantic"),
        ]
    )

    assert code == 2
