from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.cli.__main__ import main
from tests.locate_tracking.appearance_test_utils import tiny_video
from tests.locate_tracking.test_uncertainty_monitoring import (
    _appearance,
    _fusion,
    _semantic,
    _write_tracks,
)


def test_analyze_target_uncertainty_cli_smoke(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "video.avi", frame_count=10)
    tracks = _write_tracks(tmp_path / "tracks.txt", confidence=True)
    semantic = _semantic(tmp_path / "semantic_memory.json")
    appearance = _appearance(tmp_path / "appearance.json")
    fusion = _fusion(tmp_path / "fusion.json")
    output = tmp_path / "out"

    code = main(
        [
            "analyze-target-uncertainty",
            "--source-video",
            str(video),
            "--tracks",
            str(tracks),
            "--semantic-memory",
            str(semantic),
            "--appearance-result",
            str(appearance),
            "--fusion-result",
            str(fusion),
            "--output-dir",
            str(output),
            "--current-track-id",
            "7",
            "--overwrite",
        ]
    )

    assert code == 0
    assert (output / "uncertainty_events.jsonl").is_file()


def test_plan_event_grounding_cli_from_jsonl(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event_id": "event_1",
                "event_type": "target_absent",
                "severity": "warning",
                "frame_start": 100,
                "frame_end": 200,
                "trigger_frame": 200,
                "raw_track_id": 7,
                "signal_ids": ["signal_1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"

    code = main(
        [
            "plan-event-grounding",
            "--events",
            str(events),
            "--query",
            "player",
            "--source-video",
            "video.avi",
            "--output",
            str(plan),
            "--overwrite",
        ]
    )

    assert code == 0
    assert plan.is_file()
