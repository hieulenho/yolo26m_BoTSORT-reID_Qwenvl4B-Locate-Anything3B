from __future__ import annotations

import json
from pathlib import Path

from scripts.data.diagnose_video_tracks import diagnose_tracks


def test_diagnostics_use_realtime_processed_frame_count(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.txt"
    tracks.write_text("1,1,0,0,10,10,0.9,-1,-1,-1\n", encoding="utf-8")
    metadata = tmp_path / "metrics.json"
    metadata.write_text(json.dumps({"frames": 120}), encoding="utf-8")

    result = diagnose_tracks(tracks, metadata, None)

    assert result["frame_count"] == 120
    assert result["frame_track_coverage_percent"] == 0.833
