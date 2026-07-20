from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.adaptive_tracking.realtime import (
    RealtimeTrackingError,
    _capture_source,
    _timing_summary,
    _validate_output_paths,
)


def test_realtime_source_parses_camera_index_but_preserves_urls() -> None:
    assert _capture_source("0") == 0
    assert _capture_source("rtsp://example/stream") == "rtsp://example/stream"


def test_realtime_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    video = tmp_path / "tracked.mp4"
    video.write_bytes(b"existing")

    with pytest.raises(RealtimeTrackingError, match="overwrite=false"):
        _validate_output_paths(video, None, None, overwrite=False)

    _validate_output_paths(video, None, None, overwrite=True)


def test_realtime_timing_reports_latency_and_fps() -> None:
    result = _timing_summary([0.01, 0.02, 0.03], total_seconds=0.06)

    assert result["end_to_end_fps"] == pytest.approx(50.0)
    assert result["processing_fps"] == pytest.approx(50.0)
    assert result["warmup_frame_count"] == 2
    assert result["steady_state_processing_fps"] == pytest.approx(1 / 0.03)
    assert result["frame_latency_ms_mean"] == pytest.approx(20.0)
    assert result["frame_latency_ms_p95"] == pytest.approx(30.0)
