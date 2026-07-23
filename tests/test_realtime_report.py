from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.benchmarking.realtime_report import (
    RealtimeReportError,
    build_realtime_report,
)


def _write_metrics(path: Path, *, processing_fps: float, source_fps: float) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "realtime",
                "frames": 90,
                "source_frames_consumed": 100,
                "dropped_late_frames": 10,
                "late_frame_drop_rate": 0.1,
                "timing": {
                    "processing_fps": processing_fps,
                    "steady_state_processing_fps": processing_fps + 1,
                    "source_progress_fps": source_fps,
                    "source_fps": 30.0,
                    "frame_latency_ms_p50": 30.0,
                    "frame_latency_ms_p95": 45.0,
                    "frame_latency_ms_p99": 60.0,
                    "startup_seconds": 5.0,
                    "detector_fps": 40.0,
                    "tracker_fps": 200.0,
                },
                "resources": {"peak_process_rss_bytes": 2 * 1024**3},
                "cuda_memory": {"peak_allocated_bytes": 512 * 1024**2},
                "hardware": {
                    "gpu_name": "test GPU",
                    "gpu_memory_total_bytes": 8 * 1024**3,
                    "processor": "test CPU",
                    "logical_cpu_count": 8,
                },
            }
        ),
        encoding="utf-8",
    )


def test_realtime_report_preserves_latency_drop_and_hardware(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    bounded = tmp_path / "bounded.json"
    _write_metrics(baseline, processing_fps=20.0, source_fps=20.0)
    _write_metrics(bounded, processing_fps=28.0, source_fps=30.0)

    result = build_realtime_report(
        [("baseline_r1", baseline), ("bounded_r1", bounded)],
        tmp_path / "report",
    )

    assert result["summary"]["best_source_progress_run"] == "bounded_r1"
    assert result["summary"]["hardware"]["gpu"] == "test GPU"
    assert result["summary"]["runs"][1]["drop_rate"] == pytest.approx(0.1)
    assert Path(result["paths"]["json"]).is_file()
    assert Path(result["paths"]["csv"]).is_file()
    assert "Use the no-drop profile" in Path(result["paths"]["markdown"]).read_text(
        encoding="utf-8"
    )
    assert result["summary"]["profiles"][0]["profile"] == "baseline"
    assert result["summary"]["profiles"][0]["repeat_count"] == 1


def test_realtime_report_aggregates_repeated_profiles(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_metrics(first, processing_fps=20.0, source_fps=25.0)
    _write_metrics(second, processing_fps=24.0, source_fps=27.0)

    result = build_realtime_report(
        [("bounded_r1", first), ("bounded_r2", second)],
        tmp_path / "report",
    )

    profile = result["summary"]["profiles"][0]
    assert profile["profile"] == "bounded"
    assert profile["repeat_count"] == 2
    assert profile["processing_fps_mean"] == pytest.approx(22.0)
    assert profile["processing_fps_std"] == pytest.approx(2.0)


def test_realtime_report_rejects_non_realtime_artifact(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"mode":"offline"}', encoding="utf-8")

    with pytest.raises(RealtimeReportError, match="Not a realtime metrics artifact"):
        build_realtime_report([("invalid", invalid)], tmp_path / "report")
