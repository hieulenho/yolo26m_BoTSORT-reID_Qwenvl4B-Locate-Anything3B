from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.reporting.tracking_run_report import write_tracking_run_report
from football_tracking.tracking.timing import TrackingTiming


def test_tracking_report_uses_actual_tracker_name(tmp_path: Path) -> None:
    payload = {"sequences": [{"sequence_name": "clip", "frame_count": 2}]}

    paths = write_tracking_run_report(payload, tmp_path, "FastTracker")

    assert paths["json"].name == "fasttracker_tracking_run.json"
    assert paths["csv"].name == "fasttracker_tracking_per_sequence.csv"
    assert json.loads(paths["json"].read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "deepsort_tracking_run.json").exists()


def test_tracking_timing_separates_cold_start_and_steady_state() -> None:
    timing = TrackingTiming(
        model_load_seconds=2.0,
        total_pipeline_seconds=4.0,
        processed_frames=120,
    )

    payload = timing.to_dict()

    assert payload["cold_start_total_seconds"] == 6.0
    assert payload["steady_state_fps"] == pytest.approx(30.0)
    assert payload["cold_start_fps"] == pytest.approx(20.0)
