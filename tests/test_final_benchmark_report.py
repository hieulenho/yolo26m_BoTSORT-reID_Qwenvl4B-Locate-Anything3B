from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.benchmarking.final_report import (
    FinalReportError,
    _runtime_row,
    _validate_idsw,
    _validate_tracking,
)


def test_idsw_taxonomy_accepts_complete_category_partition() -> None:
    payload = {
        "summaries": [
            {
                "sequence": "__overall__",
                "tracker": "fasttrack",
                "total_id_switches_recomputed": 10,
                "fragmentation_count": 2,
                "fragmentation_percent": 20.0,
                "identity_swap_count": 2,
                "identity_swap_percent": 20.0,
                "re_identification_failure_count": 2,
                "re_identification_failure_percent": 20.0,
                "association_error_count": 2,
                "association_error_percent": 20.0,
                "appearance_confusion_count": 2,
                "appearance_confusion_percent": 20.0,
            }
        ]
    }
    issues: list[dict] = []

    _validate_idsw(payload, [{"tracker": "fasttrack"}], issues)

    assert issues == []


def test_idsw_taxonomy_rejects_incomplete_partition() -> None:
    payload = {
        "summaries": [
            {
                "sequence": "__overall__",
                "tracker": "fasttrack",
                "total_id_switches_recomputed": 10,
                "fragmentation_count": 1,
                "fragmentation_percent": 10.0,
            }
        ]
    }
    issues: list[dict] = []

    _validate_idsw(payload, [{"tracker": "fasttrack"}], issues)

    assert {issue["code"] for issue in issues} == {"idsw_count_sum", "idsw_percent_sum"}


def test_runtime_row_records_stage_fps_and_peak_cuda_memory(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    validation = tmp_path / "validation.json"
    metrics.write_text(
        json.dumps(
            {
                "route": {"checkpoint": "model.pt", "checkpoint_type": "test"},
                "tracker": "adaptive_routed",
                "frames": 120,
                "detections": 10,
                "tracker_detections": 9,
                "detection_only_boxes": 1,
                "track_boxes": 8,
                "unique_tracks": 2,
                "timing": {
                    "end_to_end_fps": 25.0,
                    "processing_fps": 30.0,
                    "steady_state_processing_fps": 35.0,
                    "frame_latency_ms_p95": 40.0,
                    "startup_seconds": 1.0,
                    "detector_fps": 50.0,
                    "tracker_fps": 200.0,
                    "render_fps": 150.0,
                },
                "hardware": {
                    "gpu_name": "test GPU",
                    "gpu_memory_total_bytes": 8 * 1024**3,
                    "system_memory_total_bytes": 16 * 1024**3,
                },
                "cuda_memory": {
                    "peak_allocated_bytes": 128 * 1024**2,
                    "peak_reserved_bytes": 192 * 1024**2,
                },
                "detector": {"backend": "ultralytics", "checkpoint_name": "model.pt"},
            }
        ),
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps(
            {
                "frame_count": 120,
                "frames_with_tracks": 119,
                "frame_track_coverage_percent": 99.167,
                "track_length": {
                    "median": 61.0,
                    "shorter_than_30_percent": 32.432,
                },
                "track_gaps": {"tracks_with_gaps": 12},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    row = _runtime_row(
        {
            "id": "test",
            "name": "Test route",
            "metrics": metrics,
            "validation": validation,
        },
        {"runtime_frames": 120},
    )

    assert row["detector_fps"] == 50.0
    assert row["detector_stack"] == "model.pt"
    assert row["tracker_fps"] == 200.0
    assert row["render_fps"] == 150.0
    assert row["peak_cuda_allocated_mib"] == 128.0
    assert row["peak_cuda_reserved_mib"] == 192.0
    assert row["frame_track_coverage_percent"] == 99.167
    assert row["median_track_length"] == 61.0
    assert row["shorter_than_30_percent"] == 32.432
    assert row["tracks_with_gaps"] == 12
    assert row["validation_warning_count"] == 0


def test_runtime_row_requires_track_validation(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")

    with pytest.raises(FinalReportError, match="requires a validation JSON path"):
        _runtime_row(
            {"id": "test", "name": "Test route", "metrics": metrics},
            {"runtime_frames": 120},
        )


def test_tracking_snapshot_keeps_metric_validation_and_reports_scope() -> None:
    rows = [
        {
            "tracker": "tracktrack",
            "sequence_count": 30,
            "frame_count": 20171,
            "HOTA": 71.058,
            "DetA": 83.864,
            "AssA": 60.273,
            "MOTA": 91.511,
            "IDF1": 71.341,
            "tracker_fps": 21.658,
        }
    ]
    issues: list[dict] = []

    _validate_tracking(
        rows,
        None,
        None,
        {
            "tracker_count": 1,
            "sequence_count": 30,
            "frame_count": 20171,
        },
        issues,
    )

    assert [issue["code"] for issue in issues] == ["tracking_snapshot_scope"]
    assert issues[0]["severity"] == "WARNING"
