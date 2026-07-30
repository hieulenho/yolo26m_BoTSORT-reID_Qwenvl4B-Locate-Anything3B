from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.benchmarking.task_research_report import (
    TaskResearchReportError,
    _aggregate_runtime,
    _idsw_overall,
    _validate_review_status,
    _validate_runtime,
)


def test_runtime_aggregate_uses_frame_weighted_elapsed_time(tmp_path: Path) -> None:
    paths = []
    for index, (frames, fps, latency) in enumerate(((100, 10.0, 20.0), (300, 30.0, 40.0))):
        path = tmp_path / f"run_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "frames": frames,
                    "timing": {"processing_fps": fps, "frame_latency_ms_p95": latency},
                    "cuda_memory": {"peak_allocated_bytes": (index + 1) * 1024**3},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    result = _aggregate_runtime(paths)

    assert result["frame_count"] == 400
    assert result["processing_fps"] == pytest.approx(20.0)
    assert result["p95_latency_ms"] == 40.0
    assert result["peak_vram_gb"] == 2.0


def test_idsw_partition_requires_complete_tracker_set() -> None:
    payload = {
        "summaries": [
            {
                "sequence": "__overall__",
                "tracker": "tracktrack",
                "total_id_switches_recomputed": 5,
                "fragmentation_count": 1,
                "fragmentation_percent": 20.0,
                "identity_swap_count": 1,
                "identity_swap_percent": 20.0,
                "re_identification_failure_count": 1,
                "re_identification_failure_percent": 20.0,
                "association_error_count": 1,
                "association_error_percent": 20.0,
                "appearance_confusion_count": 1,
                "appearance_confusion_percent": 20.0,
            }
        ]
    }

    with pytest.raises(TaskResearchReportError, match="does not cover"):
        _idsw_overall(payload, [{"tracker": "tracktrack"}, {"tracker": "ocsort"}])


def test_review_status_rejects_inconsistent_counts() -> None:
    with pytest.raises(TaskResearchReportError, match="inconsistent"):
        _validate_review_status(
            {"track_count": 10, "reviewed_track_count": 3, "remaining_track_count": 8}
        )


def test_runtime_validation_enforces_repeat_count() -> None:
    payload = {
        "runs": [{"processing_fps": 10.0}] * 4,
        "profiles": [{"profile": "test", "repeat_count": 2}],
    }

    with pytest.raises(TaskResearchReportError, match="minimum repeat count"):
        _validate_runtime(payload, minimum_repeats=3)
