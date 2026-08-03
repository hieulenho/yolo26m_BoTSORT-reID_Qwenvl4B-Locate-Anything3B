from __future__ import annotations

from scripts.benchmarks.build_traffic_quality_report import (
    _detector_checkpoint,
    _tracker_name,
)


def test_report_uses_runtime_tracker_type_instead_of_stale_display_name() -> None:
    runtime = {
        "tracker": "botsort_reid",
        "tracker_runtime_config": {
            "tracker_type": "tracktrack",
            "with_reid": True,
        },
    }

    assert _tracker_name(runtime) == ("tracktrack", True)


def test_report_reads_primary_checkpoint_from_composite_detector() -> None:
    runtime = {
        "detector": {
            "backend": "routed_composite",
            "primary": {"checkpoint": "yolo26s.pt"},
        }
    }

    assert _detector_checkpoint(runtime) == "yolo26s.pt"
