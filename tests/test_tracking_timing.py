from __future__ import annotations

from football_tracking.tracking.timing import TrackingTiming


def test_tracking_timing_does_not_divide_by_zero() -> None:
    timing = TrackingTiming()
    assert timing.fps(0.0) is None

    timing.processed_frames = 10
    timing.detector_seconds = 2.0

    assert timing.to_dict()["detector_fps"] == 5.0
    assert timing.to_dict()["tracker_fps"] is None
