from football_tracking.detection.timing import TimingStats, safe_fps


def test_safe_fps_does_not_divide_by_zero() -> None:
    assert safe_fps(0, 1.0) is None
    assert safe_fps(10, 0.0) is None
    assert safe_fps(10, 2.0) == 5.0


def test_timing_separates_detector_and_end_to_end_fps() -> None:
    timing = TimingStats(inference_seconds=2.0, total_pipeline_seconds=5.0, warmup_seconds=10.0)

    payload = timing.to_dict(image_count=10)

    assert payload["detector_fps"] == 5.0
    assert payload["end_to_end_fps"] == 2.0
    assert payload["warmup_seconds"] == 10.0
