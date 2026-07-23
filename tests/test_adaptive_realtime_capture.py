from football_tracking.adaptive_tracking.realtime import (
    _dropped_between,
    _is_live_source,
)


def test_live_source_detection_distinguishes_camera_rtsp_and_file() -> None:
    assert _is_live_source(0) is True
    assert _is_live_source("rtsp://camera/live") is True
    assert _is_live_source("rtsps://camera/live") is True
    assert _is_live_source("video.mp4") is False


def test_dropped_between_uses_successful_capture_sequence_gap() -> None:
    assert _dropped_between(10, 11) == 0
    assert _dropped_between(10, 15) == 4
    assert _dropped_between(15, 15) == 0
