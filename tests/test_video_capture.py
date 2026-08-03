from __future__ import annotations

from typing import Any

import pytest

from football_tracking import video_capture


class _FakeCapture:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def release(self) -> None:
        self.released = True


def test_windows_camera_falls_back_to_default_backend(monkeypatch: Any) -> None:
    failed = _FakeCapture(False)
    opened = _FakeCapture(True)
    captures = iter((failed, opened))
    monkeypatch.setattr(video_capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(video_capture.cv2, "VideoCapture", lambda *_args: next(captures))

    capture, backend = video_capture.open_video_capture(1)

    assert failed.released is True
    assert capture is opened
    assert backend == "default"


def test_file_source_uses_only_default_backend(monkeypatch: Any) -> None:
    opened = _FakeCapture(True)
    calls: list[tuple[Any, ...]] = []

    def fake_capture(*args: Any) -> _FakeCapture:
        calls.append(args)
        return opened

    monkeypatch.setattr(video_capture.cv2, "VideoCapture", fake_capture)

    capture, backend = video_capture.open_video_capture("video.mp4")

    assert capture is opened
    assert backend == "default"
    assert calls == [("video.mp4",)]


def test_resolve_capture_source_reads_secret_from_environment() -> None:
    result = video_capture.resolve_capture_source(
        None,
        "TEST_RTSP_URL",
        environment={"TEST_RTSP_URL": "rtsp://admin:secret@192.0.2.1/onvif2"},
    )

    assert result == "rtsp://admin:secret@192.0.2.1/onvif2"


def test_resolve_capture_source_normalizes_camera_index() -> None:
    assert video_capture.resolve_capture_source(None, environment={}) == 0
    assert video_capture.resolve_capture_source("2", environment={}) == 2


def test_resolve_capture_source_rejects_missing_environment_value() -> None:
    with pytest.raises(ValueError, match="empty or missing"):
        video_capture.resolve_capture_source(
            None,
            "TEST_RTSP_URL",
            environment={},
        )


def test_redact_capture_source_removes_rtsp_credentials() -> None:
    result = video_capture.redact_capture_source(
        "rtsp://admin:p%40ss@192.0.2.1:554/onvif2"
    )

    assert result == "rtsp://***:***@192.0.2.1:554/onvif2"
    assert "admin" not in result
    assert "p%40ss" not in result


def test_redact_capture_source_leaves_plain_source_unchanged() -> None:
    assert video_capture.redact_capture_source(0) == "0"
    assert video_capture.redact_capture_source("video.mp4") == "video.mp4"
