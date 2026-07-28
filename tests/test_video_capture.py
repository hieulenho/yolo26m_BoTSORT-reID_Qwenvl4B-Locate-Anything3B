from __future__ import annotations

from typing import Any

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
