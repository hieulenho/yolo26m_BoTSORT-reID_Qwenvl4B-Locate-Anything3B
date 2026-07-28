"""Consistent OpenCV capture opening for files, streams, and Windows webcams."""

from __future__ import annotations

import platform
from typing import Any

import cv2


def capture_backend_candidates(source: str | int) -> tuple[tuple[str, int | None], ...]:
    if isinstance(source, int) and platform.system() == "Windows":
        return (("dshow", cv2.CAP_DSHOW), ("default", None))
    return (("default", None),)


def open_video_capture(source: str | int) -> tuple[Any | None, str | None]:
    for backend_name, backend in capture_backend_candidates(source):
        capture = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        if capture.isOpened():
            return capture, backend_name
        capture.release()
    return None, None

