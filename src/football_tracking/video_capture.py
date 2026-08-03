"""Consistent OpenCV capture opening for files, streams, and Windows webcams."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2


def resolve_capture_source(
    source: str | int | None,
    source_environment_variable: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | int:
    """Resolve a capture source without requiring stream credentials in argv."""

    variable = str(source_environment_variable or "").strip()
    if not variable:
        value: str | int = 0 if source is None else source
    else:
        values = os.environ if environment is None else environment
        value = str(values.get(variable, "")).strip()
        if not value:
            raise ValueError(
                f"Capture source environment variable is empty or missing: {variable}"
            )
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def redact_capture_source(source: str | int) -> str:
    """Remove URL user information before writing a source to logs or metadata."""

    text = str(source).strip()
    try:
        parsed = urlsplit(text)
        if parsed.username is None and parsed.password is None:
            return text
        hostname = parsed.hostname
        if not hostname:
            return text
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        redacted_netloc = f"***:***@{hostname}{port}"
        return urlunsplit(
            (
                parsed.scheme,
                redacted_netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    except ValueError:
        return text


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
