"""OpenCV video writer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VideoWriterError(RuntimeError):
    """Raised when OpenCV cannot create a video writer."""


class ManagedVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float,
        width: int,
        height: int,
        overwrite: bool = False,
    ) -> None:
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.overwrite = overwrite
        self.writer: Any | None = None

    def open(self) -> ManagedVideoWriter:
        if self.path.exists() and not self.overwrite:
            raise VideoWriterError(f"Video output exists and overwrite=false: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise VideoWriterError(f"OpenCV is required for video writing: {exc}") from exc
        for codec in ("mp4v", "avc1", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                float(self.fps or 25.0),
                (int(self.width), int(self.height)),
            )
            if writer.isOpened():
                self.writer = writer
                return self
            writer.release()
        raise VideoWriterError(f"Could not open VideoWriter: {self.path}")

    def write(self, frame: Any) -> None:
        if self.writer is None:
            raise VideoWriterError("Video writer is not open.")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> ManagedVideoWriter:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()
