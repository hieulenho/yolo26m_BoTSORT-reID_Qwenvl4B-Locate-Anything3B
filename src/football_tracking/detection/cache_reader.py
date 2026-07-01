"""Readers for JSONL detection cache files."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.detection.cache_schema import (
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.cache_writer import (
    DETECTIONS_FILENAME,
    METADATA_FILENAME,
    sequence_cache_dir,
)


class DetectionCacheReaderError(RuntimeError):
    """Raised when a detection cache cannot be read."""


class DetectionCacheReader:
    """Read cached detections for one sequence."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.detections_path = cache_dir / DETECTIONS_FILENAME
        self.metadata_path = cache_dir / METADATA_FILENAME
        self._frames: list[CachedFrameDetections] | None = None
        self._by_frame: dict[int, CachedFrameDetections] | None = None
        self._metadata: DetectionCacheMetadata | None = None

    @classmethod
    def for_sequence(cls, cache_root: Path, split: str, sequence_name: str) -> DetectionCacheReader:
        return cls(sequence_cache_dir(cache_root, split, sequence_name))

    def read_metadata(self) -> DetectionCacheMetadata:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.is_file():
            raise DetectionCacheReaderError(f"Missing cache metadata: {self.metadata_path}")
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._metadata = DetectionCacheMetadata.from_dict(payload)
        return self._metadata

    def read_frames(self) -> list[CachedFrameDetections]:
        if self._frames is not None:
            return self._frames
        if not self.detections_path.is_file():
            raise DetectionCacheReaderError(f"Missing cache detections: {self.detections_path}")
        frames: list[CachedFrameDetections] = []
        for line_number, raw_line in enumerate(
            self.detections_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                frames.append(CachedFrameDetections.from_dict(payload))
            except Exception as exc:  # noqa: BLE001
                raise DetectionCacheReaderError(
                    f"Invalid JSONL cache row at {self.detections_path}:{line_number}: {exc}"
                ) from exc
        self._frames = sorted(frames, key=lambda item: item.frame_index)
        self._by_frame = {frame.frame_index: frame for frame in self._frames}
        return self._frames

    def frame(self, frame_index: int) -> CachedFrameDetections:
        if self._by_frame is None:
            self.read_frames()
        assert self._by_frame is not None
        if frame_index not in self._by_frame:
            raise DetectionCacheReaderError(
                f"Cache is missing frame {frame_index}: {self.detections_path}"
            )
        return self._by_frame[frame_index]
