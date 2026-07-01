"""Writers for per-sequence detection cache files."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.detection.cache_schema import (
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.serialization import file_sha256

DETECTIONS_FILENAME = "detections.jsonl"
METADATA_FILENAME = "metadata.json"
VALIDATION_FILENAME = "validation.json"


class DetectionCacheWriterError(RuntimeError):
    """Raised when a detection cache cannot be written safely."""


def sequence_cache_dir(cache_root: Path, split: str, sequence_name: str) -> Path:
    return cache_root / split / sequence_name


class DetectionCacheWriter:
    """Write one complete per-sequence cache directory."""

    def __init__(self, cache_dir: Path, overwrite: bool = False) -> None:
        self.cache_dir = cache_dir
        self.overwrite = overwrite
        self.detections_path = cache_dir / DETECTIONS_FILENAME
        self.metadata_path = cache_dir / METADATA_FILENAME
        self.validation_path = cache_dir / VALIDATION_FILENAME

    def prepare(self) -> None:
        if self.detections_path.exists() and not self.overwrite:
            raise DetectionCacheWriterError(
                f"Detection cache exists and overwrite=false: {self.detections_path}"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write_frames(self, frames: list[CachedFrameDetections]) -> str | None:
        self.prepare()
        with self.detections_path.open("w", encoding="utf-8") as handle:
            for frame in sorted(frames, key=lambda item: item.frame_index):
                handle.write(json.dumps(frame.to_dict(), sort_keys=True) + "\n")
        return file_sha256(self.detections_path)

    def write_metadata(self, metadata: DetectionCacheMetadata) -> Path:
        self.metadata_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return self.metadata_path

    def write_validation(self, report: dict[str, object]) -> Path:
        self.validation_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return self.validation_path
