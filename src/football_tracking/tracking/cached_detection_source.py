"""Convert cached detections into tracker inputs."""

from __future__ import annotations

from pathlib import Path

from football_tracking.detection.cache_reader import DetectionCacheReader
from football_tracking.detection.cache_schema import CachedFrameDetections
from football_tracking.tracking.schemas import TrackerDetection


class CachedDetectionSource:
    """Frame-indexed detection source backed by one sequence cache."""

    def __init__(self, cache_dir: Path, confidence_threshold: float) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        self.reader = DetectionCacheReader(cache_dir)
        self.confidence_threshold = confidence_threshold

    def metadata(self) -> dict[str, object]:
        return self.reader.read_metadata().to_dict()

    def frame(self, frame_index: int) -> CachedFrameDetections:
        return self.reader.frame(frame_index)

    def tracker_detections(self, frame_index: int) -> list[TrackerDetection]:
        return self.frame(frame_index).to_tracker_detections(self.confidence_threshold)
