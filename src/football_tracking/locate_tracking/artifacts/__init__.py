"""Read-only adapters for existing tracking artifacts."""

from football_tracking.locate_tracking.artifacts.mot_reader import (
    MotReaderError,
    read_mot_track_file,
)
from football_tracking.locate_tracking.artifacts.mot_schemas import (
    MotTrackFile,
    MotTrackObservation,
)
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex

__all__ = [
    "FrameTrackIndex",
    "MotReaderError",
    "MotTrackFile",
    "MotTrackObservation",
    "read_mot_track_file",
]
