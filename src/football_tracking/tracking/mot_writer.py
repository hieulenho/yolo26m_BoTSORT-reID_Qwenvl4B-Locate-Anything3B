"""MOTChallenge prediction writer for tracker outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.tracking.schemas import TrackOutput


class MotWriterError(RuntimeError):
    """Raised when MOT prediction output is invalid."""


@dataclass
class MotPredictionWriter:
    path: Path
    metadata_path: Path | None = None
    rows: list[TrackOutput] = field(default_factory=list)
    confidence_unavailable: bool = False

    def add_tracks(self, tracks: list[TrackOutput]) -> None:
        for track in tracks:
            if track.confidence is None:
                self.confidence_unavailable = True
            if track.track_id <= 0:
                raise MotWriterError(f"Track ID must be positive: {track.track_id}")
            if track.bbox_ltwh.width <= 0 or track.bbox_ltwh.height <= 0:
                raise MotWriterError(f"Track width/height must be positive: {track}")
            self.rows.append(track)

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sorted_rows = sorted(self.rows, key=lambda item: (item.frame_index, item.track_id))
        seen: set[tuple[int, int]] = set()
        lines: list[str] = []
        for track in sorted_rows:
            key = (track.frame_index, track.track_id)
            if key in seen:
                raise MotWriterError(f"Duplicate frame-track pair: {key}")
            seen.add(key)
            lines.append(format_mot_row(track))
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return self.path

    def write_metadata(self, metadata: dict[str, Any]) -> Path | None:
        if self.metadata_path is None:
            return None
        payload = {
            **metadata,
            "confidence_unavailable": self.confidence_unavailable,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self.metadata_path


def format_mot_row(track: TrackOutput) -> str:
    values = [
        float(track.bbox_ltwh.x),
        float(track.bbox_ltwh.y),
        float(track.bbox_ltwh.width),
        float(track.bbox_ltwh.height),
        -1.0 if track.confidence is None else float(track.confidence),
        1.0,
    ]
    if any(not math.isfinite(value) for value in values):
        raise MotWriterError(f"MOT row contains non-finite value: {track}")
    confidence = -1.0 if track.confidence is None else float(track.confidence)
    return (
        f"{track.frame_index:d},{track.track_id:d},"
        f"{track.bbox_ltwh.x:.6f},{track.bbox_ltwh.y:.6f},"
        f"{track.bbox_ltwh.width:.6f},{track.bbox_ltwh.height:.6f},"
        f"{confidence:.6f},1,1.000000"
    )
