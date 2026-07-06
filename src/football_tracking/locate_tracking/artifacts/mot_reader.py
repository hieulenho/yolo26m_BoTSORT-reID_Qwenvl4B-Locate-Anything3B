"""Read-only parser for existing MOT TXT tracking artifacts."""

from __future__ import annotations

import math
from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_schemas import (
    MotArtifactError,
    MotTrackFile,
    MotTrackObservation,
)


class MotReaderError(RuntimeError):
    """Raised when a MOT tracking artifact cannot be parsed."""


def _row_error(path: Path, line_number: int, reason: str) -> MotReaderError:
    return MotReaderError(f"Invalid MOT row at {path}:{line_number}: {reason}")


def _parse_float(path: Path, line_number: int, value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise _row_error(path, line_number, f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise _row_error(path, line_number, f"{field_name} must be finite")
    return parsed


def _parse_int_from_float(path: Path, line_number: int, value: str, field_name: str) -> int:
    parsed = _parse_float(path, line_number, value, field_name)
    if not parsed.is_integer():
        raise _row_error(path, line_number, f"{field_name} must be an integer")
    return int(parsed)


def _parse_row(path: Path, line_number: int, line: str) -> MotTrackObservation:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        raise _row_error(path, line_number, "MOT row must contain at least 6 columns")
    frame_index = _parse_int_from_float(path, line_number, parts[0], "frame_index")
    track_id = _parse_int_from_float(path, line_number, parts[1], "track_id")
    left = _parse_float(path, line_number, parts[2], "left")
    top = _parse_float(path, line_number, parts[3], "top")
    width = _parse_float(path, line_number, parts[4], "width")
    height = _parse_float(path, line_number, parts[5], "height")
    confidence = None
    if len(parts) >= 7 and parts[6] != "":
        confidence_value = _parse_float(path, line_number, parts[6], "confidence")
        confidence = confidence_value if confidence_value >= 0.0 else None
    try:
        return MotTrackObservation(
            frame_index=frame_index,
            track_id=track_id,
            bbox_ltwh=(left, top, width, height),
            bbox_xyxy=(left, top, left + width, top + height),
            confidence=confidence,
            source_path=path,
            line_number=line_number,
            metadata={"raw_columns": parts},
        )
    except MotArtifactError as exc:
        raise _row_error(path, line_number, str(exc)) from exc


def read_mot_track_file(path: str | Path) -> MotTrackFile:
    resolved = Path(path)
    if not resolved.is_file():
        raise MotReaderError(f"MOT tracking file does not exist: {resolved}")
    observations: list[MotTrackObservation] = []
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        observation = _parse_row(resolved, line_number, line)
        key = (observation.frame_index, observation.track_id)
        if key in seen:
            raise _row_error(resolved, line_number, f"duplicate frame-track pair {key}")
        seen.add(key)
        observations.append(observation)
    return MotTrackFile(path=resolved, observations=tuple(observations))
