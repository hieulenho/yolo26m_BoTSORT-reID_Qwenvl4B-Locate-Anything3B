"""Serialization helpers for semantic memory artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.semantic_memory.schemas import (
    FinalLanguageTrackResolution,
    LanguageTrackQuerySession,
    SemanticMemory,
)


class SemanticSerializationError(RuntimeError):
    """Raised when semantic memory artifacts cannot be read or written."""


def load_frame_resolution(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise SemanticSerializationError(f"Frame resolution does not exist: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SemanticSerializationError(
            f"Invalid frame resolution JSON: {resolved}: {exc}"
        ) from exc
    required = {"query", "frame_index", "associations", "overall_status"}
    missing = required - set(data)
    if missing:
        raise SemanticSerializationError(
            f"Frame resolution JSON is missing required keys {sorted(missing)}: {resolved}"
        )
    return dict(data)


def load_frame_resolutions(paths: tuple[str | Path, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(load_frame_resolution(path) for path in paths)


def write_json(data: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise SemanticSerializationError(f"Output exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output


def save_semantic_memory(
    semantic_memory: SemanticMemory,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(semantic_memory.to_dict(), path, overwrite=overwrite)


def save_final_resolution(
    final_resolution: FinalLanguageTrackResolution,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(final_resolution.to_dict(), path, overwrite=overwrite)


def save_language_track_session(
    session: LanguageTrackQuerySession,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(session.to_dict(), path, overwrite=overwrite)


def load_semantic_memory(path: str | Path) -> SemanticMemory:
    resolved = Path(path)
    if not resolved.is_file():
        raise SemanticSerializationError(f"Semantic memory does not exist: {resolved}")
    return SemanticMemory.from_dict(json.loads(resolved.read_text(encoding="utf-8")))


def load_final_resolution(path: str | Path) -> FinalLanguageTrackResolution:
    resolved = Path(path)
    if not resolved.is_file():
        raise SemanticSerializationError(f"Final resolution does not exist: {resolved}")
    return FinalLanguageTrackResolution.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
