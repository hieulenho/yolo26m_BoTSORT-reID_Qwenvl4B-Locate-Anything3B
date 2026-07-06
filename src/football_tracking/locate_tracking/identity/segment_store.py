"""Persistence for semantic target identity artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.identity.schemas import SemanticTarget


class SegmentStoreError(RuntimeError):
    """Raised when semantic identity artifacts cannot be persisted."""


def load_semantic_target(path: str | Path) -> SemanticTarget:
    resolved = Path(path)
    if not resolved.is_file():
        raise SegmentStoreError(f"Semantic target artifact does not exist: {resolved}")
    return SemanticTarget.from_dict(json.loads(resolved.read_text(encoding="utf-8")))


def save_semantic_target(
    target: SemanticTarget,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise SegmentStoreError(f"Semantic target output exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(target.to_dict(), indent=2, default=str), encoding="utf-8")
    return output


def semantic_target_timeline(target: SemanticTarget) -> dict[str, object]:
    return {
        "semantic_target_id": target.semantic_target_id,
        "query": target.query,
        "state": target.state,
        "current_raw_track_id": target.current_raw_track_id,
        "last_confirmed_frame": target.last_confirmed_frame,
        "last_update_frame": target.last_update_frame,
        "segments": [
            {
                "raw_track_id": segment.raw_track_id,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "status": segment.status,
                "source": segment.source,
                "confidence": segment.confidence,
                "transition_id": segment.transition_id,
            }
            for segment in target.segments
        ],
        "note": "semantic target timeline; raw tracker IDs remain read-only",
    }


def save_semantic_target_timeline(
    target: SemanticTarget,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise SegmentStoreError(f"Semantic target timeline exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(semantic_target_timeline(target), indent=2, default=str),
        encoding="utf-8",
    )
    return output
