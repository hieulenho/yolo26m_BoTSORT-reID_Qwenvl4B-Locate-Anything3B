"""Persistent, bounded semantic observation memory for tracked identities."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from football_tracking.adaptive_tracking.semantic_fusion import TrackSemanticEvidence


class TemporalSemanticMemoryError(RuntimeError):
    """Raised when a semantic memory artifact is invalid."""


class TemporalSemanticMemory:
    def __init__(
        self,
        observations: list[TrackSemanticEvidence] | None = None,
        *,
        context_id: str | None = None,
    ) -> None:
        self.context_id = context_id
        self.observations = list(observations or ())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        context_id: str | None = None,
    ) -> TemporalSemanticMemory:
        memory_path = Path(path)
        if not memory_path.is_file():
            return cls(context_id=context_id)
        try:
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
            stored_context = payload.get("context_id")
            if context_id and stored_context and stored_context != context_id:
                raise TemporalSemanticMemoryError(
                    "Semantic memory belongs to a different video/context."
                )
            observations = [
                TrackSemanticEvidence(
                    track_id=int(row["track_id"]),
                    class_label=str(row["class_label"]),
                    confidence=float(row["confidence"]),
                    source=str(row["source"]),
                    attributes=dict(row.get("attributes", {})),
                    evidence_frames=tuple(row.get("evidence_frames", ())),
                    evidence=str(row.get("evidence", "")),
                    fine_label=str(row.get("fine_label", "unknown")),
                    fine_confidence=float(row.get("fine_confidence", 0.0)),
                    taxonomy_path=tuple(row.get("taxonomy_path", ())),
                )
                for row in payload.get("observations", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TemporalSemanticMemoryError(
                f"Invalid semantic memory artifact: {memory_path}"
            ) from exc
        return cls(observations, context_id=context_id or stored_context)

    def merge(
        self,
        evidence: list[TrackSemanticEvidence],
        *,
        max_observations_per_track: int = 32,
    ) -> None:
        if max_observations_per_track <= 0:
            raise TemporalSemanticMemoryError(
                "max_observations_per_track must be positive."
            )
        deduplicated: dict[tuple[Any, ...], TrackSemanticEvidence] = {}
        for row in (*self.observations, *evidence):
            key = (
                row.track_id,
                row.class_label,
                row.source,
                row.evidence_frames,
                round(row.confidence, 6),
                json.dumps(row.attributes, sort_keys=True, ensure_ascii=False),
                row.evidence,
                row.fine_label,
                round(row.fine_confidence, 6),
                row.taxonomy_path,
            )
            deduplicated[key] = row
        by_track: dict[int, list[TrackSemanticEvidence]] = {}
        for row in deduplicated.values():
            by_track.setdefault(row.track_id, []).append(row)
        bounded: list[TrackSemanticEvidence] = []
        for track_id in sorted(by_track):
            ordered = sorted(
                by_track[track_id],
                key=lambda row: (
                    max(row.evidence_frames) if row.evidence_frames else -1,
                    row.source,
                    row.class_label,
                ),
            )
            bounded.extend(ordered[-max_observations_per_track:])
        self.observations = bounded

    def save(self, path: str | Path) -> Path:
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        payload = {
            "schema_version": 2,
            "context_id": self.context_id,
            "updated_at": now,
            "summary": {
                "track_count": len({row.track_id for row in self.observations}),
                "observation_count": len(self.observations),
            },
            "observations": [asdict(row) for row in self.observations],
        }
        temporary = memory_path.with_suffix(
            f"{memory_path.suffix}.{uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(memory_path)
        return memory_path


__all__ = [
    "TemporalSemanticMemory",
    "TemporalSemanticMemoryError",
]
