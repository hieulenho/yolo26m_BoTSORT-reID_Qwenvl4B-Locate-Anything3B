"""Schemas for stable semantic target identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

IdentityState = Literal[
    "UNRESOLVED",
    "ACTIVE",
    "UNCERTAIN",
    "LOST",
    "REACQUIRING",
    "PROBATION",
    "REJECTED",
    "TERMINATED",
]
SegmentStatus = Literal["confirmed", "probation", "closed", "rejected"]


class IdentitySchemaError(ValueError):
    """Raised when identity schemas receive invalid values."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_identity_id(query: str, *, namespace: str = "target") -> str:
    payload = json.dumps(
        {"namespace": namespace, "query": query.strip().lower()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{namespace}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def stable_artifact_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class SemanticIdentitySegment:
    segment_id: str
    semantic_target_id: str
    raw_track_id: int
    start_frame: int
    end_frame: int | None
    source: str
    confidence: float
    status: SegmentStatus
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    transition_id: str | None = None

    def __post_init__(self) -> None:
        if int(self.raw_track_id) < 1:
            raise IdentitySchemaError("raw_track_id must be >= 1.")
        if int(self.start_frame) < 1:
            raise IdentitySchemaError("start_frame must be >= 1.")
        if self.end_frame is not None and int(self.end_frame) < int(self.start_frame):
            raise IdentitySchemaError("end_frame must be >= start_frame when provided.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise IdentitySchemaError("confidence must be in [0, 1].")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(
            self,
            "end_frame",
            None if self.end_frame is None else int(self.end_frame),
        )
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "evidence_summary", dict(self.evidence_summary))

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return self.start_frame, self.raw_track_id, self.segment_id

    def overlaps(self, other: SemanticIdentitySegment) -> bool:
        left_end = self.end_frame if self.end_frame is not None else 10**18
        right_end = other.end_frame if other.end_frame is not None else 10**18
        return self.start_frame <= right_end and other.start_frame <= left_end

    def with_updates(self, **changes: Any) -> SemanticIdentitySegment:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "semantic_target_id": self.semantic_target_id,
            "raw_track_id": self.raw_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "source": self.source,
            "confidence": self.confidence,
            "status": self.status,
            "evidence_summary": dict(self.evidence_summary),
            "transition_id": self.transition_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticIdentitySegment:
        return cls(
            segment_id=str(data["segment_id"]),
            semantic_target_id=str(data["semantic_target_id"]),
            raw_track_id=int(data["raw_track_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=data.get("end_frame"),
            source=str(data["source"]),
            confidence=float(data.get("confidence", 0.0)),
            status=str(data["status"]),  # type: ignore[arg-type]
            evidence_summary=dict(data.get("evidence_summary", {})),
            transition_id=data.get("transition_id"),
        )


@dataclass(frozen=True)
class IdentityStateTransition:
    transition_id: str
    semantic_target_id: str
    from_state: IdentityState
    to_state: IdentityState
    frame_index: int
    event_ids: tuple[str, ...]
    decision_id: str | None
    previous_raw_track_id: int | None
    new_raw_track_id: int | None
    reason: str
    evidence_reference: str | None
    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if int(self.frame_index) < 1:
            raise IdentitySchemaError("frame_index must be >= 1.")
        for field_name in ("previous_raw_track_id", "new_raw_track_id"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise IdentitySchemaError(f"{field_name} must be >= 1 when provided.")
            object.__setattr__(self, field_name, None if value is None else int(value))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "event_ids", tuple(str(item) for item in self.event_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "semantic_target_id": self.semantic_target_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "frame_index": self.frame_index,
            "event_ids": list(self.event_ids),
            "decision_id": self.decision_id,
            "previous_raw_track_id": self.previous_raw_track_id,
            "new_raw_track_id": self.new_raw_track_id,
            "reason": self.reason,
            "evidence_reference": self.evidence_reference,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityStateTransition:
        return cls(
            transition_id=str(data["transition_id"]),
            semantic_target_id=str(data["semantic_target_id"]),
            from_state=str(data["from_state"]),  # type: ignore[arg-type]
            to_state=str(data["to_state"]),  # type: ignore[arg-type]
            frame_index=int(data["frame_index"]),
            event_ids=tuple(data.get("event_ids", ())),
            decision_id=data.get("decision_id"),
            previous_raw_track_id=data.get("previous_raw_track_id"),
            new_raw_track_id=data.get("new_raw_track_id"),
            reason=str(data["reason"]),
            evidence_reference=data.get("evidence_reference"),
            timestamp=str(data["timestamp"]),
        )


@dataclass(frozen=True)
class SemanticTarget:
    semantic_target_id: str
    query: str
    query_mode: str
    state: IdentityState
    current_raw_track_id: int | None
    segments: tuple[SemanticIdentitySegment, ...]
    reference_semantic_memory: str | None
    reference_appearance_prototype: str | None
    last_confirmed_frame: int | None
    last_update_frame: int | None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.semantic_target_id.strip():
            raise IdentitySchemaError("semantic_target_id must not be empty.")
        if not self.query.strip():
            raise IdentitySchemaError("query must not be empty.")
        if self.current_raw_track_id is not None and int(self.current_raw_track_id) < 1:
            raise IdentitySchemaError("current_raw_track_id must be >= 1 when provided.")
        segments = tuple(sorted(self.segments, key=lambda item: item.sort_key))
        for segment in segments:
            if segment.semantic_target_id != self.semantic_target_id:
                raise IdentitySchemaError("segment semantic_target_id does not match target.")
        confirmed = [item for item in segments if item.status in {"confirmed", "probation"}]
        for index, left in enumerate(confirmed):
            for right in confirmed[index + 1 :]:
                if left.overlaps(right):
                    raise IdentitySchemaError("confirmed/probation segments must not overlap.")
        for field_name in ("last_confirmed_frame", "last_update_frame"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise IdentitySchemaError(f"{field_name} must be >= 1 when provided.")
            object.__setattr__(self, field_name, None if value is None else int(value))
        object.__setattr__(
            self,
            "current_raw_track_id",
            None if self.current_raw_track_id is None else int(self.current_raw_track_id),
        )
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def active_segment(self) -> SemanticIdentitySegment | None:
        for segment in reversed(self.segments):
            if segment.status in {"confirmed", "probation"} and segment.end_frame is None:
                return segment
        return None

    def with_updates(self, **changes: Any) -> SemanticTarget:
        if "updated_at" not in changes:
            changes["updated_at"] = utc_now_iso()
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_target_id": self.semantic_target_id,
            "query": self.query,
            "query_mode": self.query_mode,
            "state": self.state,
            "current_raw_track_id": self.current_raw_track_id,
            "segments": [item.to_dict() for item in self.segments],
            "reference_semantic_memory": self.reference_semantic_memory,
            "reference_appearance_prototype": self.reference_appearance_prototype,
            "last_confirmed_frame": self.last_confirmed_frame,
            "last_update_frame": self.last_update_frame,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticTarget:
        return cls(
            semantic_target_id=str(data["semantic_target_id"]),
            query=str(data["query"]),
            query_mode=str(data.get("query_mode", "single_target")),
            state=str(data["state"]),  # type: ignore[arg-type]
            current_raw_track_id=data.get("current_raw_track_id"),
            segments=tuple(
                SemanticIdentitySegment.from_dict(item) for item in data.get("segments", ())
            ),
            reference_semantic_memory=data.get("reference_semantic_memory"),
            reference_appearance_prototype=data.get("reference_appearance_prototype"),
            last_confirmed_frame=data.get("last_confirmed_frame"),
            last_update_frame=data.get("last_update_frame"),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            metadata=dict(data.get("metadata", {})),
        )
