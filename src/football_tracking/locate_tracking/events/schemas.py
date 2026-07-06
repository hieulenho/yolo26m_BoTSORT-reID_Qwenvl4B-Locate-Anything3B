"""Structured uncertainty events emitted from monitoring signals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from football_tracking.locate_tracking.monitoring.schemas import Severity, SignalType

EventType = Literal[
    "target_absent",
    "confidence_low",
    "motion_jump",
    "semantic_ambiguous",
    "appearance_drift",
    "neighborhood_ambiguous",
    "track_gap",
    "grounding_stale",
]

SIGNAL_TO_EVENT_TYPE: dict[SignalType, EventType] = {
    "TARGET_PRESENCE": "target_absent",
    "TRACK_CONFIDENCE": "confidence_low",
    "MOTION_JUMP": "motion_jump",
    "SEMANTIC_MARGIN": "semantic_ambiguous",
    "APPEARANCE_DRIFT": "appearance_drift",
    "NEIGHBOR_AMBIGUITY": "neighborhood_ambiguous",
    "TRACK_GAP": "track_gap",
    "GROUNDING_STALENESS": "grounding_stale",
}


class EventSchemaError(ValueError):
    """Raised when an event schema receives invalid values."""


@dataclass(frozen=True)
class UncertaintyEvent:
    event_id: str
    event_type: EventType
    severity: Severity
    frame_start: int
    frame_end: int
    trigger_frame: int
    raw_track_id: int
    signal_ids: tuple[str, ...]
    score: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_start) < 1 or int(self.frame_end) < int(self.frame_start):
            raise EventSchemaError("Invalid event frame range.")
        if int(self.trigger_frame) < self.frame_start or int(self.trigger_frame) > self.frame_end:
            raise EventSchemaError("trigger_frame must be inside event range.")
        if int(self.raw_track_id) < 1:
            raise EventSchemaError("raw_track_id must be >= 1.")
        object.__setattr__(self, "frame_start", int(self.frame_start))
        object.__setattr__(self, "frame_end", int(self.frame_end))
        object.__setattr__(self, "trigger_frame", int(self.trigger_frame))
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "signal_ids", tuple(str(item) for item in self.signal_ids))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "trigger_frame": self.trigger_frame,
            "raw_track_id": self.raw_track_id,
            "signal_ids": list(self.signal_ids),
            "score": self.score,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UncertaintyEvent:
        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),  # type: ignore[arg-type]
            severity=str(data["severity"]),  # type: ignore[arg-type]
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            trigger_frame=int(data["trigger_frame"]),
            raw_track_id=int(data["raw_track_id"]),
            signal_ids=tuple(data.get("signal_ids", ())),
            score=data.get("score"),
            evidence=dict(data.get("evidence", {})),
        )


def event_json_line(event: UncertaintyEvent) -> str:
    return json.dumps(event.to_dict(), sort_keys=True, default=str)
