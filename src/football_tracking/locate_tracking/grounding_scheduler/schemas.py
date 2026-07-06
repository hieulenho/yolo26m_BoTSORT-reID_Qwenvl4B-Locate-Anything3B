"""Schemas for event-triggered grounding schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from football_tracking.locate_tracking.events.schemas import EventType, UncertaintyEvent

FrameSelectionStrategy = Literal["trigger", "window_representative", "strongest_frame"]


class GroundingSchedulerSchemaError(ValueError):
    """Raised when grounding scheduler schemas receive invalid values."""


@dataclass(frozen=True)
class SchedulerConfig:
    min_severity: str = "warning"
    cooldown_frames: int = 50
    critical_overrides_cooldown: bool = True
    max_requests_per_session: int = 20
    max_frames_per_request: int = 3
    frame_strategy: str = "window_representative"
    event_type_priority: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.cooldown_frames) < 0:
            raise GroundingSchedulerSchemaError("cooldown_frames must be >= 0.")
        if int(self.max_requests_per_session) < 0:
            raise GroundingSchedulerSchemaError("max_requests_per_session must be >= 0.")
        if int(self.max_frames_per_request) < 1:
            raise GroundingSchedulerSchemaError("max_frames_per_request must be >= 1.")
        if self.frame_strategy not in {"trigger", "window_representative", "strongest_frame"}:
            raise GroundingSchedulerSchemaError("Unsupported frame_strategy.")
        object.__setattr__(self, "cooldown_frames", int(self.cooldown_frames))
        object.__setattr__(
            self,
            "max_requests_per_session",
            int(self.max_requests_per_session),
        )
        object.__setattr__(self, "max_frames_per_request", int(self.max_frames_per_request))
        object.__setattr__(
            self,
            "event_type_priority",
            tuple(str(item) for item in self.event_type_priority),
        )


@dataclass(frozen=True)
class GroundingPlanItem:
    request_id: str
    event_id: str
    event_type: EventType
    severity: str
    query: str
    raw_track_id: int
    selected_frames: tuple[int, ...]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = tuple(sorted({int(item) for item in self.selected_frames}))
        if not frames:
            raise GroundingSchedulerSchemaError("selected_frames must not be empty.")
        if any(item < 1 for item in frames):
            raise GroundingSchedulerSchemaError("selected_frames must be >= 1.")
        if int(self.raw_track_id) < 1:
            raise GroundingSchedulerSchemaError("raw_track_id must be >= 1.")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "selected_frames", frames)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "query": self.query,
            "raw_track_id": self.raw_track_id,
            "selected_frames": list(self.selected_frames),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingPlanItem:
        return cls(
            request_id=str(data["request_id"]),
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),  # type: ignore[arg-type]
            severity=str(data["severity"]),
            query=str(data["query"]),
            raw_track_id=int(data["raw_track_id"]),
            selected_frames=tuple(data.get("selected_frames", ())),
            reason=str(data["reason"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class GroundingPlan:
    query: str
    source_video: Path | None
    items: tuple[GroundingPlanItem, ...]
    suppressed_events: tuple[dict[str, Any], ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_video",
            None if self.source_video is None else Path(self.source_video),
        )
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(
            self,
            "suppressed_events",
            tuple(dict(item) for item in self.suppressed_events),
        )
        object.__setattr__(self, "config", dict(self.config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source_video": str(self.source_video) if self.source_video else None,
            "items": [item.to_dict() for item in self.items],
            "suppressed_events": [dict(item) for item in self.suppressed_events],
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingPlan:
        return cls(
            query=str(data["query"]),
            source_video=Path(data["source_video"]) if data.get("source_video") else None,
            items=tuple(GroundingPlanItem.from_dict(item) for item in data.get("items", ())),
            suppressed_events=tuple(data.get("suppressed_events", ())),
            config=dict(data.get("config", {})),
        )


@dataclass(frozen=True)
class GroundingExecutionManifest:
    plan_path: Path
    source_video: Path
    output_dir: Path
    executed_requests: tuple[dict[str, Any], ...]
    skipped_requests: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "source_video": str(self.source_video),
            "output_dir": str(self.output_dir),
            "executed_requests": [dict(item) for item in self.executed_requests],
            "skipped_requests": [dict(item) for item in self.skipped_requests],
            "note": "grounding results are saved only; no track reacquisition is performed",
        }


def event_to_request_id(event: UncertaintyEvent, frames: tuple[int, ...]) -> str:
    from football_tracking.locate_tracking.monitoring.signal_utils import stable_id

    return stable_id(
        "ground",
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "raw_track_id": event.raw_track_id,
            "frames": frames,
        },
    )
