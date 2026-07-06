"""Build grounding plans from uncertainty events."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.events.severity import severity_at_least
from football_tracking.locate_tracking.grounding_scheduler.budget import apply_session_budget
from football_tracking.locate_tracking.grounding_scheduler.cooldown import is_allowed_by_cooldown
from football_tracking.locate_tracking.grounding_scheduler.frame_selector import select_event_frames
from football_tracking.locate_tracking.grounding_scheduler.priority import sort_events_by_priority
from football_tracking.locate_tracking.grounding_scheduler.schemas import (
    GroundingPlan,
    GroundingPlanItem,
    SchedulerConfig,
    event_to_request_id,
)


class GroundingPlannerError(RuntimeError):
    """Raised when a grounding plan cannot be written."""


def build_grounding_plan(
    *,
    events: tuple[UncertaintyEvent, ...],
    query: str,
    source_video: str | Path | None,
    config: SchedulerConfig,
) -> GroundingPlan:
    eligible = [
        event for event in events if severity_at_least(event.severity, config.min_severity)
    ]
    accepted: list[GroundingPlanItem] = []
    suppressed: list[dict[str, object]] = []
    trigger_frames: list[int] = []
    for event in sort_events_by_priority(
        tuple(eligible),
        event_type_priority=config.event_type_priority,
    ):
        if not is_allowed_by_cooldown(
            event,
            tuple(trigger_frames),
            cooldown_frames=config.cooldown_frames,
            critical_overrides=config.critical_overrides_cooldown,
        ):
            suppressed.append({"event_id": event.event_id, "reason": "cooldown"})
            continue
        frames = select_event_frames(
            event,
            strategy=config.frame_strategy,
            max_frames=config.max_frames_per_request,
        )
        accepted.append(
            GroundingPlanItem(
                request_id=event_to_request_id(event, frames),
                event_id=event.event_id,
                event_type=event.event_type,
                severity=event.severity,
                query=query,
                raw_track_id=event.raw_track_id,
                selected_frames=frames,
                reason=f"{event.event_type}:{event.severity}",
                metadata={
                    "trigger_frame": event.trigger_frame,
                    "event_frame_start": event.frame_start,
                    "event_frame_end": event.frame_end,
                },
            )
        )
        trigger_frames.append(event.trigger_frame)
    budgeted, over_budget = apply_session_budget(
        tuple(accepted),
        max_requests_per_session=config.max_requests_per_session,
    )
    suppressed.extend({"event_id": item.event_id, "reason": "budget"} for item in over_budget)
    return GroundingPlan(
        query=query,
        source_video=Path(source_video) if source_video is not None else None,
        items=budgeted,
        suppressed_events=tuple(suppressed),
        config={
            "min_severity": config.min_severity,
            "cooldown_frames": config.cooldown_frames,
            "critical_overrides_cooldown": config.critical_overrides_cooldown,
            "max_requests_per_session": config.max_requests_per_session,
            "max_frames_per_request": config.max_frames_per_request,
            "frame_strategy": config.frame_strategy,
            "event_type_priority": list(config.event_type_priority),
        },
    )


def save_grounding_plan(
    plan: GroundingPlan,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise GroundingPlannerError(f"Grounding plan exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")
    return output


def load_grounding_plan(path: str | Path) -> GroundingPlan:
    resolved = Path(path)
    if not resolved.is_file():
        raise GroundingPlannerError(f"Grounding plan does not exist: {resolved}")
    return GroundingPlan.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
