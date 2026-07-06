from __future__ import annotations

from football_tracking.locate_tracking.events.deduplicator import deduplicate_events
from football_tracking.locate_tracking.events.event_detector import detect_uncertainty_events
from football_tracking.locate_tracking.events.event_store import (
    read_events_jsonl,
    write_events_jsonl,
)
from football_tracking.locate_tracking.grounding_scheduler.cooldown import is_allowed_by_cooldown
from football_tracking.locate_tracking.grounding_scheduler.frame_selector import select_event_frames
from football_tracking.locate_tracking.grounding_scheduler.planner import build_grounding_plan
from football_tracking.locate_tracking.grounding_scheduler.priority import sort_events_by_priority
from football_tracking.locate_tracking.grounding_scheduler.schemas import SchedulerConfig
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def _event(frame_start: int, frame_end: int, *, severity: str = "warning"):
    signal = make_signal(
        signal_type="TARGET_PRESENCE",
        frame_start=frame_start,
        frame_end=frame_end,
        frame_index=frame_end,
        raw_track_id=7,
        score=float(frame_end - frame_start + 1),
        severity=severity,  # type: ignore[arg-type]
        threshold=2.0,
        triggered=True,
    )
    return detect_uncertainty_events((signal,))[0]


def test_event_deduplicator_merges_overlapping_same_type() -> None:
    events = (_event(10, 20), _event(15, 25))
    merged = deduplicate_events(events)

    assert len(merged) == 1
    assert merged[0].frame_start == 10
    assert merged[0].frame_end == 25


def test_event_store_roundtrip(tmp_path) -> None:
    events = (_event(10, 20),)
    path = write_events_jsonl(events, tmp_path / "events.jsonl", overwrite=True)

    assert read_events_jsonl(path) == events


def test_cooldown_boundary_and_critical_override() -> None:
    warning = _event(510, 510, severity="warning")
    critical = _event(510, 510, severity="critical")

    assert not is_allowed_by_cooldown(
        warning,
        (500,),
        cooldown_frames=50,
        critical_overrides=True,
    )
    assert is_allowed_by_cooldown(
        _event(560, 560, severity="warning"),
        (500,),
        cooldown_frames=50,
        critical_overrides=True,
    )
    assert is_allowed_by_cooldown(
        critical,
        (500,),
        cooldown_frames=50,
        critical_overrides=True,
    )


def test_frame_selection_window_representative() -> None:
    assert select_event_frames(
        _event(100, 200),
        strategy="window_representative",
        max_frames=3,
    ) == (
        100,
        150,
        200,
    )
    assert select_event_frames(
        _event(100, 101),
        strategy="window_representative",
        max_frames=3,
    ) == (
        100,
        101,
    )


def test_grounding_planner_applies_priority_cooldown_and_budget() -> None:
    events = (
        _event(100, 100, severity="warning"),
        _event(110, 110, severity="critical"),
        _event(300, 300, severity="warning"),
    )
    config = SchedulerConfig(
        cooldown_frames=50,
        max_requests_per_session=2,
        max_frames_per_request=1,
        frame_strategy="trigger",
        event_type_priority=("target_absent",),
    )

    plan = build_grounding_plan(
        events=events,
        query="player",
        source_video="video.avi",
        config=config,
    )

    assert [item.severity for item in plan.items] == ["critical", "warning"]
    assert any(item["reason"] == "cooldown" for item in plan.suppressed_events)
    assert sort_events_by_priority(events)[0].severity == "critical"

    budgeted = build_grounding_plan(
        events=events,
        query="player",
        source_video="video.avi",
        config=SchedulerConfig(
            cooldown_frames=0,
            max_requests_per_session=1,
            max_frames_per_request=1,
            frame_strategy="trigger",
        ),
    )
    assert len(budgeted.items) == 1
    assert any(item["reason"] == "budget" for item in budgeted.suppressed_events)
