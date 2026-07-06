"""Budget helpers for grounding plans."""

from __future__ import annotations

from football_tracking.locate_tracking.grounding_scheduler.schemas import GroundingPlanItem


def apply_session_budget(
    items: tuple[GroundingPlanItem, ...],
    *,
    max_requests_per_session: int,
) -> tuple[tuple[GroundingPlanItem, ...], tuple[GroundingPlanItem, ...]]:
    if max_requests_per_session < 0:
        raise ValueError("max_requests_per_session must be >= 0.")
    return items[:max_requests_per_session], items[max_requests_per_session:]
