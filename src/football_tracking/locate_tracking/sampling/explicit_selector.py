"""Explicit frame-list selector."""

from __future__ import annotations

from collections.abc import Iterable

from football_tracking.locate_tracking.sampling.schemas import (
    FrameSamplingError,
    FrameSamplingRequest,
    SelectedFrame,
)


def parse_explicit_frames(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        raw_items: Iterable[object] = [item.strip() for item in value.split(",")]
    else:
        raw_items = value
    parsed: list[int] = []
    for item in raw_items:
        try:
            frame_index = int(item)
        except (TypeError, ValueError) as exc:
            raise FrameSamplingError(f"Invalid explicit frame index: {item}") from exc
        parsed.append(frame_index)
    return tuple(parsed)


def select_explicit_frames(request: FrameSamplingRequest) -> tuple[SelectedFrame, ...]:
    if not request.explicit_frames:
        raise FrameSamplingError("explicit frame list must not be empty.")
    selected = sorted(set(request.explicit_frames))
    for frame_index in selected:
        if frame_index < 1:
            raise FrameSamplingError("explicit frame indices must be >= 1.")
        if frame_index > request.total_frames:
            raise FrameSamplingError(
                f"explicit frame {frame_index} exceeds total_frames {request.total_frames}."
            )
        if frame_index < request.start_frame or frame_index > (
            request.end_frame or request.total_frames
        ):
            raise FrameSamplingError(
                f"explicit frame {frame_index} is outside the configured frame range."
            )
    return tuple(
        SelectedFrame(
            frame_index=frame_index,
            selection_order=order,
            selection_reason="explicit_user_frame",
        )
        for order, frame_index in enumerate(selected, 1)
    )
