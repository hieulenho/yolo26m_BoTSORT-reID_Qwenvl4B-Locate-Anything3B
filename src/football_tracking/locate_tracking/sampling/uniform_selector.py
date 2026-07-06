"""Uniform temporal sampling with deterministic 1-based frame indices."""

from __future__ import annotations

from football_tracking.locate_tracking.sampling.schemas import (
    FrameSamplingError,
    FrameSamplingRequest,
    SelectedFrame,
)


def select_uniform_frames(request: FrameSamplingRequest) -> tuple[SelectedFrame, ...]:
    """Select sorted unique frames across the inclusive request range."""

    start = request.start_frame
    end = request.end_frame or request.total_frames
    available = end - start + 1
    if available < 1:
        raise FrameSamplingError("uniform sampling range is empty.")
    if available <= request.max_frames:
        indices = list(range(start, end + 1))
    elif request.max_frames == 1:
        indices = [start]
    else:
        step = (available - 1) / (request.max_frames - 1)
        indices = [round(start + step * offset) for offset in range(request.max_frames)]
        indices[0] = start
        indices[-1] = end
        indices = sorted(set(indices))
        if len(indices) < request.max_frames:
            used = set(indices)
            for candidate in range(start, end + 1):
                if candidate not in used:
                    indices.append(candidate)
                    used.add(candidate)
                if len(indices) == request.max_frames:
                    break
            indices = sorted(indices)
    return tuple(
        SelectedFrame(
            frame_index=frame_index,
            selection_order=order,
            selection_reason="uniform_temporal_sample",
        )
        for order, frame_index in enumerate(indices, 1)
    )
