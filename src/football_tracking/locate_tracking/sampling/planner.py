"""Build deterministic frame sampling plans."""

from __future__ import annotations

from football_tracking.locate_tracking.sampling.explicit_selector import select_explicit_frames
from football_tracking.locate_tracking.sampling.schemas import (
    FrameSamplingPlan,
    FrameSamplingRequest,
)
from football_tracking.locate_tracking.sampling.uniform_selector import select_uniform_frames


def build_frame_sampling_plan(request: FrameSamplingRequest) -> FrameSamplingPlan:
    selected = (
        select_explicit_frames(request)
        if request.effective_mode == "explicit"
        else select_uniform_frames(request)
    )
    return FrameSamplingPlan(request=request, selected_frames=selected)
