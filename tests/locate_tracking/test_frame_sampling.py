from __future__ import annotations

import pytest

from football_tracking.locate_tracking.sampling.explicit_selector import parse_explicit_frames
from football_tracking.locate_tracking.sampling.planner import build_frame_sampling_plan
from football_tracking.locate_tracking.sampling.schemas import (
    FrameSamplingError,
    FrameSamplingRequest,
)


def _indices(request: FrameSamplingRequest) -> tuple[int, ...]:
    return build_frame_sampling_plan(request).frame_indices


def test_uniform_one_frame_video() -> None:
    assert _indices(FrameSamplingRequest(total_frames=1, max_frames=5)) == (1,)


def test_uniform_short_range_returns_all_frames() -> None:
    assert _indices(FrameSamplingRequest(total_frames=4, max_frames=10)) == (1, 2, 3, 4)


def test_uniform_normal_range_matches_expected_example() -> None:
    assert _indices(FrameSamplingRequest(total_frames=10, max_frames=3)) == (1, 6, 10)


def test_uniform_respects_start_boundary() -> None:
    assert _indices(FrameSamplingRequest(total_frames=10, start_frame=3, max_frames=2)) == (3, 10)


def test_uniform_respects_last_frame() -> None:
    assert _indices(FrameSamplingRequest(total_frames=10, end_frame=8, max_frames=3))[-1] == 8


def test_uniform_has_no_duplicates_and_is_deterministic() -> None:
    request = FrameSamplingRequest(total_frames=11, max_frames=6)
    assert _indices(request) == _indices(request)
    assert len(_indices(request)) == len(set(_indices(request)))


def test_invalid_start_frame_rejected() -> None:
    with pytest.raises(FrameSamplingError):
        FrameSamplingRequest(total_frames=10, start_frame=0)


def test_invalid_end_frame_rejected() -> None:
    with pytest.raises(FrameSamplingError):
        FrameSamplingRequest(total_frames=10, end_frame=11)


def test_start_after_end_rejected() -> None:
    with pytest.raises(FrameSamplingError):
        FrameSamplingRequest(total_frames=10, start_frame=8, end_frame=4)


def test_explicit_list_is_sorted_and_deduplicated() -> None:
    request = FrameSamplingRequest(total_frames=10, explicit_frames=(5, 2, 5, 8))
    assert _indices(request) == (2, 5, 8)


def test_explicit_rejects_zero() -> None:
    request = FrameSamplingRequest(total_frames=10, explicit_frames=(0, 2))
    with pytest.raises(FrameSamplingError):
        build_frame_sampling_plan(request)


def test_explicit_rejects_out_of_range() -> None:
    request = FrameSamplingRequest(total_frames=10, explicit_frames=(2, 12))
    with pytest.raises(FrameSamplingError):
        build_frame_sampling_plan(request)


def test_explicit_overrides_uniform_mode() -> None:
    request = FrameSamplingRequest(total_frames=10, max_frames=3, explicit_frames=(4, 9))
    plan = build_frame_sampling_plan(request)
    assert plan.frame_indices == (4, 9)
    assert plan.selected_frames[0].selection_reason == "explicit_user_frame"


def test_parse_explicit_frames_from_csv() -> None:
    assert parse_explicit_frames("3, 1, 3") == (3, 1, 3)


def test_parse_invalid_explicit_frames() -> None:
    with pytest.raises(FrameSamplingError):
        parse_explicit_frames("1,nope")
