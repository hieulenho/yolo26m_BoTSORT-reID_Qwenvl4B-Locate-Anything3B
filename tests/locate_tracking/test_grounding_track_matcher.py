from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.matcher import GroundingTrackMatcher
from football_tracking.locate_tracking.association.schemas import AssociationConfig
from football_tracking.locate_tracking.grounding.schemas import (
    GroundedBox,
    GroundingRequest,
    GroundingResult,
    GroundingRuntimeInfo,
)


def _request(query: str = "player") -> GroundingRequest:
    return GroundingRequest(
        image_path=Path("frame.jpg"),
        query=query,
        backend="mock",
        model_id="mock",
    )


def _grounding(boxes) -> GroundingResult:
    return GroundingResult(
        request=_request(),
        image_width=200,
        image_height=200,
        boxes=tuple(boxes),
        raw_response="mock",
        runtime_info=GroundingRuntimeInfo(backend="mock", model_id="mock"),
    )


def _box(x1, y1, x2, y2, label="player") -> GroundedBox:
    return GroundedBox(
        label=label,
        bbox_xyxy=(x1, y1, x2, y2),
        normalized_bbox=(1, 1, 999, 999),
        confidence=None,
        query="player",
    )


def _obs(track_id: int, bbox, frame: int = 1) -> MotTrackObservation:
    x1, y1, x2, y2 = bbox
    return MotTrackObservation(
        frame_index=frame,
        track_id=track_id,
        bbox_ltwh=(x1, y1, x2 - x1, y2 - y1),
        bbox_xyxy=bbox,
        confidence=None,
        source_path=Path("tracks.txt"),
        line_number=track_id,
    )


def _match(grounding, tracks, config=None):
    return GroundingTrackMatcher(config).match(
        grounding_result=grounding,
        track_observations=tuple(tracks),
        frame_width=200,
        frame_height=200,
        source_video="video.mp4",
        tracks_path="tracks.txt",
        frame_index=1,
    )


def test_matcher_perfect_resolution() -> None:
    result = _match(_grounding([_box(10, 10, 50, 50)]), [_obs(7, (10, 10, 50, 50))])

    assert result.overall_status == "resolved"
    assert result.associations[0].selected_track_id == 7


def test_matcher_partial_overlap_ranking() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(1, (0, 0, 30, 30)), _obs(2, (0, 0, 80, 80))],
    )

    assert result.associations[0].candidates[0].track_id == 2


def test_matcher_coarse_grounding_box_passes_center_coverage_gate() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(4, (40, 40, 50, 50))],
        AssociationConfig(min_iou=0.9, min_track_coverage=0.9),
    )

    assert result.associations[0].status == "resolved"
    assert result.associations[0].selected_track_id == 4


def test_matcher_no_active_tracks_and_no_grounding_boxes() -> None:
    assert _match(_grounding([_box(0, 0, 10, 10)]), []).overall_status == "not_found"
    assert _match(_grounding([]), [_obs(1, (0, 0, 10, 10))]).overall_status == "not_found"


def test_matcher_ambiguous_when_two_candidates_close() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(1, (0, 0, 60, 60)), _obs(2, (40, 40, 100, 100))],
        AssociationConfig(ambiguity_margin=0.5),
    )

    assert result.associations[0].status == "ambiguous"
    assert result.associations[0].selected_track_id is None


def test_matcher_clear_winner() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(1, (0, 0, 100, 100)), _obs(2, (120, 120, 160, 160))],
    )

    assert result.associations[0].status == "resolved"
    assert result.associations[0].selected_track_id == 1


def test_matcher_multiple_grounding_boxes() -> None:
    result = _match(
        _grounding([_box(0, 0, 20, 20), _box(100, 100, 150, 150)]),
        [_obs(1, (0, 0, 20, 20)), _obs(2, (100, 100, 150, 150))],
    )

    assert [association.selected_track_id for association in result.associations] == [1, 2]


def test_matcher_duplicate_assignment_conflict() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100), _box(0, 0, 100, 100)]),
        [_obs(7, (0, 0, 100, 100))],
    )

    assert result.associations[0].status == "resolved"
    assert result.associations[1].status == "ambiguous"


def test_matcher_deterministic_tie_breaks_by_track_id() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(2, (0, 0, 100, 100)), _obs(1, (0, 0, 100, 100))],
        AssociationConfig(ambiguity_margin=0.0),
    )

    assert result.associations[0].candidates[0].track_id == 1


def test_matcher_below_minimum_score() -> None:
    result = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(1, (0, 0, 100, 100))],
        AssociationConfig(min_score=1.0),
    )

    assert result.associations[0].status == "resolved"

    strict = _match(
        _grounding([_box(0, 0, 100, 100)]),
        [_obs(1, (0, 0, 50, 50))],
        AssociationConfig(min_score=0.99),
    )
    assert strict.associations[0].status == "not_found"
