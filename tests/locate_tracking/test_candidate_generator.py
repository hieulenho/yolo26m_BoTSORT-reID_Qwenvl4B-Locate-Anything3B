from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.association.candidate_generator import generate_candidates
from football_tracking.locate_tracking.association.schemas import AssociationConfig
from football_tracking.locate_tracking.grounding.schemas import GroundedBox


def _box() -> GroundedBox:
    return GroundedBox(
        label="player",
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        normalized_bbox=(0, 0, 1000, 1000),
        confidence=None,
        query="player",
    )


def _obs(track_id: int, bbox) -> MotTrackObservation:
    x1, y1, x2, y2 = bbox
    return MotTrackObservation(
        frame_index=1,
        track_id=track_id,
        bbox_ltwh=(x1, y1, x2 - x1, y2 - y1),
        bbox_xyxy=bbox,
        confidence=None,
        source_path=Path("tracks.txt"),
        line_number=track_id,
    )


def test_candidate_generator_ranks_passed_candidates() -> None:
    candidates = generate_candidates(
        grounded_box=_box(),
        grounded_box_index=0,
        track_observations=(_obs(7, (0, 0, 100, 100)), _obs(3, (0, 0, 50, 50))),
        frame_width=100,
        frame_height=100,
        config=AssociationConfig(),
    )

    passed = [candidate for candidate in candidates if candidate.passed_gate]
    assert [candidate.track_id for candidate in passed] == [7, 3]
    assert [candidate.rank for candidate in passed] == [1, 2]


def test_candidate_generator_allows_coarse_grounding_center_coverage_gate() -> None:
    candidates = generate_candidates(
        grounded_box=_box(),
        grounded_box_index=0,
        track_observations=(_obs(5, (40, 40, 50, 50)),),
        frame_width=100,
        frame_height=100,
        config=AssociationConfig(min_iou=0.9, min_track_coverage=0.9),
    )

    assert candidates[0].passed_gate
    assert "coverage" in candidates[0].gate_reason


def test_candidate_generator_records_rejected_track_reason() -> None:
    candidates = generate_candidates(
        grounded_box=_box(),
        grounded_box_index=0,
        track_observations=(_obs(1, (150, 150, 160, 160)),),
        frame_width=100,
        frame_height=100,
        config=AssociationConfig(),
    )

    assert not candidates[0].passed_gate
    assert "outside" in candidates[0].gate_reason
