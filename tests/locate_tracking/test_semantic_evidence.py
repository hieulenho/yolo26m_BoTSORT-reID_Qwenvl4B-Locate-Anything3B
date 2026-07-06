from __future__ import annotations

from football_tracking.locate_tracking.semantic_memory.evidence import (
    evidence_from_frame_resolution,
)
from tests.locate_tracking.semantic_test_utils import (
    ambiguous_frame,
    association,
    candidate,
    frame_resolution,
)


def test_resolved_frame_creates_full_positive_support() -> None:
    result = frame_resolution(
        1,
        (
            association(
                frame_index=1,
                status="resolved",
                selected_track_id=7,
                candidates=(candidate(7, 1, 0.8),),
            ),
        ),
    )

    evidence = evidence_from_frame_resolution(result)

    assert len(evidence) == 1
    assert evidence[0].raw_track_id == 7
    assert evidence[0].selected_in_frame is True
    assert evidence[0].is_positive_support is True
    assert evidence[0].evidence_weight == 1.0


def test_ambiguous_frame_preserves_weak_candidates_without_positive_vote() -> None:
    evidence = evidence_from_frame_resolution(ambiguous_frame(2))

    assert {item.raw_track_id for item in evidence} == {3, 7}
    assert all(not item.is_positive_support for item in evidence)
    assert all(item.evidence_weight == 0.25 for item in evidence)


def test_not_found_frame_does_not_create_positive_support() -> None:
    result = frame_resolution(
        3,
        (
            association(
                frame_index=3,
                status="not_found",
                selected_track_id=None,
                candidates=(candidate(11, 3, 0.1, passed=False),),
            ),
        ),
    )

    evidence = evidence_from_frame_resolution(result)

    assert len(evidence) == 1
    assert evidence[0].is_positive_support is False
    assert evidence[0].evidence_weight == 0.0


def test_no_grounding_boxes_produce_no_evidence() -> None:
    assert evidence_from_frame_resolution(frame_resolution(4, ())) == ()


def test_multiple_candidates_are_preserved() -> None:
    result = frame_resolution(
        5,
        (
            association(
                frame_index=5,
                status="resolved",
                selected_track_id=1,
                candidates=(candidate(1, 5, 0.8), candidate(2, 5, 0.4, rank=2)),
            ),
        ),
    )

    evidence = evidence_from_frame_resolution(result)

    assert [item.raw_track_id for item in evidence] == [1, 2]
    assert evidence[1].evidence_reason == "non_selected_passed_candidate"
