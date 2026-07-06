from __future__ import annotations

from football_tracking.locate_tracking.appearance.prototype_bank import build_track_prototype
from football_tracking.locate_tracking.appearance.verifier import (
    AppearanceVerifierConfig,
    score_appearance_prototypes,
    score_track_appearance,
)
from tests.locate_tracking.appearance_test_utils import embedding_sample


def test_appearance_verifier_marks_consistent_track_verified() -> None:
    prototype = build_track_prototype(
        raw_track_id=7,
        samples=(
            embedding_sample((1.0, 0.0), frame_index=1),
            embedding_sample((0.99, 0.01), frame_index=2),
        ),
    )

    score = score_track_appearance(prototype)

    assert score.evidence_status == "verified"
    assert score.appearance_score is not None
    assert score.appearance_score > 0.9


def test_appearance_verifier_reports_insufficient_evidence_for_one_sample() -> None:
    prototype = build_track_prototype(
        raw_track_id=7,
        samples=(embedding_sample((1.0, 0.0), frame_index=1),),
    )

    score = score_track_appearance(prototype)

    assert score.evidence_status == "insufficient_appearance_evidence"
    assert score.appearance_score is None


def test_appearance_verifier_multiple_candidates_are_deterministically_ranked() -> None:
    consistent = build_track_prototype(
        raw_track_id=7,
        samples=(
            embedding_sample((1.0, 0.0), frame_index=1),
            embedding_sample((0.99, 0.01), frame_index=2),
        ),
    )
    weak = build_track_prototype(
        raw_track_id=11,
        samples=(
            embedding_sample((1.0, 0.0), track_id=11, frame_index=1),
            embedding_sample((0.0, 1.0), track_id=11, frame_index=2),
        ),
    )

    scores = score_appearance_prototypes(
        (weak, consistent),
        AppearanceVerifierConfig(min_verified_score=0.7),
    )

    assert [score.raw_track_id for score in scores] == [7, 11]
