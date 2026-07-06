from __future__ import annotations

import numpy as np
import pytest

from football_tracking.locate_tracking.appearance.prototype_bank import (
    PrototypeBuildError,
    build_track_prototype,
    internal_consistency_score,
    leave_one_out_prototype_vector,
)
from tests.locate_tracking.appearance_test_utils import embedding_sample


def test_mean_prototype_is_normalized_and_preserves_metadata() -> None:
    samples = (
        embedding_sample((1.0, 0.0, 0.0), frame_index=1),
        embedding_sample((0.98, 0.02, 0.0), frame_index=6),
    )

    prototype = build_track_prototype(raw_track_id=7, samples=samples)

    assert prototype.sample_count == 2
    assert prototype.sample_frame_indices == (1, 6)
    assert np.isclose(np.linalg.norm(np.array(prototype.prototype_vector)), 1.0)


def test_quality_weighted_prototype_rejects_all_zero_weights() -> None:
    samples = (
        embedding_sample((1.0, 0.0), quality_weight=0.0),
        embedding_sample((0.0, 1.0), frame_index=2, quality_weight=0.0),
    )

    with pytest.raises(PrototypeBuildError):
        build_track_prototype(
            raw_track_id=7,
            samples=samples,
            strategy="quality_weighted_mean",
        )


def test_embedding_dimension_mismatch_rejected() -> None:
    samples = (
        embedding_sample((1.0, 0.0)),
        embedding_sample((1.0, 0.0, 0.0), frame_index=2),
    )

    with pytest.raises(PrototypeBuildError):
        build_track_prototype(raw_track_id=7, samples=samples)


def test_leave_one_out_prevents_self_similarity_leakage() -> None:
    samples = (
        embedding_sample((1.0, 0.0, 0.0), frame_index=1),
        embedding_sample((0.0, 1.0, 0.0), frame_index=2),
        embedding_sample((0.0, 0.0, 1.0), frame_index=3),
    )

    loo = leave_one_out_prototype_vector(samples, exclude_index=2)

    expected = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(loo, expected)


def test_internal_consistency_high_low_and_unavailable() -> None:
    high = (
        embedding_sample((1.0, 0.0), frame_index=1),
        embedding_sample((0.99, 0.01), frame_index=2),
        embedding_sample((0.98, 0.02), frame_index=3),
    )
    low = (
        embedding_sample((1.0, 0.0), frame_index=1),
        embedding_sample((0.99, 0.01), frame_index=2),
        embedding_sample((0.0, 1.0), frame_index=3),
    )

    assert internal_consistency_score(high) > internal_consistency_score(low)
    assert internal_consistency_score((embedding_sample((1.0, 0.0)),)) is None
