from __future__ import annotations

import numpy as np
import pytest

from football_tracking.locate_tracking.appearance.embedding_utils import (
    EmbeddingValidationError,
)
from football_tracking.locate_tracking.appearance.similarity import (
    cosine_similarity,
    cosine_similarity_01,
    score_candidates_against_prototype,
)


def test_cosine_similarity_for_identical_and_orthogonal_vectors() -> None:
    assert np.isclose(cosine_similarity(np.array([1, 0]), np.array([2, 0])), 1.0)
    assert np.isclose(cosine_similarity_01(np.array([1, 0]), np.array([0, 1])), 0.5)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingValidationError):
        cosine_similarity(np.array([1, 0]), np.array([1, 0, 0]))


def test_batch_candidate_scoring() -> None:
    scores = score_candidates_against_prototype(
        np.array([1, 0]),
        (np.array([1, 0]), np.array([0, 1])),
    )

    assert scores == (1.0, 0.5)
