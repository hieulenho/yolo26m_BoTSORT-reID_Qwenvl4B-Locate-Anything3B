from __future__ import annotations

import numpy as np
import pytest

from football_tracking.locate_tracking.appearance.embedding_utils import (
    EmbeddingValidationError,
    batch_l2_normalize,
    embedding_dimension_check,
    l2_norm,
    l2_normalize,
    validate_embedding,
)


def test_l2_normalize_returns_unit_vector() -> None:
    normalized = l2_normalize([3, 4])

    assert np.isclose(l2_norm(normalized), 1.0)


def test_rejects_nan_infinite_empty_and_zero_vectors() -> None:
    for vector in ([float("nan")], [float("inf")], []):
        with pytest.raises(EmbeddingValidationError):
            validate_embedding(vector)
    with pytest.raises(EmbeddingValidationError):
        l2_normalize([0, 0, 0])


def test_batch_normalize_and_dimension_check() -> None:
    normalized = batch_l2_normalize(([1, 0], [0, 2]))

    assert len(normalized) == 2
    assert embedding_dimension_check(normalized[0], normalized[1]) == 2


def test_dimension_mismatch_rejected() -> None:
    with pytest.raises(EmbeddingValidationError):
        embedding_dimension_check([1, 0], [1, 0, 0])
