"""Cosine similarity utilities for appearance embeddings."""

from __future__ import annotations

import numpy as np

from football_tracking.locate_tracking.appearance.embedding_utils import (
    embedding_dimension_check,
    l2_normalize,
)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    embedding_dimension_check(first, second, first_name="first", second_name="second")
    first_norm = l2_normalize(first, name="first")
    second_norm = l2_normalize(second, name="second")
    score = float(np.dot(first_norm, second_norm))
    return max(-1.0, min(1.0, score))


def cosine_similarity_01(first: np.ndarray, second: np.ndarray) -> float:
    return (cosine_similarity(first, second) + 1.0) / 2.0


def score_candidates_against_prototype(
    prototype: np.ndarray,
    candidates: tuple[np.ndarray, ...],
) -> tuple[float, ...]:
    return tuple(cosine_similarity_01(prototype, candidate) for candidate in candidates)
