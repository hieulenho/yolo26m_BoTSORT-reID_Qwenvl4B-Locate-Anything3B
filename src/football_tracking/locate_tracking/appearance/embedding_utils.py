"""Pure embedding validation and normalization utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


class EmbeddingValidationError(ValueError):
    """Raised when an embedding vector is invalid."""


def validate_embedding(
    vector: Iterable[float] | np.ndarray, *, name: str = "embedding"
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise EmbeddingValidationError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise EmbeddingValidationError(f"{name} must contain finite values.")
    return array


def l2_norm(vector: Iterable[float] | np.ndarray, *, name: str = "embedding") -> float:
    array = validate_embedding(vector, name=name)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm):
        raise EmbeddingValidationError(f"{name} norm must be finite.")
    return norm


def l2_normalize(vector: Iterable[float] | np.ndarray, *, name: str = "embedding") -> np.ndarray:
    array = validate_embedding(vector, name=name)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise EmbeddingValidationError(f"{name} must not be a zero vector.")
    return (array / norm).astype(np.float32)


def batch_l2_normalize(vectors: Iterable[Iterable[float] | np.ndarray]) -> list[np.ndarray]:
    return [
        l2_normalize(vector, name=f"embedding[{index}]") for index, vector in enumerate(vectors)
    ]


def embedding_dimension_check(
    first: Iterable[float] | np.ndarray,
    second: Iterable[float] | np.ndarray,
    *,
    first_name: str = "first",
    second_name: str = "second",
) -> int:
    first_array = validate_embedding(first, name=first_name)
    second_array = validate_embedding(second, name=second_name)
    if first_array.shape != second_array.shape:
        raise EmbeddingValidationError(
            f"Embedding dimension mismatch: {first_name}={first_array.size}, "
            f"{second_name}={second_array.size}."
        )
    return int(first_array.size)


def vectors_to_tuple(vector: Iterable[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in validate_embedding(vector))
