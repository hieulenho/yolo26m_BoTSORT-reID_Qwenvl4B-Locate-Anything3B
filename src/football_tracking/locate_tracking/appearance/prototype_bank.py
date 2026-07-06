"""Track appearance prototype construction."""

from __future__ import annotations

from statistics import mean

import numpy as np

from football_tracking.locate_tracking.appearance.embedding_utils import (
    EmbeddingValidationError,
    embedding_dimension_check,
    l2_normalize,
    vectors_to_tuple,
)
from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceSchemaError,
    PrototypeAggregationStrategy,
    TrackAppearancePrototype,
    TrackEmbeddingSample,
)
from football_tracking.locate_tracking.appearance.similarity import cosine_similarity_01


class PrototypeBuildError(RuntimeError):
    """Raised when a track appearance prototype cannot be built."""


def _sample_vectors(samples: tuple[TrackEmbeddingSample, ...]) -> list[np.ndarray]:
    vectors = [np.asarray(sample.embedding.vector, dtype=np.float32) for sample in samples]
    if not vectors:
        raise PrototypeBuildError("At least one embedding sample is required.")
    first = vectors[0]
    for index, vector in enumerate(vectors[1:], 1):
        try:
            embedding_dimension_check(
                first, vector, first_name="sample[0]", second_name=f"sample[{index}]"
            )
        except EmbeddingValidationError as exc:
            raise PrototypeBuildError(str(exc)) from exc
    return vectors


def aggregate_prototype_vector(
    samples: tuple[TrackEmbeddingSample, ...],
    *,
    strategy: PrototypeAggregationStrategy = "mean",
) -> np.ndarray:
    vectors = _sample_vectors(samples)
    matrix = np.stack(vectors, axis=0)
    if strategy == "mean":
        raw = matrix.mean(axis=0)
    elif strategy == "quality_weighted_mean":
        weights = np.asarray([sample.quality_weight for sample in samples], dtype=np.float32)
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise PrototypeBuildError("quality weights must be non-negative and sum to > 0.")
        raw = np.average(matrix, axis=0, weights=weights)
    else:
        raise PrototypeBuildError(f"Unsupported prototype aggregation strategy: {strategy}")
    return l2_normalize(raw, name="prototype")


def build_track_prototype(
    *,
    raw_track_id: int,
    samples: tuple[TrackEmbeddingSample, ...],
    strategy: PrototypeAggregationStrategy = "mean",
) -> TrackAppearancePrototype:
    vector = aggregate_prototype_vector(samples, strategy=strategy)
    backends = {sample.embedding.backend for sample in samples}
    model_ids = {sample.embedding.model_id for sample in samples}
    if len(backends) != 1 or len(model_ids) != 1:
        raise PrototypeBuildError("All prototype samples must share backend and model_id.")
    return TrackAppearancePrototype(
        raw_track_id=raw_track_id,
        sample_embeddings=samples,
        sample_frame_indices=tuple(sample.crop_reference.frame_index for sample in samples),
        prototype_vector=vectors_to_tuple(vector),
        sample_count=len(samples),
        aggregation_strategy=strategy,
        quality_metadata={
            "quality_weights": [sample.quality_weight for sample in samples],
            "mean_quality_score": mean(sample.quality_weight for sample in samples),
        },
        backend=next(iter(backends)),
        model_id=next(iter(model_ids)),
        embedding_dimension=int(vector.size),
    )


def leave_one_out_prototype_vector(
    samples: tuple[TrackEmbeddingSample, ...],
    *,
    exclude_index: int,
    strategy: PrototypeAggregationStrategy = "mean",
) -> np.ndarray:
    if len(samples) < 2:
        raise PrototypeBuildError("Leave-one-out prototype requires at least two samples.")
    if not 0 <= exclude_index < len(samples):
        raise AppearanceSchemaError("exclude_index out of range.")
    remaining = tuple(sample for index, sample in enumerate(samples) if index != exclude_index)
    return aggregate_prototype_vector(remaining, strategy=strategy)


def internal_consistency_score(
    samples: tuple[TrackEmbeddingSample, ...],
    *,
    strategy: PrototypeAggregationStrategy = "mean",
) -> float | None:
    if len(samples) < 2:
        return None
    scores = []
    for index, sample in enumerate(samples):
        prototype = leave_one_out_prototype_vector(samples, exclude_index=index, strategy=strategy)
        scores.append(cosine_similarity_01(prototype, np.asarray(sample.embedding.vector)))
    return float(mean(scores))
