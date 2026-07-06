"""Deterministic mock appearance embedding provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from football_tracking.locate_tracking.appearance.embedding_utils import (
    l2_normalize,
    vectors_to_tuple,
)
from football_tracking.locate_tracking.appearance.schemas import AppearanceEmbedding


class MockAppearanceEmbeddingProvider:
    """Test backend that maps track/frame keys or image statistics to vectors."""

    def __init__(
        self,
        embeddings: Mapping[tuple[int, int], Sequence[float]] | None = None,
        *,
        default_embedding: Sequence[float] = (1.0, 0.0, 0.0),
        model_id: str = "mock-appearance",
        normalize: bool = True,
    ) -> None:
        self._embeddings = {key: tuple(value) for key, value in (embeddings or {}).items()}
        self._default_embedding = tuple(default_embedding)
        self._model_id = model_id
        self.normalize = bool(normalize)
        self.call_count = 0
        self.closed = False

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return self._model_id

    def inference_config(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "response_count": len(self._embeddings),
            "normalize": self.normalize,
        }

    def _vector_for(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[float, ...]:
        metadata = dict(metadata or {})
        track_id = metadata.get("source_track_id")
        frame_index = metadata.get("source_frame_index")
        if track_id is not None and frame_index is not None:
            key = (int(track_id), int(frame_index))
            if key in self._embeddings:
                return self._embeddings[key]
        # Deterministic fallback based on image brightness, useful for smoke tests.
        mean_value = float(np.asarray(image).mean()) if np.asarray(image).size else 0.0
        return (mean_value + 1.0, 1.0, 0.5) if not self._embeddings else self._default_embedding

    def embed_crop(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> AppearanceEmbedding:
        self.call_count += 1
        vector = np.asarray(self._vector_for(image, metadata), dtype=np.float32)
        if self.normalize:
            vector = l2_normalize(vector, name="mock embedding")
        metadata_dict = dict(metadata or {})
        return AppearanceEmbedding(
            backend=self.backend_name,
            model_id=self.model_id,
            dimension=int(vector.size),
            vector=vectors_to_tuple(vector),
            normalized=self.normalize,
            source_track_id=metadata_dict.get("source_track_id"),
            source_frame_index=metadata_dict.get("source_frame_index"),
            metadata={"mock": True, **metadata_dict},
        )

    def embed_crops(
        self,
        images: Sequence[np.ndarray],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[AppearanceEmbedding]:
        metadata_items = metadata if metadata is not None else [None] * len(images)
        return [
            self.embed_crop(image, metadata=item)
            for image, item in zip(images, metadata_items, strict=True)
        ]

    def close(self) -> None:
        self.closed = True
