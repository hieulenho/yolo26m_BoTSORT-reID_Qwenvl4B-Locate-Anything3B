"""Appearance embedding provider abstraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from football_tracking.locate_tracking.appearance.schemas import AppearanceEmbedding


class AppearanceEmbeddingProvider(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def inference_config(self) -> dict[str, Any]: ...

    def embed_crop(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> AppearanceEmbedding: ...

    def embed_crops(
        self,
        images: Sequence[np.ndarray],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[AppearanceEmbedding]: ...

    def close(self) -> None: ...
