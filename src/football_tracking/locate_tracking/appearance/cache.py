"""JSON cache for appearance embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_tracking.locate_tracking.appearance.schemas import AppearanceEmbedding


@dataclass(frozen=True)
class AppearanceCacheLookup:
    embedding: AppearanceEmbedding | None
    cache_hit: bool
    cache_key: str | None
    error: str | None = None


class AppearanceEmbeddingCache:
    def __init__(
        self, directory: str | Path, *, enabled: bool = True, overwrite: bool = False
    ) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.overwrite = bool(overwrite)

    def cache_key(
        self,
        *,
        image: np.ndarray,
        backend_name: str,
        model_id: str,
        inference_config: dict[str, Any],
    ) -> str:
        payload = {
            "image_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "backend_name": backend_name,
            "model_id": model_id,
            "inference_config": inference_config,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(
        self,
        *,
        image: np.ndarray,
        backend_name: str,
        model_id: str,
        inference_config: dict[str, Any],
    ) -> AppearanceCacheLookup:
        if not self.enabled:
            return AppearanceCacheLookup(None, False, None)
        key = self.cache_key(
            image=image,
            backend_name=backend_name,
            model_id=model_id,
            inference_config=inference_config,
        )
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return AppearanceCacheLookup(None, False, key)
        try:
            embedding = AppearanceEmbedding.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            return AppearanceCacheLookup(None, False, key, error=str(exc))
        return AppearanceCacheLookup(embedding, True, key)

    def set(self, embedding: AppearanceEmbedding, cache_key: str | None) -> Path | None:
        if not self.enabled or cache_key is None:
            return None
        path = self.directory / f"{cache_key}.json"
        if path.exists() and not self.overwrite:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(embedding.to_dict(include_vector=True), indent=2, default=str),
            encoding="utf-8",
        )
        return path
