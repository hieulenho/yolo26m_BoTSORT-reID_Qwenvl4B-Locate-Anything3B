from __future__ import annotations

import numpy as np

from football_tracking.locate_tracking.appearance.cache import AppearanceEmbeddingCache
from football_tracking.locate_tracking.appearance.mock_backend import (
    MockAppearanceEmbeddingProvider,
)


def test_appearance_cache_hits_and_misses(tmp_path) -> None:
    image = np.ones((8, 8, 3), dtype=np.uint8)
    cache = AppearanceEmbeddingCache(tmp_path / "cache")
    backend = MockAppearanceEmbeddingProvider()
    config = backend.inference_config()
    lookup = cache.get(
        image=image,
        backend_name=backend.backend_name,
        model_id=backend.model_id,
        inference_config=config,
    )
    assert lookup.cache_hit is False
    embedding = backend.embed_crop(image)
    cache.set(embedding, lookup.cache_key)

    hit = cache.get(
        image=image,
        backend_name=backend.backend_name,
        model_id=backend.model_id,
        inference_config=config,
    )

    assert hit.cache_hit is True
    assert hit.embedding == embedding
    assert backend.call_count == 1


def test_appearance_cache_key_changes_with_model_crop_and_config(tmp_path) -> None:
    cache = AppearanceEmbeddingCache(tmp_path / "cache")
    image = np.ones((8, 8, 3), dtype=np.uint8)
    changed_image = np.zeros((8, 8, 3), dtype=np.uint8)

    base = cache.cache_key(
        image=image,
        backend_name="mock",
        model_id="a",
        inference_config={"normalize": True},
    )

    assert base != cache.cache_key(
        image=image,
        backend_name="mock",
        model_id="b",
        inference_config={"normalize": True},
    )
    assert base != cache.cache_key(
        image=changed_image,
        backend_name="mock",
        model_id="a",
        inference_config={"normalize": True},
    )
    assert base != cache.cache_key(
        image=image,
        backend_name="mock",
        model_id="a",
        inference_config={"normalize": False},
    )
