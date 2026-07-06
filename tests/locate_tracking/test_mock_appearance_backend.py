from __future__ import annotations

import numpy as np

from football_tracking.locate_tracking.appearance.mock_backend import (
    MockAppearanceEmbeddingProvider,
)


def test_mock_backend_returns_configured_normalized_embedding() -> None:
    backend = MockAppearanceEmbeddingProvider({(7, 1): (3, 4, 0)})

    embedding = backend.embed_crop(
        np.zeros((8, 8, 3), dtype=np.uint8),
        metadata={"source_track_id": 7, "source_frame_index": 1},
    )

    assert embedding.backend == "mock"
    assert embedding.dimension == 3
    assert np.isclose(np.linalg.norm(np.array(embedding.vector)), 1.0)
    assert backend.call_count == 1


def test_mock_backend_batch_and_close() -> None:
    backend = MockAppearanceEmbeddingProvider()
    embeddings = backend.embed_crops([np.zeros((8, 8, 3), dtype=np.uint8)] * 2)
    backend.close()

    assert len(embeddings) == 2
    assert backend.call_count == 2
    assert backend.closed is True
