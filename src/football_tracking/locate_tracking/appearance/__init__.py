"""Appearance verification utilities for LocateAnything-guided tracking."""

from football_tracking.locate_tracking.appearance.backend import (
    AppearanceEmbeddingProvider,
)
from football_tracking.locate_tracking.appearance.mock_backend import (
    MockAppearanceEmbeddingProvider,
)
from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceCandidateScore,
    AppearanceEmbedding,
    AppearanceVerificationResult,
    CropQualityMetrics,
    CropReference,
    TrackAppearancePrototype,
)

__all__ = [
    "AppearanceCandidateScore",
    "AppearanceEmbedding",
    "AppearanceEmbeddingProvider",
    "AppearanceVerificationResult",
    "CropQualityMetrics",
    "CropReference",
    "MockAppearanceEmbeddingProvider",
    "TrackAppearancePrototype",
]
