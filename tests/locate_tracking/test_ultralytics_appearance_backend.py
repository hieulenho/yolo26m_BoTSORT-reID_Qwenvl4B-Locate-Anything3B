from __future__ import annotations

from football_tracking.locate_tracking.appearance.ultralytics_backend import (
    UltralyticsAppearanceEmbeddingProvider,
)


def test_ultralytics_backend_is_lazy_at_construction() -> None:
    provider = UltralyticsAppearanceEmbeddingProvider(model_id="yolo26n-cls.pt")

    assert provider.backend_name == "ultralytics"
    assert provider.model_loaded is False
    assert provider.inference_config()["public_api"] == "YOLO.embed"
