"""Lazy Ultralytics-backed appearance embedding provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from football_tracking.locate_tracking.appearance.embedding_utils import (
    l2_normalize,
    vectors_to_tuple,
)
from football_tracking.locate_tracking.appearance.schemas import AppearanceEmbedding


class UltralyticsAppearanceBackendError(RuntimeError):
    """Raised when the Ultralytics appearance backend cannot embed crops."""


class UltralyticsAppearanceEmbeddingProvider:
    """Public-API wrapper around ``YOLO(...).embed`` with lazy model loading."""

    def __init__(
        self,
        *,
        model_id: str = "yolo26n-cls.pt",
        device: str | None = "cuda",
        batch_size: int = 8,
        normalize: bool = True,
        cleanup_cuda_on_close: bool = False,
    ) -> None:
        self._model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.cleanup_cuda_on_close = bool(cleanup_cuda_on_close)
        self._model: Any | None = None

    @property
    def backend_name(self) -> str:
        return "ultralytics"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def inference_config(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "public_api": "YOLO.embed",
        }

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_id)
        return self._model

    def _embed_batch_raw(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        model = self._load_model()
        kwargs: dict[str, Any] = {"verbose": False}
        if self.device:
            kwargs["device"] = self.device
        try:
            output = model.embed(list(images), **kwargs)
        except TypeError:
            output = model.embed(list(images), verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise UltralyticsAppearanceBackendError(
                f"Ultralytics appearance embedding failed: {exc}"
            ) from exc
        vectors: list[np.ndarray] = []
        for item in output:
            if hasattr(item, "detach"):
                item = item.detach().cpu().numpy()
            vector = np.asarray(item, dtype=np.float32).reshape(-1)
            vectors.append(l2_normalize(vector) if self.normalize else vector)
        return vectors

    def embed_crop(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> AppearanceEmbedding:
        return self.embed_crops([image], metadata=[metadata])[0]

    def embed_crops(
        self,
        images: Sequence[np.ndarray],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[AppearanceEmbedding]:
        metadata_items = metadata if metadata is not None else [None] * len(images)
        outputs: list[AppearanceEmbedding] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = list(images[start : start + self.batch_size])
            batch_metadata = list(metadata_items[start : start + self.batch_size])
            for vector, item in zip(
                self._embed_batch_raw(batch_images),
                batch_metadata,
                strict=True,
            ):
                metadata_dict = dict(item or {})
                outputs.append(
                    AppearanceEmbedding(
                        backend=self.backend_name,
                        model_id=self.model_id,
                        dimension=int(vector.size),
                        vector=vectors_to_tuple(vector),
                        normalized=self.normalize,
                        source_track_id=metadata_dict.get("source_track_id"),
                        source_frame_index=metadata_dict.get("source_frame_index"),
                        metadata={"ultralytics_public_api": "YOLO.embed", **metadata_dict},
                    )
                )
        return outputs

    def close(self) -> None:
        self._model = None
        if self.cleanup_cuda_on_close:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
