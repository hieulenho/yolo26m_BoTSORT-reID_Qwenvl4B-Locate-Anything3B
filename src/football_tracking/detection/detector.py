"""YOLOv8 detector wrapper for the pretrained baseline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


KNOWN_ULTRALYTICS_CHECKPOINTS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
}


class DetectorError(RuntimeError):
    """Raised when detector loading or inference fails."""


def resolve_device(device: str) -> str:
    normalized = str(device).lower()
    if normalized == "auto":
        try:
            import torch  # type: ignore[import-not-found]

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    if normalized in {"cpu", "cuda"} or normalized.isdigit() or normalized.startswith("cuda:"):
        if normalized != "cpu":
            try:
                import torch  # type: ignore[import-not-found]

                if not torch.cuda.is_available():
                    raise DetectorError(
                        f"CUDA device requested but CUDA is not available: {device}"
                    )
            except DetectorError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DetectorError(f"Could not validate CUDA availability: {exc}") from exc
        return str(device)
    raise DetectorError(f"Unsupported device value: {device}")


def validate_checkpoint(weights: str | Path) -> None:
    weights_path = Path(str(weights))
    if weights_path.is_file():
        return
    if str(weights) in KNOWN_ULTRALYTICS_CHECKPOINTS:
        return
    raise DetectorError(
        f"Checkpoint does not exist and is not a known Ultralytics checkpoint name: {weights}"
    )


class YOLOv8Detector:
    def __init__(
        self,
        weights: str | Path = "yolov8m.pt",
        device: str = "auto",
        half: bool = False,
        model_factory: Any | None = None,
    ) -> None:
        self.weights = weights
        self.requested_device = device
        self.device = resolve_device(device)
        self.half = bool(half and self.device != "cpu")
        self.model_factory = model_factory
        self.model: Any | None = None

    @property
    def model_name(self) -> str:
        return Path(str(self.weights)).name

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model
        validate_checkpoint(self.weights)
        try:
            if self.model_factory is not None:
                self.model = self.model_factory(str(self.weights))
            else:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                self.model = YOLO(str(self.weights))
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Failed to load YOLO model {self.weights}: {exc}") from exc
        LOGGER.info("Loaded YOLO model %s on %s", self.weights, self.device)
        return self.model

    def predict_batch(
        self,
        image_paths: Sequence[Path],
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        batch: int,
        verbose: bool = False,
    ) -> list[Any]:
        if not image_paths:
            return []
        model = self.load_model()
        try:
            results = model(
                [str(path) for path in image_paths],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                batch=batch,
                device=self.device,
                half=self.half,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"YOLO inference failed: {exc}") from exc
        return list(results) if isinstance(results, Sequence) else [results]

    def predict_image(self, image_path: Path, **kwargs: Any) -> Any:
        return self.predict_batch([image_path], **kwargs)[0]

    def predict_sequence(self, image_paths: Sequence[Path], **kwargs: Any) -> list[Any]:
        return self.predict_batch(image_paths, **kwargs)

    def predict_dataset_split(self, image_paths: Sequence[Path], **kwargs: Any) -> list[Any]:
        return self.predict_batch(image_paths, **kwargs)
