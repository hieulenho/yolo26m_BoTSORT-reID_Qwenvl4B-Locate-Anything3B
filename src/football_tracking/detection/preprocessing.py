"""Image inspection helpers before YOLO inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ImagePreprocessingError(RuntimeError):
    """Raised when an image cannot be inspected."""


@dataclass(frozen=True)
class ImageMetadata:
    image_path: Path
    sequence_name: str
    frame_index: int
    width: int
    height: int


@dataclass(frozen=True)
class FramePreprocessingResult:
    """One detector input plus auditable image-quality measurements."""

    frame: Any
    applied: bool
    mode: str
    luminance_mean: float
    luminance_std: float


class FramePreprocessor:
    """Apply conservative, geometry-preserving enhancement before detection."""

    MODES = {"none", "auto_low_light", "clahe"}

    def __init__(
        self,
        *,
        mode: str = "none",
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        low_light_threshold: float = 70.0,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in self.MODES:
            raise ImagePreprocessingError(
                f"Unsupported frame preprocessing mode: {mode}"
            )
        if clahe_clip_limit <= 0 or clahe_grid_size < 2:
            raise ImagePreprocessingError("Invalid CLAHE preprocessing parameters.")
        if not 0.0 <= low_light_threshold <= 255.0:
            raise ImagePreprocessingError("low_light_threshold must be in [0, 255].")
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise ImagePreprocessingError(
                "OpenCV is required for frame preprocessing."
            ) from exc
        self.mode = normalized
        self.low_light_threshold = float(low_light_threshold)
        self._cv2 = cv2
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_grid_size), int(clahe_grid_size)),
        )

    def process(self, frame: Any) -> FramePreprocessingResult:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ImagePreprocessingError("Detector frame must be a valid image array.")
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        luminance_mean = float(gray.mean())
        luminance_std = float(gray.std())
        should_apply = self.mode == "clahe" or (
            self.mode == "auto_low_light"
            and luminance_mean < self.low_light_threshold
        )
        if not should_apply:
            return FramePreprocessingResult(
                frame=frame,
                applied=False,
                mode=self.mode,
                luminance_mean=luminance_mean,
                luminance_std=luminance_std,
            )
        lab = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = self._cv2.split(lab)
        enhanced_lightness = self._clahe.apply(lightness)
        enhanced = self._cv2.cvtColor(
            self._cv2.merge((enhanced_lightness, channel_a, channel_b)),
            self._cv2.COLOR_LAB2BGR,
        )
        return FramePreprocessingResult(
            frame=enhanced,
            applied=True,
            mode=self.mode,
            luminance_mean=luminance_mean,
            luminance_std=luminance_std,
        )


def parse_sequence_frame(image_path: Path) -> tuple[str, int]:
    stem = image_path.stem
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return image_path.parent.name, 1


def inspect_image(
    image_path: Path,
    sequence_name: str | None = None,
    frame_index: int | None = None,
) -> ImageMetadata:
    if not image_path.is_file():
        raise ImagePreprocessingError(f"Image does not exist: {image_path}")
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImagePreprocessingError("OpenCV is required to inspect image dimensions.") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise ImagePreprocessingError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]
    parsed_sequence, parsed_frame = parse_sequence_frame(image_path)
    return ImageMetadata(
        image_path=image_path,
        sequence_name=sequence_name or parsed_sequence,
        frame_index=frame_index or parsed_frame,
        width=int(width),
        height=int(height),
    )
