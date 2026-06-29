"""Image inspection helpers before YOLO inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ImagePreprocessingError(RuntimeError):
    """Raised when an image cannot be inspected."""


@dataclass(frozen=True)
class ImageMetadata:
    image_path: Path
    sequence_name: str
    frame_index: int
    width: int
    height: int


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
