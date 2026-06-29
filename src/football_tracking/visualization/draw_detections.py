"""Render prediction overlays for baseline smoke checks."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from football_tracking.detection.schemas import Detection

LOGGER = logging.getLogger(__name__)


def draw_detection_samples(
    detections: list[Detection],
    output_dir: Path,
    max_samples: int = 30,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("OpenCV is not available for drawing detections: %s", exc)
        cv2 = None

    by_image: dict[str, list[Detection]] = {}
    for detection in detections:
        image_path = detection.metadata.get("image_path")
        if isinstance(image_path, str):
            by_image.setdefault(image_path, []).append(detection)

    written: list[Path] = []
    for image_path_text, items in list(sorted(by_image.items()))[:max_samples]:
        image_path = Path(image_path_text)
        if not image_path.is_file():
            continue
        destination = output_dir / f"{image_path.stem}{image_path.suffix}"
        if cv2 is None:
            shutil.copy2(image_path, destination)
            written.append(destination)
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for detection in items:
            box = detection.bbox_xyxy
            color = (0, 220, 120)
            cv2.rectangle(
                image,
                (int(round(box.x1)), int(round(box.y1))),
                (int(round(box.x2)), int(round(box.y2))),
                color,
                2,
            )
            cv2.putText(
                image,
                f"{detection.target_class_name} {detection.confidence:.2f}",
                (max(0, int(round(box.x1))), max(12, int(round(box.y1)) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(destination), image)
        written.append(destination)
    return written
