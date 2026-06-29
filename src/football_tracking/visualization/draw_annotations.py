"""Render annotation samples without modifying source images."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from football_tracking.data.schemas import SequenceInfo

LOGGER = logging.getLogger(__name__)


def _track_color(track_id: int | str) -> tuple[int, int, int]:
    value = abs(hash(str(track_id)))
    return value % 255, (value // 255) % 255, (value // (255 * 255)) % 255


def draw_annotation_samples(
    sequences: list[SequenceInfo],
    output_dir: Path,
    num_sequences: int = 2,
    frames_per_sequence: int = 2,
    seed: int = 42,
    draw_ignored: bool = False,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected_sequences = list(sequences)
    rng.shuffle(selected_sequences)
    selected_sequences = selected_sequences[:num_sequences]
    written: list[Path] = []

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("OpenCV is not available for drawing annotations: %s", exc)
        cv2 = None

    for sequence in selected_sequences:
        frames = list(sequence.annotations)
        rng.shuffle(frames)
        for frame in sorted(frames[:frames_per_sequence], key=lambda item: item.frame_index):
            if not frame.image_path.is_file():
                LOGGER.warning("Cannot draw missing image: %s", frame.image_path)
                continue
            destination = (
                output_dir / f"{sequence.name}_{frame.frame_index:06d}{frame.image_path.suffix}"
            )
            if cv2 is None:
                shutil.copy2(frame.image_path, destination)
                written.append(destination)
                continue
            image = cv2.imread(str(frame.image_path))
            if image is None:
                LOGGER.warning("OpenCV could not read image: %s", frame.image_path)
                shutil.copy2(frame.image_path, destination)
                written.append(destination)
                continue
            for annotation in frame.objects:
                if annotation.is_ignored and not draw_ignored:
                    continue
                x1 = int(round(annotation.bbox_xyxy.x1))
                y1 = int(round(annotation.bbox_xyxy.y1))
                x2 = int(round(annotation.bbox_xyxy.x2))
                y2 = int(round(annotation.bbox_xyxy.y2))
                color = _track_color(annotation.track_id)
                label = (
                    f"{annotation.target_class or annotation.source_class} "
                    f"#{annotation.track_id}"
                )
                cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
                cv2.putText(
                    image,
                    f"{sequence.name} f{frame.frame_index} {label}",
                    (max(0, x1), max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    max(1, line_thickness),
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(destination), image)
            written.append(destination)
    return written
