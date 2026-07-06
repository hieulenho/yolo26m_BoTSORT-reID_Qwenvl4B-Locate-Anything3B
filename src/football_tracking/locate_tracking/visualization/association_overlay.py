"""Optional single-frame association debug overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.association.schemas import FrameQueryResolution


def _draw_box(
    cv2: Any,
    image: Any,
    box: tuple[float, float, float, float],
    color,
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def render_association_overlay(
    frame_image: Any,
    result: FrameQueryResolution,
    output_path: str | Path,
) -> Path:
    import cv2  # type: ignore[import-not-found]

    image = frame_image.copy()
    for association in result.associations:
        for candidate in association.candidates:
            if candidate.matching_track_bbox is None:
                continue
            if association.selected_track_id == candidate.track_id:
                color = (0, 255, 0)
            elif candidate.rank == 2:
                color = (0, 200, 255)
            else:
                color = (180, 180, 180)
            _draw_box(
                cv2,
                image,
                candidate.matching_track_bbox,
                color,
                f"id {candidate.track_id} s={candidate.final_score:.2f}",
            )
        grounding_box = association.candidates[0].grounding_bbox if association.candidates else None
        if grounding_box is not None:
            _draw_box(
                cv2,
                image,
                grounding_box,
                (255, 0, 255),
                f"{association.grounded_label}: {association.status}",
            )
    cv2.putText(
        image,
        result.query[:100],
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write association overlay: {path}")
    return path
