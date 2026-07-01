"""Render tracked boxes and trajectories."""

from __future__ import annotations

from typing import Any

from football_tracking.tracking.schemas import TrackOutput
from football_tracking.tracking.trajectory import TrajectoryStore
from football_tracking.utils.colors import stable_color


def draw_tracks(
    frame: Any,
    tracks: list[TrackOutput],
    trajectory_store: TrajectoryStore | None = None,
    show_confidence: bool = True,
    show_class: bool = True,
    show_track_id: bool = True,
    show_trajectory: bool = True,
    show_fps: bool = True,
    fps: float | None = None,
    frame_index: int | None = None,
    sequence_name: str | None = None,
    tracker_name: str | None = None,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> Any:
    import cv2  # type: ignore[import-not-found]

    rendered = frame.copy()
    for track in tracks:
        color = stable_color(track.track_id)
        x1 = int(round(track.bbox_xyxy.x1))
        y1 = int(round(track.bbox_xyxy.y1))
        x2 = int(round(track.bbox_xyxy.x2))
        y2 = int(round(track.bbox_xyxy.y2))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, line_thickness)
        parts: list[str] = []
        if show_track_id:
            parts.append(f"ID {track.track_id}")
        if show_class:
            parts.append(track.class_name)
        if show_confidence and track.confidence is not None:
            parts.append(f"{track.confidence:.2f}")
        label = " | ".join(parts)
        if label:
            cv2.putText(
                rendered,
                label,
                (max(0, x1), max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, line_thickness),
                cv2.LINE_AA,
            )
        if show_trajectory and trajectory_store is not None:
            points = trajectory_store.get(track.track_id)
            for start, end in zip(points, points[1:], strict=False):
                cv2.line(
                    rendered,
                    (int(round(start[0])), int(round(start[1]))),
                    (int(round(end[0])), int(round(end[1]))),
                    color,
                    max(1, line_thickness - 1),
                    cv2.LINE_AA,
                )
    overlay = []
    if tracker_name:
        overlay.append(tracker_name)
    if sequence_name:
        overlay.append(sequence_name)
    if frame_index is not None:
        overlay.append(f"frame {frame_index}")
    if show_fps and fps is not None:
        overlay.append(f"{fps:.1f} FPS")
    if overlay:
        cv2.putText(
            rendered,
            " | ".join(overlay),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return rendered
