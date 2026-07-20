"""Render fused open-domain semantic labels on MOT tracking outputs."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from football_tracking.vlm.tracking_context import MotTrackRow, read_mot_tracks


class SemanticRenderError(RuntimeError):
    """Raised when a semantic tracking video cannot be rendered."""


def render_semantic_video(
    *,
    source_video: str | Path,
    tracks_path: str | Path,
    semantics_path: str | Path,
    output_video: str | Path,
    overwrite: bool = False,
    show_confidence: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    source = Path(source_video)
    tracks = Path(tracks_path)
    semantics = Path(semantics_path)
    output = Path(output_video)
    for path, name in ((source, "source video"), (tracks, "tracks"), (semantics, "semantics")):
        if not path.is_file():
            raise SemanticRenderError(f"Missing {name}: {path}")
    if output.exists() and not overwrite:
        raise SemanticRenderError(f"Output video exists and overwrite=false: {output}")
    if max_frames is not None and max_frames <= 0:
        raise SemanticRenderError("max_frames must be positive when provided.")
    semantic_data = json.loads(semantics.read_text(encoding="utf-8"))
    labels = {int(row["track_id"]): row for row in semantic_data.get("tracks", [])}
    rows = read_mot_tracks(tracks)
    rows_by_frame: dict[int, list[MotTrackRow]] = defaultdict(list)
    for row in rows:
        rows_by_frame[row.frame_index].append(row)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SemanticRenderError(f"Could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SemanticRenderError(f"Invalid source video metadata: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.stem}.partial{output.suffix}")
    if temporary_output.exists():
        temporary_output.unlink()
    writer = cv2.VideoWriter(
        str(temporary_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SemanticRenderError(f"Could not open video writer: {temporary_output}")
    started = time.perf_counter()
    rendered_frames = 0
    box_count = 0
    accepted_boxes = 0
    render_succeeded = False
    try:
        frame_index = 1
        while True:
            if max_frames is not None and rendered_frames >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            for row in rows_by_frame.get(frame_index, []):
                semantic = labels.get(row.track_id, {})
                accepted = bool(semantic.get("accepted", False))
                label = str(semantic.get("class_label", "unknown")) if accepted else "unknown"
                confidence = float(semantic.get("confidence", 0.0))
                color = _track_color(row.track_id, accepted=accepted)
                x1, y1, x2, y2 = _clip_bbox(row.bbox_xyxy(), width, height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                text = f"ID {row.track_id} | {label}"
                attributes = _short_attributes(semantic.get("attributes", {}))
                if attributes:
                    text += f" | {attributes}"
                if show_confidence and accepted:
                    text += f" {confidence:.2f}"
                _draw_label(frame, text, x1, y1, color)
                box_count += 1
                accepted_boxes += int(accepted)
            writer.write(frame)
            rendered_frames += 1
            frame_index += 1
        render_succeeded = True
    finally:
        writer.release()
        capture.release()
        if not render_succeeded and temporary_output.exists():
            temporary_output.unlink()
    if rendered_frames == 0 or not temporary_output.is_file():
        if temporary_output.exists():
            temporary_output.unlink()
        raise SemanticRenderError("Semantic render produced no video frames.")
    if temporary_output.stat().st_size == 0:
        temporary_output.unlink()
        raise SemanticRenderError("Semantic render produced an empty video file.")
    temporary_output.replace(output)
    elapsed = time.perf_counter() - started
    unique_track_ids = {row.track_id for row in rows}
    accepted_ids = {
        track_id
        for track_id in unique_track_ids
        if bool(labels.get(track_id, {}).get("accepted", False))
    }
    result = {
        "source_video": str(source.resolve()),
        "tracks": str(tracks.resolve()),
        "semantics": str(semantics.resolve()),
        "output_video": str(output.resolve()),
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "source_frame_count": frame_count,
            "rendered_frame_count": rendered_frames,
            "requested_max_frames": max_frames,
        },
        "timing": {
            "seconds": elapsed,
            "render_fps": rendered_frames / elapsed if elapsed > 0 else 0.0,
        },
        "semantics_summary": {
            "track_count": len(unique_track_ids),
            "accepted_track_count": len(accepted_ids),
            "unknown_track_count": len(unique_track_ids) - len(accepted_ids),
            "track_coverage": (
                len(accepted_ids) / len(unique_track_ids) if unique_track_ids else 0.0
            ),
            "box_count": box_count,
            "accepted_box_count": accepted_boxes,
            "box_coverage": accepted_boxes / box_count if box_count else 0.0,
        },
    }
    metadata_path = output.with_name(f"{output.stem}.semantic.metadata.json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    result["metadata"] = str(metadata_path.resolve())
    return result


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        min(max(x1, 0), width - 1),
        min(max(y1, 0), height - 1),
        min(max(x2, 1), width),
        min(max(y2, 1), height),
    )


def _track_color(track_id: int, *, accepted: bool) -> tuple[int, int, int]:
    if not accepted:
        return (128, 128, 128)
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 79) % 192,
        64 + (track_id * 113) % 192,
    )


def _short_attributes(attributes: Any) -> str:
    if not isinstance(attributes, dict):
        return ""
    pieces = [f"{key}={value}" for key, value in list(attributes.items())[:2]]
    return ",".join(pieces)


def _draw_label(
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(y - text_height - baseline - 6, 0)
    right = min(x + text_width + 8, frame.shape[1] - 1)
    cv2.rectangle(frame, (x, top), (right, y), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (x + 4, max(y - baseline - 3, text_height)),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
