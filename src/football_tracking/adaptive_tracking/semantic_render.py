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
    tracking_metadata_path: str | Path | None = None,
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
    detector_labels: dict[int, dict[str, Any]] = {}
    tracking_metadata: Path | None = None
    if tracking_metadata_path is not None:
        tracking_metadata = Path(tracking_metadata_path)
        if not tracking_metadata.is_file():
            raise SemanticRenderError(f"Missing tracking metadata: {tracking_metadata}")
        tracking_data = json.loads(tracking_metadata.read_text(encoding="utf-8"))
        detector_labels = {
            int(row["track_id"]): row for row in tracking_data.get("track_classes", [])
        }
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
    fallback_boxes = 0
    render_succeeded = False
    try:
        frame_index = 1
        while True:
            if max_frames is not None and rendered_frames >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_rows = rows_by_frame.get(frame_index, [])
            crowded_frame = len(frame_rows) >= 4
            occupied_labels: list[tuple[int, int, int, int]] = []
            for row in frame_rows:
                semantic = labels.get(row.track_id, {})
                modeled = row.track_id in labels
                accepted = bool(semantic.get("accepted", False))
                detector = detector_labels.get(row.track_id, {})
                fallback = (
                    not accepted
                    and not modeled
                    and bool(detector.get("class_name"))
                )
                if accepted:
                    label = str(
                        semantic.get("display_label", semantic.get("class_label", "unknown"))
                    )
                    confidence = float(semantic.get("confidence", 0.0))
                elif modeled:
                    label = "unknown"
                    confidence = float(semantic.get("confidence", 0.0))
                elif fallback:
                    label = str(detector["class_name"])
                    confidence = float(detector.get("class_consensus", 0.0))
                else:
                    label = "unknown"
                    confidence = 0.0
                color = _track_color(row.track_id, accepted=accepted, fallback=fallback)
                x1, y1, x2, y2 = _clip_bbox(row.bbox_xyxy(), width, height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                base_text = f"ID {row.track_id} | {_truncate_label(label)}"
                attributes = _short_attributes(semantic.get("attributes", {}))
                confidence_text = (
                    f" {confidence:.2f}"
                    if show_confidence and (accepted or fallback)
                    else ""
                )
                candidates = []
                if attributes and not crowded_frame:
                    candidates.append(f"{base_text} | {attributes}{confidence_text}")
                candidates.extend((f"{base_text}{confidence_text}", base_text))
                text = _select_fitting_text(
                    candidates,
                    max_width=max(min(width - 12, int(width * 0.55)), 24),
                )
                label_rect = _draw_label(
                    frame,
                    text,
                    bbox=(x1, y1, x2, y2),
                    color=color,
                    occupied=occupied_labels,
                )
                occupied_labels.append(label_rect)
                box_count += 1
                accepted_boxes += int(accepted)
                fallback_boxes += int(fallback)
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
    modeled_ids = unique_track_ids & set(labels)
    rejected_ids = modeled_ids - accepted_ids
    unmodeled_ids = unique_track_ids - modeled_ids
    fallback_ids = {
        track_id
        for track_id in unmodeled_ids
        if bool(detector_labels.get(track_id, {}).get("class_name"))
    }
    labeled_ids = accepted_ids | fallback_ids
    result = {
        "source_video": str(source.resolve()),
        "tracks": str(tracks.resolve()),
        "semantics": str(semantics.resolve()),
        "tracking_metadata": (
            str(tracking_metadata.resolve()) if tracking_metadata is not None else None
        ),
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
            "modeled_track_count": len(modeled_ids),
            "unmodeled_track_count": len(unmodeled_ids),
            "accepted_track_count": len(accepted_ids),
            "semantic_rejected_track_count": len(rejected_ids),
            "semantic_unaccepted_track_count": len(unique_track_ids) - len(accepted_ids),
            "unknown_track_count": len(unique_track_ids) - len(labeled_ids),
            "track_coverage": (
                len(accepted_ids) / len(unique_track_ids) if unique_track_ids else 0.0
            ),
            "box_count": box_count,
            "accepted_box_count": accepted_boxes,
            "box_coverage": accepted_boxes / box_count if box_count else 0.0,
            "detector_fallback_track_count": len(fallback_ids),
            "detector_fallback_box_count": fallback_boxes,
            "labeled_track_count": len(labeled_ids),
            "unlabeled_track_count": len(unique_track_ids) - len(labeled_ids),
            "label_track_coverage": (
                len(labeled_ids) / len(unique_track_ids) if unique_track_ids else 0.0
            ),
            "labeled_box_count": accepted_boxes + fallback_boxes,
            "label_box_coverage": (
                (accepted_boxes + fallback_boxes) / box_count if box_count else 0.0
            ),
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


def _track_color(
    track_id: int,
    *,
    accepted: bool,
    fallback: bool = False,
) -> tuple[int, int, int]:
    if not accepted and not fallback:
        return (128, 128, 128)
    if fallback:
        return (96, 170, 220)
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


def _truncate_label(label: str, *, max_characters: int = 42) -> str:
    value = " ".join(str(label).split())
    if len(value) <= max_characters:
        return value
    return value[: max_characters - 3].rstrip() + "..."


def _select_fitting_text(candidates: list[str], *, max_width: int) -> str:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    for candidate in candidates:
        text_width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if text_width + 8 <= max_width:
            return candidate
    value = candidates[-1] if candidates else ""
    suffix = "..."
    while value:
        candidate = value.rstrip() + suffix
        text_width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if text_width + 8 <= max_width:
            return candidate
        value = value[:-1]
    return suffix


def _draw_label(
    frame: Any,
    text: str,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = bbox
    label_width = text_width + 8
    label_height = text_height + baseline + 6
    candidates = (
        (x1, y1 - label_height),
        (x1, y2),
        (x2 - label_width, y1 - label_height),
        (x2, y1),
        (x1, y1),
    )
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    rectangles = [
        _clip_label_rect(left, top, label_width, label_height, frame_width, frame_height)
        for left, top in candidates
    ]
    left, top, right, bottom = min(
        rectangles,
        key=lambda rect: (
            sum(_rect_overlap(rect, other) for other in occupied),
            rectangles.index(rect),
        ),
    )
    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (left + 4, min(top + text_height + 2, bottom - baseline)),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return (left, top, right, bottom)


def _clip_label_rect(
    left: int,
    top: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    left = min(max(left, 0), max(frame_width - width, 0))
    top = min(max(top, 0), max(frame_height - height, 0))
    return (left, top, min(left + width, frame_width - 1), min(top + height, frame_height - 1))


def _rect_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height
