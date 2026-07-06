"""Render a semantic target overlay without changing raw MOT output."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.identity.schemas import SemanticTarget
from football_tracking.locate_tracking.identity.segment_store import load_semantic_target


class SemanticTargetRenderError(RuntimeError):
    """Raised when semantic target rendering fails."""


def _segment_for_frame(target: SemanticTarget, frame_index: int):
    for segment in target.segments:
        end = segment.end_frame if segment.end_frame is not None else 10**18
        if (
            segment.start_frame <= frame_index <= end
            and segment.status in {"confirmed", "probation"}
        ):
            return segment
    return None


def render_semantic_target_video(
    *,
    source_video: str | Path,
    tracks_path: str | Path,
    semantic_target_path: str | Path,
    output_video: str | Path,
    debug_raw_id: bool = True,
) -> Path:
    import cv2  # type: ignore[import-not-found]

    target = load_semantic_target(semantic_target_path)
    mot = read_mot_track_file(tracks_path)
    index = FrameTrackIndex.from_observations(mot.observations)
    capture = cv2.VideoCapture(str(source_video))
    try:
        if not capture.isOpened():
            raise SemanticTargetRenderError(f"Could not open video: {source_video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if width <= 0 or height <= 0:
            raise SemanticTargetRenderError("Invalid video dimensions.")
        output = Path(output_video)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise SemanticTargetRenderError(f"Could not create video: {output}")
        frame_index = 1
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            segment = _segment_for_frame(target, frame_index)
            if segment is not None:
                row = next(
                    (
                        item
                        for item in index.get_frame(frame_index)
                        if item.track_id == segment.raw_track_id
                    ),
                    None,
                )
                if row is not None:
                    x1, y1, x2, y2 = (int(value) for value in row.bbox_xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = target.semantic_target_id
                    if debug_raw_id:
                        label = f"{label} [raw={segment.raw_track_id}]"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            frame_index += 1
        writer.release()
        return output
    finally:
        capture.release()
