"""Build VLM-ready context from MOT tracking outputs and video frames."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_tracking.detection.serialization import runtime_versions
from football_tracking.vlm.config import VlmConfigError, VlmTrackingConfig, load_vlm_tracking_config


class VlmAnalysisError(RuntimeError):
    """Raised when tracked-video VLM analysis cannot be prepared."""


@dataclass(frozen=True)
class MotTrackRow:
    frame_index: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float | None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.x + self.width)),
            int(round(self.y + self.height)),
        )


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        if self.fps <= 0 or self.frame_count <= 0:
            return None
        return self.frame_count / self.fps

    def time_seconds(self, frame_index: int) -> float | None:
        if self.fps <= 0:
            return None
        return max(frame_index - 1, 0) / self.fps

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
        }


def run_vlm_analysis(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_vlm_tracking_config(config_path, overrides=overrides)
    if dry_run:
        return _dry_run_plan(config)
    _validate_inputs(config)
    if (config.output_dir / "vlm_context.json").exists() and not config.overwrite:
        raise VlmAnalysisError(
            f"VLM output exists and overwrite=false: {config.output_dir / 'vlm_context.json'}"
        )

    rows = read_mot_tracks(config.tracks_path)
    video_info = _read_video_info(config.source_video)
    metadata = _read_optional_json(config.metadata_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.keyframes_dir.mkdir(parents=True, exist_ok=True)
    config.crops_dir.mkdir(parents=True, exist_ok=True)

    rows_by_frame = _group_rows_by_frame(rows)
    keyframes = _write_keyframes(config, rows_by_frame, video_info)
    all_track_summaries = _summarize_tracks(rows, video_info)
    track_summaries = all_track_summaries[: config.max_tracks]
    crops = _write_track_crops(
        config,
        rows,
        video_info,
        allowed_track_ids={int(row["track_id"]) for row in track_summaries},
    )
    context = _build_context(
        config,
        rows,
        video_info,
        metadata,
        all_track_summaries,
        track_summaries,
        keyframes,
        crops,
    )
    context_path = config.output_dir / "vlm_context.json"
    prompt_path = config.output_dir / "prompt.md"
    context_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    prompt_text = build_prompt(config, context)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    model_result: dict[str, Any]
    if config.run_model:
        from football_tracking.vlm.qwen_runner import QwenRunnerError, run_qwen_vlm

        try:
            image_paths = [Path(row["path"]) for row in keyframes]
            model_result = run_qwen_vlm(config, prompt_text, image_paths)
        except QwenRunnerError as exc:
            model_result = {"status": "failed", "error": str(exc)}
        answer_path = config.output_dir / "vlm_answer.md"
        answer_json_path = config.output_dir / "vlm_answer.json"
        answer_path.write_text(str(model_result.get("answer", "")), encoding="utf-8")
        answer_json_path.write_text(
            json.dumps(model_result, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        model_result = {
            "status": "prepared_only",
            "run_model": False,
            "reason": "Set model.run_model=true or pass --run-model to execute Qwen.",
        }

    run_status = (
        "model_failed"
        if config.run_model and model_result.get("status") == "failed"
        else "ok"
    )
    return {
        "status": run_status,
        "model": {
            "provider": "qwen",
            "model_id": config.model_id,
            "run_model": config.run_model,
            "quantization": config.quantization,
            "torch_dtype": config.torch_dtype,
        },
        "summary": {
            "track_count": context["tracking_summary"]["track_count"],
            "track_observation_count": context["tracking_summary"]["track_observation_count"],
            "keyframe_count": len(keyframes),
            "crop_count": len(crops),
        },
        "model_result": model_result,
        "paths": {
            "context_json": str(context_path),
            "prompt_md": str(prompt_path),
            "keyframes_dir": str(config.keyframes_dir),
            "crops_dir": str(config.crops_dir),
            "answer_md": str(config.output_dir / "vlm_answer.md") if config.run_model else None,
            "answer_json": str(config.output_dir / "vlm_answer.json") if config.run_model else None,
        },
    }


def read_mot_tracks(path: Path) -> list[MotTrackRow]:
    rows: list[MotTrackRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 6:
            raise VlmAnalysisError(f"Invalid MOT row at {path}:{line_number}: {line}")
        try:
            frame_index = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, width, height = (float(value) for value in parts[2:6])
            confidence = float(parts[6]) if len(parts) >= 7 else None
        except ValueError as exc:
            raise VlmAnalysisError(f"Invalid numeric MOT value at {path}:{line_number}") from exc
        if frame_index < 1 or track_id < 1:
            raise VlmAnalysisError(f"Frame and track IDs must be positive at {path}:{line_number}")
        if width <= 0 or height <= 0:
            raise VlmAnalysisError(
                f"Track box width/height must be positive at {path}:{line_number}"
            )
        if any(not math.isfinite(value) for value in (x, y, width, height)):
            raise VlmAnalysisError(f"Track box contains non-finite values at {path}:{line_number}")
        rows.append(
            MotTrackRow(
                frame_index=frame_index,
                track_id=track_id,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=confidence if confidence is not None and confidence >= 0 else None,
            )
        )
    return sorted(rows, key=lambda row: (row.frame_index, row.track_id))


def build_prompt(config: VlmTrackingConfig, context: dict[str, Any]) -> str:
    context_json = json.dumps(
        _compact_prompt_context(context),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "# Tracking VLM Analysis Task",
            "",
            config.task_prompt,
            "",
            "Primary goal: audit the tracking quality. Keep match commentary short.",
            "Use only track IDs that appear in the metadata below.",
            "If an issue is not visible or not supported by metadata, "
            "say that it is not confirmed.",
            "Each provided keyframe image is annotated with tracking IDs.",
            "Cross-check the visual evidence with this tracking metadata and diagnostics:",
            "",
            "```json",
            context_json,
            "```",
            "",
            "Return your answer in Vietnamese using exactly these sections:",
            "1. Tom tat tracking trong 2 cau",
            "2. Track on dinh nhat kem bang chung",
            "3. Track can kiem tra lai kem bang chung",
            "4. Chuyen dong noi bat theo track_id",
            "5. Ket luan va buoc kiem tra tiep theo",
            "For every track claim, include evidence fields such as obs, dur_s, "
            "disp_px, conf, gaps, max_gap, or visible keyframe IDs.",
            "",
        ]
    )


def _compact_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    video = context["video"]
    summary = context["tracking_summary"]
    diagnostics = context["tracking_diagnostics"]
    return {
        "video": {
            "frames": video.get("frame_count"),
            "fps": _rounded(video.get("fps"), 2),
            "duration_s": _rounded(video.get("duration_seconds"), 2),
        },
        "summary": {
            "track_count": summary.get("track_count"),
            "observations": summary.get("track_observation_count"),
            "frames_with_tracks": summary.get("frames_with_tracks"),
            "mean_tracks_per_frame": _rounded(summary.get("mean_tracks_per_frame"), 2),
        },
        "tracking_diagnostics": {
            "stable": _prompt_tracks(diagnostics.get("stable_long_tracks", []), limit=3),
            "large_motion": _prompt_tracks(
                diagnostics.get("largest_displacement_tracks", []),
                limit=3,
            ),
            "fragmented": _prompt_tracks(diagnostics.get("fragmented_tracks", []), limit=3),
            "low_conf": _prompt_tracks(diagnostics.get("low_confidence_tracks", []), limit=3),
            "short": _prompt_tracks(diagnostics.get("short_tracks", []), limit=3),
            "visible_selected_ids": diagnostics.get("selected_track_ids_visible_in_keyframes", []),
            "not_visible_selected_ids": diagnostics.get(
                "selected_track_ids_not_visible_in_keyframes",
                [],
            ),
        },
        "selected_tracks": _prompt_tracks(context["tracks"], limit=6),
        "keyframes": [
            {
                "frame": row.get("frame_index"),
                "time_s": _rounded(row.get("time_seconds"), 2),
                "track_count": row.get("track_count"),
                "visible_ids": row.get("visible_track_ids", []),
            }
            for row in context["keyframes"]
        ],
    }


def _prompt_tracks(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [_prompt_track(row) for row in rows[:limit]]


def _prompt_track(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("track_id"),
        "obs": row.get("observation_count"),
        "dur_s": _rounded(row.get("duration_seconds"), 2),
        "conf": _rounded(row.get("mean_confidence"), 3),
        "cover": _rounded(row.get("coverage_ratio"), 3),
        "disp_px": _rounded(row.get("displacement_pixels"), 1),
        "gaps": row.get("gap_count"),
        "max_gap": row.get("max_gap_frames"),
        "t0": _rounded(row.get("time_start_seconds"), 2),
        "t1": _rounded(row.get("time_end_seconds"), 2),
    }


def _rounded(value: Any, digits: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(float(value), digits)
    return value


def _dry_run_plan(config: VlmTrackingConfig) -> dict[str, Any]:
    return {
        "dry_run": True,
        "input": {
            "source_video": str(config.source_video),
            "source_video_exists": config.source_video.is_file(),
            "tracked_video": str(config.tracked_video) if config.tracked_video else None,
            "tracks": str(config.tracks_path),
            "tracks_exists": config.tracks_path.is_file(),
            "metadata": str(config.metadata_path) if config.metadata_path else None,
            "metadata_exists": config.metadata_path.is_file() if config.metadata_path else None,
        },
        "output": {
            "dir": str(config.output_dir),
            "keyframes_dir": str(config.keyframes_dir),
            "crops_dir": str(config.crops_dir),
        },
        "model": {
            "model_id": config.model_id,
            "run_model": config.run_model,
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "quantization": config.quantization,
        },
        "action": "validated config shape; no frames, crops, or model outputs were written",
    }


def _validate_inputs(config: VlmTrackingConfig) -> None:
    if not config.source_video.is_file():
        raise VlmAnalysisError(f"Source video does not exist: {config.source_video}")
    if not config.tracks_path.is_file():
        raise VlmAnalysisError(f"MOT tracks file does not exist: {config.tracks_path}")
    if config.metadata_path is not None and not config.metadata_path.is_file():
        raise VlmAnalysisError(f"Tracking metadata does not exist: {config.metadata_path}")
    if config.tracked_video is not None and not config.tracked_video.is_file():
        raise VlmAnalysisError(f"Tracked video does not exist: {config.tracked_video}")


def _read_video_info(path: Path) -> VideoInfo:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VlmAnalysisError(f"Could not open source video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise VlmAnalysisError(f"Video has invalid dimensions: {path}")
    return VideoInfo(
        width=width,
        height=height,
        fps=fps if fps > 0 else 25.0,
        frame_count=frame_count,
    )


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VlmAnalysisError(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise VlmAnalysisError(f"Metadata JSON must contain an object: {path}")
    return payload


def _group_rows_by_frame(rows: list[MotTrackRow]) -> dict[int, list[MotTrackRow]]:
    grouped: dict[int, list[MotTrackRow]] = {}
    for row in rows:
        grouped.setdefault(row.frame_index, []).append(row)
    return grouped


def _group_rows_by_track(rows: list[MotTrackRow]) -> dict[int, list[MotTrackRow]]:
    grouped: dict[int, list[MotTrackRow]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    return grouped


def _summarize_tracks(
    rows: list[MotTrackRow],
    video_info: VideoInfo,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for track_id, track_rows in _group_rows_by_track(rows).items():
        ordered = sorted(track_rows, key=lambda row: row.frame_index)
        first = ordered[0]
        last = ordered[-1]
        first_center = first.center
        last_center = last.center
        frame_span = last.frame_index - first.frame_index + 1
        duration_seconds = frame_span / video_info.fps if video_info.fps > 0 else None
        frame_gaps = [
            right.frame_index - left.frame_index - 1
            for left, right in zip(ordered, ordered[1:], strict=False)
            if right.frame_index - left.frame_index > 1
        ]
        delta_x = last_center[0] - first_center[0]
        delta_y = last_center[1] - first_center[1]
        displacement = math.hypot(delta_x, delta_y)
        confidences = [row.confidence for row in ordered if row.confidence is not None]
        summaries.append(
            {
                "track_id": track_id,
                "frame_start": first.frame_index,
                "frame_end": last.frame_index,
                "time_start_seconds": video_info.time_seconds(first.frame_index),
                "time_end_seconds": video_info.time_seconds(last.frame_index),
                "duration_seconds": duration_seconds,
                "observation_count": len(ordered),
                "span_frame_count": frame_span,
                "coverage_ratio": round(len(ordered) / frame_span, 4) if frame_span else None,
                "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                "first_center": [round(first_center[0], 2), round(first_center[1], 2)],
                "last_center": [round(last_center[0], 2), round(last_center[1], 2)],
                "delta_xy": [round(delta_x, 2), round(delta_y, 2)],
                "displacement_pixels": round(displacement, 2),
                "gap_count": len(frame_gaps),
                "max_gap_frames": max(frame_gaps) if frame_gaps else 0,
            }
        )
    summaries.sort(key=lambda item: (-int(item["observation_count"]), int(item["track_id"])))
    return summaries


def _track_diagnostics(
    all_track_summaries: list[dict[str, Any]],
    selected_track_summaries: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    selected_ids = {int(row["track_id"]) for row in selected_track_summaries}
    visible_keyframe_ids = {
        int(track_id)
        for keyframe in keyframes
        for track_id in keyframe.get("visible_track_ids", [])
    }
    visible_selected_ids = sorted(selected_ids & visible_keyframe_ids)
    return {
        "note": (
            "Heuristic diagnostics from MOT metadata. Use keyframes to confirm visual claims."
        ),
        "stable_long_tracks": _top_tracks(
            selected_track_summaries,
            key=lambda row: (
                int(row.get("observation_count") or 0),
                float(row.get("coverage_ratio") or 0.0),
                float(row.get("mean_confidence") or 0.0),
            ),
            limit=limit,
        ),
        "largest_displacement_tracks": _top_tracks(
            selected_track_summaries,
            key=lambda row: float(row.get("displacement_pixels") or 0.0),
            limit=limit,
        ),
        "fragmented_tracks": _top_tracks(
            all_track_summaries,
            key=lambda row: (
                int(row.get("gap_count") or 0),
                int(row.get("max_gap_frames") or 0),
            ),
            limit=limit,
            require_positive="gap_count",
        ),
        "low_confidence_tracks": _top_tracks(
            all_track_summaries,
            key=lambda row: -(float(row.get("mean_confidence") or 1.0)),
            limit=limit,
        ),
        "short_tracks": _top_tracks(
            all_track_summaries,
            key=lambda row: -(int(row.get("observation_count") or 0)),
            limit=limit,
        ),
        "selected_track_ids_visible_in_keyframes": visible_selected_ids,
        "selected_track_ids_not_visible_in_keyframes": sorted(selected_ids - visible_keyframe_ids),
    }


def _top_tracks(
    rows: list[dict[str, Any]],
    *,
    key: Any,
    limit: int,
    require_positive: str | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if require_positive is None or float(row.get(require_positive) or 0) > 0
    ]
    return [_compact_track(row) for row in sorted(filtered, key=key, reverse=True)[:limit]]


def _compact_track(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "track_id",
        "observation_count",
        "duration_seconds",
        "mean_confidence",
        "coverage_ratio",
        "displacement_pixels",
        "gap_count",
        "max_gap_frames",
        "time_start_seconds",
        "time_end_seconds",
    )
    return {field: row.get(field) for field in fields if field in row}


def _select_keyframe_indices(
    rows_by_frame: dict[int, list[MotTrackRow]],
    video_info: VideoInfo,
    interval_seconds: float,
    max_keyframes: int,
) -> list[int]:
    frames = sorted(rows_by_frame)
    if not frames:
        return [1]
    stride = max(1, int(round(video_info.fps * interval_seconds)))
    selected: list[int] = []
    last = -stride
    for frame_index in frames:
        if frame_index - last >= stride:
            selected.append(frame_index)
            last = frame_index
    if len(selected) <= max_keyframes:
        return selected
    return _evenly_select(selected, max_keyframes)


def _evenly_select(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[position] for position in positions]


def _write_keyframes(
    config: VlmTrackingConfig,
    rows_by_frame: dict[int, list[MotTrackRow]],
    video_info: VideoInfo,
) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(config.source_video))
    keyframes: list[dict[str, Any]] = []
    try:
        for frame_index in _select_keyframe_indices(
            rows_by_frame,
            video_info,
            config.keyframe_interval_seconds,
            config.max_keyframes,
        ):
            frame = _read_frame(capture, frame_index)
            if frame is None:
                continue
            tracks = rows_by_frame.get(frame_index, [])
            annotated = _draw_track_boxes(frame, tracks)
            path = config.keyframes_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(path), annotated)
            keyframes.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": video_info.time_seconds(frame_index),
                    "path": str(path),
                    "visible_track_ids": [row.track_id for row in tracks],
                    "track_count": len(tracks),
                }
            )
    finally:
        capture.release()
    return keyframes


def _write_track_crops(
    config: VlmTrackingConfig,
    rows: list[MotTrackRow],
    video_info: VideoInfo,
    allowed_track_ids: set[int],
) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    crops: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(config.source_video))
    try:
        for track_id, track_rows in _group_rows_by_track(rows).items():
            if track_id not in allowed_track_ids:
                continue
            selected = _evenly_select(
                [row.frame_index for row in sorted(track_rows, key=lambda row: row.frame_index)],
                config.max_crops_per_track,
            )
            rows_by_selected_frame = {
                row.frame_index: row for row in track_rows if row.frame_index in selected
            }
            for frame_index in selected:
                row = rows_by_selected_frame.get(frame_index)
                if row is None:
                    continue
                frame = _read_frame(capture, frame_index)
                if frame is None:
                    continue
                crop = _crop_with_padding(frame, row, config.crop_padding)
                track_dir = config.crops_dir / f"track_{track_id:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)
                path = track_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(path), crop)
                crops.append(
                    {
                        "track_id": track_id,
                        "frame_index": frame_index,
                        "time_seconds": video_info.time_seconds(frame_index),
                        "path": str(path),
                    }
                )
    finally:
        capture.release()
    return crops


def _read_frame(capture: Any, frame_index: int) -> Any | None:
    import cv2  # type: ignore[import-not-found]

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
    ok, frame = capture.read()
    return frame if ok else None


def _draw_track_boxes(frame: Any, tracks: list[MotTrackRow]) -> Any:
    import cv2  # type: ignore[import-not-found]

    annotated = frame.copy()
    for row in tracks:
        x1, y1, x2, y2 = row.bbox_xyxy()
        color = _track_color(row.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {row.track_id}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id + 80) % 255),
        int((17 * track_id + 160) % 255),
        int((29 * track_id + 40) % 255),
    )


def _crop_with_padding(frame: Any, row: MotTrackRow, padding: float) -> Any:
    height, width = frame.shape[:2]
    pad_x = row.width * padding
    pad_y = row.height * padding
    x1 = max(0, int(math.floor(row.x - pad_x)))
    y1 = max(0, int(math.floor(row.y - pad_y)))
    x2 = min(width, int(math.ceil(row.x + row.width + pad_x)))
    y2 = min(height, int(math.ceil(row.y + row.height + pad_y)))
    return frame[y1:y2, x1:x2]


def _build_context(
    config: VlmTrackingConfig,
    rows: list[MotTrackRow],
    video_info: VideoInfo,
    metadata: dict[str, Any] | None,
    all_track_summaries: list[dict[str, Any]],
    track_summaries: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    crops: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_frame = _group_rows_by_frame(rows)
    track_ids = {row.track_id for row in rows}
    frame_counts = [len(items) for items in rows_by_frame.values()]
    return {
        "schema": "football_tracking.vlm_context.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_video": str(config.source_video),
        "tracked_video": str(config.tracked_video) if config.tracked_video else None,
        "tracks_path": str(config.tracks_path),
        "metadata_path": str(config.metadata_path) if config.metadata_path else None,
        "video": video_info.to_dict(),
        "tracking_summary": {
            "track_count": len(track_ids),
            "track_observation_count": len(rows),
            "frames_with_tracks": len(rows_by_frame),
            "mean_tracks_per_frame": sum(frame_counts) / len(frame_counts) if frame_counts else 0.0,
        },
        "tracking_diagnostics": _track_diagnostics(
            all_track_summaries,
            track_summaries,
            keyframes,
        ),
        "tracks": track_summaries,
        "keyframes": keyframes,
        "crops": crops,
        "tracking_metadata": metadata,
        "runtime": runtime_versions(),
    }


__all__ = [
    "MotTrackRow",
    "VlmAnalysisError",
    "VlmConfigError",
    "build_prompt",
    "read_mot_tracks",
    "run_vlm_analysis",
]
