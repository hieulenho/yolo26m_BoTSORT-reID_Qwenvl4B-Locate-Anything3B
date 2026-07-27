"""Build VLM-ready context from MOT tracking outputs and video frames."""

from __future__ import annotations

import json
import math
import re
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
    grounding = _read_optional_json(config.grounding_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.keyframes_dir.mkdir(parents=True, exist_ok=True)
    config.crops_dir.mkdir(parents=True, exist_ok=True)

    rows_by_frame = _group_rows_by_frame(rows)
    keyframes = _write_keyframes(config, rows_by_frame, video_info)
    all_track_summaries = _summarize_tracks(rows, video_info)
    track_summaries = _select_track_summaries(all_track_summaries, config)
    selected_track_ids = {int(row["track_id"]) for row in track_summaries}
    crops = _write_track_crops(
        config,
        rows,
        video_info,
        allowed_track_ids=selected_track_ids,
    )
    track_contexts = _write_track_context_frames(
        config,
        rows,
        video_info,
        allowed_track_ids=selected_track_ids,
    )
    context = _build_context(
        config,
        rows,
        video_info,
        metadata,
        grounding,
        all_track_summaries,
        track_summaries,
        keyframes,
        crops,
    )
    context["track_contexts"] = track_contexts
    model_crops = _merge_grounded_crops(
        crops,
        context["grounding_evidence"],
        max_crops_per_track=config.max_crops_per_track,
    )
    model_crops = _compose_track_evidence_panels(
        model_crops,
        output_dir=config.output_dir / "model_panels",
        panel_size=config.crop_output_size,
    )
    model_batches = _build_model_batches(
        context,
        keyframes,
        model_crops,
        track_contexts=track_contexts,
        max_images=config.max_model_images,
        max_tracks=config.max_tracks_per_batch,
    )
    context["model_batches"] = [
        {
            "batch_id": batch["batch_id"],
            "track_ids": batch["track_ids"],
            "image_paths": [str(path) for path in batch["image_paths"]],
            "image_labels": batch["image_labels"],
            "evidence_frames_by_track": batch["evidence_frames_by_track"],
        }
        for batch in model_batches
    ]
    context_path = config.output_dir / "vlm_context.json"
    prompt_path = config.output_dir / "prompt.md"
    context_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    prompt_text = build_prompt(config, context)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    model_result: dict[str, Any]
    if config.run_model:
        from football_tracking.vlm.qwen_runner import (
            QwenRunnerError,
            run_qwen_vlm_batches,
        )

        try:
            prompt_batch_dir = config.output_dir / "prompt_batches"
            prompt_batch_dir.mkdir(parents=True, exist_ok=True)
            jobs: list[dict[str, Any]] = []
            for batch in model_batches:
                batch_context = _context_for_track_ids(context, set(batch["track_ids"]))
                batch_prompt = build_prompt(config, batch_context)
                batch_prompt_path = prompt_batch_dir / f"{batch['batch_id']}.md"
                batch_prompt_path.write_text(batch_prompt, encoding="utf-8")
                jobs.append(
                    {
                        "batch_id": batch["batch_id"],
                        "prompt": batch_prompt,
                        "image_paths": batch["image_paths"],
                        "image_labels": batch["image_labels"],
                        "track_ids": batch["track_ids"],
                        "evidence_frames_by_track": batch[
                            "evidence_frames_by_track"
                        ],
                        "prompt_path": str(batch_prompt_path),
                    }
                )
            if jobs:
                raw_batches = run_qwen_vlm_batches(config, jobs)
                model_result = _merge_qwen_batch_results(raw_batches, jobs)
            else:
                model_result = {
                    "status": "skipped_no_tracks",
                    "reason": "No tracked object was available for semantic inference.",
                    "batch_count": 0,
                    "image_count": 0,
                    "track_predictions": [],
                }
        except QwenRunnerError as exc:
            model_result = {"status": "failed", "error": str(exc)}
        answer_path = config.output_dir / "vlm_answer.md"
        answer_json_path = config.output_dir / "vlm_answer.json"
        answer = model_result.get("answer", "")
        answer_text = (
            json.dumps(answer, indent=2, ensure_ascii=False)
            if isinstance(answer, dict)
            else str(answer)
        )
        answer_path.write_text(answer_text, encoding="utf-8")
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
            "track_context_count": len(track_contexts),
            "selected_track_count": len(track_summaries),
            "model_batch_count": len(model_batches),
            "modeled_track_count": len(
                {
                    int(track_id)
                    for batch in model_batches
                    for track_id in batch["track_ids"]
                }
            ),
        },
        "model_result": model_result,
        "paths": {
            "context_json": str(context_path),
            "prompt_md": str(prompt_path),
            "keyframes_dir": str(config.keyframes_dir),
            "crops_dir": str(config.crops_dir),
            "track_contexts_dir": str(config.output_dir / "track_contexts"),
            "answer_md": str(config.output_dir / "vlm_answer.md") if config.run_model else None,
            "answer_json": str(config.output_dir / "vlm_answer.json") if config.run_model else None,
        },
    }


def _select_track_summaries(
    all_track_summaries: list[dict[str, Any]],
    config: VlmTrackingConfig,
) -> list[dict[str, Any]]:
    if config.track_ids is None:
        return (
            all_track_summaries
            if config.max_tracks == 0
            else all_track_summaries[: config.max_tracks]
        )
    by_id = {int(row["track_id"]): row for row in all_track_summaries}
    missing = [track_id for track_id in config.track_ids if track_id not in by_id]
    if missing:
        raise VlmAnalysisError(
            "Requested track IDs do not exist in the MOT input: "
            + ", ".join(str(track_id) for track_id in missing)
        )
    return [by_id[track_id] for track_id in config.track_ids]


def _merge_grounded_crops(
    crops: list[dict[str, Any]],
    grounding: dict[str, Any],
    *,
    max_crops_per_track: int,
) -> list[dict[str, Any]]:
    """Prefer one Locate-grounded ROI, then retain independent temporal crops."""

    by_track: dict[int, list[dict[str, Any]]] = {}
    for row in crops:
        item = {**row, "source": "tracking_crop"}
        by_track.setdefault(int(row["track_id"]), []).append(item)
    grounded_by_track = {
        int(row["track_id"]): row
        for row in grounding.get("tracks", [])
        if isinstance(row, dict) and int(row.get("track_id", 0)) > 0
    }
    merged: list[dict[str, Any]] = []
    for track_id in sorted(by_track):
        original = by_track[track_id]
        grounded = grounded_by_track.get(track_id)
        selected: list[dict[str, Any]] = []
        if grounded:
            grounded_path = Path(str(grounded.get("crop_path", "")))
            if grounded_path.is_file():
                selected.append(
                    {
                        "track_id": track_id,
                        "frame_index": grounded.get("frame_index"),
                        "path": str(grounded_path),
                        "source": "locateanything",
                    }
                )
        grounded_frame = (
            int(grounded.get("frame_index", -1)) if grounded is not None else -1
        )
        selected.extend(
            row for row in original if int(row.get("frame_index", -2)) != grounded_frame
        )
        if len(selected) < max_crops_per_track:
            selected.extend(
                row for row in original if row not in selected
            )
        merged.extend(selected[:max_crops_per_track])
    return merged


def _compose_track_evidence_panels(
    crops: list[dict[str, Any]],
    *,
    output_dir: Path,
    panel_size: int,
) -> list[dict[str, Any]]:
    """Pack multi-time crops for one identity into one bounded model image."""

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    by_track: dict[int, list[dict[str, Any]]] = {}
    for row in crops:
        by_track.setdefault(int(row["track_id"]), []).append(row)
    panels: list[dict[str, Any]] = []
    for track_id, rows in sorted(by_track.items()):
        valid: list[tuple[dict[str, Any], Any]] = []
        for row in rows:
            image = cv2.imread(str(row["path"]))
            if image is not None:
                valid.append((row, image))
        if len(valid) < 2:
            panels.extend(rows[:1])
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        header_height = 28
        canvas = np.full(
            (panel_size + header_height, panel_size * len(valid), 3),
            24,
            dtype=np.uint8,
        )
        frame_indices: list[int] = []
        sources: list[str] = []
        for index, (row, image) in enumerate(valid):
            tile = _letterbox_image(image, panel_size)
            left = index * panel_size
            canvas[header_height:, left : left + panel_size] = tile
            frame_index = int(row.get("frame_index", 0))
            source = str(row.get("source", "tracking_crop"))
            frame_indices.append(frame_index)
            sources.append(source)
            label = (
                f"ID {track_id} | frame {frame_index} | "
                f"{'Locate' if source == 'locateanything' else 'track'}"
            )
            cv2.putText(
                canvas,
                label,
                (left + 5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        panel_path = output_dir / f"track_{track_id:06d}.jpg"
        if not cv2.imwrite(str(panel_path), canvas):
            raise VlmAnalysisError(f"Could not write model evidence panel: {panel_path}")
        panels.append(
            {
                "track_id": track_id,
                "frame_index": frame_indices[0],
                "frame_indices": frame_indices,
                "path": str(panel_path),
                "source": "evidence_panel",
                "panel_sources": sources,
            }
        )
    return panels


def _letterbox_image(image: Any, size: int) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    height, width = image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _build_model_batches(
    context: dict[str, Any],
    keyframes: list[dict[str, Any]],
    crops: list[dict[str, Any]],
    *,
    track_contexts: list[dict[str, Any]] | None = None,
    max_images: int,
    max_tracks: int = 3,
) -> list[dict[str, Any]]:
    """Pack selected tracks with local scene context into bounded batches."""
    if max_images < 1:
        raise VlmAnalysisError("sampling.max_model_images must be positive.")
    if max_tracks < 1:
        raise VlmAnalysisError("sampling.max_tracks_per_batch must be positive.")
    context_rows = list(track_contexts or [])
    # A target-local full frame is more useful than one repeated early keyframe
    # for role and fine-class inference later in a long video.
    global_limit = 0 if context_rows else min(len(keyframes), max(max_images - 1, 0))
    global_rows = keyframes[:global_limit]
    global_images = [Path(row["path"]) for row in global_rows]
    global_labels = [
        (
            f"Global keyframe at frame {row.get('frame_index', 'unknown')}; "
            f"visible track IDs: {row.get('visible_track_ids', [])}."
        )
        for row in global_rows
    ]
    crop_capacity = max_images - len(global_images)
    if crop_capacity < 1:
        raise VlmAnalysisError(
            "At least one model image slot must remain for a track crop."
        )
    global_frames_by_track: dict[int, set[int]] = {}
    for row in global_rows:
        frame_index = _positive_frame_index(row.get("frame_index"))
        if frame_index is None:
            continue
        for visible_track_id in row.get("visible_track_ids", []):
            try:
                track_id = int(visible_track_id)
            except (TypeError, ValueError):
                continue
            global_frames_by_track.setdefault(track_id, set()).add(frame_index)

    evidence_by_track: dict[int, list[tuple[Path, str, tuple[int, ...]]]] = {}
    for row in context_rows:
        track_id = int(row["track_id"])
        frame_index = _positive_frame_index(row.get("frame_index"))
        evidence_by_track.setdefault(track_id, []).append(
            (
                Path(row["path"]),
                (
                    f"Track-local scene context for target ID {track_id} at frame "
                    f"{row.get('frame_index', 'unknown')}; the highlighted box is "
                    "the target track."
                ),
                (frame_index,) if frame_index is not None else (),
            )
        )
    for row in crops:
        track_id = int(row["track_id"])
        source = str(row.get("source", "tracking_crop"))
        if source == "evidence_panel":
            frames = [
                frame_index
                for value in row.get("frame_indices", [])
                if (frame_index := _positive_frame_index(value)) is not None
            ]
            description = (
                f"Multi-time evidence panel for track ID {track_id}; "
                f"frames left-to-right: {frames}."
            )
        else:
            frame_index = _positive_frame_index(row.get("frame_index"))
            frames = [frame_index] if frame_index is not None else []
            description = (
                f"{'Locate-grounded' if source == 'locateanything' else 'Appearance'} "
                f"crop for track ID {track_id} at frame "
                f"{row.get('frame_index', 'unknown')}."
            )
        evidence_by_track.setdefault(track_id, []).append(
            (
                Path(row["path"]),
                description,
                tuple(frames),
            )
        )
    selected_ids = [int(row["track_id"]) for row in context["tracks"]]
    batches: list[dict[str, Any]] = []
    current_ids: list[int] = []
    current_crops: list[Path] = []
    current_crop_labels: list[str] = []
    current_frames: dict[int, set[int]] = {}

    def flush() -> None:
        if not current_ids:
            return
        batches.append(
            {
                "batch_id": f"batch_{len(batches) + 1:03d}",
                "track_ids": list(current_ids),
                "image_paths": [*global_images, *current_crops],
                "image_labels": [*global_labels, *current_crop_labels],
                "evidence_frames_by_track": {
                    str(track_id): sorted(current_frames.get(track_id, set()))
                    for track_id in current_ids
                },
            }
        )
        current_ids.clear()
        current_crops.clear()
        current_crop_labels.clear()
        current_frames.clear()

    for track_id in selected_ids:
        track_crop_items = evidence_by_track.get(track_id, [])[:crop_capacity]
        if not track_crop_items:
            continue
        if current_ids and (
            len(current_ids) >= max_tracks
            or len(current_crops) + len(track_crop_items) > crop_capacity
        ):
            flush()
        current_ids.append(track_id)
        current_frames[track_id] = set(global_frames_by_track.get(track_id, set()))
        current_crops.extend(path for path, _label, _frames in track_crop_items)
        current_crop_labels.extend(
            label for _path, label, _frames in track_crop_items
        )
        for _path, _label, frames in track_crop_items:
            current_frames[track_id].update(frames)
    flush()
    if selected_ids and not batches:
        raise VlmAnalysisError("No valid track crops were available for Qwen inference.")
    return batches


def _context_for_track_ids(
    context: dict[str, Any],
    track_ids: set[int],
) -> dict[str, Any]:
    subset = dict(context)
    subset["tracks"] = [
        row for row in context["tracks"] if int(row["track_id"]) in track_ids
    ]
    subset["crops"] = [
        row for row in context["crops"] if int(row["track_id"]) in track_ids
    ]
    subset["track_contexts"] = [
        row
        for row in context.get("track_contexts", [])
        if int(row["track_id"]) in track_ids
    ]
    subset["keyframes"] = [
        {
            **row,
            "visible_track_ids": [
                int(track_id)
                for track_id in row.get("visible_track_ids", [])
                if int(track_id) in track_ids
            ],
        }
        for row in context["keyframes"]
    ]
    summary = dict(context["tracking_summary"])
    summary["track_count"] = len(track_ids)
    summary["track_observation_count"] = sum(
        int(row.get("observation_count") or 0) for row in subset["tracks"]
    )
    subset["tracking_summary"] = summary
    diagnostics = dict(context["tracking_diagnostics"])
    for key in (
        "stable_long_tracks",
        "largest_displacement_tracks",
        "fragmented_tracks",
        "low_confidence_tracks",
        "short_tracks",
    ):
        diagnostics[key] = [
            row
            for row in diagnostics.get(key, [])
            if int(row.get("track_id", -1)) in track_ids
        ]
    for key in (
        "selected_track_ids_visible_in_keyframes",
        "selected_track_ids_not_visible_in_keyframes",
    ):
        diagnostics[key] = [
            int(track_id)
            for track_id in diagnostics.get(key, [])
            if int(track_id) in track_ids
        ]
    subset["tracking_diagnostics"] = diagnostics
    grounding = dict(context.get("grounding_evidence") or {})
    grounding["tracks"] = [
        row
        for row in grounding.get("tracks", [])
        if int(row.get("track_id", -1)) in track_ids
    ]
    subset["grounding_evidence"] = grounding
    return subset


def _merge_qwen_batch_results(
    raw_result: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    jobs_by_id = {str(job["batch_id"]): job for job in jobs}
    predictions: dict[int, dict[str, Any]] = {}
    notes: list[str] = []
    parse_failures: list[dict[str, str]] = []
    validation_warnings: list[dict[str, Any]] = []
    expected_ids = {
        int(track_id) for job in jobs for track_id in job.get("track_ids", [])
    }
    for batch in raw_result.get("batches", []):
        batch_id = str(batch.get("batch_id", ""))
        allowed_ids = {
            int(track_id)
            for track_id in jobs_by_id.get(batch_id, {}).get("track_ids", [])
        }
        evidence_map = jobs_by_id.get(batch_id, {}).get(
            "evidence_frames_by_track"
        )
        try:
            parsed = _parse_qwen_json_object(str(batch.get("answer", "")))
        except ValueError as exc:
            parse_failures.append({"batch_id": batch_id, "error": str(exc)})
            continue
        rows = parsed.get("track_predictions", [])
        if not isinstance(rows, list):
            parse_failures.append(
                {"batch_id": batch_id, "error": "track_predictions is not a list"}
            )
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("track_id") is None:
                continue
            try:
                track_id = int(row["track_id"])
            except (TypeError, ValueError):
                continue
            if track_id not in allowed_ids:
                continue
            allowed_frames = _allowed_evidence_frames(evidence_map, track_id)
            sanitized, row_warnings = _sanitize_qwen_prediction(
                row,
                track_id=track_id,
                allowed_frames=allowed_frames,
            )
            validation_warnings.extend(
                {
                    "batch_id": batch_id,
                    "track_id": track_id,
                    **warning,
                }
                for warning in row_warnings
            )
            previous = predictions.get(track_id)
            if previous is None or float(sanitized.get("confidence", 0.0)) > float(
                previous.get("confidence", 0.0)
            ):
                predictions[track_id] = sanitized
        parsed_notes = parsed.get("notes", [])
        if isinstance(parsed_notes, list):
            notes.extend(str(note) for note in parsed_notes)
    missing_ids = sorted(expected_ids - set(predictions))
    for track_id in missing_ids:
        predictions[track_id] = {
            "track_id": track_id,
            "class_label": "unknown",
            "attributes": {},
            "confidence": 0.0,
            "fine_label": "unknown",
            "fine_label_type": "unknown",
            "fine_confidence": 0.0,
            "evidence_frames": [],
            "evidence": "",
            "unknown_reason": "no_valid_prediction_returned_by_qwen",
        }
    answer = {
        "track_predictions": [predictions[track_id] for track_id in sorted(predictions)],
        "notes": notes,
    }
    return {
        **raw_result,
        "status": "partial" if parse_failures or missing_ids else "ok",
        "answer": answer,
        "coverage": {
            "expected_track_count": len(expected_ids),
            "predicted_track_count": len(expected_ids) - len(missing_ids),
            "missing_track_ids": missing_ids,
        },
        "parse_failures": parse_failures,
        "validation_warnings": validation_warnings,
        "jobs": [
            {
                "batch_id": job["batch_id"],
                "track_ids": job["track_ids"],
                "image_count": len(job["image_paths"]),
                "prompt_path": job["prompt_path"],
                "evidence_frames_by_track": job.get(
                    "evidence_frames_by_track", {}
                ),
            }
            for job in jobs
        ],
    }


def _positive_frame_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return parsed if parsed > 0 else None


def _allowed_evidence_frames(
    evidence_map: Any,
    track_id: int,
) -> set[int] | None:
    # A missing map denotes a legacy caller. Production jobs always carry the
    # exact frames represented by their model images.
    if not isinstance(evidence_map, dict):
        return None
    raw_frames = evidence_map.get(str(track_id), evidence_map.get(track_id, []))
    if not isinstance(raw_frames, list | tuple | set):
        return set()
    return {
        frame_index
        for value in raw_frames
        if (frame_index := _positive_frame_index(value)) is not None
    }


def _sanitize_qwen_prediction(
    row: dict[str, Any],
    *,
    track_id: int,
    allowed_frames: set[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sanitized = dict(row)
    warnings: list[dict[str, Any]] = []
    sanitized["track_id"] = track_id
    class_label = str(row.get("class_label", "")).strip()
    sanitized["class_label"] = class_label or "unknown"
    if not isinstance(row.get("attributes", {}), dict):
        sanitized["attributes"] = {}
        warnings.append({"code": "invalid_attributes_replaced"})
    else:
        sanitized["attributes"] = dict(row.get("attributes", {}))

    for field in ("confidence", "fine_confidence"):
        bounded, changed = _bounded_confidence(row.get(field, 0.0))
        sanitized[field] = bounded
        if changed:
            warnings.append(
                {
                    "code": "invalid_confidence_bounded",
                    "field": field,
                    "value": row.get(field),
                }
            )

    fine_label = str(row.get("fine_label", "unknown")).strip() or "unknown"
    fine_type = (
        str(row.get("fine_label_type", "unknown"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    allowed_fine_types = {
        "role",
        "subtype",
        "species",
        "breed",
        "make",
        "model",
        "make_model",
        "medical",
        "diagnosis",
        "identity",
        "unknown",
    }
    if fine_type not in allowed_fine_types:
        warnings.append(
            {
                "code": "unsupported_fine_label_type",
                "value": fine_type,
            }
        )
        fine_type = "unknown"
        fine_label = "unknown"
        sanitized["fine_confidence"] = 0.0
    sanitized["fine_label"] = fine_label
    sanitized["fine_label_type"] = fine_type

    raw_frames = row.get("evidence_frames", [])
    if not isinstance(raw_frames, list | tuple | set):
        raw_frames = []
        warnings.append({"code": "invalid_evidence_frames_replaced"})
    parsed_frames = {
        frame_index
        for value in raw_frames
        if (frame_index := _positive_frame_index(value)) is not None
    }
    if allowed_frames is None:
        validated_frames = parsed_frames
    else:
        validated_frames = parsed_frames & allowed_frames
        unsupported = sorted(parsed_frames - allowed_frames)
        if unsupported:
            warnings.append(
                {
                    "code": "unsupported_evidence_frames_removed",
                    "values": unsupported,
                }
            )
    sanitized["evidence_frames"] = sorted(validated_frames)
    return sanitized, warnings


def _bounded_confidence(value: Any) -> tuple[float, bool]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0, True
    if not math.isfinite(parsed):
        return 0.0, True
    bounded = min(max(parsed, 0.0), 1.0)
    return bounded, bounded != parsed


def _parse_qwen_json_object(answer: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Qwen batch answer.")
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Qwen batch JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Qwen batch answer root must be an object.")
    return payload


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
    requested_rows = context.get("selected_tracks") or context.get("tracks") or []
    requested_track_ids = sorted(
        int(row["track_id"] if "track_id" in row else row["id"])
        for row in requested_rows
        if isinstance(row, dict) and ("track_id" in row or "id" in row)
    )
    requested_ids_text = ", ".join(str(track_id) for track_id in requested_track_ids)
    context_json = json.dumps(
        _compact_prompt_context(context),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    lines = [
            "# Tracking VLM Analysis Task",
            "",
            config.task_prompt,
            "",
            "Use only track IDs that appear in the metadata below.",
            f"This batch requests exactly these track IDs: [{requested_ids_text}].",
            "Return exactly one track_predictions entry for each requested ID and no "
            "entries for any other visible ID.",
            "Use visual evidence from keyframes and crops. Do not infer a semantic "
            "label from track duration, confidence, or motion alone.",
            "LocateAnything evidence was produced before this inference. It verifies "
            "where a tracked object remains visible, including partial visibility, but "
            "its class query is not fine-label ground truth.",
            "Treat detector_label as the stable base class unless the images clearly "
            "contradict it. Infer role, subtype, or species only from visual cues.",
            "When visual evidence is insufficient, return unknown.",
            "Each provided keyframe image is annotated with tracking IDs.",
            "The metadata below is supporting context, not semantic ground truth:",
            "",
            "```json",
            context_json,
            "```",
            "",
        ]
    if config.output_schema == "dynamic":
        lines.extend(
            [
                "Use a two-level open vocabulary. class_label is the stable base class "
                "used by detection/tracking; fine_label is the most specific visually "
                "supported subtype, species, breed, role, make, or model.",
                "The candidate vocabulary in the task is a non-binding hint. Preserve a "
                "visually supported class that is absent from it.",
                "Keep color, clothing, state, and action in attributes. A role belongs in "
                "fine_label only when it is the requested taxonomy facet.",
                "Set fine_label_type to exactly one of: role, subtype, species, breed, "
                "make, model, make_model, medical, diagnosis, identity, or unknown.",
                "Never infer species, breed, brand, model, medical diagnosis, or personal "
                "identity from context alone. Prefer fine_label=unknown over guessing.",
                "A fine label needs a discriminative visual cue visible in at least two "
                "independent appearance views of that requested track. The views may be "
                "separate crops or tiles in a multi-time evidence panel. Repeated global "
                "keyframes do not count as independent fine-label evidence. If this "
                "condition is not met, set fine_label=unknown and fine_confidence=0.",
                "Set class_label=unknown only when the base class is unclear. A clear base "
                "class may coexist with fine_label=unknown.",
                "Keep the response compact so JSON is never truncated. Do not return "
                "observations, taxonomy_path, alternatives, notes, or prose evidence.",
                "Return one compact JSON object only, without Markdown fences or prose:",
                '{"track_predictions":[{"track_id":7,"class_label":"car",'
                '"fine_label":"sedan",'
                '"fine_label_type":"subtype",'
                '"attributes":{"color":"red"},"confidence":0.85,'
                '"fine_confidence":0.72,"evidence_frames":[5,20],'
                '"unknown_reason":null,"fine_unknown_reason":null}]}',
                "Explain unknown_reason and fine_unknown_reason whenever their corresponding "
                "label is unknown, using at most 12 words per reason.",
                "Use evidence_frames to list every visually usable supplied crop in frame order.",
                "Include every selected track that has usable visual evidence.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Return one JSON object only, without Markdown fences or prose:",
            '{"track_predictions":[{"track_id":7,"team_label":"light_blue",'
            '"role_label":"player","confidence":0.85,"evidence_frames":[5],'
            '"evidence":"short visual reason"}],"notes":[]}',
            "Include every selected track that has usable visual evidence.",
            "Allowed team_label values: light_blue, dark_blue, yellow_kit, dark_kit, "
            "goalkeeper_green, goalkeeper_red, goalkeeper_orange, referee_black, unknown.",
            "Allowed role_label values: goalkeeper, defender, midfielder, forward, "
            "player, referee, unknown.",
            "",
            ]
        )
    return "\n".join(lines)


def _compact_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    video = context["video"]
    summary = context["tracking_summary"]
    diagnostics = context["tracking_diagnostics"]
    selected_ids = {
        int(row["track_id"])
        for row in context["tracks"]
        if isinstance(row, dict) and row.get("track_id") is not None
    }
    metadata = context.get("tracking_metadata") or {}
    detector_labels = [
        {
            "track_id": int(row["track_id"]),
            "detector_label": row.get("class_name"),
            "consensus": _rounded(row.get("class_consensus"), 3),
        }
        for row in metadata.get("track_classes", [])
        if isinstance(row, dict) and int(row.get("track_id", -1)) in selected_ids
    ]
    grounding = context.get("grounding_evidence") or {}
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
        "selected_tracks": _prompt_tracks(context["tracks"], limit=40),
        "detector_labels": detector_labels,
        "locateanything": {
            "status": grounding.get("status"),
            "tracks": [
                {
                    "track_id": row.get("track_id"),
                    "query_class": row.get("class_label"),
                    "confidence": _rounded(row.get("confidence"), 3),
                    "iou": _rounded(row.get("iou"), 3),
                    "partially_visible": row.get("partially_visible"),
                    "frame": row.get("frame_index"),
                }
                for row in grounding.get("tracks", [])
                if int(row.get("track_id", -1)) in selected_ids
            ],
        },
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
            "grounding": str(config.grounding_path) if config.grounding_path else None,
            "grounding_exists": (
                config.grounding_path.is_file() if config.grounding_path else None
            ),
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
    if config.grounding_path is not None and not config.grounding_path.is_file():
        raise VlmAnalysisError(f"Grounding result does not exist: {config.grounding_path}")
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


def _write_track_context_frames(
    config: VlmTrackingConfig,
    rows: list[MotTrackRow],
    video_info: VideoInfo,
    *,
    allowed_track_ids: set[int],
) -> list[dict[str, Any]]:
    """Write one target-highlighted full frame per track for scene-aware semantics."""
    import cv2  # type: ignore[import-not-found]

    output_dir = config.output_dir / "track_contexts"
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(config.source_video))
    try:
        for track_id, track_rows in _group_rows_by_track(rows).items():
            if track_id not in allowed_track_ids:
                continue
            representative = _select_representative_track_rows(
                track_rows,
                1,
                video_info,
            )[0]
            frame = _read_frame(capture, representative.frame_index)
            if frame is None:
                continue
            annotated = _draw_track_boxes(frame, [representative])
            banner = f"TARGET ID {track_id} | frame {representative.frame_index}"
            cv2.rectangle(annotated, (0, 0), (min(420, annotated.shape[1]), 34), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                banner,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (40, 240, 255),
                2,
                cv2.LINE_AA,
            )
            path = output_dir / (
                f"track_{track_id:06d}_frame_{representative.frame_index:06d}.jpg"
            )
            if not cv2.imwrite(str(path), annotated):
                raise VlmAnalysisError(f"Could not write track context frame: {path}")
            contexts.append(
                {
                    "track_id": track_id,
                    "frame_index": representative.frame_index,
                    "time_seconds": video_info.time_seconds(
                        representative.frame_index
                    ),
                    "path": str(path),
                    "selection_score": round(
                        _track_crop_score(representative, video_info),
                        6,
                    ),
                }
            )
    finally:
        capture.release()
    return contexts


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
            selected_rows = _select_representative_track_rows(
                track_rows,
                config.max_crops_per_track,
                video_info,
            )
            for row in selected_rows:
                frame_index = row.frame_index
                frame = _read_frame(capture, frame_index)
                if frame is None:
                    continue
                crop = _crop_with_padding(frame, row, config.crop_padding)
                native_height, native_width = crop.shape[:2]
                crop = _prepare_crop_for_vlm(crop, config.crop_output_size)
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
                        "native_width": native_width,
                        "native_height": native_height,
                        "output_width": int(crop.shape[1]),
                        "output_height": int(crop.shape[0]),
                        "selection_score": round(
                            _track_crop_score(row, video_info), 6
                        ),
                    }
                )
    finally:
        capture.release()
    return crops


def _select_representative_track_rows(
    rows: list[MotTrackRow],
    limit: int,
    video_info: VideoInfo,
) -> list[MotTrackRow]:
    """Choose one high-quality crop from each temporal segment of a track."""
    ordered = sorted(rows, key=lambda row: row.frame_index)
    if len(ordered) <= limit:
        return ordered
    selected: list[MotTrackRow] = []
    segment_count = min(limit, len(ordered))
    for segment_index in range(segment_count):
        start = segment_index * len(ordered) // segment_count
        end = (segment_index + 1) * len(ordered) // segment_count
        segment = ordered[start:end]
        selected.append(
            max(
                segment,
                key=lambda row: (
                    _track_crop_score(row, video_info),
                    -row.frame_index,
                ),
            )
        )
    return sorted(selected, key=lambda row: row.frame_index)


def _track_crop_score(row: MotTrackRow, video_info: VideoInfo) -> float:
    area = max(row.width, 0.0) * max(row.height, 0.0)
    if area <= 0:
        return 0.0
    x1 = max(row.x, 0.0)
    y1 = max(row.y, 0.0)
    x2 = min(row.x + row.width, float(video_info.width))
    y2 = min(row.y + row.height, float(video_info.height))
    visible_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    visible_ratio = visible_area / area
    confidence = row.confidence if row.confidence is not None else 0.5
    return math.log1p(area) * visible_ratio * (0.5 + 0.5 * confidence)


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


def _prepare_crop_for_vlm(crop: Any, output_size: int) -> Any:
    import cv2  # type: ignore[import-not-found]

    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise VlmAnalysisError("Cannot prepare an empty track crop.")
    scale = output_size / max(height, width)
    resized_width = max(1, min(output_size, int(round(width * scale))))
    resized_height = max(1, min(output_size, int(round(height * scale))))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    top = (output_size - resized_height) // 2
    bottom = output_size - resized_height - top
    left = (output_size - resized_width) // 2
    right = output_size - resized_width - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def _grounding_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"status": "not_provided", "summary": {}, "tracks": []}
    request_inputs = {
        str(row.get("request_id")): dict(row.get("grounding_input") or {})
        for row in payload.get("grounding_results", [])
        if isinstance(row, dict)
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in payload.get("associations", []):
        if not isinstance(row, dict) or not row.get("accepted_for_fusion", False):
            continue
        track_id = int(row.get("track_id", 0))
        if track_id <= 0:
            continue
        grouped.setdefault(track_id, []).append(row)
    tracks: list[dict[str, Any]] = []
    for track_id, rows in sorted(grouped.items()):
        best = max(rows, key=lambda row: float(row.get("confidence", 0.0)))
        prepared = request_inputs.get(str(best.get("request_id")), {})
        tracks.append(
            {
                "track_id": track_id,
                "class_label": str(best.get("class_label", "unknown")),
                "confidence": float(best.get("confidence", 0.0)),
                "iou": float(best.get("iou", 0.0)),
                "frame_index": int(best.get("frame_index", 0)),
                "crop_path": prepared.get("image_path"),
                "input_mode": prepared.get("mode"),
                "partially_visible": float(best.get("iou", 0.0)) < 0.5,
            }
        )
    return {
        "status": "available",
        "summary": dict(payload.get("summary") or {}),
        "tracks": tracks,
    }


def _build_context(
    config: VlmTrackingConfig,
    rows: list[MotTrackRow],
    video_info: VideoInfo,
    metadata: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
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
        "grounding_path": str(config.grounding_path) if config.grounding_path else None,
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
        "grounding_evidence": _grounding_context(grounding),
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
