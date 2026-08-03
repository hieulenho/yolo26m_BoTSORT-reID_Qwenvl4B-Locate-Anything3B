"""Non-blocking semantic event queue and bounded Qwen worker."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from football_tracking.adaptive_tracking.ontology import VocabularyRegistry
from football_tracking.adaptive_tracking.semantic_fusion import (
    SemanticFusionError,
    TrackSemanticEvidence,
    fuse_track_semantics,
    normalize_semantic_evidence,
    parse_qwen_answer,
)
from football_tracking.adaptive_tracking.temporal_memory import TemporalSemanticMemory
from football_tracking.paths import get_project_root
from football_tracking.tracking.schemas import TrackOutput
from football_tracking.vlm.config import load_vlm_tracking_config
from football_tracking.vlm.qwen_runner import run_qwen_vlm_batches


class SemanticQueueError(RuntimeError):
    """Raised when realtime semantic queue data is invalid."""


_VISIBLE_COLOR_LABELS = {
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
    "beige",
    "maroon",
    "purple",
    "pink",
    "gold",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _replace_file_with_retry(temporary, path)


def _replace_file_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 8,
    initial_delay_seconds: float = 0.005,
) -> Path:
    """Handle brief Windows sharing violations between queue readers and writers."""
    if attempts < 1:
        raise SemanticQueueError("File replace attempts must be positive.")
    for attempt in range(attempts):
        try:
            return source.replace(target)
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(initial_delay_seconds * (attempt + 1))
    raise AssertionError("unreachable")


class SemanticEventQueue:
    """Persist track crops without blocking the detector/tracker loop."""

    def __init__(
        self,
        root: str | Path,
        *,
        context_id: str,
        max_pending_events: int = 256,
        semantic_task: dict[str, Any] | None = None,
    ) -> None:
        if max_pending_events < 1:
            raise SemanticQueueError("max_pending_events must be positive.")
        self.root = Path(root)
        self.context_id = str(context_id)
        self.pending_dir = self.root / "pending"
        self.processing_dir = self.root / "processing"
        self.processed_dir = self.root / "processed"
        self.failed_dir = self.root / "failed"
        self.crops_dir = self.root / "crops"
        self.panels_dir = self.root / "panels"
        self._last_frame_by_track: dict[int, int] = {}
        self._pending_path_by_track: dict[int, Path] = {}
        self._processing_track_ids: set[int] = set()
        self._processing_refresh_seconds = 0.0
        self._evidence_by_track: dict[int, list[dict[str, Any]]] = {}
        self.semantic_task = dict(semantic_task or {})
        self.max_evidence_images = int(self.semantic_task.get("max_evidence_images", 2))
        self.evidence_interval_frames = int(
            self.semantic_task.get("evidence_interval_frames", 0)
        )
        self.evidence_collection_delay_seconds = float(
            self.semantic_task.get("evidence_collection_delay_seconds", 0.0)
        )
        if self.evidence_interval_frames < 0:
            raise SemanticQueueError("evidence_interval_frames cannot be negative.")
        if not 0.0 <= self.evidence_collection_delay_seconds <= 5.0:
            raise SemanticQueueError(
                "evidence_collection_delay_seconds must be in [0, 5]."
            )
        self.evidence_layout = str(
            self.semantic_task.get("evidence_layout", "panel")
        ).strip().lower()
        self.evidence_panel_width = int(
            self.semantic_task.get("evidence_panel_width", 512)
        )
        self.evidence_panel_height = int(
            self.semantic_task.get("evidence_panel_height", 384)
        )
        self.evidence_context_fraction = float(
            self.semantic_task.get("evidence_context_fraction", 0.55)
        )
        self.crop_padding = float(self.semantic_task.get("crop_padding", 0.15))
        self.crop_size = int(self.semantic_task.get("crop_size", 256))
        self.minimum_crop_quality = float(
            self.semantic_task.get("minimum_crop_quality", 0.0)
        )
        self.replacement_quality_margin = float(
            self.semantic_task.get("replacement_quality_margin", 0.08)
        )
        self.max_pending_events = int(max_pending_events)
        self.dropped_full = 0
        self.replaced_pending = 0
        self.rejected_low_quality = 0
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.processed_dir,
            self.failed_dir,
            self.crops_dir,
            self.panels_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._pending_estimate = len(list(self.pending_dir.glob("*.json")))
        self._restore_pending_index()

    @property
    def pending_count(self) -> int:
        self._pending_estimate = len(list(self.pending_dir.glob("*.json")))
        return self._pending_estimate

    @property
    def pending_track_ids(self) -> set[int]:
        stale = [
            track_id
            for track_id, path in self._pending_path_by_track.items()
            if not path.is_file()
        ]
        for track_id in stale:
            self._pending_path_by_track.pop(track_id, None)
        self._pending_estimate = len(self._pending_path_by_track)
        self._refresh_processing_track_ids()
        return set(self._pending_path_by_track) | self._processing_track_ids

    def diagnostics(self) -> dict[str, Any]:
        self._refresh_processing_track_ids(force=True)
        return {
            "pending_events": self.pending_count,
            "pending_tracks": len(self.pending_track_ids),
            "processing_tracks": len(self._processing_track_ids),
            "inflight_tracks": len(self.pending_track_ids),
            "dropped_full": self.dropped_full,
            "replaced_pending": self.replaced_pending,
            "rejected_low_quality": self.rejected_low_quality,
            "minimum_crop_quality": self.minimum_crop_quality,
            "max_evidence_images": self.max_evidence_images,
            "evidence_interval_frames": self.evidence_interval_frames,
            "evidence_collection_delay_seconds": (
                self.evidence_collection_delay_seconds
            ),
            "evidence_layout": self.evidence_layout,
            "evidence_panel_size": [
                self.evidence_panel_width,
                self.evidence_panel_height,
            ],
            "evidence_context_fraction": self.evidence_context_fraction,
            "crop_padding": self.crop_padding,
            "crop_size": self.crop_size,
        }

    def enqueue(
        self,
        *,
        frame: Any,
        frame_index: int,
        track: TrackOutput,
        reason: str,
        minimum_frame_gap: int = 90,
        crop_padding: float | None = None,
        crop_size: int | None = None,
    ) -> Path | None:
        crop_padding = self.crop_padding if crop_padding is None else crop_padding
        crop_size = self.crop_size if crop_size is None else crop_size
        if minimum_frame_gap < 1:
            raise SemanticQueueError("minimum_frame_gap must be positive.")
        if not 0.0 <= crop_padding <= 1.0:
            raise SemanticQueueError("crop_padding must be in [0, 1].")
        if crop_size < 64:
            raise SemanticQueueError("crop_size must be at least 64.")
        track_id = int(track.track_id)
        existing_path = self._pending_path_by_track.get(track_id)
        if existing_path is not None and not existing_path.is_file():
            self._pending_path_by_track.pop(track_id, None)
            self._pending_estimate = max(self._pending_estimate - 1, 0)
            existing_path = None
        self._refresh_processing_track_ids()
        if track_id in self._processing_track_ids:
            self._last_frame_by_track[track_id] = frame_index
            return None
        last_frame = self._last_frame_by_track.get(track_id)
        evidence_count = len(self._evidence_by_track.get(track_id, []))
        effective_frame_gap = minimum_frame_gap
        if (
            existing_path is not None
            and evidence_count < self.max_evidence_images
            and self.evidence_interval_frames > 0
        ):
            effective_frame_gap = min(
                minimum_frame_gap,
                self.evidence_interval_frames,
            )
        within_gap = (
            last_frame is not None
            and frame_index - last_frame < effective_frame_gap
        )
        if within_gap:
            return None
        inflight_estimate = self._pending_estimate + len(self._processing_track_ids)
        if existing_path is None and inflight_estimate >= self.max_pending_events:
            self._pending_estimate = len(list(self.pending_dir.glob("*.json")))
            self._refresh_processing_track_ids(force=True)
            inflight_estimate = self._pending_estimate + len(
                self._processing_track_ids
            )
        if existing_path is None and inflight_estimate >= self.max_pending_events:
            self.dropped_full += 1
            self._last_frame_by_track[track_id] = frame_index
            return None

        crop_result = _track_crop(
            frame,
            track,
            padding=crop_padding,
            output_size=crop_size,
        )
        if crop_result is None:
            return None
        crop, target_bbox = crop_result
        quality = _crop_quality(frame, crop, track, target_bbox)
        if quality["score"] < self.minimum_crop_quality:
            self.rejected_low_quality += 1
            self._last_frame_by_track[track_id] = frame_index
            return None
        existing_event = _read_json_if_present(existing_path)
        existing_quality = float(
            ((existing_event or {}).get("crop_quality") or {}).get("score", 0.0)
        )
        replacement = existing_path is not None
        existing_frames = {
            int(value)
            for value in (existing_event or {}).get("evidence_frame_indices", [])
        }
        needs_temporal_evidence = (
            replacement
            and frame_index not in existing_frames
            and len(existing_frames) < self.max_evidence_images
        )
        if replacement and (
            not needs_temporal_evidence
            and quality["score"] < existing_quality + self.replacement_quality_margin
        ):
            self._last_frame_by_track[track_id] = frame_index
            return None
        superseded_path: Path | None = None
        if replacement and existing_path is not None:
            superseded_dir = self.root / "superseded"
            superseded_dir.mkdir(parents=True, exist_ok=True)
            superseded_path = superseded_dir / existing_path.name
            try:
                _replace_file_with_retry(existing_path, superseded_path)
            except FileNotFoundError:
                self._pending_path_by_track.pop(track_id, None)
                return None
        event_id = f"f{frame_index:09d}_t{track_id:07d}"
        crop_path = self.crops_dir / f"{event_id}.jpg"
        panel_path = self.panels_dir / f"{event_id}.jpg"
        event_path = self.pending_dir / f"{event_id}.json"
        if not cv2.imwrite(str(crop_path), crop):
            if superseded_path is not None and existing_path is not None:
                _replace_file_with_retry(superseded_path, existing_path)
            raise SemanticQueueError(f"Could not write semantic crop: {crop_path}")
        evidence = self._updated_evidence(
            track_id,
            {
                "frame_index": frame_index,
                "crop_path": str(crop_path.resolve()),
                "quality": quality["score"],
            },
        )
        if self.evidence_layout == "panel":
            panel = _build_evidence_panel(
                frame,
                track,
                evidence,
                width=self.evidence_panel_width,
                height=self.evidence_panel_height,
                context_fraction=self.evidence_context_fraction,
            )
            if not cv2.imwrite(str(panel_path), panel):
                crop_path.unlink(missing_ok=True)
                if superseded_path is not None and existing_path is not None:
                    _replace_file_with_retry(superseded_path, existing_path)
                raise SemanticQueueError(
                    f"Could not write semantic evidence panel: {panel_path}"
                )
            qwen_image_paths = [str(panel_path.resolve())]
            qwen_image_labels = [
                (
                    f"Track {track_id} evidence panel: highlighted scene context and "
                    f"{len(evidence)} temporally selected crop(s)."
                )
            ]
        else:
            qwen_image_paths = [row["crop_path"] for row in evidence]
            qwen_image_labels = [
                f"Track {track_id} crop at frame {int(row['frame_index'])}."
                for row in evidence
            ]
        now = time.time()
        ready_after = float(
            (existing_event or {}).get(
                "ready_after_unix_seconds",
                now + self.evidence_collection_delay_seconds,
            )
        )
        detector_class_name = str(
            track.metadata.get("base_detector_class_name", track.class_name)
        )
        fast_semantic_label = track.metadata.get("fast_semantic_label")
        fast_visual_color = track.metadata.get("fast_visual_color")
        payload = {
            "schema_version": "2.0",
            "event_id": event_id,
            "context_id": self.context_id,
            "frame_index": frame_index,
            "track_id": track_id,
            "detector_class_id": track.class_id,
            "detector_class_name": detector_class_name,
            "fast_semantic_proposal": (
                {
                    "class_label": (
                        str(fast_semantic_label)
                        if fast_semantic_label is not None
                        else None
                    ),
                    "confidence": float(
                        track.metadata.get("fast_semantic_confidence", 0.0)
                    ),
                    "visual_color": (
                        str(fast_visual_color)
                        if fast_visual_color is not None
                        else None
                    ),
                    "visual_color_confidence": float(
                        track.metadata.get(
                            "fast_visual_color_confidence",
                            0.0,
                        )
                    ),
                    "source": str(
                        track.metadata.get(
                            "fast_semantic_source",
                            "supplemental_detector",
                        )
                    ),
                }
                if fast_semantic_label is not None
                or fast_visual_color is not None
                else None
            ),
            "track_confidence": track.confidence,
            "reason": reason,
            "priority": _event_priority(reason, quality["score"], track.confidence),
            "created_unix_seconds": now,
            "ready_after_unix_seconds": ready_after,
            "crop_path": str(crop_path.resolve()),
            "crop_quality": quality,
            "target_bbox_in_crop_xyxy": list(target_bbox),
            "evidence": evidence,
            "evidence_layout": self.evidence_layout,
            "evidence_panel_path": (
                str(panel_path.resolve()) if self.evidence_layout == "panel" else None
            ),
            "qwen_image_paths": qwen_image_paths,
            "qwen_image_labels": qwen_image_labels,
            "evidence_frame_indices": [row["frame_index"] for row in evidence],
            "semantic_task": self.semantic_task,
        }
        try:
            _atomic_json(event_path, payload)
        except Exception:
            crop_path.unlink(missing_ok=True)
            panel_path.unlink(missing_ok=True)
            if superseded_path is not None and existing_path is not None:
                _replace_file_with_retry(superseded_path, existing_path)
            raise
        if superseded_path is not None:
            superseded_path.unlink(missing_ok=True)
            self.replaced_pending += 1
        else:
            self._pending_estimate += 1
        self._pending_path_by_track[track_id] = event_path
        self._last_frame_by_track[track_id] = frame_index
        return event_path

    def _refresh_processing_track_ids(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._processing_refresh_seconds < 0.25:
            return
        self._processing_track_ids = {
            track_id
            for path in self.processing_dir.glob("*.json")
            if (track_id := _event_track_id_from_path(path)) is not None
        }
        self._processing_refresh_seconds = now

    def _restore_pending_index(self) -> None:
        for path in sorted(self.pending_dir.glob("*.json")):
            event = _read_json_if_present(path)
            if not event:
                continue
            try:
                track_id = int(event["track_id"])
                frame_index = int(event["frame_index"])
            except (KeyError, TypeError, ValueError):
                continue
            self._pending_path_by_track[track_id] = path
            self._last_frame_by_track[track_id] = max(
                frame_index,
                self._last_frame_by_track.get(track_id, frame_index),
            )
            evidence = event.get("evidence")
            if isinstance(evidence, list):
                self._evidence_by_track[track_id] = [
                    dict(row) for row in evidence if isinstance(row, dict)
                ]

    def _updated_evidence(
        self,
        track_id: int,
        new_row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates = [*self._evidence_by_track.get(track_id, []), dict(new_row)]
        unique = {
            (int(row["frame_index"]), str(row["crop_path"])): row
            for row in candidates
            if Path(str(row.get("crop_path", ""))).is_file()
        }
        selected = _select_temporal_evidence(
            list(unique.values()),
            max_images=self.max_evidence_images,
        )
        self._evidence_by_track[track_id] = selected
        return selected


def _read_json_if_present(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _event_priority(reason: str, quality: float, confidence: float | None) -> float:
    reason_weight = {
        "new_track": 3.0,
        "unknown_track": 3.0,
        "low_confidence": 2.5,
        "periodic_refresh": 1.0,
    }.get(str(reason), 2.0)
    track_confidence = float(confidence if confidence is not None else 0.0)
    return round(reason_weight + quality + 0.25 * track_confidence, 6)


def _event_track_id_from_path(path: Path) -> int | None:
    match = re.search(r"_t(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def _crop_quality(
    frame: Any,
    crop: Any,
    track: TrackOutput,
    target_bbox: tuple[float, float, float, float],
) -> dict[str, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(sharpness_raw / 250.0, 1.0)
    brightness_raw = float(gray.mean())
    brightness = max(0.0, 1.0 - abs(brightness_raw - 127.5) / 127.5)
    target_width = max(float(target_bbox[2] - target_bbox[0]), 1.0)
    target_height = max(float(target_bbox[3] - target_bbox[1]), 1.0)
    size = min(min(target_width, target_height) / 96.0, 1.0)
    frame_height, frame_width = frame.shape[:2]
    box = track.bbox_xyxy
    clipped_width = max(min(box.x2, frame_width) - max(box.x1, 0.0), 0.0)
    clipped_height = max(min(box.y2, frame_height) - max(box.y1, 0.0), 0.0)
    original_area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1.0)
    visible_fraction = min(max(clipped_width * clipped_height / original_area, 0.0), 1.0)
    confidence = min(max(float(track.confidence or 0.0), 0.0), 1.0)
    score = (
        0.30 * sharpness
        + 0.20 * brightness
        + 0.20 * size
        + 0.20 * confidence
        + 0.10 * visible_fraction
    )
    return {
        "score": round(score, 6),
        "sharpness": round(sharpness, 6),
        "sharpness_raw": round(sharpness_raw, 3),
        "brightness": round(brightness, 6),
        "brightness_raw": round(brightness_raw, 3),
        "size": round(size, 6),
        "visible_fraction": round(visible_fraction, 6),
        "detector_confidence": round(confidence, 6),
    }


def _select_temporal_evidence(
    rows: list[dict[str, Any]],
    *,
    max_images: int,
) -> list[dict[str, Any]]:
    if max_images < 1:
        raise SemanticQueueError("max_images must be positive.")
    if len(rows) <= max_images:
        return sorted(rows, key=lambda row: int(row["frame_index"]))
    ordered = sorted(
        rows,
        key=lambda row: (-float(row.get("quality", 0.0)), int(row["frame_index"])),
    )
    selected = [ordered.pop(0)]
    frame_span = max(int(row["frame_index"]) for row in rows) - min(
        int(row["frame_index"]) for row in rows
    )
    while ordered and len(selected) < max_images:
        def selection_score(row: dict[str, Any]) -> tuple[float, float, int]:
            frame = int(row["frame_index"])
            temporal_distance = min(
                abs(frame - int(chosen["frame_index"])) for chosen in selected
            )
            diversity = temporal_distance / max(frame_span, 1)
            quality = float(row.get("quality", 0.0))
            return quality + 0.25 * diversity, quality, -frame

        choice = max(ordered, key=selection_score)
        selected.append(choice)
        ordered.remove(choice)
    return sorted(selected, key=lambda row: int(row["frame_index"]))


def _track_crop(
    frame: Any,
    track: TrackOutput,
    *,
    padding: float,
    output_size: int,
) -> tuple[Any, tuple[float, float, float, float]] | None:
    height, width = frame.shape[:2]
    box = track.bbox_xyxy
    box_width = max(box.x2 - box.x1, 1.0)
    box_height = max(box.y2 - box.y1, 1.0)
    x1 = max(int(box.x1 - box_width * padding), 0)
    y1 = max(int(box.y1 - box_height * padding), 0)
    x2 = min(int(box.x2 + box_width * padding + 0.5), width)
    y2 = min(int(box.y2 + box_height * padding + 0.5), height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    target_bbox = (
        max(float(box.x1) - x1, 0.0),
        max(float(box.y1) - y1, 0.0),
        min(float(box.x2) - x1, float(crop.shape[1])),
        min(float(box.y2) - y1, float(crop.shape[0])),
    )
    scale = min(output_size / crop.shape[1], output_size / crop.shape[0])
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        target_bbox = tuple(value * scale for value in target_bbox)
    return crop, target_bbox


def _build_evidence_panel(
    frame: Any,
    track: TrackOutput,
    evidence: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    context_fraction: float = 0.55,
) -> Any:
    """Compose scene context and temporal crops into one bounded Qwen image."""
    import numpy as np

    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    if not 0.0 < context_fraction < 1.0:
        raise SemanticQueueError("context_fraction must be between 0 and 1.")
    context_width = max(int(round(width * context_fraction)), 1)
    context = frame.copy()
    box = track.bbox_xyxy
    cv2.rectangle(
        context,
        (max(0, int(round(box.x1))), max(0, int(round(box.y1)))),
        (
            min(context.shape[1] - 1, int(round(box.x2))),
            min(context.shape[0] - 1, int(round(box.y2))),
        ),
        (0, 255, 255),
        4,
    )
    cv2.putText(
        context,
        f"TARGET ID {int(track.track_id)}",
        (max(4, int(round(box.x1))), max(22, int(round(box.y1)) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    panel[:, :context_width] = _letterbox_image(
        context,
        context_width,
        height,
        background=24,
    )

    crop_width = width - context_width
    slot_count = max(len(evidence), 1)
    slot_height = max(height // slot_count, 1)
    for index, row in enumerate(evidence):
        crop = cv2.imread(str(row.get("crop_path", "")))
        if crop is None or crop.size == 0:
            continue
        y1 = index * slot_height
        y2 = height if index == slot_count - 1 else min((index + 1) * slot_height, height)
        slot = _letterbox_image(crop, crop_width, y2 - y1, background=36)
        panel[y1:y2, context_width:] = slot
        cv2.putText(
            panel,
            f"crop f{int(row.get('frame_index', 0))}",
            (context_width + 5, min(y1 + 18, y2 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def _letterbox_image(
    image: Any,
    width: int,
    height: int,
    *,
    background: int,
) -> Any:
    import numpy as np

    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image is None or image.size == 0 or width < 1 or height < 1:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, min(width, int(round(image.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(image.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    x1 = (width - resized_width) // 2
    y1 = (height - resized_height) // 2
    canvas[y1 : y1 + resized_height, x1 : x1 + resized_width] = resized
    return canvas


class SemanticCacheView:
    """Reload accepted semantic labels only when an atomic cache changes."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._mtime_ns: int | None = None
        self.labels: dict[int, dict[str, Any]] = {}

    def refresh(self) -> bool:
        if self.path is None or not self.path.is_file():
            return False
        stat = self.path.stat()
        if stat.st_mtime_ns == self._mtime_ns:
            return False
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.labels = {
            int(row["track_id"]): dict(row) for row in payload.get("tracks", [])
        }
        self._mtime_ns = stat.st_mtime_ns
        return True

    def accepted(self, track_id: int) -> dict[str, Any] | None:
        row = self.labels.get(track_id)
        return row if row and bool(row.get("accepted")) else None

    def decorate(
        self,
        tracks: list[TrackOutput],
        *,
        pending_track_ids: set[int] | None = None,
        semantic_enabled: bool = False,
    ) -> list[TrackOutput]:
        pending = pending_track_ids or set()
        decorated: list[TrackOutput] = []
        for track in tracks:
            semantic = self.accepted(track.track_id)
            if semantic is None:
                row = self.labels.get(track.track_id)
                status = (
                    "unknown"
                    if row is not None
                    else "pending"
                    if track.track_id in pending
                    else "waiting"
                    if semantic_enabled
                    else "base"
                )
                label = (
                    f"{track.class_name} | unknown"
                    if semantic_enabled and status == "unknown"
                    else track.class_name
                )
                decorated.append(
                    replace(
                        track,
                        class_name=label,
                        metadata={
                            **track.metadata,
                            "detector_class_name": track.class_name,
                            "semantic_status": status,
                        },
                    )
                )
                continue
            base_label = str(
                semantic.get("detector_class_name", track.class_name)
            )
            deep_label = _semantic_detail_label(
                semantic,
                base_label=base_label,
                fast_label=str(track.metadata.get("fast_semantic_label", "")),
                fast_color=str(track.metadata.get("fast_visual_color", "")),
            )
            label = (
                base_label
                if deep_label.casefold() == base_label.casefold()
                else f"{base_label} | {deep_label}"
            )
            decorated.append(
                replace(
                    track,
                    class_name=label,
                    metadata={
                        **track.metadata,
                        "detector_class_name": track.class_name,
                        "semantic_label": label,
                        "semantic_deep_label": deep_label,
                        "semantic_status": "accepted",
                        "semantic_confidence": semantic.get("confidence"),
                        "semantic_base_class": semantic.get("class_label"),
                        "semantic_fine_label": semantic.get("fine_label", "unknown"),
                        "semantic_fine_confidence": semantic.get("fine_confidence", 0.0),
                    },
                )
            )
        return decorated


def _semantic_detail_label(
    semantic: dict[str, Any],
    *,
    base_label: str,
    fast_label: str = "",
    fast_color: str = "",
) -> str:
    base = str(base_label).strip() or "object"
    class_label = str(semantic.get("class_label", "unknown")).strip()
    fine_label = str(semantic.get("fine_label", "unknown")).strip()
    if bool(semantic.get("fine_accepted")) and fine_label.casefold() not in {
        "",
        "unknown",
        base.casefold(),
        class_label.casefold(),
    }:
        detail = fine_label
    elif class_label.casefold() not in {"", "unknown", base.casefold()}:
        detail = class_label
    elif fast_label.casefold() not in {"", "unknown", base.casefold()}:
        detail = fast_label
    else:
        detail = base
    attributes = semantic.get("attributes", {})
    semantic_color = (
        str(attributes.get("color", "")).strip().lower()
        if isinstance(attributes, dict)
        else ""
    )
    color = semantic_color or str(fast_color).strip().lower()
    allowed_colors = {
        "black",
        "white",
        "gray",
        "grey",
        "silver",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "brown",
        "beige",
        "maroon",
        "purple",
        "pink",
        "gold",
    }
    if color in allowed_colors and color not in detail.casefold().split():
        return f"{color} {detail}"
    return detail


def _event_prompt(event: dict[str, Any]) -> str:
    task = event.get("semantic_task") or {}
    label_mode = str(task.get("label_mode", "open")).strip().lower()
    enable_fine_labels = bool(
        task.get("enable_fine_labels", label_mode != "closed")
    )
    allowed_labels = [str(item) for item in task.get("allowed_labels", [])]
    fine_taxonomy = _fine_taxonomy_payload(task)
    allowed_note = (
        "Choose class_label only from this closed taxonomy: "
        + json.dumps(allowed_labels, ensure_ascii=False)
        + "."
        if label_mode == "closed" and allowed_labels
        else "Use an open hierarchical vocabulary, but prefer stable common names."
    )
    attributes = [str(item) for item in task.get("attributes", [])]
    attribute_note = (
        "Only return these attributes: " + json.dumps(attributes, ensure_ascii=False) + "."
        if attributes
        else "Return an empty attributes object unless an attribute is requested."
    )
    instruction = str(
        task.get(
            "instruction",
            "Assign the most specific visually supported stable label.",
        )
    ).strip()
    unknown_threshold = float(task.get("unknown_threshold", 0.70))
    evidence_frames = [
        int(value) for value in event.get("evidence_frame_indices", [])
    ]
    if not evidence_frames:
        evidence_frames = [int(event["frame_index"])]
    layout_note = (
        "The evidence panel contains scene context with the target track highlighted, "
        "plus temporally selected crops of that same track."
        if event.get("evidence_layout") == "panel"
        else "The supplied images are temporally selected crops of the same track."
    )
    fast_proposal = event.get("fast_semantic_proposal")
    proposal_note = (
        "A sparse open-vocabulary detector proposed "
        + json.dumps(fast_proposal, ensure_ascii=False, separators=(",", ":"))
        + "; verify it from the images instead of copying it blindly."
        if isinstance(fast_proposal, dict)
        else "No fast semantic proposal is available."
    )
    fine_note = (
        "Use class_label only for the parent class and choose fine_label from this "
        f"parent-to-subtype taxonomy: {json.dumps(fine_taxonomy, ensure_ascii=False)}. "
        "Put visible color only in attributes.color, never in class_label or fine_label."
        if enable_fine_labels and fine_taxonomy
        else ""
    )
    response_schema = (
        (
            f'{{"track_predictions":[{{"track_id":{int(event["track_id"])},'
            '"class_label":"...","fine_label":"...","fine_label_type":"subtype",'
            '"attributes":{},"confidence":0.0,"fine_confidence":0.0,'
            f'"evidence_frames":{json.dumps(evidence_frames)}}}]}}'
        )
        if enable_fine_labels
        else (
            f'{{"track_predictions":[{{"track_id":{int(event["track_id"])},'
            '"class_label":"...","attributes":{},"confidence":0.0,'
            f'"evidence_frames":{json.dumps(evidence_frames)}}}]}}'
        )
    )
    return f"""
Task: {task.get('task_name', task.get('task_id', 'tracked-object semantics'))}.
Instruction: {instruction}
Analyze all supplied images as temporal evidence for the same track_id. Do not count them as
different objects. {allowed_note} {attribute_note}
{fine_note}
Allowed evidence frame_index values: {json.dumps(evidence_frames)}. Return only visually usable
values in evidence_frames and never invent another frame_index.
Return class_label="unknown" when confidence is below {unknown_threshold:.2f}. Do not guess a
fine label, identity, role, species, make, or model from scene context alone. When fine_label is
enabled, use "unknown" unless the subtype is directly supported by the target crops.
{layout_note}
{proposal_note}
Keep the answer compact. Return JSON only using this schema:
{response_schema}
Detector hint (not ground truth): {event.get('detector_class_name', 'unknown')}.
""".strip()


def prepare_pending_events_with_locate(
    *,
    queue_dir: str | Path,
    max_events: int = 0,
    model_id: str = "nvidia/LocateAnything-3B",
    device: str = "cuda",
    quantization: str = "8bit",
    max_new_tokens: int = 256,
    image_max_pixels: int = 256 * 256,
    minimum_association_score: float = 0.10,
    service: Any | None = None,
) -> dict[str, Any]:
    """Run Locate before Qwen without holding both large models in VRAM."""

    if max_events < 0:
        raise SemanticQueueError("max_events must be non-negative.")
    if not 0.0 <= minimum_association_score <= 1.0:
        raise SemanticQueueError(
            "minimum_association_score must be between 0 and 1."
        )
    root = Path(queue_dir)
    # Match the exact claim order used by process_semantic_queue. Otherwise a
    # bounded Locate pass can prepare one track while Qwen claims another.
    pending_paths = sorted(
        (root / "pending").glob("*.json"),
        key=_pending_claim_order,
    )
    candidates = []
    for path in pending_paths:
        event = json.loads(path.read_text(encoding="utf-8"))
        if (event.get("locateanything") or {}).get("status") == "completed":
            continue
        candidates.append((path, event))
        if max_events > 0 and len(candidates) >= max_events:
            break
    if not candidates:
        return {
            "status": "idle",
            "prepared_event_count": 0,
            "accepted_event_count": 0,
        }

    owned_backend: Any | None = None
    if service is None:
        from football_tracking.locate_tracking.grounding.cache import (
            GroundingCache,
        )
        from football_tracking.locate_tracking.grounding.locate_anything_backend import (
            LocateAnythingBackend,
        )
        from football_tracking.locate_tracking.grounding.service import (
            GroundingService,
        )

        owned_backend = LocateAnythingBackend(
            model_id=model_id,
            device=device,
            torch_dtype="auto",
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            image_max_pixels=image_max_pixels,
        )
        service = GroundingService(
            backend=owned_backend,
            cache=GroundingCache(root / "locate_cache"),
            overwrite=True,
        )

    started = time.perf_counter()
    accepted_count = 0
    try:
        for path, event in candidates:
            crop_path = Path(str(event.get("crop_path", "")))
            image = cv2.imread(str(crop_path))
            if image is None:
                raise SemanticQueueError(
                    f"Semantic crop does not exist or is unreadable: {crop_path}"
                )
            height, width = image.shape[:2]
            target_bbox = _validated_target_bbox(
                event.get("target_bbox_in_crop_xyxy"),
                width=width,
                height=height,
            )
            detector_class = str(
                event.get("detector_class_name") or "object"
            ).strip()
            query = f"the {detector_class}"
            result = service.ground_image(
                image_path=crop_path,
                query=query,
            )
            selected = _best_grounded_box(
                getattr(result, "boxes", ()),
                target_bbox=target_bbox,
            )
            accepted = bool(
                selected
                and selected["association_score"] >= minimum_association_score
            )
            existing_paths = [
                str(Path(str(value)).resolve())
                for value in (
                    event.get("qwen_image_paths")
                    or [event.get("evidence_panel_path") or crop_path]
                )
                if value and Path(str(value)).is_file()
            ]
            # Keep the temporal evidence panel as Qwen's primary image. Locate
            # contributes a second, tighter view instead of replacing history.
            qwen_paths = existing_paths[:1] or [str(crop_path.resolve())]
            qwen_labels = [
                str(value) for value in event.get("qwen_image_labels", [])[:1]
            ]
            if not qwen_labels:
                qwen_labels = [
                    f"Track {int(event['track_id'])} temporal evidence panel."
                ]
            grounded_crop_path: str | None = None
            if accepted and selected is not None:
                grounded_crop = _crop_grounded_region(
                    image,
                    selected["bbox_xyxy"],
                    padding=0.10,
                )
                if grounded_crop is not None:
                    locate_dir = root / "locate_crops"
                    locate_dir.mkdir(parents=True, exist_ok=True)
                    output_path = locate_dir / f"{event['event_id']}.jpg"
                    if not cv2.imwrite(str(output_path), grounded_crop):
                        raise SemanticQueueError(
                            f"Could not write Locate crop: {output_path}"
                        )
                    grounded_crop_path = str(output_path.resolve())
                    qwen_paths.append(grounded_crop_path)
                    qwen_labels.append(
                        f"LocateAnything-refined crop for track {int(event['track_id'])}."
                    )
                accepted_count += 1
            runtime = getattr(result, "runtime_info", None)
            locate_payload = {
                "status": "completed",
                "accepted": accepted,
                "query": query,
                "association_score": (
                    round(float(selected["association_score"]), 6)
                    if selected is not None
                    else 0.0
                ),
                "minimum_association_score": minimum_association_score,
                "bbox_xyxy": (
                    list(selected["bbox_xyxy"])
                    if selected is not None
                    else None
                ),
                "confidence": (
                    selected["confidence"] if selected is not None else None
                ),
                "grounded_crop_path": grounded_crop_path,
                "runtime": (
                    runtime.to_dict()
                    if runtime is not None and hasattr(runtime, "to_dict")
                    else None
                ),
            }
            _atomic_json(
                path,
                {
                    **event,
                    "locateanything": locate_payload,
                    "qwen_image_paths": qwen_paths,
                    "qwen_image_labels": qwen_labels,
                },
            )
    finally:
        if owned_backend is not None:
            owned_backend.close()
    return {
        "status": "ok",
        "prepared_event_count": len(candidates),
        "accepted_event_count": accepted_count,
        "prepared_event_ids": [str(event["event_id"]) for _path, event in candidates],
        "prepared_track_ids": [int(event["track_id"]) for _path, event in candidates],
        "elapsed_seconds": time.perf_counter() - started,
        "model_id": model_id,
        "quantization": quantization,
    }


def _validated_target_bbox(
    value: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        values = ()
    if (
        len(values) != 4
        or not all(math.isfinite(item) for item in values)
        or values[2] <= values[0]
        or values[3] <= values[1]
        or values[2] <= 0.0
        or values[3] <= 0.0
        or values[0] >= float(width)
        or values[1] >= float(height)
    ):
        return (0.0, 0.0, float(width), float(height))
    clipped = (
        min(max(values[0], 0.0), float(width - 1)),
        min(max(values[1], 0.0), float(height - 1)),
        min(max(values[2], 1.0), float(width)),
        min(max(values[3], 1.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return (0.0, 0.0, float(width), float(height))
    return clipped


def _best_grounded_box(
    boxes: Any,
    *,
    target_bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for box in boxes:
        try:
            bbox = tuple(float(value) for value in box.bbox_xyxy)
        except (AttributeError, TypeError, ValueError):
            continue
        score = _association_score(bbox, target_bbox)
        candidates.append(
            {
                "bbox_xyxy": bbox,
                "association_score": score,
                "confidence": getattr(box, "confidence", None),
            }
        )
    return (
        max(
            candidates,
            key=lambda row: (
                float(row["association_score"]),
                float(
                    row["confidence"]
                    if row["confidence"] is not None
                    else 1.0
                ),
            ),
        )
        if candidates
        else None
    )


def _association_score(
    grounded: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    x1 = max(grounded[0], target[0])
    y1 = max(grounded[1], target[1])
    x2 = min(grounded[2], target[2])
    y2 = min(grounded[3], target[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    grounded_area = max(grounded[2] - grounded[0], 0.0) * max(
        grounded[3] - grounded[1], 0.0
    )
    target_area = max(target[2] - target[0], 0.0) * max(
        target[3] - target[1], 0.0
    )
    union = grounded_area + target_area - intersection
    iou = intersection / union if union > 0 else 0.0
    target_coverage = intersection / target_area if target_area > 0 else 0.0
    grounding_coverage = (
        intersection / grounded_area if grounded_area > 0 else 0.0
    )
    return max(iou, math.sqrt(target_coverage * grounding_coverage))


def _crop_grounded_region(
    image: Any,
    bbox: tuple[float, float, float, float],
    *,
    padding: float,
) -> Any | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    crop_x1 = max(int(math.floor(x1 - box_width * padding)), 0)
    crop_y1 = max(int(math.floor(y1 - box_height * padding)), 0)
    crop_x2 = min(int(math.ceil(x2 + box_width * padding)), width)
    crop_y2 = min(int(math.ceil(y2 + box_height * padding)), height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop if crop.size else None


def _qwen_job(event: dict[str, Any]) -> dict[str, Any]:
    raw_paths = event.get("qwen_image_paths") or [event["crop_path"]]
    image_paths = [Path(str(path)) for path in raw_paths]
    image_count = len(image_paths)
    frame_indices = [
        int(value) for value in event.get("evidence_frame_indices", [])
    ]
    if len(frame_indices) != image_count:
        frame_indices = [int(event["frame_index"])] * image_count
    labels = [str(value) for value in event.get("qwen_image_labels", [])]
    if len(labels) != image_count:
        labels = [
            (
                f"Track {event['track_id']} at frame {event['frame_index']} "
                f"evidence frame {frame_indices[index]} ({index + 1}/{image_count})."
            )
            for index in range(image_count)
        ]
    return {
        "batch_id": event["event_id"],
        "prompt": _event_prompt(event),
        "image_paths": image_paths,
        "image_labels": labels,
    }


def _qwen_group_job(events: list[dict[str, Any]]) -> dict[str, Any]:
    if len(events) < 2:
        raise SemanticQueueError("A grouped Qwen job requires at least two events.")
    image_paths: list[Path] = []
    image_labels: list[str] = []
    for event in events:
        event_job = _qwen_job(event)
        image_paths.extend(event_job["image_paths"])
        image_labels.extend(
            f"Expected track_id {int(event['track_id'])}. {label}"
            for label in event_job["image_labels"]
        )
    event_ids = "_".join(str(int(event["track_id"])) for event in events)
    return {
        "batch_id": f"group_tracks_{event_ids}",
        "prompt": _group_event_prompt(events),
        "image_paths": image_paths,
        "image_labels": image_labels,
    }


def _group_event_prompt(events: list[dict[str, Any]]) -> str:
    task = events[0].get("semantic_task") or {}
    if any((event.get("semantic_task") or {}) != task for event in events[1:]):
        raise SemanticQueueError("Grouped events must use the same semantic task.")
    label_mode = str(task.get("label_mode", "open")).strip().lower()
    enable_fine_labels = bool(
        task.get("enable_fine_labels", label_mode != "closed")
    )
    requested_attributes = {
        str(item).strip().casefold() for item in task.get("attributes", [])
    }
    request_color = "color" in requested_attributes
    allowed_labels = [str(item) for item in task.get("allowed_labels", [])]
    fine_taxonomy = _fine_taxonomy_payload(task)
    allowed_note = (
        "Choose class_label only from this closed taxonomy: "
        + json.dumps(allowed_labels, ensure_ascii=False)
        + "."
        if label_mode == "closed" and allowed_labels
        else "Use an open hierarchical vocabulary, but prefer stable common names."
    )
    expected: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    for event in events:
        evidence_frames = [
            int(value) for value in event.get("evidence_frame_indices", [])
        ] or [int(event["frame_index"])]
        expected.append(
            {
                "track_id": int(event["track_id"]),
                "detector_hint": str(
                    event.get("detector_class_name", "unknown")
                ),
                "fast_semantic_hint": _compact_fast_semantic_hint(event),
                "allowed_evidence_frames": evidence_frames,
            }
        )
        row: dict[str, Any] = {
            "id": int(event["track_id"]),
            "class": "...",
        }
        if enable_fine_labels:
            row["subtype"] = "..."
        if request_color:
            row["color"] = "..."
        row["q"] = 0.0
        if enable_fine_labels:
            row["sq"] = 0.0
        schema_rows.append(row)
    instruction = str(
        task.get(
            "instruction",
            "Assign the most specific visually supported stable label.",
        )
    ).strip()
    unknown_threshold = float(task.get("unknown_threshold", 0.70))
    color_instruction = (
        "For color, return one of black, white, gray, silver, red, blue, green, "
        "yellow, orange, brown, beige, maroon, purple, pink, gold, or unknown "
        "using only visible target pixels."
        if request_color
        else ""
    )
    fine_instruction = (
        "Choose class only from the base taxonomy above. Choose subtype only from "
        f"the matching parent entry here: "
        f"{json.dumps(fine_taxonomy, ensure_ascii=False, separators=(',', ':'))}. "
        "Never place a subtype in class and never place a color in subtype."
        if enable_fine_labels and fine_taxonomy
        else (
            "A subtype must be more specific than class; never repeat class in subtype."
            if enable_fine_labels
            else ""
        )
    )
    return f"""
Task: {task.get('task_name', task.get('task_id', 'tracked-object semantics'))}.
Instruction: {instruction}
Each supplied image label identifies exactly one target track. Analyze each target separately,
do not transfer appearance or labels between track IDs, and return exactly one prediction for
every expected track below.
Expected tracks: {json.dumps(expected, ensure_ascii=False, separators=(',', ':'))}
{allowed_note}
Use only the allowed evidence frames listed for that track. Return class_label="unknown" below
confidence {unknown_threshold:.2f}. Fine labels must be directly visible in the target crops;
otherwise return subtype="unknown". {fine_instruction}
{color_instruction}
Use q and sq for your own confidence rounded to one decimal place. Do not copy confidence
from a detector hint. The system already records evidence frames, so do not return them.
Do not return descriptions, unrequested attributes, or extra keys.
Keep the answer compact and return JSON only:
{json.dumps({"p": schema_rows}, ensure_ascii=False, separators=(',', ':'))}
""".strip()


def _compact_fast_semantic_hint(event: dict[str, Any]) -> dict[str, Any] | None:
    proposal = event.get("fast_semantic_proposal")
    if not isinstance(proposal, dict):
        return None
    task = event.get("semantic_task")
    task_payload = task if isinstance(task, dict) else {}
    label_threshold = float(task_payload.get("fast_label_hint_threshold", 0.65))
    color_threshold = float(task_payload.get("fast_color_hint_threshold", 0.55))
    output: dict[str, Any] = {}
    label = proposal.get("class_label")
    if (
        label not in {None, "", "unknown"}
        and float(proposal.get("confidence", 0.0)) >= label_threshold
    ):
        output["label"] = label
    color = proposal.get("visual_color")
    if (
        color not in {None, "", "unknown"}
        and float(proposal.get("visual_color_confidence", 0.0))
        >= color_threshold
    ):
        output["color"] = color
    return output or None


def _requires_focused_qwen_inference(event: dict[str, Any]) -> bool:
    proposal = event.get("fast_semantic_proposal")
    if not isinstance(proposal, dict):
        return False
    label = proposal.get("class_label")
    if label in {None, "", "unknown"}:
        return False
    task = event.get("semantic_task")
    task_payload = task if isinstance(task, dict) else {}
    threshold = float(task_payload.get("fast_label_hint_threshold", 0.65))
    confidence = float(proposal.get("confidence", 0.0))
    return 0.0 < confidence < threshold


def _fine_taxonomy_payload(task: dict[str, Any]) -> dict[str, list[str]]:
    raw = task.get("fine_label_taxonomy")
    if not isinstance(raw, dict):
        return {}
    output: dict[str, list[str]] = {}
    for parent, values in raw.items():
        if not isinstance(values, list | tuple):
            continue
        parent_label = str(parent).strip()
        fine_labels = [str(value).strip() for value in values if str(value).strip()]
        if parent_label and fine_labels:
            output[parent_label] = fine_labels
    return output


def _fine_alias_payload(task: dict[str, Any]) -> dict[str, str]:
    raw = task.get("fine_label_aliases")
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias).strip().casefold().replace("_", " "): str(target)
        .strip()
        .casefold()
        .replace("_", " ")
        for alias, target in raw.items()
        if str(alias).strip() and str(target).strip()
    }


def _label_alias_payload(task: dict[str, Any]) -> dict[str, str]:
    raw = task.get("label_aliases")
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias).strip().casefold().replace("_", " "): str(target)
        .strip()
        .casefold()
        .replace("_", " ")
        for alias, target in raw.items()
        if str(alias).strip() and str(target).strip()
    }


def _pending_claim_order(path: Path) -> tuple[float, int, str]:
    event = _read_json_if_present(path) or {}
    priority = float(event.get("priority", 0.0))
    frame_index = int(event.get("frame_index", 0))
    return -priority, frame_index, path.name


def _validated_evidence_frames(
    row: TrackSemanticEvidence,
    event: dict[str, Any],
) -> tuple[int, ...]:
    allowed = {
        int(value) for value in event.get("evidence_frame_indices", [])
    }
    allowed.add(int(event["frame_index"]))
    validated = tuple(frame for frame in row.evidence_frames if frame in allowed)
    return validated or tuple(sorted(allowed))


def _enforce_event_taxonomy(
    rows: list[TrackSemanticEvidence],
    event: dict[str, Any],
) -> list[TrackSemanticEvidence]:
    task = event.get("semantic_task") or {}
    allowed_attributes = {
        str(item).strip().casefold() for item in task.get("attributes", [])
    }
    label_mode = str(task.get("label_mode", "open")).strip().casefold()
    allowed_labels = {
        str(item).strip().casefold().replace("_", " "): str(item)
        .strip()
        .lower()
        .replace("_", " ")
        for item in task.get("allowed_labels", [])
    }
    fine_taxonomy = _fine_taxonomy_payload(task)
    label_aliases = _label_alias_payload(task)
    fine_aliases = _fine_alias_payload(task)
    fine_to_parent: dict[str, tuple[str, str]] = {}
    for parent, fine_labels in fine_taxonomy.items():
        parent_key = parent.casefold().replace("_", " ")
        canonical_parent = allowed_labels.get(parent_key, parent_key)
        for fine_label in fine_labels:
            fine_key = fine_label.casefold().replace("_", " ")
            fine_to_parent[fine_key] = (
                canonical_parent,
                fine_key,
            )
    normalized: list[TrackSemanticEvidence] = []
    for row in rows:
        attributes = {
            key: value
            for key, value in row.attributes.items()
            if not allowed_attributes or str(key).strip().casefold() in allowed_attributes
        }
        class_label = row.class_label
        confidence = row.confidence
        fine_label = row.fine_label
        fine_confidence = row.fine_confidence
        fine_label_type = row.fine_label_type
        class_key = class_label.casefold().replace("_", " ")
        fine_key = fine_label.casefold().replace("_", " ")
        class_key = label_aliases.get(class_key, class_key)
        class_key = fine_aliases.get(class_key, class_key)
        fine_key = fine_aliases.get(fine_key, fine_key)
        if fine_key in _VISIBLE_COLOR_LABELS:
            if "color" in allowed_attributes:
                attributes.setdefault("color", fine_key)
            fine_label = "unknown"
            fine_key = "unknown"
            fine_confidence = 0.0
            fine_label_type = "unknown"
        if class_key in _VISIBLE_COLOR_LABELS:
            if "color" in allowed_attributes:
                attributes.setdefault("color", class_key)
            class_label = "unknown"
            class_key = "unknown"
        if label_mode == "closed" and allowed_labels:
            if class_key in allowed_labels:
                class_label = allowed_labels[class_key]
            elif class_key in fine_to_parent:
                class_label, promoted_fine = fine_to_parent[class_key]
                fine_label = promoted_fine
                fine_confidence = max(confidence, fine_confidence)
                fine_label_type = "subtype"
            elif fine_key in allowed_labels:
                class_label = allowed_labels[fine_key]
                confidence = max(confidence, row.fine_confidence)
                fine_label = "unknown"
                fine_confidence = 0.0
                fine_label_type = "unknown"
            else:
                class_label = "unknown"
                confidence = max(1.0 - confidence, 0.0)
        if fine_to_parent:
            fine_key = fine_label.casefold().replace("_", " ")
            expected = fine_to_parent.get(fine_key)
            if expected is not None and expected[0].casefold() == class_label.casefold():
                fine_label = expected[1]
                fine_label_type = "subtype"
            else:
                fine_label = "unknown"
                fine_confidence = 0.0
                fine_label_type = "unknown"
        if class_label == "unknown":
            fine_label = "unknown"
            fine_confidence = 0.0
            fine_label_type = "unknown"
        color = str(attributes.get("color", "")).strip().casefold()
        if color and color not in _VISIBLE_COLOR_LABELS:
            attributes.pop("color", None)
        normalized.append(
            replace(
                row,
                class_label=class_label,
                confidence=confidence,
                attributes=attributes,
                fine_label=fine_label,
                fine_confidence=fine_confidence,
                fine_label_type=fine_label_type,
            )
        )
    return normalized


def process_semantic_queue(
    *,
    queue_dir: str | Path,
    vlm_config_path: str | Path,
    semantic_output: str | Path,
    memory_path: str | Path,
    registry_path: str | Path = "configs/ontology/vocabulary_registry.yaml",
    max_events: int = 8,
    group_events: bool = False,
    max_group_images: int = 2,
    max_memory_observations_per_track: int = 32,
    runner: Callable[[Any, list[dict[str, Any]]], dict[str, Any]] = run_qwen_vlm_batches,
) -> dict[str, Any]:
    if max_events < 1:
        raise SemanticQueueError("max_events must be positive.")
    if max_group_images < 1:
        raise SemanticQueueError("max_group_images must be positive.")
    root = Path(queue_dir)
    pending_dir = root / "pending"
    processing_dir = root / "processing"
    processing_dir.mkdir(parents=True, exist_ok=True)
    claimed: list[Path] = []
    pending_paths = sorted(
        pending_dir.glob("*.json"),
        key=_pending_claim_order,
    )
    now = time.time()
    deferred_until: list[float] = []
    for pending_path in pending_paths:
        pending_event = _read_json_if_present(pending_path)
        ready_after = float(
            (pending_event or {}).get("ready_after_unix_seconds", 0.0)
        )
        if ready_after > now:
            deferred_until.append(ready_after)
            continue
        claimed_path = processing_dir / pending_path.name
        try:
            _replace_file_with_retry(pending_path, claimed_path)
        except FileNotFoundError:
            continue
        claimed.append(claimed_path)
        if len(claimed) >= max_events:
            break
    if not claimed:
        remaining = len(list(pending_dir.glob("*.json")))
        return {
            "status": (
                "waiting_for_evidence"
                if remaining and deferred_until
                else "idle"
            ),
            "processed_event_count": 0,
            "remaining_event_count": remaining,
            "next_ready_seconds": (
                max(min(deferred_until) - time.time(), 0.0)
                if deferred_until
                else 0.0
            ),
        }
    try:
        events = [json.loads(path.read_text(encoding="utf-8")) for path in claimed]
        context_ids = {str(event.get("context_id", "")) for event in events}
        if len(context_ids) != 1 or not next(iter(context_ids)):
            raise SemanticQueueError(
                "A worker batch must contain one non-empty context_id."
            )
        for event in events:
            image_paths = event.get("qwen_image_paths") or [event["crop_path"]]
            for image_path in image_paths:
                resolved = Path(str(image_path))
                if not resolved.is_file():
                    raise SemanticQueueError(
                        f"Semantic evidence image does not exist: {resolved}"
                    )
        config = load_vlm_tracking_config(
            vlm_config_path,
            overrides={"run_model": True},
        )
        total_image_count = sum(
            len(event.get("qwen_image_paths") or [event["crop_path"]])
            for event in events
        )
        grouped = (
            group_events
            and len(events) > 1
            and total_image_count <= max_group_images
            and not any(
                _requires_focused_qwen_inference(event) for event in events
            )
        )
        jobs = (
            [_qwen_group_job(events)]
            if grouped
            else [_qwen_job(event) for event in events]
        )
        inference = runner(config, jobs)
    except Exception:
        _requeue_claims(claimed, pending_dir)
        raise
    batches = inference.get("batches", [])
    expected_batch_count = 1 if grouped else len(events)
    if len(batches) != expected_batch_count:
        _requeue_claims(claimed, pending_dir)
        raise SemanticQueueError("Qwen worker returned an unexpected batch count.")
    event_batches = [batches[0]] * len(events) if grouped else batches
    evidence = []
    processed: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    failed: list[tuple[Path, dict[str, Any], dict[str, Any], str]] = []
    for path, event, batch in zip(claimed, events, event_batches, strict=True):
        try:
            parsed = [
                row
                for row in parse_qwen_answer({"answer": batch.get("answer", "")})
                if row.track_id == int(event["track_id"])
            ]
        except SemanticFusionError as exc:
            failed.append((path, event, batch, f"invalid_model_output: {exc}"))
            continue
        if not parsed:
            failed.append(
                (
                    path,
                    event,
                    batch,
                    "no_valid_evidence_for_expected_track_id",
                )
            )
            continue
        parsed = _enforce_event_taxonomy(parsed, event)
        parsed = [
            replace(row, evidence_frames=_validated_evidence_frames(row, event))
            for row in parsed
        ]
        evidence.extend(parsed)
        # LocateAnything verifies and tightens the target region; it does not
        # independently classify that region. Voting the detector hint as a
        # Locate semantic label suppresses valid Qwen refinements such as
        # detector "car" -> visually supported "van".
        processed.append((path, event, batch))

    failed_dir = root / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    for path, event, batch, reason in failed:
        _atomic_json(
            failed_dir / path.name,
            {
                **event,
                "failure_reason": reason,
                "model_result": batch,
            },
        )
        path.unlink()
    if not processed:
        return {
            "status": "no_valid_evidence",
            "processed_event_count": 0,
            "failed_event_count": len(failed),
            "remaining_event_count": len(list(pending_dir.glob("*.json"))),
        }

    try:
        project_root = get_project_root()
        registry_candidate = Path(registry_path)
        resolved_registry = (
            registry_candidate.resolve()
            if registry_candidate.is_absolute()
            else (project_root / registry_candidate).resolve()
        )
        task = events[0].get("semantic_task") or {}
        if str(task.get("label_mode", "open")).strip().casefold() != "closed":
            evidence = normalize_semantic_evidence(
                evidence,
                VocabularyRegistry.load(resolved_registry),
            )
        context_id = next(iter(context_ids))
        memory = TemporalSemanticMemory.load(memory_path, context_id=context_id)
        memory.merge(
            evidence,
            max_observations_per_track=max_memory_observations_per_track,
        )
        memory.save(memory_path)
        fused = fuse_track_semantics(
            list(memory.observations),
            unknown_threshold=float(task.get("unknown_threshold", 0.70)),
            fine_unknown_threshold=float(
                task.get("fine_unknown_threshold", 0.85)
            ),
            fine_minimum_margin=float(task.get("fine_minimum_margin", 0.15)),
            fine_minimum_temporal_stability=float(
                task.get("fine_minimum_temporal_stability", 0.67)
            ),
        )
        detector_labels = {
            int(event["track_id"]): str(event.get("detector_class_name", "object"))
            for event in events
        }
        for row in fused.get("tracks", []):
            track_id = int(row["track_id"])
            if track_id in detector_labels:
                row["detector_class_name"] = detector_labels[track_id]
            row["task_id"] = task.get("task_id")
        fused["runtime"] = {
            "mode": "realtime_semantic_worker",
            "context_id": context_id,
            "processed_event_count": len(processed),
            "qwen_call_count": len(batches),
            "grouped_events": grouped,
            "model": inference.get("model_id"),
            "quantization": inference.get("quantization"),
            "timing": inference.get("timing"),
            "cuda_memory": inference.get("cuda_memory"),
        }
        _atomic_json(Path(semantic_output), fused)
    except Exception:
        _requeue_claims([path for path, _event, _batch in processed], pending_dir)
        raise
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for path, event, batch in processed:
        completed = {
            **{
                key: value
                for key, value in event.items()
                if key not in {"failure_reason", "model_result"}
            },
            "model_result": batch,
        }
        completed_path = processed_dir / path.name
        _atomic_json(completed_path, completed)
        path.unlink()
    return {
        "status": "ok",
        "processed_event_count": len(processed),
        "failed_event_count": len(failed),
        "remaining_event_count": len(list(pending_dir.glob("*.json"))),
        "qwen_call_count": len(batches),
        "grouped_events": grouped,
        "semantic_output": str(Path(semantic_output).resolve()),
        "memory_path": str(Path(memory_path).resolve()),
        "fusion_summary": fused["summary"],
    }


def rebuild_semantic_cache_from_processed(
    *,
    processed_dir: str | Path,
    semantic_output: str | Path,
    registry_path: str | Path = "configs/ontology/vocabulary_registry.yaml",
) -> dict[str, Any]:
    """Rebuild fusion from saved Qwen outputs without another model call."""

    started = time.perf_counter()
    paths = sorted(Path(processed_dir).glob("*.json"))
    if not paths:
        raise SemanticQueueError(
            f"No processed semantic events found: {Path(processed_dir)}"
        )
    events = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    context_ids = {str(event.get("context_id", "")) for event in events}
    if len(context_ids) != 1 or not next(iter(context_ids)):
        raise SemanticQueueError(
            "Processed semantic events must share one non-empty context_id."
        )
    tasks = [event.get("semantic_task") or {} for event in events]
    if any(task != tasks[0] for task in tasks[1:]):
        raise SemanticQueueError("Processed semantic events use different tasks.")
    task = tasks[0]
    evidence: list[TrackSemanticEvidence] = []
    detector_labels: dict[int, str] = {}
    invalid_events: list[str] = []
    for event in events:
        batch = event.get("model_result") or {}
        try:
            parsed = [
                row
                for row in parse_qwen_answer({"answer": batch.get("answer", "")})
                if row.track_id == int(event["track_id"])
            ]
        except SemanticFusionError:
            parsed = []
        if not parsed:
            invalid_events.append(str(event.get("event_id", "unknown")))
            continue
        parsed = _enforce_event_taxonomy(parsed, event)
        evidence.extend(
            replace(row, evidence_frames=_validated_evidence_frames(row, event))
            for row in parsed
        )
        detector_labels[int(event["track_id"])] = str(
            event.get("detector_class_name", "object")
        )
    if not evidence:
        raise SemanticQueueError("Saved Qwen outputs contain no valid track evidence.")

    if str(task.get("label_mode", "open")).strip().casefold() != "closed":
        project_root = get_project_root()
        registry_candidate = Path(registry_path)
        resolved_registry = (
            registry_candidate.resolve()
            if registry_candidate.is_absolute()
            else (project_root / registry_candidate).resolve()
        )
        evidence = normalize_semantic_evidence(
            evidence,
            VocabularyRegistry.load(resolved_registry),
        )
    fused = fuse_track_semantics(
        evidence,
        unknown_threshold=float(task.get("unknown_threshold", 0.70)),
        fine_unknown_threshold=float(task.get("fine_unknown_threshold", 0.85)),
        fine_minimum_margin=float(task.get("fine_minimum_margin", 0.15)),
        fine_minimum_temporal_stability=float(
            task.get("fine_minimum_temporal_stability", 0.67)
        ),
    )
    for row in fused.get("tracks", []):
        track_id = int(row["track_id"])
        row["detector_class_name"] = detector_labels.get(track_id, "object")
        row["task_id"] = task.get("task_id")
    fused["runtime"] = {
        "mode": "rebuild_from_saved_qwen_outputs",
        "context_id": next(iter(context_ids)),
        "processed_event_count": len(events),
        "valid_event_count": len(events) - len(invalid_events),
        "invalid_event_ids": invalid_events,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(semantic_output)
    _atomic_json(output, fused)
    return {
        "status": "ok",
        "processed_event_count": len(events),
        "valid_event_count": len(events) - len(invalid_events),
        "invalid_event_ids": invalid_events,
        "semantic_output": str(output.resolve()),
        "fusion_summary": fused["summary"],
    }


def _requeue_claims(claimed: list[Path], pending_dir: Path) -> None:
    pending_dir.mkdir(parents=True, exist_ok=True)
    for claimed_path in claimed:
        if not claimed_path.is_file():
            continue
        target = pending_dir / claimed_path.name
        if target.exists():
            raise SemanticQueueError(f"Could not requeue duplicate event: {target}")
        _replace_file_with_retry(claimed_path, target)
