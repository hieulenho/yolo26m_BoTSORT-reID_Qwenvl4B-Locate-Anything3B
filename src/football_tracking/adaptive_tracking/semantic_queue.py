"""Non-blocking semantic event queue and bounded Qwen worker."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from football_tracking.adaptive_tracking.ontology import VocabularyRegistry
from football_tracking.adaptive_tracking.semantic_fusion import (
    SemanticFusionError,
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class SemanticEventQueue:
    """Persist track crops without blocking the detector/tracker loop."""

    def __init__(
        self,
        root: str | Path,
        *,
        context_id: str,
        max_pending_events: int = 256,
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
        self._last_frame_by_track: dict[int, int] = {}
        self.max_pending_events = int(max_pending_events)
        self.dropped_full = 0
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.processed_dir,
            self.failed_dir,
            self.crops_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._pending_estimate = len(list(self.pending_dir.glob("*.json")))

    @property
    def pending_count(self) -> int:
        self._pending_estimate = len(list(self.pending_dir.glob("*.json")))
        return self._pending_estimate

    def enqueue(
        self,
        *,
        frame: Any,
        frame_index: int,
        track: TrackOutput,
        reason: str,
        minimum_frame_gap: int = 90,
        crop_padding: float = 0.15,
        crop_size: int = 256,
    ) -> Path | None:
        if minimum_frame_gap < 1:
            raise SemanticQueueError("minimum_frame_gap must be positive.")
        if not 0.0 <= crop_padding <= 1.0:
            raise SemanticQueueError("crop_padding must be in [0, 1].")
        if crop_size < 64:
            raise SemanticQueueError("crop_size must be at least 64.")
        last_frame = self._last_frame_by_track.get(track.track_id)
        if last_frame is not None and frame_index - last_frame < minimum_frame_gap:
            return None
        if self._pending_estimate >= self.max_pending_events:
            self._pending_estimate = len(list(self.pending_dir.glob("*.json")))
        if self._pending_estimate >= self.max_pending_events:
            self.dropped_full += 1
            self._last_frame_by_track[track.track_id] = frame_index
            return None
        crop = _track_crop(frame, track, padding=crop_padding, output_size=crop_size)
        if crop is None:
            return None
        event_id = f"f{frame_index:09d}_t{track.track_id:07d}"
        crop_path = self.crops_dir / f"{event_id}.jpg"
        event_path = self.pending_dir / f"{event_id}.json"
        if not cv2.imwrite(str(crop_path), crop):
            raise SemanticQueueError(f"Could not write semantic crop: {crop_path}")
        _atomic_json(
            event_path,
            {
                "schema_version": "1.0",
                "event_id": event_id,
                "context_id": self.context_id,
                "frame_index": frame_index,
                "track_id": track.track_id,
                "detector_class_id": track.class_id,
                "detector_class_name": track.class_name,
                "track_confidence": track.confidence,
                "reason": reason,
                "crop_path": str(crop_path.resolve()),
            },
        )
        self._last_frame_by_track[track.track_id] = frame_index
        self._pending_estimate += 1
        return event_path


def _track_crop(
    frame: Any,
    track: TrackOutput,
    *,
    padding: float,
    output_size: int,
) -> Any | None:
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
    scale = min(output_size / crop.shape[1], output_size / crop.shape[0])
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


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

    def decorate(self, tracks: list[TrackOutput]) -> list[TrackOutput]:
        decorated: list[TrackOutput] = []
        for track in tracks:
            semantic = self.accepted(track.track_id)
            if semantic is None:
                decorated.append(track)
                continue
            label = str(semantic.get("display_label", semantic.get("class_label", "unknown")))
            decorated.append(
                replace(
                    track,
                    class_name=label,
                    metadata={
                        **track.metadata,
                        "detector_class_name": track.class_name,
                        "semantic_label": label,
                        "semantic_confidence": semantic.get("confidence"),
                        "semantic_base_class": semantic.get("class_label"),
                        "semantic_fine_label": semantic.get("fine_label", "unknown"),
                        "semantic_fine_confidence": semantic.get("fine_confidence", 0.0),
                    },
                )
            )
        return decorated


def _event_prompt(event: dict[str, Any]) -> str:
    return f"""
Analyze this crop from a tracked object using an open, hierarchical vocabulary. Infer a stable
base class and, separately, the most specific visually supported subtype/species/make/model.
Do not guess a fine label from context. A clear base class may have fine_label "unknown".
Return JSON only using this schema:
{{"track_predictions":[{{"track_id":{int(event['track_id'])},
"class_label":"...","fine_label":"...","taxonomy_path":[],
"attributes":{{}},"confidence":0.0,"fine_confidence":0.0,
"observations":[{{"frame_index":{int(event['frame_index'])},
"class_label":"...","fine_label":"...","attributes":{{}},
"confidence":0.0,"fine_confidence":0.0}}]}}]}}
Detector hint (not ground truth): {event.get('detector_class_name', 'unknown')}.
""".strip()


def process_semantic_queue(
    *,
    queue_dir: str | Path,
    vlm_config_path: str | Path,
    semantic_output: str | Path,
    memory_path: str | Path,
    registry_path: str | Path = "configs/ontology/vocabulary_registry.yaml",
    max_events: int = 8,
    max_memory_observations_per_track: int = 32,
    runner: Callable[[Any, list[dict[str, Any]]], dict[str, Any]] = run_qwen_vlm_batches,
) -> dict[str, Any]:
    if max_events < 1:
        raise SemanticQueueError("max_events must be positive.")
    root = Path(queue_dir)
    pending_dir = root / "pending"
    processing_dir = root / "processing"
    processing_dir.mkdir(parents=True, exist_ok=True)
    claimed: list[Path] = []
    for pending_path in sorted(pending_dir.glob("*.json")):
        claimed_path = processing_dir / pending_path.name
        try:
            pending_path.replace(claimed_path)
        except FileNotFoundError:
            continue
        claimed.append(claimed_path)
        if len(claimed) >= max_events:
            break
    if not claimed:
        return {"status": "idle", "processed_event_count": 0}
    try:
        events = [json.loads(path.read_text(encoding="utf-8")) for path in claimed]
        context_ids = {str(event.get("context_id", "")) for event in events}
        if len(context_ids) != 1 or not next(iter(context_ids)):
            raise SemanticQueueError(
                "A worker batch must contain one non-empty context_id."
            )
        for event in events:
            crop_path = Path(event["crop_path"])
            if not crop_path.is_file():
                raise SemanticQueueError(f"Semantic crop does not exist: {crop_path}")
        config = load_vlm_tracking_config(
            vlm_config_path,
            overrides={"run_model": True},
        )
        jobs = [
            {
                "batch_id": event["event_id"],
                "prompt": _event_prompt(event),
                "image_paths": [Path(event["crop_path"])],
                "image_labels": [
                    f"Track {event['track_id']} at frame {event['frame_index']}."
                ],
            }
            for event in events
        ]
        inference = runner(config, jobs)
    except Exception:
        _requeue_claims(claimed, pending_dir)
        raise
    batches = inference.get("batches", [])
    if len(batches) != len(events):
        _requeue_claims(claimed, pending_dir)
        raise SemanticQueueError("Qwen worker returned an unexpected batch count.")
    evidence = []
    processed: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    failed: list[tuple[Path, dict[str, Any], dict[str, Any], str]] = []
    for path, event, batch in zip(claimed, events, batches, strict=True):
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
        evidence.extend(parsed)
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
        fused = fuse_track_semantics(list(memory.observations))
        fused["runtime"] = {
            "mode": "realtime_semantic_worker",
            "context_id": context_id,
            "processed_event_count": len(processed),
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
            **event,
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
        "semantic_output": str(Path(semantic_output).resolve()),
        "memory_path": str(Path(memory_path).resolve()),
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
        claimed_path.replace(target)
