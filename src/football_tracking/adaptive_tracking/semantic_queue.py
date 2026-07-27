"""Non-blocking semantic event queue and bounded Qwen worker."""

from __future__ import annotations

import json
import math
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
        crop_result = _track_crop(
            frame,
            track,
            padding=crop_padding,
            output_size=crop_size,
        )
        if crop_result is None:
            return None
        crop, target_bbox = crop_result
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
                "target_bbox_in_crop_xyxy": list(target_bbox),
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
    locate = event.get("locateanything") or {}
    locate_note = (
        "LocateAnything spatially verified the detector query in this target crop "
        f"(association_score={float(locate.get('association_score', 0.0)):.3f}). "
        "Treat this as geometric support, not fine-label ground truth."
        if locate.get("accepted") is True
        else "LocateAnything did not verify this target crop. Rely only on visible pixels."
    )
    return f"""
Analyze this crop from a tracked object using an open, hierarchical vocabulary. Infer a stable
base class and, separately, the most specific visually supported subtype/species/make/model.
Do not guess a fine label from context. A clear base class may have fine_label "unknown".
{locate_note}
Return JSON only using this schema:
{{"track_predictions":[{{"track_id":{int(event['track_id'])},
"class_label":"...","fine_label":"...","taxonomy_path":[],
"attributes":{{}},"confidence":0.0,"fine_confidence":0.0,
"observations":[{{"frame_index":{int(event['frame_index'])},
"class_label":"...","fine_label":"...","attributes":{{}},
"confidence":0.0,"fine_confidence":0.0}}]}}]}}
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
    pending_paths = sorted((root / "pending").glob("*.json"))
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
            qwen_paths = [str(crop_path.resolve())]
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
                },
            )
    finally:
        if owned_backend is not None:
            owned_backend.close()
    return {
        "status": "ok",
        "prepared_event_count": len(candidates),
        "accepted_event_count": accepted_count,
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
    return {
        "batch_id": event["event_id"],
        "prompt": _event_prompt(event),
        "image_paths": image_paths,
        "image_labels": [
            (
                f"Track {event['track_id']} at frame {event['frame_index']} "
                f"({index + 1}/{image_count})."
            )
            for index in range(image_count)
        ],
    }


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
        jobs = [_qwen_job(event) for event in events]
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
        event_frame = int(event["frame_index"])
        parsed = [
            replace(row, evidence_frames=(event_frame,))
            for row in parsed
        ]
        evidence.extend(parsed)
        locate = event.get("locateanything") or {}
        locate_class = str(
            event.get("detector_class_name") or "unknown"
        ).strip()
        if (
            locate.get("accepted") is True
            and locate_class.casefold() not in {"", "unknown", "object"}
        ):
            locate_confidence = locate.get("confidence")
            evidence.append(
                TrackSemanticEvidence(
                    track_id=int(event["track_id"]),
                    class_label=locate_class,
                    confidence=(
                        float(locate_confidence)
                        if locate_confidence is not None
                        else 1.0
                    )
                    * float(locate.get("association_score", 0.0)),
                    source="locateanything",
                    evidence_frames=(event_frame,),
                    evidence="event-triggered target-crop grounding",
                )
            )
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
