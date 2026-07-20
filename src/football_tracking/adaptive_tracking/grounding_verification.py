"""Event-triggered LocateAnything grounding and track association."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.schemas import SceneDiscovery
from football_tracking.adaptive_tracking.semantic_fusion import parse_qwen_answer
from football_tracking.detection.serialization import runtime_versions
from football_tracking.vlm.tracking_context import MotTrackRow, read_mot_tracks


def build_grounding_plan(
    discovery: SceneDiscovery,
    *,
    output_path: str | Path,
    confidence_trigger: float = 0.65,
    max_classes: int = 4,
    max_keyframes_per_class: int = 2,
    max_expected_tracks_per_class: int = 3,
    qwen_answer: str | Path | None = None,
    semantic_context: str | Path | None = None,
    verify_track_ids: list[int] | tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for item in discovery.detector_objects:
        if not item.open_vocabulary and item.confidence >= confidence_trigger:
            continue
        candidates[item.canonical_name] = {
            "class_label": item.canonical_name,
            "display_name": item.display_name,
            "trigger": (
                "open_vocabulary_class"
                if item.open_vocabulary
                else "low_discovery_confidence"
            ),
            "expected_track_ids": [],
        }
    if max_expected_tracks_per_class <= 0:
        raise ValueError("max_expected_tracks_per_class must be positive.")
    uncertain_track_ids = {int(track_id) for track_id in verify_track_ids or ()}
    if any(track_id <= 0 for track_id in uncertain_track_ids):
        raise ValueError("verify_track_ids must contain positive track IDs.")
    semantic_keyframes: list[dict[str, Any]] = []
    semantic_crops: list[dict[str, Any]] = []
    context_path = Path(semantic_context) if semantic_context is not None else None
    if context_path is not None:
        if not context_path.is_file():
            raise FileNotFoundError(f"Semantic context does not exist: {context_path}")
        context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        semantic_keyframes = [dict(row) for row in context_payload.get("keyframes", [])]
        semantic_crops = [dict(row) for row in context_payload.get("crops", [])]
    if qwen_answer is not None and Path(qwen_answer).is_file():
        qwen_path = Path(qwen_answer)
        qwen_payload = json.loads(qwen_path.read_text(encoding="utf-8"))
        sibling_context = qwen_path.parent / "vlm_context.json"
        if context_path is None and sibling_context.is_file():
            context_payload = json.loads(sibling_context.read_text(encoding="utf-8"))
            semantic_keyframes = [
                dict(row) for row in context_payload.get("keyframes", [])
            ]
            semantic_crops = [dict(row) for row in context_payload.get("crops", [])]
        for evidence in parse_qwen_answer(qwen_payload):
            if evidence.class_label == "unknown":
                uncertain_track_ids.add(evidence.track_id)
                continue
            if evidence.confidence >= confidence_trigger:
                continue
            candidate = candidates.setdefault(
                evidence.class_label,
                {
                    "class_label": evidence.class_label,
                    "display_name": evidence.class_label,
                    "trigger": "low_qwen_semantic_confidence",
                    "expected_track_ids": [],
                },
            )
            candidate["expected_track_ids"].append(evidence.track_id)
    if uncertain_track_ids:
        for item in discovery.tracking_objects:
            candidate = candidates.setdefault(
                item.canonical_name,
                {
                    "class_label": item.canonical_name,
                    "display_name": item.display_name,
                    "trigger": "qwen_unknown_track",
                    "expected_track_ids": [],
                },
            )
            candidate["expected_track_ids"].extend(uncertain_track_ids)
    selected_candidates = list(candidates.values())[:max_classes]
    requests = _grounding_requests(
        selected_candidates,
        discovery_keyframes=list(discovery.keyframes),
        semantic_keyframes=semantic_keyframes,
        semantic_crops=semantic_crops,
        max_keyframes_per_class=max_keyframes_per_class,
        max_expected_tracks_per_class=max_expected_tracks_per_class,
    )
    result = {
        "schema_version": "1.0",
        "policy": "explicit_track_benchmark" if verify_track_ids else "event_triggered",
        "source_video": discovery.source_video,
        "requests": requests,
        "summary": {
            "candidate_class_count": len(selected_candidates),
            "request_count": len(requests),
            "uncertain_track_count": len(uncertain_track_ids),
            "explicit_track_count": len(set(verify_track_ids or ())),
            "skipped": not requests,
            "skip_reason": (
                "No open class, low-confidence class, or unknown Qwen track needs grounding."
                if not requests
                else None
            ),
        },
    }
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Grounding plan exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def _grounding_requests(
    candidates: list[dict[str, Any]],
    *,
    discovery_keyframes: list[dict[str, Any]],
    semantic_keyframes: list[dict[str, Any]],
    semantic_crops: list[dict[str, Any]],
    max_keyframes_per_class: int,
    max_expected_tracks_per_class: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for class_index, item in enumerate(candidates):
        expected_ids = sorted(set(item["expected_track_ids"]))[
            :max_expected_tracks_per_class
        ]
        if not expected_ids:
            for frame_index, keyframe in enumerate(
                discovery_keyframes[:max_keyframes_per_class]
            ):
                requests.append(
                    _grounding_request(
                        request_id=f"{class_index:02d}_{frame_index:02d}",
                        item=item,
                        keyframe=keyframe,
                        expected_track_ids=[],
                    )
                )
            continue
        for track_id in expected_ids:
            visible = [
                row
                for row in semantic_keyframes
                if track_id in {int(value) for value in row.get("visible_track_ids", [])}
            ]
            track_crops = [
                row
                for row in semantic_crops
                if int(row.get("track_id", -1)) == track_id
            ]
            has_target_frame = bool(visible or track_crops)
            keyframes = (visible or track_crops or discovery_keyframes)[
                :max_keyframes_per_class
            ]
            for frame_index, keyframe in enumerate(keyframes):
                requests.append(
                    _grounding_request(
                        request_id=(
                            f"{class_index:02d}_t{track_id:04d}_{frame_index:02d}"
                        ),
                        item=item,
                        keyframe=keyframe,
                        expected_track_ids=[track_id],
                        target_track_id=track_id if has_target_frame else None,
                    )
                )
    return requests


def _grounding_request(
    *,
    request_id: str,
    item: dict[str, Any],
    keyframe: dict[str, Any],
    expected_track_ids: list[int],
    target_track_id: int | None = None,
) -> dict[str, Any]:
    query = f"the {item['display_name']}"
    if target_track_id is not None:
        query += f" inside the tracking box labeled ID {target_track_id}"
    return {
        "request_id": request_id,
        "class_label": item["class_label"],
        "query": query,
        "localized_query": f"the {item['display_name']}",
        "frame_index": int(keyframe["frame_index"]),
        "image_path": str(keyframe["path"]),
        "trigger": item["trigger"],
        "expected_track_ids": expected_track_ids,
        "target_track_id": target_track_id,
    }


def execute_grounding_plan(
    *,
    plan_path: str | Path,
    tracks_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    model_id: str = "nvidia/LocateAnything-3B",
    device: str = "cuda",
    torch_dtype: str = "auto",
    quantization: str = "8bit",
    max_new_tokens: int = 512,
    minimum_iou: float = 0.10,
    target_crop_padding: float = 1.0,
    target_crop_size: int = 384,
    overwrite: bool = False,
) -> dict[str, Any]:
    from football_tracking.locate_tracking.grounding.cache import GroundingCache
    from football_tracking.locate_tracking.grounding.locate_anything_backend import (
        LocateAnythingBackend,
    )
    from football_tracking.locate_tracking.grounding.service import GroundingService

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    requests = plan.get("requests", [])
    rows_by_frame: dict[int, list[MotTrackRow]] = {}
    for row in read_mot_tracks(Path(tracks_path)):
        rows_by_frame.setdefault(row.frame_index, []).append(row)
    if not requests:
        result = _grounding_execution_payload(plan, [], [], skipped=True)
        result["runtime"] = runtime_versions()
        return _write_result(result, output_path, overwrite)

    backend = LocateAnythingBackend(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
    )
    service = GroundingService(
        backend=backend,
        cache=GroundingCache(cache_dir),
        overwrite=overwrite,
    )
    raw_results: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    frame_cache: dict[int, Any] = {}
    _reset_peak_cuda_memory()
    execution_started = time.perf_counter()
    for request in requests:
        frame_index = int(request["frame_index"])
        frame_rows = rows_by_frame.get(frame_index, [])
        prepared = _prepare_grounding_input(
            request=request,
            source_video=plan.get("source_video"),
            frame_rows=frame_rows,
            output_dir=Path(output_path).parent / "target_crops",
            frame_cache=frame_cache,
            crop_padding=target_crop_padding,
            crop_size=target_crop_size,
        )
        request_started = time.perf_counter()
        result = service.ground_image(
            image_path=prepared["image_path"],
            query=prepared["query"],
        )
        raw_results.append(
            {
                "request_id": request["request_id"],
                "result": result.to_dict(),
                "grounding_input": prepared,
                "seconds": time.perf_counter() - request_started,
            }
        )
        for box in result.boxes:
            global_bbox = _map_bbox_to_source(box.bbox_xyxy, prepared.get("roi"))
            matched = _best_track(global_bbox, frame_rows)
            if matched is None or matched[1] < minimum_iou:
                continue
            track, overlap = matched
            grounding_confidence = box.confidence if box.confidence is not None else 1.0
            expected_track_ids = {
                int(track_id) for track_id in request.get("expected_track_ids", [])
            }
            matches_expected = not expected_track_ids or track.track_id in expected_track_ids
            associations.append(
                {
                    "request_id": request["request_id"],
                    "frame_index": frame_index,
                    "track_id": track.track_id,
                    "class_label": request["class_label"],
                    "iou": round(overlap, 6),
                    "grounding_confidence": box.confidence,
                    "confidence": round(float(grounding_confidence) * overlap, 6),
                    "expected_track_ids": sorted(expected_track_ids),
                    "matches_expected_track": matches_expected,
                    "accepted_for_fusion": matches_expected,
                    "grounded_bbox_xyxy": list(global_bbox),
                    "grounded_bbox_xyxy_local": list(box.bbox_xyxy),
                    "track_bbox_xyxy": list(track.bbox_xyxy()),
                }
            )
    cold_start_total = time.perf_counter() - execution_started
    model_load_seconds = float(backend.model_load_seconds or 0.0)
    cache_statuses = [
        str(row.get("result", {}).get("runtime_info", {}).get("cache_status", "unknown"))
        for row in raw_results
    ]
    result = _grounding_execution_payload(plan, raw_results, associations, skipped=False)
    result["timing"] = {
        "model_load_seconds": model_load_seconds,
        "execution_seconds": max(cold_start_total - model_load_seconds, 0.0),
        "cold_start_total_seconds": cold_start_total,
        "total_seconds": cold_start_total,
        "mean_request_seconds": (
            sum(float(row["seconds"]) for row in raw_results) / len(raw_results)
            if raw_results
            else None
        ),
        "cache_hit_count": cache_statuses.count("hit"),
        "cache_miss_count": cache_statuses.count("miss"),
    }
    result["cuda_memory"] = _peak_cuda_memory()
    result["runtime"] = runtime_versions()
    return _write_result(result, output_path, overwrite)


def _prepare_grounding_input(
    *,
    request: dict[str, Any],
    source_video: str | None,
    frame_rows: list[MotTrackRow],
    output_dir: Path,
    frame_cache: dict[int, Any],
    crop_padding: float,
    crop_size: int,
) -> dict[str, Any]:
    if crop_padding < 0:
        raise ValueError("target_crop_padding must be non-negative.")
    if crop_size < 64:
        raise ValueError("target_crop_size must be at least 64 pixels.")
    target_track_id = request.get("target_track_id")
    if target_track_id is None:
        return {
            "mode": "full_keyframe",
            "image_path": str(request["image_path"]),
            "query": str(request["query"]),
            "roi": None,
        }
    target = next(
        (row for row in frame_rows if row.track_id == int(target_track_id)),
        None,
    )
    if target is None:
        return {
            "mode": "full_keyframe_fallback",
            "image_path": str(request["image_path"]),
            "query": str(request["query"]),
            "roi": None,
            "fallback_reason": "target_track_missing_at_frame",
        }

    frame_index = int(request["frame_index"])
    frame = frame_cache.get(frame_index)
    if frame is None:
        frame = _read_source_frame(source_video, frame_index)
        if frame is None:
            import cv2

            frame = cv2.imread(str(request["image_path"]))
        if frame is None:
            return {
                "mode": "full_keyframe_fallback",
                "image_path": str(request["image_path"]),
                "query": str(request["query"]),
                "roi": None,
                "fallback_reason": "source_frame_unreadable",
            }
        frame_cache[frame_index] = frame

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in target.bbox_xyxy())
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    crop_x1 = max(0, math.floor(x1 - box_width * crop_padding))
    crop_y1 = max(0, math.floor(y1 - box_height * crop_padding))
    crop_x2 = min(frame_width, math.ceil(x2 + box_width * crop_padding))
    crop_y2 = min(frame_height, math.ceil(y2 + box_height * crop_padding))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return {
            "mode": "full_keyframe_fallback",
            "image_path": str(request["image_path"]),
            "query": str(request["query"]),
            "roi": None,
            "fallback_reason": "empty_target_crop",
        }

    import cv2

    native_height, native_width = crop.shape[:2]
    scale = max(float(crop_size) / max(native_width, native_height), 1.0)
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (
                max(1, round(native_width * scale)),
                max(1, round(native_height * scale)),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{request['request_id']}.jpg"
    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"Could not write grounding target crop: {crop_path}")
    return {
        "mode": "target_crop",
        "image_path": str(crop_path.resolve()),
        "query": str(request.get("localized_query") or request["query"]),
        "roi": {
            "source_x": crop_x1,
            "source_y": crop_y1,
            "source_width": native_width,
            "source_height": native_height,
            "scale": scale,
            "target_track_id": int(target_track_id),
        },
    }


def _read_source_frame(source_video: str | None, frame_index: int) -> Any | None:
    if not source_video or not Path(source_video).is_file():
        return None
    import cv2

    capture = cv2.VideoCapture(str(source_video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _map_bbox_to_source(
    bbox: tuple[float, float, float, float],
    roi: dict[str, Any] | None,
) -> tuple[float, float, float, float]:
    if not roi:
        return bbox
    scale = max(float(roi.get("scale", 1.0)), 1e-9)
    source_x = float(roi["source_x"])
    source_y = float(roi["source_y"])
    return (
        source_x + bbox[0] / scale,
        source_y + bbox[1] / scale,
        source_x + bbox[2] / scale,
        source_y + bbox[3] / scale,
    )


def _grounding_execution_payload(
    plan: dict[str, Any],
    results: list[dict[str, Any]],
    associations: list[dict[str, Any]],
    *,
    skipped: bool,
) -> dict[str, Any]:
    accepted_associations = [
        row for row in associations if row.get("accepted_for_fusion", True)
    ]
    matched_requests = {row["request_id"] for row in accepted_associations}
    return {
        "schema_version": "1.0",
        "plan_summary": plan.get("summary", {}),
        "summary": {
            "skipped": skipped,
            "request_count": len(plan.get("requests", [])),
            "result_count": len(results),
            "association_count": len(associations),
            "accepted_association_count": len(accepted_associations),
            "rejected_association_count": len(associations)
            - len(accepted_associations),
            "matched_request_count": len(matched_requests),
        },
        "associations": associations,
        "grounding_results": results,
    }


def _best_track(
    bbox: tuple[float, float, float, float],
    rows: list[MotTrackRow],
) -> tuple[MotTrackRow, float] | None:
    if not rows:
        return None
    scored = [(row, _iou(bbox, tuple(float(value) for value in row.bbox_xyxy()))) for row in rows]
    return max(scored, key=lambda item: item[1])


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _write_result(
    result: dict[str, Any],
    output_path: str | Path,
    overwrite: bool,
) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Grounding result exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def _reset_peak_cuda_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _peak_cuda_memory() -> dict[str, int | None]:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except (ImportError, RuntimeError):
        pass
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
