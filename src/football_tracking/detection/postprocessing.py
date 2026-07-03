"""Convert model predictions into stable detection records."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.schemas import Detection


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _rows_from_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    if "boxes" in mapping:
        boxes = mapping["boxes"]
        if isinstance(boxes, dict):
            return _rows_from_mapping(boxes)
        if isinstance(boxes, Iterable) and not isinstance(boxes, str | bytes):
            rows: list[dict[str, Any]] = []
            for item in boxes:
                if isinstance(item, dict):
                    rows.extend(_rows_from_mapping(item))
            return rows
    xyxy = _to_python(mapping.get("xyxy", mapping.get("bbox_xyxy")))
    conf = _to_python(mapping.get("conf", mapping.get("confidence")))
    cls = _to_python(mapping.get("cls", mapping.get("class_id", mapping.get("source_class_id"))))
    if xyxy is None:
        return []
    if xyxy and isinstance(xyxy[0], int | float):
        xyxy = [xyxy]
        conf = [conf]
        cls = [cls]
    return [
        {"xyxy": box, "confidence": conf[index], "class_id": cls[index]}
        for index, box in enumerate(xyxy)
    ]


def _rows_from_ultralytics_boxes(boxes: Any) -> list[dict[str, Any]]:
    xyxy = _to_python(getattr(boxes, "xyxy", []))
    conf = _to_python(getattr(boxes, "conf", []))
    cls = _to_python(getattr(boxes, "cls", []))
    rows = []
    for index, box in enumerate(xyxy):
        rows.append({"xyxy": box, "confidence": conf[index], "class_id": cls[index]})
    return rows


def iter_prediction_rows(raw_prediction: Any) -> list[dict[str, Any]]:
    if raw_prediction is None:
        return []
    if isinstance(raw_prediction, dict):
        return _rows_from_mapping(raw_prediction)
    boxes = getattr(raw_prediction, "boxes", None)
    if boxes is not None:
        if isinstance(boxes, dict):
            return _rows_from_mapping(boxes)
        return _rows_from_ultralytics_boxes(boxes)
    if isinstance(raw_prediction, Iterable) and not isinstance(raw_prediction, str | bytes):
        rows: list[dict[str, Any]] = []
        for item in raw_prediction:
            if isinstance(item, dict):
                rows.extend(_rows_from_mapping(item))
        return rows
    return []


def _finite_values(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def postprocess_detections(
    raw_prediction: Any,
    frame_index: int,
    sequence_name: str,
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    coco_person_class_id: int = 0,
    target_class_id: int = 0,
    target_class_name: str = "player",
    keep_only_person: bool = True,
    allowed_class_ids: Iterable[int] | None = None,
    source_class_names: dict[int, str] | None = None,
    preserve_source_class: bool = False,
    image_path: str | Path | None = None,
) -> list[Detection]:
    detections: list[Detection] = []
    allowed_classes = set(allowed_class_ids) if allowed_class_ids is not None else None
    class_names = source_class_names or {}
    for row in iter_prediction_rows(raw_prediction):
        try:
            source_class_id = int(float(row["class_id"]))
            confidence = float(row["confidence"])
            coords = [float(value) for value in row["xyxy"]]
        except (KeyError, TypeError, ValueError):
            continue
        if len(coords) != 4 or not _finite_values([*coords, confidence]):
            continue
        if keep_only_person and source_class_id != coco_person_class_id:
            continue
        if allowed_classes is not None and source_class_id not in allowed_classes:
            continue
        if confidence < confidence_threshold:
            continue
        clipped = clip_xyxy_to_image(
            BoundingBoxXYXY(coords[0], coords[1], coords[2], coords[3]),
            image_width,
            image_height,
        )
        if not is_valid_bbox(clipped):
            continue
        source_class_name = class_names.get(
            source_class_id,
            "person" if source_class_id == coco_person_class_id else f"class_{source_class_id}",
        )
        emitted_class_id = source_class_id if preserve_source_class else target_class_id
        emitted_class_name = source_class_name if preserve_source_class else target_class_name
        detections.append(
            Detection(
                frame_index=frame_index,
                sequence_name=sequence_name,
                bbox_xyxy=clipped,
                confidence=confidence,
                source_class_id=source_class_id,
                source_class_name=source_class_name,
                target_class_id=emitted_class_id,
                target_class_name=emitted_class_name,
                image_width=image_width,
                image_height=image_height,
                metadata={"image_path": str(image_path)} if image_path is not None else {},
            )
        )
    return sorted(
        detections,
        key=lambda item: (
            item.frame_index,
            -item.confidence,
            item.bbox_xyxy.x1,
            item.bbox_xyxy.y1,
            item.bbox_xyxy.x2,
            item.bbox_xyxy.y2,
        ),
    )
