"""Convert public multi-domain tracking annotations into a common MOT layout."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


class MultiDomainGtError(RuntimeError):
    """Raised when an external tracking annotation cannot be normalized."""


def convert_multidomain_gt(
    *,
    source_format: str,
    annotation_path: str | Path,
    output_dir: str | Path,
    category_map_path: str | Path | None = None,
    media_root: str | Path | None = None,
    media_fps: float | None = None,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert BDD100K, TAO, or AnimalTrack annotations to class-aware MOT text."""

    source = Path(annotation_path).resolve()
    if not source.exists():
        raise MultiDomainGtError(f"Annotation input does not exist: {source}")
    output = Path(output_dir).resolve()
    manifest_path = output / "normalized_gt_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise MultiDomainGtError(f"Output exists and overwrite=false: {manifest_path}")
    normalized_format = source_format.strip().lower().replace("-", "_")
    if normalized_format == "bdd100k_scalabel":
        sequences, categories = _read_bdd100k(source)
    elif normalized_format == "tao_coco_video":
        sequences, categories = _read_tao(source)
    elif normalized_format == "animaltrack_mot":
        sequences, categories = _read_animaltrack(source)
    elif normalized_format == "ctc_masks_lineage":
        sequences, categories = _read_ctc_masks(source)
    elif normalized_format == "ua_detrac_xml":
        sequences, categories = _read_ua_detrac(source)
    else:
        raise MultiDomainGtError(f"Unsupported source format: {source_format}")
    if max_frames is not None:
        if max_frames < 1:
            raise MultiDomainGtError("max_frames must be positive.")
        sequences = {
            name: [row for row in rows if int(row[0]) <= max_frames]
            for name, rows in sequences.items()
        }
        sequences = {name: rows for name, rows in sequences.items() if rows}
    if category_map_path is not None:
        categories.update(_read_category_map(Path(category_map_path).resolve()))
    resolved_media_root = Path(media_root).resolve() if media_root is not None else None
    if resolved_media_root is not None and not resolved_media_root.exists():
        raise MultiDomainGtError(f"Media root does not exist: {resolved_media_root}")
    if not sequences:
        raise MultiDomainGtError("No valid tracking annotations were found.")

    output.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    for sequence_name, rows in sorted(sequences.items()):
        sequence_dir = output / _safe_name(sequence_name) / "gt"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        gt_path = sequence_dir / "gt.txt"
        rendered = "\n".join(_mot_line(row) for row in sorted(rows)) + "\n"
        _write_text_atomic(gt_path, rendered)
        sequence_info = _sequence_info(
            sequence_name,
            rows,
            resolved_media_root,
            media_fps=media_fps,
        )
        seqinfo_path = sequence_dir.parent / "seqinfo.ini"
        _write_text_atomic(seqinfo_path, _seqinfo_text(sequence_name, sequence_info))
        sequence_rows.append(
            {
                "sequence": sequence_name,
                "normalized_sequence": _safe_name(sequence_name),
                "gt_path": str(gt_path),
                "seqinfo_path": str(seqinfo_path),
                "media_path": sequence_info.get("media_path"),
                "annotation_count": len(rows),
                "frame_count": sequence_info["frame_count"],
                "track_count": len({row[1] for row in rows}),
                "class_count": len({row[7] for row in rows}),
            }
        )
    payload = {
        "schema_version": 1,
        "source_format": normalized_format,
        "source": str(source),
        "media_root": str(resolved_media_root) if resolved_media_root else None,
        "media_fps": media_fps,
        "max_frames": max_frames,
        "mot_columns": [
            "frame",
            "track_id",
            "x",
            "y",
            "width",
            "height",
            "confidence",
            "class_id",
            "visibility",
            "unused",
        ],
        "categories": {str(key): value for key, value in sorted(categories.items())},
        "sequence_count": len(sequence_rows),
        "annotation_count": sum(row["annotation_count"] for row in sequence_rows),
        "sequences": sequence_rows,
    }
    _write_text_atomic(manifest_path, json.dumps(payload, indent=2))
    return payload


def _read_bdd100k(
    source: Path,
) -> tuple[dict[str, list[tuple]], dict[int, str]]:
    frames: list[dict[str, Any]] = []
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    for path in paths:
        payload = _json(path)
        if isinstance(payload, list):
            frames.extend(row for row in payload if isinstance(row, dict))
        elif isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            frames.extend(row for row in payload["frames"] if isinstance(row, dict))
        else:
            raise MultiDomainGtError(f"Unsupported BDD100K JSON root: {path}")
    category_names = sorted(
        {
            str(label.get("category", "")).strip()
            for frame in frames
            for label in frame.get("labels", [])
            if isinstance(label, dict) and str(label.get("category", "")).strip()
        }
    )
    name_to_id = {name: index + 1 for index, name in enumerate(category_names)}
    categories = {value: key for key, value in name_to_id.items()}
    sequences: dict[str, list[tuple]] = defaultdict(list)
    track_maps: dict[str, dict[str, int]] = defaultdict(dict)
    for frame_offset, frame in enumerate(frames, start=1):
        sequence = str(
            frame.get("videoName")
            or frame.get("video_name")
            or Path(str(frame.get("name", "sequence"))).stem.split("-")[0]
        )
        frame_id = int(frame.get("frameIndex", frame.get("frame_index", frame_offset))) + 1
        for label in frame.get("labels", []):
            if not isinstance(label, dict) or not isinstance(label.get("box2d"), dict):
                continue
            external_id = str(label.get("id", "")).strip()
            category = str(label.get("category", "")).strip()
            if not external_id or category not in name_to_id:
                continue
            mapping = track_maps[sequence]
            if external_id not in mapping:
                mapping[external_id] = len(mapping) + 1
            box = label["box2d"]
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
            sequences[sequence].append(
                _row(frame_id, mapping[external_id], x1, y1, x2 - x1, y2 - y1, name_to_id[category])
            )
    return dict(sequences), categories


def _read_tao(source: Path) -> tuple[dict[str, list[tuple]], dict[int, str]]:
    payload = _json(source)
    if not isinstance(payload, dict):
        raise MultiDomainGtError("TAO annotation root must be an object.")
    categories = {
        int(row["id"]): str(row["name"])
        for row in payload.get("categories", [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    videos = {
        int(row["id"]): str(row.get("name") or row.get("file_name") or row["id"])
        for row in payload.get("videos", [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    images = {
        int(row["id"]): row
        for row in payload.get("images", [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    sequences: dict[str, list[tuple]] = defaultdict(list)
    track_maps: dict[str, dict[int, int]] = defaultdict(dict)
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image = images.get(int(annotation.get("image_id", -1)))
        bbox = annotation.get("bbox")
        if image is None or not isinstance(bbox, list) or len(bbox) < 4:
            continue
        video_id = int(image.get("video_id", -1))
        sequence = videos.get(video_id, str(video_id))
        external_track = int(annotation.get("track_id", annotation.get("id", -1)))
        mapping = track_maps[sequence]
        if external_track not in mapping:
            mapping[external_track] = len(mapping) + 1
        frame_id = int(image.get("frame_index", image.get("frame_id", 0))) + 1
        x, y, width, height = map(float, bbox[:4])
        category_id = int(annotation.get("category_id", 0))
        sequences[sequence].append(
            _row(frame_id, mapping[external_track], x, y, width, height, category_id)
        )
    return dict(sequences), categories


def _read_animaltrack(source: Path) -> tuple[dict[str, list[tuple]], dict[int, str]]:
    paths = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    sequences: dict[str, list[tuple]] = defaultdict(list)
    categories: dict[int, str] = {}
    for path in paths:
        sequence = path.stem.removesuffix("_gt")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = [value.strip() for value in line.split(",")]
            if len(fields) < 8:
                raise MultiDomainGtError(
                    f"AnimalTrack row needs at least 8 columns: {path}:{line_number}"
                )
            frame_id, track_id = int(float(fields[0])), int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            confidence = float(fields[6])
            class_id = int(float(fields[7]))
            visibility = float(fields[8]) if len(fields) > 8 else 1.0
            categories.setdefault(class_id, f"animal_class_{class_id}")
            sequences[sequence].append(
                _row(
                    frame_id,
                    track_id,
                    x,
                    y,
                    width,
                    height,
                    class_id,
                    confidence,
                    visibility,
                )
            )
    return dict(sequences), categories


def _read_ctc_masks(source: Path) -> tuple[dict[str, list[tuple]], dict[int, str]]:
    import cv2
    import numpy as np

    paths = sorted(source.rglob("man_track*.tif")) if source.is_dir() else [source]
    sequences: dict[str, list[tuple]] = defaultdict(list)
    for path in paths:
        match = re.search(r"(\d+)$", path.stem)
        if match is None:
            continue
        frame_id = int(match.group(1)) + 1
        sequence = path.parent.parent.name.removesuffix("_GT") or path.parent.name
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise MultiDomainGtError(f"Unreadable CTC tracking mask: {path}")
        if mask.ndim > 2:
            mask = mask[..., 0]
        for track_id_value in np.unique(mask):
            track_id = int(track_id_value)
            if track_id <= 0:
                continue
            ys, xs = np.where(mask == track_id)
            if not len(xs):
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            sequences[sequence].append(
                _row(
                    frame_id,
                    track_id,
                    float(x1),
                    float(y1),
                    float(x2 - x1 + 1),
                    float(y2 - y1 + 1),
                    1,
                )
            )
    return dict(sequences), {1: "cell"}


def _read_ua_detrac(source: Path) -> tuple[dict[str, list[tuple]], dict[int, str]]:
    import xml.etree.ElementTree as ET

    paths = sorted(source.rglob("*.xml")) if source.is_dir() else [source]
    category_names = ("car", "van", "bus", "others")
    name_to_id = {name: index + 1 for index, name in enumerate(category_names)}
    sequences: dict[str, list[tuple]] = defaultdict(list)
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise MultiDomainGtError(f"Invalid UA-DETRAC XML: {path}") from exc
        sequence = str(root.attrib.get("name") or path.stem)
        for frame in root.findall("frame"):
            frame_id = int(frame.attrib["num"])
            for target in frame.findall("./target_list/target"):
                box = target.find("box")
                attribute = target.find("attribute")
                if box is None or attribute is None:
                    continue
                category = str(attribute.attrib.get("vehicle_type", "others")).lower()
                category = category if category in name_to_id else "others"
                truncation = float(attribute.attrib.get("truncation_ratio", 0.0))
                sequences[sequence].append(
                    _row(
                        frame_id,
                        int(target.attrib["id"]),
                        float(box.attrib["left"]),
                        float(box.attrib["top"]),
                        float(box.attrib["width"]),
                        float(box.attrib["height"]),
                        name_to_id[category],
                        visibility=max(0.0, min(1.0, 1.0 - truncation)),
                    )
                )
    return dict(sequences), {value: key for key, value in name_to_id.items()}


def _row(
    frame_id: int,
    track_id: int,
    x: float,
    y: float,
    width: float,
    height: float,
    class_id: int,
    confidence: float = 1.0,
    visibility: float = 1.0,
) -> tuple:
    if frame_id < 1 or track_id < 1 or width <= 0 or height <= 0:
        raise MultiDomainGtError(
            f"Invalid MOT annotation: frame={frame_id}, track={track_id}, "
            f"width={width}, height={height}"
        )
    return (frame_id, track_id, x, y, width, height, confidence, class_id, visibility, -1)


def _mot_line(row: tuple) -> str:
    return ",".join(
        str(int(value)) if index in {0, 1, 7, 9} else f"{float(value):.6f}"
        for index, value in enumerate(row)
    )


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return cleaned.strip("_") or "sequence"


def _sequence_info(
    sequence_name: str,
    rows: list[tuple],
    media_root: Path | None,
    *,
    media_fps: float | None = None,
) -> dict[str, Any]:
    annotation_frames = max(int(row[0]) for row in rows)
    fallback = {
        "frame_count": annotation_frames,
        "fps": 30.0,
        "width": max(1, math.ceil(max(float(row[2]) + float(row[4]) for row in rows))),
        "height": max(1, math.ceil(max(float(row[3]) + float(row[5]) for row in rows))),
        "media_path": None,
        "image_extension": ".jpg",
    }
    if media_root is None:
        return fallback
    safe_name = _safe_name(sequence_name)
    video_candidates = [
        path
        for path in media_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        and _safe_name(path.stem) == safe_name
    ]
    if len(video_candidates) > 1:
        raise MultiDomainGtError(
            f"Multiple media files match sequence {sequence_name}: {video_candidates}"
        )
    if not video_candidates:
        return _image_sequence_info(
            media_root,
            sequence_name,
            annotation_frames,
            fallback,
            media_fps=media_fps,
        )
    import cv2

    media_path = video_candidates[0]
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise MultiDomainGtError(f"Could not open media file: {media_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if frame_count < annotation_frames or fps <= 0 or width <= 0 or height <= 0:
        raise MultiDomainGtError(
            f"Invalid media metadata for {sequence_name}: frames={frame_count}, "
            f"annotations={annotation_frames}, fps={fps}, size={width}x{height}"
        )
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "media_path": str(media_path),
        "image_extension": media_path.suffix.lower(),
    }


def _image_sequence_info(
    media_root: Path,
    sequence_name: str,
    annotation_frames: int,
    fallback: dict[str, Any],
    *,
    media_fps: float | None,
) -> dict[str, Any]:
    import cv2

    safe_name = _safe_name(sequence_name)
    directories = [
        path
        for path in media_root.rglob("*")
        if path.is_dir() and _safe_name(path.name) == safe_name
    ]
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    candidates = [
        (directory, sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ))
        for directory in directories
    ]
    candidates = [(directory, images) for directory, images in candidates if images]
    if not candidates:
        return fallback
    if len(candidates) > 1:
        raise MultiDomainGtError(
            f"Multiple image sequences match {sequence_name}: "
            f"{[directory for directory, _ in candidates]}"
        )
    directory, images = candidates[0]
    if len(images) < annotation_frames:
        raise MultiDomainGtError(
            f"Image sequence {sequence_name} has {len(images)} frames but "
            f"annotations reach frame {annotation_frames}."
        )
    first = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim < 2:
        raise MultiDomainGtError(f"Unreadable first image: {images[0]}")
    height, width = first.shape[:2]
    return {
        "frame_count": len(images),
        "fps": float(media_fps or 1.0),
        "width": int(width),
        "height": int(height),
        "media_path": str(directory),
        "image_extension": images[0].suffix.lower(),
    }


def _seqinfo_text(sequence_name: str, info: dict[str, Any]) -> str:
    return (
        "[Sequence]\n"
        f"name={_safe_name(sequence_name)}\n"
        "imDir=img1\n"
        f"frameRate={max(1, round(float(info['fps'])))}\n"
        f"seqLength={int(info['frame_count'])}\n"
        f"imWidth={int(info['width'])}\n"
        f"imHeight={int(info['height'])}\n"
        f"imExt={info.get('image_extension', '.jpg')}\n"
    )


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MultiDomainGtError(f"Invalid JSON annotation: {path}") from exc


def _read_category_map(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise MultiDomainGtError(f"Category map does not exist: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        payload = _json(path)
    else:
        payload = {
            index + 1: line.strip()
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        }
    if not isinstance(payload, dict):
        raise MultiDomainGtError("Category map must be a mapping or one-name-per-line text.")
    result = {
        int(key): str(value).strip()
        for key, value in payload.items()
        if str(value).strip()
    }
    if not result:
        raise MultiDomainGtError("Category map must not be empty.")
    return result


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
