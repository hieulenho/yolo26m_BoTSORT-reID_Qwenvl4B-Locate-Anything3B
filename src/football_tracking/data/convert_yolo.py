"""Convert internal annotations to YOLO detection format."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.bbox import (
    clip_xyxy_to_image,
    is_valid_bbox,
    xyxy_to_yolo_normalized,
)
from football_tracking.data.schemas import (
    FrameAnnotation,
    ObjectAnnotation,
    SequenceInfo,
    SplitManifest,
)

LOGGER = logging.getLogger(__name__)
YOLO_SUPPORTED_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class ConversionError(RuntimeError):
    """Raised when conversion cannot safely write output."""


def _ensure_output_root(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and not overwrite:
        meaningful = [child for child in path.rglob("*") if child.name != ".gitkeep"]
        if meaningful:
            raise ConversionError(f"Output already exists and overwrite=false: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _remove_output_path(path: Path, dry_run: bool) -> None:
    if dry_run or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _clear_generated_outputs(
    output_dir: Path,
    relative_paths: tuple[str, ...],
    dry_run: bool,
) -> None:
    for relative_path in relative_paths:
        _remove_output_path(output_dir / relative_path, dry_run=dry_run)


def _link_or_copy_image(
    source: Path,
    destination: Path,
    prefer_symlink: bool,
    copy_images: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
        return "copy"
    if prefer_symlink:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError as exc:
            LOGGER.info("Symlink failed for %s -> %s: %s", destination, source, exc)
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError as exc:
        LOGGER.info("Hardlink failed for %s -> %s: %s", destination, source, exc)
    shutil.copy2(source, destination)
    return "copy"


def _transcode_image_to_png(
    source: Path,
    destination: Path,
    expected_width: int,
    expected_height: int,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ConversionError(
            f"OpenCV is required to transcode unsupported image format: {source}"
        ) from exc
    image = cv2.imread(str(source))
    if image is None:
        raise ConversionError(f"Could not read image for YOLO conversion: {source}")
    if expected_width > 0 and expected_height > 0:
        height, width = image.shape[:2]
        if (width, height) != (expected_width, expected_height):
            image = cv2.resize(
                image,
                (expected_width, expected_height),
                interpolation=cv2.INTER_NEAREST,
            )
    if not cv2.imwrite(str(destination), image):
        raise ConversionError(f"Could not write transcoded YOLO image: {destination}")
    return "transcode_png"


def _write_yolo_image(
    source: Path,
    destination_stem: Path,
    expected_width: int,
    expected_height: int,
    prefer_symlink: bool,
    copy_images: bool,
) -> str:
    if source.suffix.lower() in YOLO_SUPPORTED_IMAGE_SUFFIXES:
        return _link_or_copy_image(
            source,
            destination_stem.with_suffix(source.suffix),
            prefer_symlink=prefer_symlink,
            copy_images=copy_images,
        )
    return _transcode_image_to_png(
        source,
        destination_stem.with_suffix(".png"),
        expected_width=expected_width,
        expected_height=expected_height,
    )


def _eligible_objects(
    frame: FrameAnnotation,
    clip_boxes: bool,
) -> list[ObjectAnnotation]:
    objects: list[ObjectAnnotation] = []
    seen_track_ids: set[int | str] = set()
    for annotation in frame.objects:
        if annotation.is_ignored or annotation.target_class_id is None:
            continue
        if annotation.track_id in seen_track_ids:
            continue
        seen_track_ids.add(annotation.track_id)
        box = (
            clip_xyxy_to_image(annotation.bbox_xyxy, frame.width, frame.height)
            if clip_boxes
            else annotation.bbox_xyxy
        )
        if not is_valid_bbox(box):
            continue
        objects.append(
            ObjectAnnotation(
                frame_index=annotation.frame_index,
                track_id=annotation.track_id,
                source_class=annotation.source_class,
                target_class=annotation.target_class,
                target_class_id=annotation.target_class_id,
                bbox_xyxy=box,
                confidence=annotation.confidence,
                visibility=annotation.visibility,
                is_ignored=annotation.is_ignored,
                metadata=annotation.metadata,
            )
        )
    return objects


def _format_label_line(
    annotation: ObjectAnnotation,
    frame: FrameAnnotation,
    decimal_places: int,
) -> str:
    values = xyxy_to_yolo_normalized(annotation.bbox_xyxy, frame.width, frame.height)
    formatted = " ".join(f"{value:.{decimal_places}f}" for value in values)
    return f"{annotation.target_class_id} {formatted}"


def convert_to_yolo(
    sequences: list[SequenceInfo],
    split_manifest: SplitManifest,
    output_dir: Path,
    class_names: dict[int, str],
    decimal_places: int = 6,
    copy_images: bool = False,
    prefer_symlink: bool = True,
    clip_boxes: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _ensure_output_root(output_dir, overwrite=overwrite, dry_run=dry_run)
    if overwrite:
        _clear_generated_outputs(
            output_dir,
            ("images", "labels", "dataset.yaml", "manifest.json"),
            dry_run=dry_run,
        )
    planned_images = 0
    planned_labels = 0
    link_methods: dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        if not dry_run:
            (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for sequence in sequences:
        split_name = split_manifest.split_for_sequence(sequence.name)
        if split_name is None:
            continue
        for frame in sequence.annotations:
            stem = f"{sequence.name}_{frame.frame_index:06d}"
            image_destination_stem = output_dir / "images" / split_name / stem
            label_destination = output_dir / "labels" / split_name / f"{stem}.txt"
            planned_images += 1
            planned_labels += 1
            labels = [
                _format_label_line(annotation, frame, decimal_places)
                for annotation in _eligible_objects(frame, clip_boxes=clip_boxes)
            ]
            if dry_run:
                continue
            if not frame.image_path.is_file():
                LOGGER.warning("Image does not exist for YOLO conversion: %s", frame.image_path)
                continue
            method = _write_yolo_image(
                frame.image_path,
                image_destination_stem,
                expected_width=frame.width,
                expected_height=frame.height,
                prefer_symlink=prefer_symlink,
                copy_images=copy_images,
            )
            link_methods[method] = link_methods.get(method, 0) + 1
            label_destination.write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )

    dataset_yaml = {
        "path": output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
        "nc": len(class_names),
    }
    manifest = {
        "images": planned_images,
        "labels": planned_labels,
        "dry_run": dry_run,
        "link_methods": link_methods,
    }
    if not dry_run:
        (output_dir / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False),
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
    return manifest
