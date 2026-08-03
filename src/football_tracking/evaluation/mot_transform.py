"""Coordinate-safe staging utilities for normalized MOT ground truth."""

from __future__ import annotations

import configparser
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


class MotTransformError(ValueError):
    """Raised when a requested MOT coordinate transform is invalid."""


def load_normalized_mot_ignore_regions(
    *,
    gt_root: Path,
    sequences: list[str],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Load static xywh ignore regions recorded by the normalized GT manifest."""
    manifest_path = gt_root / "normalized_gt_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MotTransformError(f"Invalid normalized GT manifest: {manifest_path}") from exc
    requested = set(sequences)
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in manifest.get("sequences", []):
        if not isinstance(row, dict):
            continue
        sequence = str(row.get("normalized_sequence") or row.get("sequence") or "")
        if sequence not in requested:
            continue
        configured_path = row.get("ignored_regions_path")
        path = (
            Path(str(configured_path))
            if configured_path
            else gt_root / sequence / "ignored_regions.json"
        )
        if not path.is_absolute():
            path = gt_root / path
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MotTransformError(f"Invalid ignored-region file: {path}") from exc
        regions: list[tuple[float, float, float, float]] = []
        for region in payload.get("regions", []):
            if not isinstance(region, dict):
                continue
            try:
                x = float(region["x"]) * scale_x
                y = float(region["y"]) * scale_y
                width = float(region["width"]) * scale_x
                height = float(region["height"]) * scale_y
            except (KeyError, TypeError, ValueError) as exc:
                raise MotTransformError(f"Invalid ignored region in {path}: {region}") from exc
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise MotTransformError(f"Ignored region contains non-finite values: {path}")
            if width <= 0.0 or height <= 0.0:
                raise MotTransformError(f"Ignored region must have positive area: {path}")
            regions.append((x, y, width, height))
        if regions:
            result[sequence] = regions
    return result


def filter_mot_file_by_ignore_regions(
    *,
    source: Path,
    destination: Path,
    regions: list[tuple[float, float, float, float]],
    overlap_threshold: float,
) -> dict[str, Any]:
    """Remove MOT boxes whose intersection covers the configured share of the box."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise MotTransformError("overlap_threshold must be in [0, 1].")
    if not source.is_file():
        raise MotTransformError(f"MOT file does not exist: {source}")
    kept: list[list[str]] = []
    removed = 0
    total = 0
    with source.open("r", encoding="utf-8", newline="") as source_handle:
        rows = list(csv.reader(source_handle))
    for line_number, row in enumerate(rows, start=1):
        if not row:
            continue
        if len(row) < 6:
            raise MotTransformError(
                f"MOT row {line_number} in {source} has fewer than 6 columns."
            )
        try:
            box = tuple(float(row[index]) for index in range(2, 6))
        except ValueError as exc:
            raise MotTransformError(
                f"MOT row {line_number} in {source} has invalid coordinates."
            ) from exc
        total += 1
        if any(
            _intersection_over_box_area(box, region) >= overlap_threshold
            for region in regions
        ):
            removed += 1
            continue
        kept.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination_handle:
        csv.writer(destination_handle, lineterminator="\n").writerows(kept)
    temporary.replace(destination)
    return {"input_rows": total, "kept_rows": len(kept), "removed_rows": removed}


def stage_ignore_filtered_mot_ground_truth(
    *,
    gt_root: Path,
    sequences: list[str],
    output_root: Path,
    regions_by_sequence: dict[str, list[tuple[float, float, float, float]]],
    overlap_threshold: float,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Stage GT after applying the same static ignore-region policy as predictions."""
    stats: dict[str, dict[str, Any]] = {}
    for sequence in sequences:
        source_sequence = gt_root / sequence
        destination_sequence = output_root / sequence
        source_gt = source_sequence / "gt" / "gt.txt"
        destination_gt = destination_sequence / "gt" / "gt.txt"
        regions = regions_by_sequence.get(sequence, [])
        if regions:
            stats[sequence] = filter_mot_file_by_ignore_regions(
                source=source_gt,
                destination=destination_gt,
                regions=regions,
                overlap_threshold=overlap_threshold,
            )
        else:
            destination_gt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_gt, destination_gt)
            stats[sequence] = {"input_rows": 0, "kept_rows": 0, "removed_rows": 0}
        source_seqinfo = source_sequence / "seqinfo.ini"
        if source_seqinfo.is_file():
            destination_sequence.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_seqinfo, destination_sequence / "seqinfo.ini")
    return output_root, stats


def _intersection_over_box_area(
    box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    x, y, width, height = box
    rx, ry, rwidth, rheight = region
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(x + width, rx + rwidth) - max(x, rx))
    intersection_height = max(0.0, min(y + height, ry + rheight) - max(y, ry))
    return intersection_width * intersection_height / (width * height)


def stage_scaled_mot_ground_truth(
    *,
    gt_root: Path,
    sequences: list[str],
    output_root: Path,
    scale_x: float,
    scale_y: float,
) -> Path:
    """Stage GT with x/y/width/height scaled into prediction coordinates."""
    for name, value in (("scale_x", scale_x), ("scale_y", scale_y)):
        if not math.isfinite(value) or value <= 0.0:
            raise MotTransformError(f"{name} must be finite and positive.")
    for sequence in sequences:
        source_sequence = gt_root / sequence
        source_gt = source_sequence / "gt" / "gt.txt"
        if not source_gt.is_file():
            raise MotTransformError(f"MOT ground truth does not exist: {source_gt}")
        destination_sequence = output_root / sequence
        destination_gt = destination_sequence / "gt" / "gt.txt"
        destination_gt.parent.mkdir(parents=True, exist_ok=True)
        _scale_mot_file(source_gt, destination_gt, scale_x=scale_x, scale_y=scale_y)
        source_seqinfo = source_sequence / "seqinfo.ini"
        if source_seqinfo.is_file():
            _scale_seqinfo(
                source_seqinfo,
                destination_sequence / "seqinfo.ini",
                scale_x=scale_x,
                scale_y=scale_y,
            )
    return output_root


def _scale_mot_file(
    source: Path,
    destination: Path,
    *,
    scale_x: float,
    scale_y: float,
) -> None:
    with source.open("r", encoding="utf-8", newline="") as source_handle:
        rows = list(csv.reader(source_handle))
    transformed: list[list[str]] = []
    for line_number, row in enumerate(rows, start=1):
        if not row:
            continue
        if len(row) < 6:
            raise MotTransformError(
                f"MOT row {line_number} in {source} has fewer than 6 columns."
            )
        try:
            coordinates = [float(row[index]) for index in range(2, 6)]
        except ValueError as exc:
            raise MotTransformError(
                f"MOT row {line_number} in {source} has invalid coordinates."
            ) from exc
        coordinates[0] *= scale_x
        coordinates[1] *= scale_y
        coordinates[2] *= scale_x
        coordinates[3] *= scale_y
        transformed.append(
            [*row[:2], *(f"{value:.6f}" for value in coordinates), *row[6:]]
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination_handle:
        csv.writer(destination_handle, lineterminator="\n").writerows(transformed)
    temporary.replace(destination)


def _scale_seqinfo(
    source: Path,
    destination: Path,
    *,
    scale_x: float,
    scale_y: float,
) -> None:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(source, encoding="utf-8")
    section = parser["Sequence"] if parser.has_section("Sequence") else None
    if section is None:
        shutil.copy2(source, destination)
        return
    if section.get("imWidth"):
        section["imWidth"] = str(round(float(section["imWidth"]) * scale_x))
    if section.get("imHeight"):
        section["imHeight"] = str(round(float(section["imHeight"]) * scale_y))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        parser.write(handle, space_around_delimiters=False)
    temporary.replace(destination)
