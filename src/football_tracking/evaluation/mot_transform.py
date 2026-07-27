"""Coordinate-safe staging utilities for normalized MOT ground truth."""

from __future__ import annotations

import configparser
import csv
import math
import shutil
from pathlib import Path


class MotTransformError(ValueError):
    """Raised when a requested MOT coordinate transform is invalid."""


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
