"""Build a leakage-safe YOLO dataset from normalized MOT sequences."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--train-sequence", action="append", required=True)
    parser.add_argument("--val-sequence", action="append", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        payload = prepare_mot_yolo_split(
            normalized_root=args.normalized_root,
            media_root=args.media_root,
            train_sequences=args.train_sequence,
            val_sequences=args.val_sequence,
            class_name=args.class_name,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def prepare_mot_yolo_split(
    *,
    normalized_root: Path,
    media_root: Path,
    train_sequences: list[str],
    val_sequences: list[str],
    class_name: str,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    normalized = normalized_root.resolve()
    media = media_root.resolve()
    output = output_dir.resolve()
    overlap = set(train_sequences) & set(val_sequences)
    if overlap:
        raise ValueError(f"Train/val sequence leakage: {sorted(overlap)}")
    if output.exists():
        if not overwrite:
            raise ValueError(f"Output exists and overwrite=false: {output}")
        shutil.rmtree(output)

    split_rows: dict[str, dict[str, object]] = {}
    for split, sequences in (("train", train_sequences), ("val", val_sequences)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_count = 0
        label_count = 0
        box_count = 0
        for sequence in sequences:
            source_dir = media / sequence
            gt_path = normalized / sequence / "gt" / "gt.txt"
            if not source_dir.is_dir() or not gt_path.is_file():
                raise ValueError(f"Missing sequence media or GT: {sequence}")
            rows_by_frame = _read_mot_rows(gt_path)
            images = sorted(path for path in source_dir.iterdir() if path.is_file())
            for frame_index, source_image in enumerate(images, start=1):
                image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Unreadable image: {source_image}")
                height, width = image.shape[:2]
                stem = f"{sequence}_{frame_index:06d}"
                destination_image = image_dir / f"{stem}.png"
                image_for_model = _three_channel_uint8(image)
                if not cv2.imwrite(str(destination_image), image_for_model):
                    raise OSError(f"Could not write normalized image: {destination_image}")
                labels = [
                    _yolo_line(row, width=width, height=height)
                    for row in rows_by_frame.get(frame_index, [])
                ]
                (label_dir / f"{stem}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )
                image_count += 1
                label_count += 1
                box_count += len(labels)
        split_rows[split] = {
            "sequences": sequences,
            "image_count": image_count,
            "label_count": label_count,
            "box_count": box_count,
        }

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output),
                "train": "images/train",
                "val": "images/val",
                "names": {0: class_name.strip()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "normalized_root": str(normalized),
        "media_root": str(media),
        "dataset_yaml": str(dataset_yaml),
        "class_name": class_name.strip(),
        "train_val_sequence_overlap": sorted(overlap),
        "splits": split_rows,
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _read_mot_rows(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    rows: defaultdict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        columns = raw_line.split(",")
        if len(columns) < 6:
            raise ValueError(f"Malformed MOT row at {path}:{line_number}")
        frame = int(float(columns[0]))
        rows[frame].append(tuple(float(value) for value in columns[2:6]))
    return dict(rows)


def _yolo_line(
    row: tuple[float, float, float, float], *, width: int, height: int
) -> str:
    x, y, box_width, box_height = row
    center_x = (x + box_width / 2.0) / width
    center_y = (y + box_height / 2.0) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    values = (center_x, center_y, normalized_width, normalized_height)
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"Box outside image bounds after normalization: {row}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def _three_channel_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        minimum = float(image.min())
        maximum = float(image.max())
        scale = 255.0 / max(maximum - minimum, 1.0)
        image = np.clip((image.astype(np.float32) - minimum) * scale, 0, 255).astype(
            np.uint8
        )
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported source image shape: {image.shape}")
    return image


if __name__ == "__main__":
    raise SystemExit(main())
