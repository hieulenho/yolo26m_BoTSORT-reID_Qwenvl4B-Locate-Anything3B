"""Import keremberke/football-object-detection into the project YOLO layout."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SPLITS = {"train": "train", "valid": "val", "test": "test"}
PLAYER_CLASS_NAME = "player"


def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _clip_bbox_xywh(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = bbox
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    return x1, y1, clipped_width, clipped_height


def _to_yolo_line(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str | None:
    clipped = _clip_bbox_xywh(bbox, image_width=image_width, image_height=image_height)
    if clipped is None:
        return None
    x, y, width, height = clipped
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    values = (x_center, y_center, normalized_width, normalized_height)
    if any(value < 0.0 or value > 1.0 for value in values):
        return None
    return "0 " + " ".join(f"{value:.6f}" for value in values)


def _load_annotations(split_dir: Path) -> dict[str, Any]:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def _convert_split(
    raw_root: Path, output_root: Path, source_split: str, target_split: str
) -> dict[str, Any]:
    split_dir = raw_root / source_split
    annotations = _load_annotations(split_dir)
    category_by_id = {category["id"]: category["name"] for category in annotations["categories"]}
    player_category_ids = {
        category_id
        for category_id, category_name in category_by_id.items()
        if category_name == PLAYER_CLASS_NAME
    }
    if not player_category_ids:
        raise ValueError(f"No {PLAYER_CLASS_NAME!r} category found in {split_dir}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ignored_annotations = 0
    for annotation in annotations["annotations"]:
        if annotation["category_id"] in player_category_ids:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        else:
            ignored_annotations += 1

    image_dir = output_root / "images" / target_split
    label_dir = output_root / "labels" / target_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "labels": 0,
        "player_annotations": 0,
        "ignored_annotations": ignored_annotations,
        "invalid_player_annotations": 0,
        "link_methods": {},
    }
    for image in annotations["images"]:
        file_name = str(image["file_name"])
        source_image = split_dir / file_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing image: {source_image}")
        destination_image = image_dir / file_name
        method = _link_or_copy(source_image, destination_image)
        stats["link_methods"][method] = stats["link_methods"].get(method, 0) + 1

        lines = []
        for annotation in annotations_by_image[int(image["id"])]:
            line = _to_yolo_line(
                [float(value) for value in annotation["bbox"]],
                image_width=int(image["width"]),
                image_height=int(image["height"]),
            )
            if line is None:
                stats["invalid_player_annotations"] += 1
                continue
            lines.append(line)
        (label_dir / Path(file_name).with_suffix(".txt").name).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        stats["images"] += 1
        stats["labels"] += 1
        stats["player_annotations"] += len(lines)
    return stats


def import_dataset(raw_root: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite:
        for relative_path in ("images", "labels", "dataset.yaml", "manifest.json"):
            _reset_path(output_root / relative_path)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "dataset.yaml").exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=false: {output_root}")

    split_stats = {
        target_split: _convert_split(raw_root, output_root, source_split, target_split)
        for source_split, target_split in SPLITS.items()
    }
    dataset_yaml = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "player"},
        "nc": 1,
    }
    manifest = {
        "source": "keremberke/football-object-detection",
        "source_url": "https://huggingface.co/datasets/keremberke/football-object-detection",
        "source_license": "CC BY 4.0",
        "target_classes": {"player": 0},
        "ignored_source_classes": ["football"],
        "splits": split_stats,
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/keremberke_football_object_detection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/yolo"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = import_dataset(args.raw_root, args.output_root, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
